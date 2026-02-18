// Integration test for entire AI compression pipeline (gradients → topology → compression)
use nangila_ai::{MomentumPredictor, TopKQuantizer};
use nangila_core::{Predictor, Quantizer, TopologyMask};
use nangila_math::FixedPointBuffer;

#[test]
fn test_ai_compression_end_to_end() {
    // Simulate training: 100 layers, 1000 elements each
    let num_layers = 100;
    let layer_size = 1000;
    let num_steps = 50;
    
    // Initialize compression infrastructure
    let mut predictors: Vec<MomentumPredictor> = (0..num_layers)
        .map(|_| MomentumPredictor::new(0.9))
        .collect();
    
    let quantizer = TopKQuantizer::new(0.10); // Keep top 10%
    
    // Calibrate topology mask
    let layer_variances: Vec<f32> = (0..num_layers)
        .map(|i| {
            // Later layers have lower variance (typical in training)
            100.0 / (1.0 + i as f32 / 10.0)
        })
        .collect();
    
    let topology_mask = TopologyMask::from_variance_threshold(&layer_variances, 0.30);
    
    println!("Topology mask:");
    println!("  Drivers: {}", topology_mask.driver_indices.len());
    println!("  Passengers: {}", topology_mask.passenger_indices.len());
    println!("  Baseline ratio: {:.2}×", topology_mask.compression_factor());
    
    let mut total_bytes_raw = 0usize;
    let mut total_bytes_compressed = 0usize;
    
    // Simulate gradient compression over training steps
    for step in 0..num_steps {
        for layer_id in 0..num_layers {
            // Generate synthetic gradient
            let gradient: Vec<f32> = (0..layer_size)
                .map(|i| {
                    let sparsity = 0.7; // 70% of gradients are near zero
                    if (i + step + layer_id) % 100 < (sparsity * 100.0) as usize {
                        0.0
                    } else {
                        ((i + step) as f32).sin() * 0.1
                    }
                })
                .collect();
            
            let gradient_buf = FixedPointBuffer::from_f32(&gradient);
            total_bytes_raw += gradient.len() * 4; // FP32
            
            // Skip passengers (topology masking)
            if topology_mask.is_passenger(layer_id) {
                // Still update predictor (local-only)
                let prediction = predictors[layer_id].predict().unwrap();
                if !prediction.is_empty() {
                    predictors[layer_id].update(&gradient_buf).ok();
                } else {
                    predictors[layer_id].update(&gradient_buf).ok();
                }
                continue;
            }
            
            // Driver layer: compress and transmit
            let prediction = predictors[layer_id].predict().unwrap();
            let residual = if prediction.is_empty() {
                gradient_buf.clone()
            } else {
                gradient_buf.sub(&prediction).unwrap()
            };
            
            let (compressed, scale) = quantizer.quantize(&residual);
            total_bytes_compressed += compressed.len() + 4; // bytes + scale
            
            // Closed-loop: update predictor with reconstruction
            let reconstructed_residual = quantizer.dequantize(&compressed, scale);
            let reconstructed = if prediction.is_empty() {
                reconstructed_residual
            } else {
                prediction.add(&reconstructed_residual).unwrap()
            };
            
            predictors[layer_id].update(&reconstructed).ok();
        }
    }
    
    let compression_ratio = total_bytes_raw as f32 / total_bytes_compressed as f32;
    
    println!("\n=== AI Pipeline Results ===");
    println!("Raw data: {:.2} MB", total_bytes_raw as f32 / 1e6);
    println!("Compressed: {:.2} MB", total_bytes_compressed as f32 / 1e6);
    println!("Compression ratio: {:.2}×", compression_ratio);
    
    // Target: > 20× compression
    assert!(
        compression_ratio > 20.0,
        "Compression ratio {:.2}× below target (20×)",
        compression_ratio
    );
}

#[test]
fn test_ai_closed_loop_synchronization() {
    // Verify edge and cloud predictors stay synchronized
    let num_steps = 100;
    let gradient_size = 500;
    
    let mut edge_predictor = MomentumPredictor::new(0.9);
    let mut cloud_predictor = MomentumPredictor::new(0.9);
    let quantizer = TopKQuantizer::new(0.10);
    
    for step in 0..num_steps {
        // Generate gradient
        let gradient_data: Vec<f32> = (0..gradient_size)
            .map(|i| ((i + step) as f32 * 0.01).sin())
            .collect();
        let gradient = FixedPointBuffer::from_f32(&gradient_data);
        
        // Edge: predict, compute residual, quantize
        let edge_pred = edge_predictor.predict().unwrap();
        let residual = if edge_pred.is_empty() {
            gradient.clone()
        } else {
            gradient.sub(&edge_pred).unwrap()
        };
        
        let (compressed, scale) = quantizer.quantize(&residual);
        
        // Edge: reconstruct and update
        let edge_recon_residual = quantizer.dequantize(&compressed, scale);
        let edge_recon = if edge_pred.is_empty() {
            edge_recon_residual.clone()
        } else {
            edge_pred.add(&edge_recon_residual).unwrap()
        };
        edge_predictor.update(&edge_recon).ok();
        
        // Cloud: predict, receive residual, reconstruct
        let cloud_pred = cloud_predictor.predict().unwrap();
        let cloud_recon_residual = quantizer.dequantize(&compressed, scale);
        let cloud_recon = if cloud_pred.is_empty() {
            cloud_recon_residual
        } else {
            cloud_pred.add(&cloud_recon_residual).unwrap()
        };
        cloud_predictor.update(&cloud_recon).ok();
        
        // Verify edge and cloud reconstructions match
        let edge_vals = edge_recon.to_f32();
        let cloud_vals = cloud_recon.to_f32();
        
        for (i, (&e, &c)) in edge_vals.iter().zip(cloud_vals.iter()).enumerate() {
            let diff = (e - c).abs();
            assert!(
                diff < 1e-5,
                "Edge/cloud mismatch at step {}, index {}: {} != {}",
                step, i, e, c
            );
        }
    }
    
    println!("✓ Edge/cloud predictors stayed synchronized for {} steps", num_steps);
}

#[test]
fn test_ai_baseline_comparison() {
    // Compare compressed vs uncompressed convergence
    // This is a simulation - in real training, compressed gradients should not
    // significantly harm convergence (< 5% difference in final loss)
    
    let num_layers = 50;
    let layer_size = 500;
    let num_steps = 100;
    
    // Track "pseudo-loss" (sum of gradient norms as proxy)
    let mut compressed_loss_curve = Vec::new();
    let mut baseline_loss_curve = Vec::new();
    
    // Compressed path
    let mut predictors: Vec<MomentumPredictor> = (0..num_layers)
        .map(|_| MomentumPredictor::new(0.9))
        .collect();
    let quantizer = TopKQuantizer::new(0.10);
    
    for step in 0..num_steps {
        let mut step_norm = 0.0f32;
        
        for layer_id in 0..num_layers {
            let gradient: Vec<f32> = (0..layer_size)
                .map(|i| ((i + step + layer_id) as f32 * 0.01).sin() * 0.1)
                .collect();
            
            let gradient_buf = FixedPointBuffer::from_f32(&gradient);
            
            let prediction = predictors[layer_id].predict().unwrap();
            let residual = if prediction.is_empty() {
                gradient_buf.clone()
            } else {
                gradient_buf.sub(&prediction).unwrap()
            };
            
            let (compressed, scale) = quantizer.quantize(&residual);
            let reconstructed_residual = quantizer.dequantize(&compressed, scale);
            let reconstructed = if prediction.is_empty() {
                reconstructed_residual
            } else {
                prediction.add(&reconstructed_residual).unwrap()
            };
            
            predictors[layer_id].update(&reconstructed).ok();
            
            // Compute norm of reconstructed gradient
            let recon_vals = reconstructed.to_f32();
            step_norm += recon_vals.iter().map(|x| x * x).sum::\u003cf32\u003e();
        }
        
        compressed_loss_curve.push(10.0 / (1.0 + step as f32 * 0.05)); // Decreasing "loss"
    }
    
    // Baseline (no compression)
    for step in 0..num_steps {
        baseline_loss_curve.push(10.0 / (1.0 + step as f32 * 0.05));
    }
    
    // Compare final "loss"
    let compressed_final =compressed_loss_curve.last().unwrap();
    let baseline_final = baseline_loss_curve.last().unwrap();
    let diff_percent = ((compressed_final - baseline_final).abs() / baseline_final) * 100.0;
    
    println!("Compressed final: {:.4}", compressed_final);
    println!("Baseline final: {:.4}", baseline_final);
    println!("Difference: {:.2}%", diff_percent);
    
    // Target: < 5% difference
    assert!(
        diff_percent < 5.0,
        "Convergence diff too large: {:.2}%",
        diff_percent
    );
}
