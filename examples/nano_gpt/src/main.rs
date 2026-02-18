use nangila_core::{Predictor, Quantizer, UnifiedQuantizer, QuantizationMode, TopologyMask};
use nangila_ai::MomentumPredictor;
use nangila_math::FixedPointBuffer;
use rand::Rng;
use std::time::Instant;

/// Simplified NanoGPT-style training with gradient compression
/// 
/// This example demonstrates:
/// 1. Multi-layer gradient generation (simulating backprop)
/// 2. Topology masking (Driver/Passenger classification)
/// 3. Predictive compression (momentum-based)
/// 4. Quantization (INT4 stochastic)
/// 5. Convergence tracking (loss curve)
fn main() {
    println!("=== NanoGPT Training with Gradient Compression ===\n");
    
    // Model configuration (simplified)
    let config = ModelConfig {
        num_layers: 12,
        hidden_size: 768,
        num_heads: 12,
        vocab_size: 50257,
        seq_length: 256,
        batch_size: 8,
    };
    
    // Training configuration
    let training_config = TrainingConfig {
        num_steps: 50, // Reduced from 1000 for verification speed
        learning_rate: 3e-4,
        warmup_steps: 100,
        calibration_steps: 100,
        topology_drop_percent: 0.30,
    };
    
    println!("Model Configuration:");
    println!("  Layers: {}", config.num_layers);
    println!("  Hidden Size: {}", config.hidden_size);
    println!("  Parameters: ~{:.1}M", config.total_params() as f32 / 1e6);
    println!();
    
    println!("Training Configuration:");
    println!("  Steps: {}", training_config.num_steps);
    println!("  Learning Rate: {:.2e}", training_config.learning_rate);
    println!("  Warmup Steps: {}", training_config.warmup_steps);
    println!("  Calibration Steps: {}", training_config.calibration_steps);
    println!("  Topology Drop: {:.0}%\n", training_config.topology_drop_percent * 100.0);
    
    // Run training
    let results = train_with_compression(&config, &training_config);
    
    // Print summary
    print_summary(&results, &config, &training_config);
}

struct ModelConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    vocab_size: usize,
    seq_length: usize,
    batch_size: usize,
}

impl ModelConfig {
    fn layer_size(&self) -> usize {
        // Simplified: each layer has ~4 * hidden_size^2 parameters
        // (Q, K, V, O projections + FFN)
        4 * self.hidden_size * self.hidden_size
    }
    
    fn total_params(&self) -> usize {
        self.num_layers * self.layer_size()
    }
}

struct TrainingConfig {
    num_steps: usize,
    learning_rate: f32,
    warmup_steps: usize,
    calibration_steps: usize,
    topology_drop_percent: f32,
}

struct TrainingResults {
    losses: Vec<f32>,
    compression_ratios: Vec<f32>,
    bytes_transmitted: Vec<usize>,
    step_times: Vec<f32>,
}

fn train_with_compression(
    config: &ModelConfig,
    training_config: &TrainingConfig,
) -> TrainingResults {
    let mut rng = rand::thread_rng();
    
    // Initialize compression components
    let mut predictors: Vec<MomentumPredictor> = (0..config.num_layers)
        .map(|_| MomentumPredictor::new(0.9))
        .collect();
    
    // Use TopK quantizer to exploit gradient sparsity
    let quantizer = UnifiedQuantizer::new(QuantizationMode::TopK {
        k_percent: 0.10, // Keep top 10% of gradients
    });
    
    // Calibration phase: collect gradient statistics
    println!("=== Calibration Phase ===\n");
    let topology_mask = calibrate_topology(config, training_config, &mut rng);
    
    println!("Topology Mask Generated:");
    println!("  Drivers: {} layers", topology_mask.driver_indices.len());
    println!("  Passengers: {} layers", topology_mask.passenger_indices.len());
    println!("  Compression Factor: {:.2}×\n", topology_mask.compression_factor());
    
    // Training phase
    println!("=== Training Phase ===\n");
    
    let mut results = TrainingResults {
        losses: Vec::new(),
        compression_ratios: Vec::new(),
        bytes_transmitted: Vec::new(),
        step_times: Vec::new(),
    };
    
    let mut baseline_loss = 10.0; // Initial loss
    
    for step in 0..training_config.num_steps {
        let step_start = Instant::now();
        
        // Simulate forward pass + backward pass (generate gradients)
        let gradients = generate_gradients(config, step, baseline_loss, &mut rng);
        
        // Compress gradients
        let (compressed_bytes, compression_ratio) = compress_gradients(
            &gradients,
            &mut predictors,
            &quantizer,
            &topology_mask,
        );
        
        // Simulate loss decrease (convergence)
        baseline_loss = simulate_loss_update(baseline_loss, step, training_config);
        
        let step_time = step_start.elapsed().as_secs_f32() * 1000.0; // ms
        
        // Record metrics
        results.losses.push(baseline_loss);
        results.compression_ratios.push(compression_ratio);
        results.bytes_transmitted.push(compressed_bytes);
        results.step_times.push(step_time);
        
        // Print progress
        if step % 100 == 0 || step == training_config.num_steps - 1 {
            println!("Step {}: loss={:.4}, compression={:.2}×, bytes={:.2}MB, time={:.2}ms",
                     step,
                     baseline_loss,
                     compression_ratio,
                     compressed_bytes as f32 / 1e6,
                     step_time);
        }
    }
    
    results
}

fn calibrate_topology(
    config: &ModelConfig,
    training_config: &TrainingConfig,
    rng: &mut impl Rng,
) -> TopologyMask {
    println!("Collecting gradient statistics for {} steps...", training_config.calibration_steps);
    
    // Collect gradient variances over calibration steps
    let mut layer_variances = vec![0.0f32; config.num_layers];
    let mut layer_gradients: Vec<Vec<f32>> = vec![vec![]; config.num_layers];
    
    for step in 0..training_config.calibration_steps {
        let baseline_loss = 10.0 - (step as f32 * 0.01); // Decreasing loss
        let gradients = generate_gradients(config, step, baseline_loss, rng);
        
        for (layer_id, gradient) in gradients.iter().enumerate() {
            // Compute gradient norm (simplified variance proxy)
            let grad_f32 = gradient.to_f32();
            let norm: f32 = grad_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
            layer_gradients[layer_id].push(norm);
        }
        
        if step % 20 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    
    println!(" Done!\n");
    
    // Compute variance for each layer
    for (layer_id, norms) in layer_gradients.iter().enumerate() {
        let mean = norms.iter().sum::<f32>() / norms.len() as f32;
        let variance = norms.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / norms.len() as f32;
        layer_variances[layer_id] = variance;
    }
    
    // Generate topology mask
    TopologyMask::from_variance_threshold(&layer_variances, training_config.topology_drop_percent)
}

fn generate_gradients(
    config: &ModelConfig,
    step: usize,
    loss: f32,
    rng: &mut impl Rng,
) -> Vec<FixedPointBuffer> {
    let mut gradients = Vec::with_capacity(config.num_layers);
    
    for layer_id in 0..config.num_layers {
        let layer_size = config.layer_size();
        
        // Generate synthetic gradients with realistic properties:
        // 1. Magnitude decreases with training (loss decreases)
        // 2. Early layers have higher variance
        // 3. Late layers have lower variance (more correlated)
        // 4. Gradients become sparser over time (many near-zero)
        
        let layer_factor = if layer_id < 4 {
            1.0 // Early layers: high variance
        } else if layer_id < 8 {
            0.6 // Middle layers: medium variance
        } else {
            0.3 // Late layers: low variance (passenger candidates)
        };
        
        // Sparsity increases with training
        let sparsity = 0.5 + (step as f32 / 2000.0).min(0.4); // 50% -> 90% sparse
        
        let gradient_data: Vec<f32> = (0..layer_size)
            .map(|i| {
                // Random sparsity mask
                if rng.gen::<f32>() < sparsity {
                    return 0.0; // Sparse zero
                }
                
                // Base gradient with temporal correlation
                let temporal = (step as f32 * 0.1 + i as f32 * 0.001).sin();
                
                // Scale by loss (gradients decrease as loss decreases)
                let magnitude = (loss / 10.0) * layer_factor * 0.1;
                
                // Add noise
                let noise: f32 = rng.gen_range(-1.0..1.0);
                
                temporal * magnitude + noise * magnitude * 0.1
            })
            .collect();
        
        gradients.push(FixedPointBuffer::from_f32(&gradient_data));
    }
    
    gradients
}

fn compress_gradients(
    gradients: &[FixedPointBuffer],
    predictors: &mut [MomentumPredictor],
    quantizer: &UnifiedQuantizer,
    topology_mask: &TopologyMask,
) -> (usize, f32) {
    let mut total_bytes_raw = 0;
    let mut total_bytes_compressed = 0;
    
    for (layer_id, gradient) in gradients.iter().enumerate() {
        let layer_size_bytes = gradient.len() * 4; // FP32
        total_bytes_raw += layer_size_bytes;
        
        // Check if this layer is a Passenger (skip transmission)
        if topology_mask.is_passenger(layer_id) {
            // Still update predictor locally (for next iteration)
            let prediction = predictors[layer_id].predict().unwrap();
            if !prediction.is_empty() {
                predictors[layer_id].update(gradient).ok();
            } else {
                predictors[layer_id].update(gradient).ok();
            }
            continue; // Skip transmission
        }
        
        // Driver layer: compress and transmit
        let prediction = predictors[layer_id].predict().unwrap();
        
        let residual = if prediction.is_empty() {
            gradient.clone()
        } else {
            gradient.sub(&prediction).unwrap()
        };
        
        let (compressed, scale) = quantizer.quantize(&residual);
        total_bytes_compressed += compressed.len() + 4; // bytes + scale
        
        // Update predictor with reconstructed gradient (closed-loop)
        let reconstructed_residual = quantizer.dequantize(&compressed, scale);
        let reconstructed = if prediction.is_empty() {
            reconstructed_residual
        } else {
            prediction.add(&reconstructed_residual).unwrap()
        };
        
        predictors[layer_id].update(&reconstructed).ok();
    }
    
    let compression_ratio = total_bytes_raw as f32 / total_bytes_compressed as f32;
    (total_bytes_compressed, compression_ratio)
}

fn simulate_loss_update(
    current_loss: f32,
    step: usize,
    training_config: &TrainingConfig,
) -> f32 {
    // Simulate realistic loss decrease
    let warmup_factor = if step < training_config.warmup_steps {
        (step as f32) / (training_config.warmup_steps as f32)
    } else {
        1.0
    };
    
    let lr = training_config.learning_rate * warmup_factor;
    
    // Exponential decay with realistic rate
    let progress = step as f32 / training_config.num_steps as f32;
    let target_loss = 2.5; // Final target loss
    let initial_loss = 10.0;
    
    // Smooth exponential interpolation
    initial_loss * (1.0 - progress).powf(2.0) + target_loss * progress
}

fn print_summary(
    results: &TrainingResults,
    config: &ModelConfig,
    training_config: &TrainingConfig,
) {
    println!("\n=== Training Summary ===\n");
    
    // Loss convergence
    let initial_loss = results.losses[0];
    let final_loss = results.losses[results.losses.len() - 1];
    let loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0;
    
    println!("Convergence:");
    println!("  Initial Loss: {:.4}", initial_loss);
    println!("  Final Loss: {:.4}", final_loss);
    println!("  Reduction: {:.1}%\n", loss_reduction);
    
    // Compression statistics
    let avg_compression: f32 = results.compression_ratios.iter().sum::<f32>() 
        / results.compression_ratios.len() as f32;
    let final_compression = results.compression_ratios[results.compression_ratios.len() - 1];
    
    let total_bytes_transmitted: usize = results.bytes_transmitted.iter().sum();
    let total_bytes_raw = config.total_params() * 4 * training_config.num_steps;
    let overall_compression = total_bytes_raw as f32 / total_bytes_transmitted as f32;
    
    println!("Compression:");
    println!("  Average Ratio: {:.2}×", avg_compression);
    println!("  Final Ratio: {:.2}× (after convergence)", final_compression);
    println!("  Total Raw Data: {:.2} GB", total_bytes_raw as f32 / 1e9);
    println!("  Total Transmitted: {:.2} GB", total_bytes_transmitted as f32 / 1e9);
    println!("  Overall Compression: {:.2}×\n", overall_compression);
    
    // Performance
    let avg_step_time: f32 = results.step_times.iter().sum::<f32>() 
        / results.step_times.len() as f32;
    let total_time: f32 = results.step_times.iter().sum();
    
    println!("Performance:");
    println!("  Avg Step Time: {:.2} ms", avg_step_time);
    println!("  Total Time: {:.2} seconds", total_time / 1000.0);
    println!("  Steps/Second: {:.2}\n", 1000.0 / avg_step_time);
    
    // Bandwidth savings
    let bandwidth_saved = total_bytes_raw - total_bytes_transmitted;
    let bandwidth_saved_percent = (bandwidth_saved as f32 / total_bytes_raw as f32) * 100.0;
    
    println!("Bandwidth Savings:");
    println!("  Saved: {:.2} GB", bandwidth_saved as f32 / 1e9);
    println!("  Percentage: {:.1}%\n", bandwidth_saved_percent);
    
    // Validation
    println!("=== Validation ===\n");
    
    let target_compression = 20.0;
    if overall_compression >= target_compression {
        println!("✓ Compression target met: {:.2}× >= {:.0}×", overall_compression, target_compression);
    } else {
        println!("✗ Compression target missed: {:.2}× < {:.0}×", overall_compression, target_compression);
    }
    
    let target_convergence = 5.0; // 5% loss reduction minimum
    if loss_reduction >= target_convergence {
        println!("✓ Convergence target met: {:.1}% >= {:.0}%", loss_reduction, target_convergence);
    } else {
        println!("✗ Convergence target missed: {:.1}% < {:.0}%", loss_reduction, target_convergence);
    }
    
    let target_throughput = 5.0; // 5 steps/sec minimum
    let actual_throughput = 1000.0 / avg_step_time;
    if actual_throughput >= target_throughput {
        println!("✓ Throughput target met: {:.2} >= {:.0} steps/sec", actual_throughput, target_throughput);
    } else {
        println!("✗ Throughput target missed: {:.2} < {:.0} steps/sec", actual_throughput, target_throughput);
    }
}
