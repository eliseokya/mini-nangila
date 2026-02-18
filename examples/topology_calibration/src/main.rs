use nangila_core::TopologyMask;
use nangila_math::FixedPointBuffer;
use rand::Rng;
use std::fs;

/// Simulate gradient collection for topology calibration
/// 
/// In a real scenario, this would be actual gradients from training.
/// Here we simulate:
/// - 12 layers (like a small transformer)
/// - 100 calibration steps
/// - Varying gradient variance per layer
fn main() {
    println!("=== Topology Calibration Example ===\n");
    
    // Configuration
    let num_layers = 12;
    let calibration_steps = 100;
    let layer_size = 1024; // Parameters per layer
    
    println!("Configuration:");
    println!("  Layers: {}", num_layers);
    println!("  Calibration Steps: {}", calibration_steps);
    println!("  Layer Size: {} parameters\n", layer_size);
    
    // Simulate gradient collection
    println!("Collecting gradient statistics...");
    let layer_variances = collect_gradient_statistics(num_layers, layer_size, calibration_steps);
    
    // Display statistics
    println!("\nLayer Variance Statistics:");
    for (i, &var) in layer_variances.iter().enumerate() {
        println!("  Layer {:2}: variance = {:.6}", i, var);
    }
    
    // Generate masks with different drop percentages
    println!("\n=== Generating Topology Masks ===\n");
    
    let drop_percentages = vec![0.0, 0.2, 0.3, 0.4, 0.5];
    
    for &drop_pct in &drop_percentages {
        let mask = TopologyMask::from_variance_threshold(&layer_variances, drop_pct);
        
        println!("Drop Percentage: {:.0}%", drop_pct * 100.0);
        println!("  Drivers: {} layers {:?}", 
                 mask.driver_indices.len(), 
                 mask.driver_indices);
        println!("  Passengers: {} layers {:?}", 
                 mask.passenger_indices.len(), 
                 mask.passenger_indices);
        println!("  Compression Factor: {:.2}×", mask.compression_factor());
        println!("  Threshold Variance: {:.6}", mask.threshold);
        
        // Estimate bandwidth savings
        let original_size = num_layers * layer_size * 4; // FP32
        let compressed_size = mask.driver_indices.len() * layer_size * 4;
        let bandwidth_reduction = original_size as f32 / compressed_size as f32;
        println!("  Bandwidth Reduction: {:.2}× (topology only)", bandwidth_reduction);
        println!();
    }
    
    // Save recommended mask (30% drop)
    let recommended_mask = TopologyMask::from_variance_threshold(&layer_variances, 0.3);
    let mask_bytes = recommended_mask.to_bytes().unwrap();
    
    fs::write("topology_mask.bin", &mask_bytes).expect("Failed to write mask");
    println!("✓ Saved recommended mask (30% drop) to topology_mask.bin");
    println!("  Size: {} bytes", mask_bytes.len());
    
    // Verify serialization
    let loaded_mask = TopologyMask::from_bytes(&mask_bytes).unwrap();
    assert_eq!(recommended_mask.driver_indices, loaded_mask.driver_indices);
    println!("✓ Verified mask serialization\n");
    
    // Demonstrate usage in training loop
    println!("=== Example Training Loop Usage ===\n");
    demonstrate_training_usage(&recommended_mask, layer_size);
}

/// Simulate collecting gradient statistics over multiple training steps
fn collect_gradient_statistics(
    num_layers: usize, 
    layer_size: usize, 
    steps: usize
) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    
    // Simulate different layer behaviors:
    // - Early layers (0-3): High variance (learning features)
    // - Middle layers (4-7): Medium variance
    // - Late layers (8-11): Low variance (correlated with middle)
    
    let mut layer_gradients: Vec<Vec<f32>> = vec![vec![]; num_layers];
    
    for step in 0..steps {
        for layer_id in 0..num_layers {
            // Generate gradient with layer-specific characteristics
            let base_variance = if layer_id < 4 {
                0.5 // High variance (important layers)
            } else if layer_id < 8 {
                0.3 // Medium variance
            } else {
                0.1 // Low variance (passenger candidates)
            };
            
            // Add temporal correlation (gradients evolve smoothly)
            let temporal_noise = (step as f32 * 0.1).sin() * 0.1;
            
            // Sample gradient magnitude (simplified)
            let gradient_norm: f32 = rng.gen_range(0.0..1.0) * base_variance + temporal_noise;
            
            layer_gradients[layer_id].push(gradient_norm);
        }
    }
    
    // Compute variance for each layer
    layer_gradients.iter().map(|grads| {
        let mean = grads.iter().sum::<f32>() / grads.len() as f32;
        let variance = grads.iter()
            .map(|&g| (g - mean).powi(2))
            .sum::<f32>() / grads.len() as f32;
        variance
    }).collect()
}

/// Demonstrate how to use the mask in a training loop
fn demonstrate_training_usage(mask: &TopologyMask, layer_size: usize) {
    let mut rng = rand::thread_rng();
    
    println!("Simulating 5 training steps with topology masking:\n");
    
    for step in 0..5 {
        let mut total_transmitted = 0;
        let mut total_skipped = 0;
        
        println!("Step {}:", step);
        
        for layer_id in 0..mask.total_layers {
            // Generate fake gradient
            let gradient_data: Vec<f32> = (0..layer_size)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            let gradient = FixedPointBuffer::from_f32(&gradient_data);
            
            if mask.is_driver(layer_id) {
                // Transmit this layer
                let size_bytes = gradient.len() * 4; // FP32
                total_transmitted += size_bytes;
                println!("  Layer {:2}: TRANSMIT ({} bytes)", layer_id, size_bytes);
            } else {
                // Skip this layer (Passenger)
                let size_bytes = gradient.len() * 4;
                total_skipped += size_bytes;
                println!("  Layer {:2}: SKIP (saved {} bytes)", layer_id, size_bytes);
            }
        }
        
        let total_size = total_transmitted + total_skipped;
        let compression = total_size as f32 / total_transmitted as f32;
        
        println!("  Total: {} KB transmitted, {} KB skipped", 
                 total_transmitted / 1024, 
                 total_skipped / 1024);
        println!("  Compression: {:.2}× (topology only)\n", compression);
    }
    
    println!("Note: In practice, Drivers would also be compressed with");
    println!("      predictive coding + quantization for additional 8-20× reduction.");
}
