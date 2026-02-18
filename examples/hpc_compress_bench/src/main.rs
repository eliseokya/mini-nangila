use nangila_hpc::{LinearPredictor, ErrorBoundedQuantizer};
use nangila_core::{Predictor, Quantizer};
use nangila_math::FixedPointBuffer;
use std::time::Instant;

fn main() {
    println!("Running Mini-Nangila HPC Compression Benchmark (Mock LAMMPS)");
    
    // Config
    let num_particles = 100_000;
    let steps = 100;
    let error_bound = 0.001; // Strict scientific tolerance
    
    // Components
    let mut predictor = LinearPredictor::new();
    let quantizer = ErrorBoundedQuantizer::new(error_bound);
    
    // Stats
    let mut total_bytes_raw = 0;
    let mut total_bytes_compressed = 0;
    let mut max_absolute_error = 0.0f32;
    
    let start = Instant::now();
    
    for t in 0..steps {
        // 1. Generate Trajectory (Harmonic Oscillator + slight noise)
        // x(t) = A * cos(omega * t)
        let data: Vec<f32> = (0..num_particles)
            .map(|i| {
                let phase = (t as f32) * 0.1 + (i as f32) * 0.01;
                let noise = (i % 100) as f32 * 0.0001;
                (phase).cos() + noise
            })
            .collect();
            
        let state_t = FixedPointBuffer::from_f32(&data);
        total_bytes_raw += num_particles * 4; // f32
        
        // 2. Predict (Linear Extrapolation)
        let pred = predictor.predict().unwrap();
        
        // 3. Residual
        let residual = if pred.is_empty() {
             state_t.clone()
        } else {
             state_t.sub(&pred).unwrap()
        };
        
        // 4. Quantize (Error Bounded)
        let (compressed, scale) = quantizer.quantize(&residual);
        total_bytes_compressed += compressed.len() + 4; // bytes + scale
        
        // 5. Reconstruct & Update
        let dequantized = quantizer.dequantize(&compressed, scale);
        let reconstructed = if pred.is_empty() {
             dequantized
        } else {
             pred.add(&dequantized).unwrap()
        };
        
        // Verify Error
        let original_f32 = state_t.to_f32();
        let recon_f32 = reconstructed.to_f32();
        
        for i in 0..original_f32.len() {
            let err = (original_f32[i] - recon_f32[i]).abs();
            if err > max_absolute_error {
                max_absolute_error = err;
            }
        }
        
        predictor.update(&reconstructed).unwrap();
        
        if t % 10 == 0 {
            let ratio = total_bytes_raw as f32 / total_bytes_compressed as f32;
            println!("Step {}: Ratio {:.2}x | Max Error {:.6} (Bound: {:.6})", t, ratio, max_absolute_error, error_bound);
        }
    }
    
    let duration = start.elapsed();
    let throughput = (total_bytes_raw as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
    
    println!("--------------------------------------------------");
    println!("Benchmark Complete");
    println!("Total Raw Data: {:.2} MB", total_bytes_raw as f64 / 1024.0 / 1024.0);
    println!("Total Compressed: {:.2} MB", total_bytes_compressed as f64 / 1024.0 / 1024.0);
    println!("Final Compression Ratio: {:.2}x", total_bytes_raw as f32 / total_bytes_compressed as f32);
    println!("Max Absolute Error: {:.6}", max_absolute_error);
    println!("Throughput: {:.2} MB/s", throughput);
    
    if max_absolute_error <= error_bound + 1e-5 { // Allow small float drift
        println!("SUCCESS: Error bound respected.");
    } else {
        println!("FAILURE: Error bound violated!");
    }
}
