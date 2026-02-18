use nangila_ai::{MomentumPredictor, StochasticQuantizer};
use nangila_core::{Predictor, Quantizer};
use nangila_math::FixedPointBuffer;
use std::time::Instant;

fn main() {
    println!("Running Mini-Nangila Throughput Benchmark (AI Mode)");
    
    // Config
    let batch_size = 100;
    let vector_dim = 1024 * 1024; // 1M params ~ NanoGPT small layer
    let steps = 100;
    
    // Components
    let mut predictor = MomentumPredictor::new(0.9);
    let quantizer = StochasticQuantizer::new(42);
    
    // Generate synthetic gradient stream (simulating training dynamics)
    // g_t = sin(t/10) + noise
    let mut total_bytes_raw = 0;
    let mut total_bytes_compressed = 0;
    
    let start = Instant::now();
    
    for t in 0..steps {
        // 1. Generate Gradient
        let data: Vec<f32> = (0..vector_dim)
            .map(|i| {
                let phase = (t as f32) * 0.1;
                let noise = (i % 100) as f32 * 0.001;
                (phase + noise).sin()
            })
            .collect();
            
        let g_t = FixedPointBuffer::from_f32(&data);
        total_bytes_raw += vector_dim * 4; // f32
        
        // 2. Predict
        let pred = predictor.predict().unwrap();
        
        // 3. Residual
        let residual = if pred.is_empty() {
             g_t.clone()
        } else {
             g_t.sub(&pred).unwrap()
        };
        
        // 4. Quantize
        let (compressed, scale) = quantizer.quantize(&residual);
        total_bytes_compressed += compressed.len() + 4; // bytes + scale(f32)
        
        // 5. Update Predictor (Feedback Loop)
        let dequantized = quantizer.dequantize(&compressed, scale);
        let reconstructed = if pred.is_empty() {
             dequantized
        } else {
             pred.add(&dequantized).unwrap()
        };
        predictor.update(&reconstructed).unwrap();
        
        if t % 10 == 0 {
            let ratio = total_bytes_raw as f32 / total_bytes_compressed as f32;
            println!("Step {}: Compression Ratio {:.2}x", t, ratio);
        }
    }
    
    let duration = start.elapsed();
    let throughput = (total_bytes_raw as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
    
    println!("--------------------------------------------------");
    println!("Benchmark Complete");
    println!("Total Raw Data: {:.2} MB", total_bytes_raw as f64 / 1024.0 / 1024.0);
    println!("Total Compressed: {:.2} MB", total_bytes_compressed as f64 / 1024.0 / 1024.0);
    println!("Final Compression Ratio: {:.2}x (Target: >8x for INT4)", total_bytes_raw as f32 / total_bytes_compressed as f32);
    println!("Throughput (Raw): {:.2} MB/s", throughput);
}
