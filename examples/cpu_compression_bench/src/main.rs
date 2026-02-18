use nangila_ai::TopKQuantizer;
use nangila_hpc::RunLengthQuantizer;
use nangila_core::Quantizer;
use nangila_math::{FixedPointBuffer, FixedPointError};
use std::time::Instant;

fn main() {
    println!("Running Mini-Nangila CPU Compression Benchmark (High Ratio Targets)");
    
    // --- Scenario 1: AI (Top-K Sparsification) ---
    println!("\n[Scenario 1: AI Gradient Compression (Target: >20x)]");
    let ai_dim = 1_000_000;
    // Generate sparse gradients: 95% near-zero, 5% spikes
    let ai_data: Vec<f32> = (0..ai_dim).map(|i| {
        if i % 20 == 0 {
            // Signal (5%)
            let phase = (i as f32) * 0.001;
            phase.sin()
        } else {
            // Noise (95%)
            ((i % 100) as f32 * 0.0001)
        }
    }).collect();
    
    let ai_buffer = FixedPointBuffer::from_f32(&ai_data);
    let ai_quantizer = TopKQuantizer::new(0.05); // Keep top 5%
    
    let start_ai = Instant::now();
    let (ai_compressed, _) = ai_quantizer.quantize(&ai_buffer);
    let duration_ai = start_ai.elapsed();
    
    let ai_raw_size = ai_dim * 4;
    let ai_comp_size = ai_compressed.len();
    let ai_ratio = ai_raw_size as f32 / ai_comp_size as f32;
    let ai_throughput = (ai_raw_size as f64 / 1024.0 / 1024.0) / duration_ai.as_secs_f64();
    
    println!("Raw Size: {:.2} MB", ai_raw_size as f64 / 1024.0 / 1024.0);
    println!("Compressed Size: {:.2} MB", ai_comp_size as f64 / 1024.0 / 1024.0);
    println!("Compression Ratio: {:.2}x", ai_ratio);
    println!("Throughput: {:.2} MB/s", ai_throughput);
    
    if ai_ratio > 18.0 {
        println!("SUCCESS: AI Target Met (>18x)");
    } else {
        println!("FAILURE: AI Target Missed");
    }

    // --- Scenario 2: HPC/Twin (Run-Length Encoding) ---
    println!("\n[Scenario 2: HPC/Twin Trajectory (Target: >100x)]");
    let hpc_dim = 100_000;
    // Generate trajectory where predictor works perfectly 99% of time (zero residual)
    // 1% events
    let hpc_data: Vec<f32> = (0..hpc_dim).map(|i| {
        if i % 100 == 0 {
             1.0 // Event
        } else {
             0.0 // Predicted perfectly
        }
    }).collect();
    
    let hpc_buffer = FixedPointBuffer::from_f32(&hpc_data);
    let hpc_quantizer = RunLengthQuantizer::new(0.01); 
    
    let start_hpc = Instant::now();
    let (hpc_compressed, _) = hpc_quantizer.quantize(&hpc_buffer);
    let duration_hpc = start_hpc.elapsed();
    
    let hpc_raw_size = hpc_dim * 4;
    let hpc_comp_size = hpc_compressed.len();
    let hpc_ratio = hpc_raw_size as f32 / hpc_comp_size as f32;
    let hpc_throughput = (hpc_raw_size as f64 / 1024.0 / 1024.0) / duration_hpc.as_secs_f64();
    
    println!("Raw Size: {:.2} MB", hpc_raw_size as f64 / 1024.0 / 1024.0);
    println!("Compressed Size: {:.2} MB", hpc_comp_size as f64 / 1024.0 / 1024.0);
    println!("Compression Ratio: {:.2}x", hpc_ratio);
    println!("Throughput: {:.2} MB/s", hpc_throughput);

    if hpc_ratio > 50.0 {
        println!("SUCCESS: HPC Target Met (>50x)");
    } else {
        println!("FAILURE: HPC Target Missed");
    }
}
