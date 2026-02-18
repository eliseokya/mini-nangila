use clap::Parser;
use nangila_core::{UnifiedQuantizer, QuantizationMode, Quantizer, Predictor};
use nangila_ai::MomentumPredictor;
use nangila_hpc::{BlockSparseQuantizer, BlockSparseRleQuantizer};
use nangila_math::FixedPointBuffer;
use std::time::Instant;

/// Demonstrate runtime mode switching between AI and HPC quantization
/// 
/// Scenario: A training run that periodically checkpoints to disk
/// - During training: Use Stochastic INT4 (fast, lossy)
/// - During checkpoint: Switch to ErrorBounded (strict guarantees)
#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// Number of training steps
    #[arg(long, default_value_t = 50)]
    num_steps: usize,

    /// Checkpoint interval in steps
    #[arg(long, default_value_t = 10)]
    checkpoint_interval: usize,

    /// Gradient size (number of parameters)
    #[arg(long, default_value_t = 10_000)]
    gradient_size: usize,

    /// Epsilon for checkpoint error-bounded quantization
    #[arg(long, default_value_t = 1e-3)]
    epsilon: f32,

    /// Use error-bounded BlockSparse at checkpoints (otherwise plain error-bounded)
    #[arg(long, default_value_t = true)]
    checkpoint_block_sparse: bool,

    /// Use RLE over BlockSparse masks at checkpoints
    #[arg(long, default_value_t = false)]
    checkpoint_bsrle: bool,

    /// Optional TopK percentage for training (0.0 disables, falls back to INT4)
    #[arg(long, default_value_t = 0.0)]
    train_topk: f32,
}

fn main() {
    let args = Args::parse();
    println!("=== Unified Quantizer Mode Switching Example ===\n");
    
    // Simulate a training scenario
    let num_steps = args.num_steps;
    let checkpoint_interval = args.checkpoint_interval;
    let gradient_size = args.gradient_size;
    
    println!("Scenario: Training with periodic checkpointing");
    println!("  Training steps: {}", num_steps);
    println!("  Checkpoint every: {} steps", checkpoint_interval);
    println!("  Gradient size: {} parameters\n", gradient_size);
    
    // Initialize predictor and training quantizer (Stochastic INT4 or TopK)
    let mut predictor = MomentumPredictor::new(0.9);
    let mut train_quant = if args.train_topk > 0.0 {
        UnifiedQuantizer::new(QuantizationMode::TopK {
            k_percent: args.train_topk,
        })
    } else {
        UnifiedQuantizer::new(QuantizationMode::Stochastic {
            seed: 42,
            bits: 4,
        })
    };
    
    println!(
        "Initial mode: {}\n",
        train_quant.mode_description()
    );
    
    // Statistics
    let mut total_training_bytes = 0;
    let mut total_checkpoint_bytes = 0;
    let mut training_time = 0.0;
    let mut checkpoint_time = 0.0;
    
    println!("=== Training Loop ===\n");
    
    for step in 0..num_steps {
        // Generate synthetic gradient (simulating training)
        let gradient_data: Vec<f32> = (0..gradient_size)
            .map(|i| {
                let phase = (step as f32) * 0.1 + (i as f32) * 0.001;
                phase.sin() * 0.5
            })
            .collect();
        
        let gradient = FixedPointBuffer::from_f32(&gradient_data);
        
        // Check if this is a checkpoint step
        let is_checkpoint = (step + 1) % checkpoint_interval == 0;
        
        if is_checkpoint {
            println!("Step {}: CHECKPOINT", step);

            // Predict and residual
            let prediction = predictor.predict().unwrap();
            let residual = if prediction.is_empty() {
                gradient.clone()
            } else {
                gradient.sub(&prediction).unwrap()
            };

            // Choose checkpoint codec
            let start = Instant::now();
            let (compressed, scale, codec_label) = if args.checkpoint_bsrle {
                let q = BlockSparseRleQuantizer::new(args.epsilon);
                let (b, s) = q.quantize(&residual);
                (b, s, "ErrorBounded+BlockSparse+RLE")
            } else if args.checkpoint_block_sparse {
                let q = BlockSparseQuantizer::new(args.epsilon);
                let (b, s) = q.quantize(&residual);
                (b, s, "ErrorBounded+BlockSparse")
            } else {
                // Fallback to unified error-bounded
                let q = UnifiedQuantizer::new(QuantizationMode::ErrorBounded {
                    epsilon: args.epsilon,
                });
                let (b, s) = q.quantize(&residual);
                (b, s, "ErrorBounded")
            };
            let duration = start.elapsed();

            total_checkpoint_bytes += compressed.len();
            checkpoint_time += duration.as_secs_f64();

            // Reconstruct using the same codec (deterministic path)
            let reconstructed_residual = if args.checkpoint_bsrle {
                let q = BlockSparseRleQuantizer::new(args.epsilon);
                q.dequantize(&compressed, scale)
            } else if args.checkpoint_block_sparse {
                let q = BlockSparseQuantizer::new(args.epsilon);
                q.dequantize(&compressed, scale)
            } else {
                let q = UnifiedQuantizer::new(QuantizationMode::ErrorBounded { epsilon: args.epsilon });
                q.dequantize(&compressed, scale)
            };

            let reconstructed = if prediction.is_empty() {
                reconstructed_residual
            } else {
                prediction.add(&reconstructed_residual).unwrap()
            };

            // Verify error bound
            let orig = gradient.to_f32();
            let recon = reconstructed.to_f32();
            let max_error = orig.iter().zip(recon.iter()).map(|(o, r)| (o - r).abs()).fold(0.0f32, f32::max);

            println!(
                "  Codec: {} | Compressed: {} bytes (raw: {} bytes)",
                codec_label,
                compressed.len(),
                gradient_size * 4
            );
            println!(
                "  Compression ratio: {:.2}×",
                (gradient_size * 4) as f32 / compressed.len() as f32
            );
            println!("  Max error: {:.2e} (bound: {:.1e})", max_error, args.epsilon);
            println!("  Time: {:.2} ms", duration.as_secs_f64() * 1000.0);

            // Update predictor with reconstructed gradient
            predictor.update(&reconstructed).unwrap();

            // Report current training mode again
            println!("  Continue with training mode: {}\n", train_quant.mode_description());

        } else {
            // Regular training step with Stochastic mode
            let start = Instant::now();
            
            // Predict and compress
            let prediction = predictor.predict().unwrap();
            let residual = if prediction.is_empty() {
                gradient.clone()
            } else {
                gradient.sub(&prediction).unwrap()
            };
            
            let (compressed, scale) = train_quant.quantize(&residual);
            let duration = start.elapsed();
            
            total_training_bytes += compressed.len();
            training_time += duration.as_secs_f64();
            
            // Reconstruct and update
            let reconstructed_residual = train_quant.dequantize(&compressed, scale);
            let reconstructed = if prediction.is_empty() {
                reconstructed_residual
            } else {
                prediction.add(&reconstructed_residual).unwrap()
            };
            
            predictor.update(&reconstructed).unwrap();
            
            if step % 10 == 0 && step > 0 {
                println!("Step {}: Training (compressed: {} bytes)", step, compressed.len());
            }
        }
    }
    
    println!("\n=== Summary ===\n");
    
    let num_training_steps = num_steps - (num_steps / checkpoint_interval);
    let num_checkpoint_steps = num_steps / checkpoint_interval;
    
    println!("Training Steps: {}", num_training_steps);
    println!("  Total bytes: {} KB", total_training_bytes / 1024);
    println!("  Avg bytes/step: {} bytes", total_training_bytes / num_training_steps);
    println!("  Avg time/step: {:.2} ms", (training_time / num_training_steps as f64) * 1000.0);
    println!("  Mode: {}\n", train_quant.mode_description());
    
    println!("Checkpoint Steps: {}", num_checkpoint_steps);
    println!("  Total bytes: {} KB", total_checkpoint_bytes / 1024);
    println!("  Avg bytes/step: {} bytes", total_checkpoint_bytes / num_checkpoint_steps);
    println!("  Avg time/step: {:.2} ms", (checkpoint_time / num_checkpoint_steps as f64) * 1000.0);
    println!("  Mode: {} (strict ε-guarantee)\n",
        if args.checkpoint_bsrle { "ErrorBounded+BlockSparse+RLE" }
        else if args.checkpoint_block_sparse { "ErrorBounded+BlockSparse" } else { "ErrorBounded" }
    );
    
    let total_bytes = total_training_bytes + total_checkpoint_bytes;
    let raw_bytes = num_steps * gradient_size * 4;
    let overall_compression = raw_bytes as f32 / total_bytes as f32;
    
    println!("Overall:");
    println!("  Raw data: {} MB", raw_bytes / (1024 * 1024));
    println!("  Compressed: {} KB", total_bytes / 1024);
    println!("  Compression ratio: {:.2}×", overall_compression);
    println!("  Total time: {:.2} ms", (training_time + checkpoint_time) * 1000.0);
    
    println!("\n=== Mode Switching Benefits ===\n");
    println!("✓ Fast compression during training (INT4)");
    println!("✓ Strict error bounds for checkpoints (ε-guarantee)");
    println!("✓ Zero code changes (runtime mode switch)");
    println!("✓ Single unified interface for all modes");
}
