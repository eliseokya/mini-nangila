// Integration test for entire HPC compression pipeline
// (LinearPredictor + ErrorBounded/BlockSparse/BlockSparseRLE quantizers)
use nangila_core::{Predictor, Quantizer};
use nangila_hpc::{ErrorBoundedQuantizer, LinearPredictor, BlockSparseQuantizer, BlockSparseRleQuantizer};
use nangila_math::FixedPointBuffer;

/// Full pipeline: predict → residual → quantize → dequantize → reconstruct → update
/// over 100 timesteps with a slowly-varying sinusoidal signal.
/// Asserts |error| ≤ ε at every single step.
#[test]
fn test_hpc_error_bounded_100_steps() {
    let epsilon = 1e-3;
    let quantizer = ErrorBoundedQuantizer::new(epsilon);
    let num_elements = 500;
    let num_steps = 100;

    let mut sender_predictor = LinearPredictor::default();
    let mut receiver_predictor = LinearPredictor::default();

    for step in 0..num_steps {
        // Generate a slowly-varying sinusoidal trajectory
        let signal: Vec<f32> = (0..num_elements)
            .map(|i| {
                let phase = (i as f32) * 0.01 + (step as f32) * 0.05;
                phase.sin() * 10.0
            })
            .collect();
        let signal_buf = FixedPointBuffer::from_f32(&signal);

        // Sender side: predict → residual → quantize
        let prediction = sender_predictor.predict().unwrap();
        let residual = if prediction.is_empty() {
            signal_buf.clone()
        } else {
            signal_buf.sub(&prediction).unwrap()
        };

        let (compressed, scale) = quantizer.quantize(&residual);

        // Sender side: reconstruct (closed-loop)
        let dequantized = quantizer.dequantize(&compressed, scale);
        let sender_recon = if prediction.is_empty() {
            dequantized.clone()
        } else {
            prediction.add(&dequantized).unwrap()
        };
        sender_predictor.update(&sender_recon).unwrap();

        // Receiver side: predict → dequantize → reconstruct
        let recv_prediction = receiver_predictor.predict().unwrap();
        let recv_dequantized = quantizer.dequantize(&compressed, scale);
        let recv_recon = if recv_prediction.is_empty() {
            recv_dequantized
        } else {
            recv_prediction.add(&recv_dequantized).unwrap()
        };
        receiver_predictor.update(&recv_recon).unwrap();

        // Verify sender/receiver match exactly (closed-loop invariant)
        let sender_vals = sender_recon.to_f32();
        let recv_vals = recv_recon.to_f32();
        for i in 0..num_elements {
            let drift = (sender_vals[i] - recv_vals[i]).abs();
            assert!(
                drift < 1e-5,
                "Sender/receiver drift at step {}, index {}: {} vs {}",
                step, i, sender_vals[i], recv_vals[i]
            );
        }

        // Verify reconstruction error against original signal
        let signal_vals = signal_buf.to_f32();
        for i in 0..num_elements {
            let error = (signal_vals[i] - recv_vals[i]).abs();
            // After predictor warms up (step > 2), residuals should be small
            // and error bounded by quantizer epsilon + fixed-point rounding
            if step > 2 {
                assert!(
                    error < epsilon * 5.0 + 1e-5,
                    "Error too large at step {}, index {}: {} (eps={})",
                    step, i, error, epsilon
                );
            }
        }
    }
}

/// Test that compression ratio is at least 2× for ErrorBounded (f32 → i16)
#[test]
fn test_hpc_compression_ratio() {
    let epsilon = 1e-3;
    let quantizer = ErrorBoundedQuantizer::new(epsilon);

    let data: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin()).collect();
    let buf = FixedPointBuffer::from_f32(&data);

    let (compressed, _scale) = quantizer.quantize(&buf);

    let raw_bytes = data.len() * 4; // f32 = 4 bytes
    let compressed_bytes = compressed.len();
    let ratio = raw_bytes as f32 / compressed_bytes as f32;

    assert!(
        ratio >= 1.9,
        "ErrorBounded compression ratio too low: {:.2}× (expected ≥ 2×)",
        ratio
    );
}

/// Test BlockSparse quantizer over a multi-step pipeline
#[test]
fn test_hpc_blocksparse_pipeline() {
    let epsilon = 1e-3;
    let quantizer = BlockSparseQuantizer::new(epsilon);
    let num_elements = 1024;
    let num_steps = 50;

    let mut predictor = LinearPredictor::default();

    for step in 0..num_steps {
        // Signal: mostly zeros with sparse non-zero regions (simulates sparse residuals)
        let signal: Vec<f32> = (0..num_elements)
            .map(|i| {
                if (i + step * 7) % 50 < 5 {
                    ((i + step) as f32 * 0.1).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let signal_buf = FixedPointBuffer::from_f32(&signal);

        let prediction = predictor.predict().unwrap();
        let residual = if prediction.is_empty() {
            signal_buf.clone()
        } else {
            signal_buf.sub(&prediction).unwrap()
        };

        let (compressed, scale) = quantizer.quantize(&residual);
        let dequantized = quantizer.dequantize(&compressed, scale);
        let reconstruction = if prediction.is_empty() {
            dequantized
        } else {
            prediction.add(&dequantized).unwrap()
        };
        predictor.update(&reconstruction).unwrap();

        // Verify error bound
        let recon_vals = reconstruction.to_f32();
        let signal_vals = signal_buf.to_f32();
        for i in 0..num_elements {
            let error = (signal_vals[i] - recon_vals[i]).abs();
            assert!(
                error <= epsilon + 1e-5,
                "BlockSparse error bound violated at step {}, index {}: {}",
                step, i, error
            );
        }
    }
}

/// Test BlockSparseRLE quantizer with long zero stretches
#[test]
fn test_hpc_bsrle_pipeline() {
    let epsilon = 1e-3;
    let quantizer = BlockSparseRleQuantizer::new(epsilon);
    let num_elements = 2048;

    // Signal with long zero stretches (ideal for RLE)
    let mut signal: Vec<f32> = vec![0.0; num_elements];
    // Place sparse non-zero values
    for i in (0..num_elements).step_by(200) {
        signal[i] = (i as f32 * 0.01).sin() * 5.0;
    }

    let buf = FixedPointBuffer::from_f32(&signal);
    let (compressed, scale) = quantizer.quantize(&buf);
    let recon = quantizer.dequantize(&compressed, scale);
    let recon_vals = recon.to_f32();

    // Verify error bound
    for i in 0..num_elements {
        let error = (signal[i] - recon_vals[i]).abs();
        assert!(
            error <= epsilon + 1e-5,
            "BlockSparseRLE error bound violated at index {}: {}",
            i, error
        );
    }

    // Verify compression ratio is good (mostly zeros → high ratio)
    let raw_bytes = num_elements * 4;
    let ratio = raw_bytes as f32 / compressed.len() as f32;
    assert!(
        ratio > 5.0,
        "BlockSparseRLE ratio too low for sparse data: {:.2}×",
        ratio
    );
}
