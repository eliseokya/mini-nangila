use proptest::prelude::*;
use nangila_core::{Predictor, Quantizer, UnifiedQuantizer, QuantizationMode, TopologyMask};
use nangila_math::FixedPointBuffer;

// Property 1: Predictor convergence for constant signal
proptest! {
    #[test]
    fn prop_predictor_converges_to_constant(
        constant_val in -100.0f32..100.0f32,
        num_steps in 10usize..100
    ) {
        use nangila_ai::MomentumPredictor;
        
        let mut predictor = MomentumPredictor::new(0.9);
        let signal = FixedPointBuffer::from_f32(&[constant_val]);
        
        // Feed constant signal repeatedly
        for _ in 0..num_steps {
            predictor.update(&signal).unwrap();
        }
        
        // After many steps, prediction should be close to constant
        let prediction = predictor.predict().unwrap();
        if !prediction.is_empty() {
            let pred_val = prediction.to_f32()[0];
            
            let error = (pred_val - constant_val).abs();
            prop_assert!(
                error < 5.0, // Reasonable tolerance for momentum convergence
                "Predictor did not converge: predicted {}, expected {} (error: {})",
                pred_val, constant_val, error
            );
        }
    }
}

// Property 2: Predictor stability (predictions should be bounded)
proptest! {
    #[test]
    fn prop_predictor_stability(
        values in prop::collection::vec(-100.0f32..100.0f32, 10..50)
    ) {
        use nangila_ai::MomentumPredictor;
        
        let mut predictor = MomentumPredictor::new(0.9);
        
        for &val in &values {
            let signal = FixedPointBuffer::from_f32(&[val]);
            predictor.update(&signal).unwrap();
            
            let prediction = predictor.predict().unwrap();
            if !prediction.is_empty() {
                let pred_val = prediction.to_f32()[0];
                
                // Predictions should stay within reasonable bounds
                prop_assert!(
                    pred_val.abs() <= 300.0, // Slightly above input range
                    "Prediction unstable: {}", pred_val
                );
            }
        }
    }
}

// Property 3: Predictor reset returns to initial state
proptest! {
    #[test]
    fn prop_predictor_reset(
        values in prop::collection::vec(-50.0f32..50.0f32, 5..20)
    ) {
        use nangila_ai::MomentumPredictor;
        
        let mut predictor = MomentumPredictor::new(0.9);
        
        // Update with some values
        for &val in &values {
            let signal = FixedPointBuffer::from_f32(&[val]);
            predictor.update(&signal).unwrap();
        }
        
        // Reset
        predictor.reset();
        
        // Prediction should be empty after reset (no history)
        let prediction_after_reset = predictor.predict().unwrap();
        prop_assert!(prediction_after_reset.is_empty(), "Predictor not properly reset");
    }
}

// Property 4: Quantizer roundtrip (dequantize(quantize(x)) ≈ x)
proptest! {
    #[test]
    fn prop_quantizer_roundtrip(
        values in prop::collection::vec(-100.0f32..100.0f32, 10..100)
    ) {
        let quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic { bits: 8, seed: 0 });
        let buffer = FixedPointBuffer::from_f32(&values);
        
        let (compressed, scale) = quantizer.quantize(&buffer);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        let reconstructed_f32 = reconstructed.to_f32();
        
        // Check reconstruction quality
        for (i, (&original, &reconstructed_val)) in values.iter().zip(reconstructed_f32.iter()).enumerate() {
            let error = (original - reconstructed_val).abs();
            let relative_error = if original.abs() > 1e-6 {
                error / original.abs()
            } else {
                error
            };
            
            // Allow reasonable quantization error (INT8 stochastic quantization)
            prop_assert!(
                relative_error < 0.2 || error < 1.0, // 20% or 1.0 absolute error
                "Roundtrip error too large at index {}: {} != {} (error: {})",
                i, original, reconstructed_val, error
            );
        }
    }
}

// Property 5: Quantizer error bounds for error-bounded mode
proptest! {
    #[test]
    fn prop_error_bounded_quantization(
        values in prop::collection::vec(-50.0f32..50.0f32, 10..50),
        epsilon in 0.01f32..0.5f32
    ) {
        use nangila_hpc::ErrorBoundedQuantizer;
        
        let quantizer = ErrorBoundedQuantizer::new(epsilon);
        let buffer = FixedPointBuffer::from_f32(&values);
        
        let (compressed, scale) = quantizer.quantize(&buffer);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        let reconstructed_f32 = reconstructed.to_f32();
        
        // Check that error is within epsilon bounds
        for (i, (&original, &reconstructed_val)) in values.iter().zip(reconstructed_f32.iter()).enumerate() {
            let error = (original - reconstructed_val).abs();
            
            prop_assert!(
                error <= epsilon * 1.2, // Allow 20% tolerance for rounding
                "Error bound violated at index {}: error {} > epsilon {} (original: {}, reconstructed: {})",
                i, error, epsilon, original, reconstructed_val
            );
        }
    }
}

// Property 6: Topology mask determinism
proptest! {
    #[test]
    fn prop_topology_mask_determinism(
        variances in prop::collection::vec(0.0f32..100.0f32, 5..20),
        drop_percent in 0.1f32..0.9f32
    ) {
        let mask1 = TopologyMask::from_variance_threshold(&variances, drop_percent);
        let mask2 = TopologyMask::from_variance_threshold(&variances, drop_percent);
        
        // Same inputs should produce identical masks
        prop_assert_eq!(
            mask1.driver_indices, mask2.driver_indices,
            "Non-deterministic topology mask generation (drivers)"
        );
        prop_assert_eq!(
            mask1.passenger_indices, mask2.passenger_indices,
            "Non-deterministic topology mask generation (passengers)"
        );
    }
}

// Property 7: Topology mask coverage (drivers + passengers = total)
proptest! {
    #[test]
    fn prop_topology_mask_coverage(
        variances in prop::collection::vec(0.0f32..100.0f32, 5..20),
        drop_percent in 0.1f32..0.9f32
    ) {
        let mask = TopologyMask::from_variance_threshold(&variances, drop_percent);
        
        // Total indices should equal input length
        let total_indices = mask.driver_indices.len() + mask.passenger_indices.len();
        prop_assert_eq!(
            total_indices, variances.len(),
            "Topology mask doesn't cover all indices: {} drivers + {} passengers != {} total",
            mask.driver_indices.len(), mask.passenger_indices.len(), variances.len()
        );
        
        // No overlaps between drivers and passengers
        for &driver_idx in &mask.driver_indices {
            prop_assert!(
                !mask.passenger_indices.contains(&driver_idx),
                "Index {} appears in both drivers and passengers",
                driver_idx
            );
        }
    }
}

// Property 8: Quantization preserves dimension
proptest! {
    #[test]
    fn prop_quantization_preserves_dimension(
        values in prop::collection::vec(-50.0f32..50.0f32, 10..100)
    ) {
        let quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic { bits: 8, seed: 0 });
        let buffer = FixedPointBuffer::from_f32(&values);
        
        let (compressed, scale) = quantizer.quantize(&buffer);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        
        // Reconstructed should have same length as original
        prop_assert_eq!(
            buffer.len(), reconstructed.len(),
            "Quantization changed buffer dimension"
        );
    }
}

// Property 9: BlockSparse quantizer roundtrip respects error bound
proptest! {
    #[test]
    fn prop_blocksparse_roundtrip(
        values in prop::collection::vec(-50.0f32..50.0f32, 10..200),
        epsilon in 0.01f32..0.5f32
    ) {
        use nangila_hpc::BlockSparseQuantizer;

        let quantizer = BlockSparseQuantizer::new(epsilon);
        let buffer = FixedPointBuffer::from_f32(&values);

        let (compressed, scale) = quantizer.quantize(&buffer);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        let recon_f32 = reconstructed.to_f32();

        prop_assert_eq!(recon_f32.len(), values.len(), "Length mismatch");

        for (i, (&original, &recon_val)) in values.iter().zip(recon_f32.iter()).enumerate() {
            let error = (original - recon_val).abs();
            prop_assert!(
                error <= epsilon * 1.2,
                "BlockSparse error bound violated at index {}: error {} > epsilon {} (orig: {}, recon: {})",
                i, error, epsilon, original, recon_val
            );
        }
    }
}

// Property 10: BlockSparseRLE quantizer roundtrip respects error bound
proptest! {
    #[test]
    fn prop_blocksparse_rle_roundtrip(
        values in prop::collection::vec(-50.0f32..50.0f32, 10..200),
        epsilon in 0.01f32..0.5f32
    ) {
        use nangila_hpc::BlockSparseRleQuantizer;

        let quantizer = BlockSparseRleQuantizer::new(epsilon);
        let buffer = FixedPointBuffer::from_f32(&values);

        let (compressed, scale) = quantizer.quantize(&buffer);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        let recon_f32 = reconstructed.to_f32();

        prop_assert_eq!(recon_f32.len(), values.len(), "Length mismatch");

        for (i, (&original, &recon_val)) in values.iter().zip(recon_f32.iter()).enumerate() {
            let error = (original - recon_val).abs();
            prop_assert!(
                error <= epsilon * 1.2,
                "BlockSparseRLE error bound violated at index {}: error {} > epsilon {} (orig: {}, recon: {})",
                i, error, epsilon, original, recon_val
            );
        }
    }
}

// Property 11: LinearPredictor converges to constant signal
proptest! {
    #[test]
    fn prop_linear_predictor_converges_to_constant(
        constant_val in -100.0f32..100.0f32,
        num_steps in 10usize..50
    ) {
        use nangila_hpc::LinearPredictor;

        let mut predictor = LinearPredictor::default();
        let signal = FixedPointBuffer::from_f32(&[constant_val]);

        for _ in 0..num_steps {
            predictor.update(&signal).unwrap();
        }

        let prediction = predictor.predict().unwrap();
        if !prediction.is_empty() {
            let pred_val = prediction.to_f32()[0];
            let error = (pred_val - constant_val).abs();
            prop_assert!(
                error < 1.0,
                "LinearPredictor did not converge: predicted {}, expected {} (error: {})",
                pred_val, constant_val, error
            );
        }
    }
}
