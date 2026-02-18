use proptest::prelude::*;
use nangila_math::FixedPointBuffer;

// Property 1: Roundtrip conversion (from_f32 → to_f32 ≈ identity within precision)
proptest! {
    #[test]
    fn prop_roundtrip_conversion(values in prop::collection::vec(
        // Q8.23 range is [-256, 256), test within this range
        -255.0f32..255.0f32,
        1..1000
    )) {
        let buffer = FixedPointBuffer::from_f32(&values);
        let reconstructed = buffer.to_f32();
        
        prop_assert_eq!(values.len(), reconstructed.len());
        
        for (i, (&original, &reconstructed_val)) in values.iter().zip(reconstructed.iter()).enumerate() {
            // Precision of Q8.23 is 2^-23 ≈ 1.19e-7
            // Allow slightly larger tolerance due to rounding
            let diff = (original - reconstructed_val).abs();
            prop_assert!(
                diff < 2e-6,
                "Roundtrip failed at index {}: {} != {} (diff: {})",
                i, original, reconstructed_val, diff
            );
        }
    }
}

// Property 2: Addition is commutative (a + b == b + a)
proptest! {
    #[test]
    fn prop_addition_commutative(
        a_vals in prop::collection::vec(-100.0f32..100.0f32, 1..100),
        b_vals in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        // Ensure same length
        let len = a_vals.len().min(b_vals.len());
        let a_vals = &a_vals[..len];
        let b_vals = &b_vals[..len];
        
        let a = FixedPointBuffer::from_f32(a_vals);
        let b = FixedPointBuffer::from_f32(b_vals);
        
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        
        // Results should be identical (bit-exact)
        prop_assert_eq!(ab.data, ba.data, "Addition is not commutative");
    }
}

// Property 3: Addition is associative ((a + b) + c ≈ a + (b + c))
proptest! {
    #[test]
    fn prop_addition_associative(
        a_vals in prop::collection::vec(-50.0f32..50.0f32, 1..50),
        b_vals in prop::collection::vec(-50.0f32..50.0f32, 1..50),
        c_vals in prop::collection::vec(-50.0f32..50.0f32, 1..50)
    ) {
        let len = a_vals.len().min(b_vals.len()).min(c_vals.len());
        let a_vals = &a_vals[..len];
        let b_vals = &b_vals[..len];
        let c_vals = &c_vals[..len];
        
        let a = FixedPointBuffer::from_f32(a_vals);
        let b = FixedPointBuffer::from_f32(b_vals);
        let c = FixedPointBuffer::from_f32(c_vals);
        
        let ab_c = a.add(&b).unwrap().add(&c).unwrap();
        let a_bc = a.add(&b.add(&c).unwrap()).unwrap();
        
        // Due to potential saturation, compare element-wise with tolerance
        let ab_c_f32 = ab_c.to_f32();
        let a_bc_f32 = a_bc.to_f32();
        
        for (i, (&v1, &v2)) in ab_c_f32.iter().zip(a_bc_f32.iter()).enumerate() {
            let diff = (v1 - v2).abs();
            prop_assert!(
                diff < 1e-5,
                "Associativity failed at index {}: {} != {} (diff: {})",
                i, v1, v2, diff
            );
        }
    }
}

// Property 4: Scalar multiplication is distributive (k * (a + b) ≈ k*a + k*b)
proptest! {
    #[test]
    fn prop_mul_scalar_distributive(
        a_vals in prop::collection::vec(-50.0f32..50.0f32, 1..50),
        b_vals in prop::collection::vec(-50.0f32..50.0f32, 1..50),
        k in -2.0f32..2.0f32
    ) {
        let len = a_vals.len().min(b_vals.len());
        let a_vals = &a_vals[..len];
        let b_vals = &b_vals[..len];
        
        let a = FixedPointBuffer::from_f32(a_vals);
        let b = FixedPointBuffer::from_f32(b_vals);
        
        let ab = a.add(&b).unwrap();
        let k_ab = ab.mul_scalar(k);
        
        let ka = a.mul_scalar(k);
        let kb = b.mul_scalar(k);
        let ka_kb = ka.add(&kb).unwrap();
        
        let k_ab_f32 = k_ab.to_f32();
        let ka_kb_f32 = ka_kb.to_f32();
        
        for (i, (&v1, &v2)) in k_ab_f32.iter().zip(ka_kb_f32.iter()).enumerate() {
            let diff = (v1 - v2).abs();
            // Allow some tolerance due to fixed-point rounding
            prop_assert!(
                diff < 1e-4,
                "Distributivity failed at index {}: {} != {} (diff: {})",
                i, v1, v2, diff
            );
        }
    }
}

// Property 5: Saturation at boundaries
proptest! {
    #[test]
    fn prop_saturation_at_boundaries(
        extreme_vals in prop::collection::vec(
            prop::num::f32::POSITIVE | prop::num::f32::NEGATIVE,
            1..10
        )
    ) {
        let buffer = FixedPointBuffer::from_f32(&extreme_vals);
        let reconstructed = buffer.to_f32();
        
        // All values should be within valid Q8.23 range or saturated to boundary
        // Note: Saturation can produce exactly 256.0 or -256.0
        for &val in &reconstructed {
            prop_assert!(
                val >= -256.0 && val <= 256.0,
                "Value {} outside saturation range [-256, 256]",
                val
            );
        }
    }
}

// Property 6: Determinism (same input always produces same output)
proptest! {
    #[test]
    fn prop_determinism(values in prop::collection::vec(-100.0f32..100.0f32, 1..100)) {
        let buffer1 = FixedPointBuffer::from_f32(&values);
        let buffer2 = FixedPointBuffer::from_f32(&values);
        
        // Should produce identical internal representation
        prop_assert_eq!(&buffer1.data, &buffer2.data, "Non-deterministic conversion");
        
        // And identical reconstructions
        let recon1 = buffer1.to_f32();
        let recon2 = buffer2.to_f32();
        prop_assert_eq!(recon1, recon2, "Non-deterministic reconstruction");
    }
}

// Property 7: Subtraction is inverse of addition
proptest! {
    #[test]
    fn prop_subtraction_inverse_of_addition(
        a_vals in prop::collection::vec(-100.0f32..100.0f32, 1..100),
        b_vals in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        let len = a_vals.len().min(b_vals.len());
        let a_vals = &a_vals[..len];
        let b_vals = &b_vals[..len];
        
        let a = FixedPointBuffer::from_f32(a_vals);
        let b = FixedPointBuffer::from_f32(b_vals);
        
        let a_plus_b = a.add(&b).unwrap();
        let result = a_plus_b.sub(&b).unwrap();
        
        let a_f32 = a.to_f32();
        let result_f32 = result.to_f32();
        
        for (i, (&original, &reconstructed)) in a_f32.iter().zip(result_f32.iter()).enumerate() {
            let diff = (original - reconstructed).abs();
            prop_assert!(
                diff < 1e-5,
                "Subtraction inverse failed at index {}: {} != {} (diff: {})",
                i, original, reconstructed, diff
            );
        }
    }
}
