use nangila_math::FixedPointBuffer;

// Determinism tests for Q8.23 fixed-point conversions and ops.
// These use rational values exactly representable in binary to avoid
// any cross-platform rounding ambiguity.

#[test]
fn test_q823_determinism_rationals() {
    let q: i32 = 1 << 23; // FRACTIONAL_BITS = 23

    let vals: [f32; 13] = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        0.25,
        -0.25,
        0.75,
        -0.75,
        1.25,
        -1.25,
        127.0,
        -128.0,
    ];

    let expected: Vec<i32> = vec![
        0,
        1 * q,
        -1 * q,
        q / 2,
        -q / 2,
        q / 4,
        -q / 4,
        (3 * q) / 4,
        -(3 * q) / 4,
        q + (q / 4),
        -q - (q / 4),
        127 * q,
        -128 * q,
    ];

    let buf = FixedPointBuffer::from_f32(&vals);
    assert_eq!(buf.data, expected, "Q8.23 encoding mismatch");

    // Round-trip back to f32 should recover the source within 1e-6.
    let out = buf.to_f32();
    for (i, (&v, &r)) in vals.iter().zip(out.iter()).enumerate() {
        assert!((v - r).abs() < 1e-6, "Round-trip mismatch at {}: {} vs {}", i, v, r);
    }
}

#[test]
fn test_q823_add_sub_and_mul_scalar_determinism() {
    let q: i32 = 1 << 23;

    // a = [1.25, -0.75], b = [0.25, 0.5]
    let a = FixedPointBuffer::from_f32(&[1.25, -0.75]);
    let b = FixedPointBuffer::from_f32(&[0.25, 0.5]);

    // a + b = [1.5, -0.25]
    let add = a.add(&b).expect("add");
    let expected_add = vec![
        (3 * q) / 2, // 1.5 * 2^23 = 3/2 * 2^23 = 3 * 2^22
        -q / 4,      // -0.25
    ];
    assert_eq!(add.data, expected_add, "add determinism");

    // a - b = [1.0, -1.25]
    let sub = a.sub(&b).expect("sub");
    let expected_sub = vec![
        1 * q,             // 1.0
        -q - (q / 4),      // -1.25
    ];
    assert_eq!(sub.data, expected_sub, "sub determinism");

    // mul_scalar(0.5) should be exact right shift by 1 in Q8.23 domain
    let half = a.mul_scalar(0.5);
    let expected_half = vec![
        (q + (q / 4)) / 2, // 1.25 / 2 = 0.625 = (1/2 + 1/8)
        -((3 * q) / 4) / 2, // -0.75 / 2 = -0.375 = -(3/8)
    ];
    assert_eq!(half.data, expected_half, "mul_scalar(0.5) determinism");
}

