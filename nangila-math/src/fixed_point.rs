use thiserror::Error;

/// Fixed-Point Arithmetic with 23 fractional bits (Q*.23)
/// Dynamic range (with i32 backing): approximately [-256.0, 256.0)
/// Precision: 2^-23 (~1e-7)
/// Zero handling: Represents exact 0.0
/// Overflow behavior: Saturating (Clamping)
pub(crate) const FRACTIONAL_BITS: i32 = 23;
const SCALE_FACTOR: f32 = (1 << FRACTIONAL_BITS) as f32;
const MIN_VAL_I32: i32 = i32::MIN;
const MAX_VAL_I32: i32 = i32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct FixedPointBuffer {
    /// Internal storage as Q8.23 signed integers
    pub data: Vec<i32>,
}

#[derive(Error, Debug)]
pub enum FixedPointError {
    #[error("Shape mismatch: {0} vs {1}")]
    ShapeMismatch(usize, usize),
}

impl FixedPointBuffer {
    /// Create a new buffer of zeros
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    /// Convert from f32 slice with deterministic quantization (Round to Nearest)
    pub fn from_f32(data: &[f32]) -> Self {
        let mut buffer = Vec::with_capacity(data.len());
        for &val in data {
            // Scale and clamp to i32 range
            let scaled = val * SCALE_FACTOR;
            
            // Deterministic rounding: round() creates nearest integer, .5 rounds away from zero
            // We clamp to i32::MIN/MAX to handle overflow explicitly
            let clamped = if scaled >= MAX_VAL_I32 as f32 {
                MAX_VAL_I32
            } else if scaled <= MIN_VAL_I32 as f32 {
                MIN_VAL_I32
            } else {
                scaled.round() as i32
            };
            
            buffer.push(clamped);
        }
        Self { data: buffer }
    }

    /// Convert back to f32
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter()
            .map(|&val| val as f32 / SCALE_FACTOR)
            .collect()
    }
    
    /// Returns the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_roundtrip() {
        let input = vec![0.0, 1.0, -1.0, 0.5, -0.5, 127.0, -128.0];
        let buffer = FixedPointBuffer::from_f32(&input);
        let output = buffer.to_f32();
        
        for (i, &val) in input.iter().enumerate() {
            let diff = (val - output[i]).abs();
            assert!(diff < 1e-6, "Mismatch at {}: {} vs {}", i, val, output[i]);
        }
    }

    #[test]
    fn test_saturation() {
        let input = vec![200.0, -200.0];
        let buffer = FixedPointBuffer::from_f32(&input);
        let output = buffer.to_f32();
        // With 23 fractional bits in an i32 container, the effective range is ~[-256, 256).
        // Values like ±200 should be representable without overflow after scaling; this
        // assertion simply ensures we didn't quantize to tiny magnitudes by mistake.
        assert!(output[0] > 127.0);
        assert!(output[1] < -127.0);
    }
}
