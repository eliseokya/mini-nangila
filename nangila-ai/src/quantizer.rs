use nangila_core::{QuantizationMode, Quantizer, UnifiedQuantizer};
use nangila_math::FixedPointBuffer;

/// Stochastic quantizer delegating to the core UnifiedQuantizer (INT4 default).
pub struct StochasticQuantizer {
    inner: UnifiedQuantizer,
}

impl StochasticQuantizer {
    pub fn new(seed: u64) -> Self {
        Self {
            // INT4 by default: 0.5 bytes/value → 8× quantization compression.
            // Matches the whitepaper's AI compression targets (20-40×).
            inner: UnifiedQuantizer::new(QuantizationMode::Stochastic { seed, bits: 4 }),
        }
    }
}

impl Quantizer for StochasticQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        self.inner.quantize(residual)
    }
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        self.inner.dequantize(compressed, scale)
    }
}

/// Backwards-compatible alias for [`StochasticQuantizer`].
///
/// The name `RefCellQuantizer` is a historical artifact — interior mutability
/// now lives inside `UnifiedQuantizer`. Prefer [`StochasticQuantizer`] instead.
#[deprecated(since = "0.1.0", note = "Use StochasticQuantizer instead")]
pub struct RefCellQuantizer {
    inner: UnifiedQuantizer,
}

#[allow(deprecated)]
impl RefCellQuantizer {
    pub fn new(seed: u64) -> Self {
        Self {
            // INT8 for backward compatibility (deprecated type)
            inner: UnifiedQuantizer::new(QuantizationMode::Stochastic { seed, bits: 8 }),
        }
    }
}

#[allow(deprecated)]
impl Quantizer for RefCellQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        self.inner.quantize(residual)
    }
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        self.inner.dequantize(compressed, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_stochastic_rounding_unbiased() {
        // Deterministic stochastic rounding via UnifiedQuantizer (hash PRNG), mean stays close.
        let quantizer = RefCellQuantizer::new(42);
        let val = 0.5f32;
        let input = FixedPointBuffer::from_f32(&vec![val; 1000]);
        let (bytes, scale) = quantizer.quantize(&input);
        let sum: f32 = bytes.iter().map(|&b| (b as i8 as f32) * scale).sum();
        let mean = sum / 1000.0;
        assert!((mean - 0.5).abs() < 0.1, "Mean {} not close to 0.5", mean);
    }
}
