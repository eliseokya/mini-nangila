use nangila_math::FixedPointBuffer;

/// Quantizer interface for predictive-residual compression.
///
/// Determinism: Implementations that rely on stochastic rounding should ensure
/// reproducibility by using deterministic PRNG state (e.g., interior mutability
/// with `RefCell`, or a hash-based counter PRNG) so that `quantize` with the
/// same inputs yields the same outputs across platforms and runs.
pub trait Quantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32); // Returns compressed bytes + error scale
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer;
}
