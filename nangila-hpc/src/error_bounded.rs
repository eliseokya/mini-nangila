use nangila_core::Quantizer;
use nangila_math::{FixedPointBuffer, FixedPointError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuantizerError {
    #[error("Error bound {0} violated, fallback required")]
    FallbackRequired(f32),
    #[error("Math error: {0}")]
    MathError(#[from] FixedPointError),
}

pub struct ErrorBoundedQuantizer {
    pub epsilon: f32,
}

impl ErrorBoundedQuantizer {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}

/// Simple Error-Bounded Quantizer (SZ-basic style)
/// Formula: q = round(x / (2 * epsilon))
/// Reconstruction: x_hat = q * (2 * epsilon)
/// Max error is effectively epsilon IF q fits in integer range.
/// We use i8 for quantization [-128, 127].
impl Quantizer for ErrorBoundedQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        // For Mini-Nangila HPC, we use i16 quantization to ensure range is sufficient
        // while maintaining error bounds. This gives 2x compression (vs f32).
        
        // Scale based on epsilon: scale = 2 * epsilon (bucket size)
        let scale = 2.0 * self.epsilon; 
        
        let mut compressed = Vec::with_capacity(residual.len() * 2);
        
        let floats = residual.to_f32();
        
        for &val in &floats {
            let q = (val / scale).round();
            
            // Clamp to i16 range [-32768, 32767]
            // With i16, max value is ~32k * 2e-3 = 64.0 (for eps=1e-3).
            // This covers almost all physical simulations unless divergent.
            let clamped = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
            
            // Push as little endian bytes
            compressed.extend_from_slice(&clamped.to_le_bytes());
        }
        
        (compressed, scale)
    }

    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        let mut reconstructed = Vec::with_capacity(compressed.len() / 2);
        
        for chunk in compressed.chunks(2) {
            if chunk.len() == 2 {
                let bytes = [chunk[0], chunk[1]];
                let val = i16::from_le_bytes(bytes);
                let q = val as f32;
                reconstructed.push(q * scale);
            }
        }
            
        FixedPointBuffer::from_f32(&reconstructed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_bound() {
        let epsilon = 1e-3;
        let quantizer = ErrorBoundedQuantizer::new(epsilon);
        
        // Create data within range
        let input = FixedPointBuffer::from_f32(&[0.0, 0.0005, -0.0005, 0.0015, -0.1]);
        
        let (bytes, scale) = quantizer.quantize(&input);
        let output = quantizer.dequantize(&bytes, scale);
        
        let in_f32 = input.to_f32();
        let out_f32 = output.to_f32();
        
        for (i, &val) in in_f32.iter().enumerate() {
            let diff = (val - out_f32[i]).abs();
            // Allow small float/fixed conversion noise on top of epsilon
            assert!(diff < epsilon + 1e-6, "Error {}: {} vs {} (diff {})", i, val, out_f32[i], diff);
        }
    }
}
