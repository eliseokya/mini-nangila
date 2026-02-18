use nangila_core::{Quantizer, UnifiedQuantizer, QuantizationMode};
use nangila_math::FixedPointBuffer;

/// Top-K Sparse Quantizer
///
/// Keeps only the largest `k` percent of gradients by magnitude,
/// using a block-based bitmask encoding for CPU efficiency.
///
/// Encoding format (per quantized buffer):
/// `[u32 n_elems]` then for each 32-wide block: `[u32 mask][i8 payload...]`
///
/// This is a thin wrapper around [`UnifiedQuantizer`] with [`QuantizationMode::TopK`].
pub struct TopKQuantizer {
    inner: UnifiedQuantizer,
}

impl TopKQuantizer {
    pub fn new(k_percent: f32) -> Self {
        Self {
            inner: UnifiedQuantizer::new(QuantizationMode::TopK { k_percent }),
        }
    }
}

impl Quantizer for TopKQuantizer {
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
    fn test_topk_compression() {
        // Data: 100 values. 95 are small (<0.1), 5 are large (>1.0).
        // 5% TopK should keep only large ones.
        
        let mut data = vec![0.01f32; 100];
        // Set spikes
        data[10] = 5.0;
        data[20] = -5.0;
        data[50] = 10.0;
        data[80] = -10.0;
        data[90] = 8.0;
        
        let buffer = FixedPointBuffer::from_f32(&data);
        let quantizer = TopKQuantizer::new(0.05); // 5%
        
        let (bytes, scale) = quantizer.quantize(&buffer);
        
        // Original size: 100 * 4 = 400 bytes.
        // Compressed: 
        // 4 blocks of 32 (last partial? code pads 32 implicitly in loop? No chunk iteration handles last).
        // 100 items -> chunks: 32, 32, 32, 4.
        // Masks: 4 * 4 = 16 bytes.
        // Values: 5 kept * 1 byte = 5 bytes.
        // Total ~ 21 bytes.
        // Ratio: 400 / 21 ~ 19x.
        
        println!("Compressed Size: {}", bytes.len());
        
        let reconstructed = quantizer.dequantize(&bytes, scale);
        let recon_f32 = reconstructed.to_f32();
        
        // Identify kept indices
        let mut non_zeros = 0;
        for (i, &val) in recon_f32.iter().enumerate() {
            if val.abs() > 0.0 {
                non_zeros += 1;
                // Check if it's one of our spikes
                let original = data[i];
                // Check error
                assert!((val - original).abs() < scale * 0.6, "Value at {} mismatch: {} vs {}", i, val, original);
            }
        }
        
        assert!(non_zeros <= 6, "Kept too many values: {}", non_zeros); 
        assert!(non_zeros >= 5, "Lost spikes: {}", non_zeros);
    }

    #[test]
    fn test_topk_delegation_equivalence() {
        // Verify TopKQuantizer produces byte-identical output to UnifiedQuantizer
        let k = 0.10;
        let topk = TopKQuantizer::new(k);
        let unified = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: k });

        let data: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.05).sin()).collect();
        let buf = FixedPointBuffer::from_f32(&data);

        let (topk_bytes, topk_scale) = topk.quantize(&buf);
        let (unified_bytes, unified_scale) = unified.quantize(&buf);

        assert_eq!(topk_scale, unified_scale, "Scales differ");
        assert_eq!(topk_bytes, unified_bytes, "Compressed bytes differ");

        // Also verify dequantize produces identical results
        let topk_recon = topk.dequantize(&topk_bytes, topk_scale);
        let unified_recon = unified.dequantize(&unified_bytes, unified_scale);
        assert_eq!(topk_recon.to_f32(), unified_recon.to_f32());
    }
}
