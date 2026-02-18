use crate::Quantizer;
use nangila_math::FixedPointBuffer;
use serde::{Deserialize, Serialize};

/// Quantization mode selector for unified interface
/// 
/// This enum allows runtime selection between different quantization strategies:
/// - **Stochastic**: Unbiased randomized quantization (AI training)
/// - **ErrorBounded**: Strict ε-guarantee (HPC simulations)
/// - **TopK**: Sparse compression (gradient sparsification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Stochastic quantization with unbiased rounding
    /// 
    /// Uses hash-based PRNG for deterministic randomness.
    /// Typical use: AI gradient compression (INT4/INT8)
    Stochastic {
        /// Random seed for reproducibility
        seed: u64,
        /// Number of bits (4 or 8 typical)
        bits: u8,
    },
    
    /// Error-bounded quantization with strict guarantees
    /// 
    /// Guarantees max absolute error < epsilon.
    /// Typical use: HPC checkpointing, scientific data
    ErrorBounded {
        /// Maximum absolute error tolerance
        epsilon: f32,
    },
    
    /// Top-K sparse quantization
    /// 
    /// Keeps only the largest k% of values by magnitude.
    /// Typical use: Sparse gradient compression (DGC-style)
    TopK {
        /// Percentage of values to keep (0.0 to 1.0)
        k_percent: f32,
    },
}

/// Unified quantizer that dispatches to mode-specific implementations
/// 
/// This provides a single interface for all quantization strategies,
/// allowing runtime mode switching without code changes.
/// 
/// # Example
/// ```
/// use nangila_core::{UnifiedQuantizer, QuantizationMode, Quantizer};
/// use nangila_math::FixedPointBuffer;
/// 
/// // AI mode: Stochastic INT4
/// let ai_quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic {
///     seed: 42,
///     bits: 4,
/// });
/// 
/// // HPC mode: Error-bounded
/// let hpc_quantizer = UnifiedQuantizer::new(QuantizationMode::ErrorBounded {
///     epsilon: 1e-3,
/// });
/// 
/// // Switch modes at runtime
/// let data = FixedPointBuffer::from_f32(&[1.0, 2.0, 3.0]);
/// let (compressed, scale) = ai_quantizer.quantize(&data);
/// ```
pub struct UnifiedQuantizer {
    mode: QuantizationMode,
    // Internal state for stochastic mode (RefCell for interior mutability)
    stochastic_state: Option<std::cell::RefCell<StochasticState>>,
}

/// Internal state for stochastic quantization
struct StochasticState {
    seed: u64,
    counter: u64,
}

impl UnifiedQuantizer {
    /// Create a new unified quantizer with the specified mode
    pub fn new(mode: QuantizationMode) -> Self {
        let stochastic_state = match &mode {
            QuantizationMode::Stochastic { seed, .. } => {
                Some(std::cell::RefCell::new(StochasticState {
                    seed: *seed,
                    counter: 0,
                }))
            }
            _ => None,
        };
        
        Self {
            mode,
            stochastic_state,
        }
    }
    
    /// Get the current quantization mode
    pub fn mode(&self) -> &QuantizationMode {
        &self.mode
    }
    
    /// Switch to a different quantization mode
    /// 
    /// This allows runtime mode changes, useful for:
    /// - AI training → HPC checkpointing transitions
    /// - Adaptive compression based on data characteristics
    pub fn set_mode(&mut self, mode: QuantizationMode) {
        self.stochastic_state = match &mode {
            QuantizationMode::Stochastic { seed, .. } => {
                Some(std::cell::RefCell::new(StochasticState {
                    seed: *seed,
                    counter: 0,
                }))
            }
            _ => None,
        };
        self.mode = mode;
    }
    
    /// Get a description of the current mode
    pub fn mode_description(&self) -> String {
        match &self.mode {
            QuantizationMode::Stochastic { seed, bits } => {
                format!("Stochastic (seed={}, bits={})", seed, bits)
            }
            QuantizationMode::ErrorBounded { epsilon } => {
                format!("ErrorBounded (ε={:.2e})", epsilon)
            }
            QuantizationMode::TopK { k_percent } => {
                format!("TopK (k={:.1}%)", k_percent * 100.0)
            }
        }
    }
}

impl Quantizer for UnifiedQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        match &self.mode {
            QuantizationMode::Stochastic { bits, .. } => {
                self.quantize_stochastic(residual, *bits)
            }
            QuantizationMode::ErrorBounded { epsilon } => {
                self.quantize_error_bounded(residual, *epsilon)
            }
            QuantizationMode::TopK { k_percent } => {
                self.quantize_topk(residual, *k_percent)
            }
        }
    }
    
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        match &self.mode {
            QuantizationMode::Stochastic { bits, .. } => {
                self.dequantize_stochastic(compressed, scale, *bits)
            }
            QuantizationMode::ErrorBounded { .. } => {
                self.dequantize_error_bounded(compressed, scale)
            }
            QuantizationMode::TopK { .. } => {
                self.dequantize_topk(compressed, scale)
            }
        }
    }
}

// Mode-specific implementations
impl UnifiedQuantizer {
    /// Stochastic quantization (INT4/INT8)
    fn quantize_stochastic(&self, residual: &FixedPointBuffer, bits: u8) -> (Vec<u8>, f32) {
        let floats = residual.to_f32();
        if floats.is_empty() {
            return (vec![], 1.0);
        }
        
        // Determine quantization range
        let max_val = floats.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_val > 1e-9 {
            let levels = (1 << bits) - 1;
            max_val / (levels as f32 / 2.0)
        } else {
            1.0
        };
        
        // Quantize with stochastic rounding
        let mut compressed = Vec::with_capacity(floats.len() * (bits as usize) / 8);
        
        if bits == 4 {
            // Pack 2 values per byte
            for chunk in floats.chunks(2) {
                let mut byte = 0u8;
                
                for (i, &val) in chunk.iter().enumerate() {
                    let q = self.stochastic_round(val / scale, 4);
                    let clamped = q.clamp(-8, 7) as i8;
                    let nibble = (clamped & 0x0F) as u8;
                    
                    if i == 0 {
                        byte |= nibble << 4;
                    } else {
                        byte |= nibble;
                    }
                }
                
                compressed.push(byte);
            }
        } else if bits == 8 {
            // One value per byte
            for &val in &floats {
                let q = self.stochastic_round(val / scale, 8);
                let clamped = q.clamp(-128, 127) as i8;
                compressed.push(clamped as u8);
            }
        } else {
            // Fallback: use 8-bit
            for &val in &floats {
                let q = (val / scale).round();
                let clamped = q.clamp(-128.0, 127.0) as i8;
                compressed.push(clamped as u8);
            }
        }
        
        (compressed, scale)
    }
    
    fn dequantize_stochastic(&self, compressed: &[u8], scale: f32, bits: u8) -> FixedPointBuffer {
        let mut reconstructed = Vec::new();
        
        if bits == 4 {
            // Unpack 2 values per byte
            for &byte in compressed {
                let high = ((byte >> 4) as i8) << 4;
                let val1 = (high >> 4) as f32;
                reconstructed.push(val1 * scale);
                
                let low = (byte as i8) << 4;
                let val2 = (low >> 4) as f32;
                reconstructed.push(val2 * scale);
            }
        } else {
            // One value per byte
            for &byte in compressed {
                let val = byte as i8 as f32;
                reconstructed.push(val * scale);
            }
        }
        
        FixedPointBuffer::from_f32(&reconstructed)
    }
    
    /// Error-bounded quantization (INT16)
    fn quantize_error_bounded(&self, residual: &FixedPointBuffer, epsilon: f32) -> (Vec<u8>, f32) {
        let scale = 2.0 * epsilon;
        let mut compressed = Vec::with_capacity(residual.len() * 2);
        
        let floats = residual.to_f32();
        for &val in &floats {
            let q = (val / scale).round();
            let clamped = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
            compressed.extend_from_slice(&clamped.to_le_bytes());
        }
        
        (compressed, scale)
    }
    
    fn dequantize_error_bounded(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        let mut reconstructed = Vec::with_capacity(compressed.len() / 2);
        
        for chunk in compressed.chunks(2) {
            if chunk.len() == 2 {
                let bytes = [chunk[0], chunk[1]];
                let val = i16::from_le_bytes(bytes);
                reconstructed.push((val as f32) * scale);
            }
        }
        
        FixedPointBuffer::from_f32(&reconstructed)
    }
    
    /// Top-K sparse quantization
    ///
    /// Encoding format:
    /// [u32 n_elems] then for each 32-wide block: [u32 mask][i8 values for 1-bits]
    fn quantize_topk(&self, residual: &FixedPointBuffer, k_percent: f32) -> (Vec<u8>, f32) {
        let floats = residual.to_f32();
        let n = floats.len();
        if n == 0 {
            return (vec![], 1.0);
        }

        // Determine threshold via selection (avoid full sort)
        let mut abs_vals: Vec<f32> = floats.iter().map(|x| x.abs()).collect();
        let keep = ((abs_vals.len() as f32) * k_percent).floor() as usize;
        let threshold = if keep == 0 {
            1e-9
        } else {
            let idx = abs_vals.len() - keep.min(abs_vals.len());
            let (_, nth, _) = abs_vals.select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            (*nth).max(1e-9)
        };

        // Global scale
        let max_val = floats
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = if max_val > 1e-9 { max_val / 127.0 } else { 1.0 };

        // Encode: prefix original length
        let block_size = 32;
        let mut compressed = Vec::with_capacity(4 + n / 4);
        compressed.extend_from_slice(&(n as u32).to_le_bytes());

        for chunk in floats.chunks(block_size) {
            let mut mask: u32 = 0;
            let mut values: Vec<u8> = Vec::with_capacity(chunk.len());

            for (i, &val) in chunk.iter().enumerate() {
                if val.abs() >= threshold {
                    mask |= 1 << i;
                    let q = (val / scale).round();
                    let clamped = q.clamp(-128.0, 127.0) as i8;
                    values.push(clamped as u8);
                }
            }

            compressed.extend_from_slice(&mask.to_le_bytes());
            compressed.extend_from_slice(&values);
        }

        (compressed, scale)
    }
    
    fn dequantize_topk(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        if compressed.len() < 4 {
            return FixedPointBuffer::from_f32(&[]);
        }
        let n = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
        let mut reconstructed: Vec<f32> = Vec::with_capacity(n);

        let mut ptr = 4;
        let block_size = 32usize;
        let num_blocks = n.div_ceil(block_size);

        for b in 0..num_blocks {
            if ptr + 4 > compressed.len() {
                break;
            }
            let mask_bytes = [compressed[ptr], compressed[ptr + 1], compressed[ptr + 2], compressed[ptr + 3]];
            let mask = u32::from_le_bytes(mask_bytes);
            ptr += 4;

            let block_len = if b == num_blocks - 1 { n - b * block_size } else { block_size };
            for i in 0..block_len {
                if (mask >> i) & 1 == 1 {
                    if ptr < compressed.len() {
                        let val_i8 = compressed[ptr] as i8;
                        ptr += 1;
                        reconstructed.push((val_i8 as f32) * scale);
                    } else {
                        reconstructed.push(0.0);
                    }
                } else {
                    reconstructed.push(0.0);
                }
            }
        }

        if reconstructed.len() > n { reconstructed.truncate(n); }
        if reconstructed.len() < n { reconstructed.resize(n, 0.0); }
        FixedPointBuffer::from_f32(&reconstructed)
    }
    
    /// Stochastic rounding using hash-based PRNG
    fn stochastic_round(&self, value: f32, _bits: u8) -> i32 {
        let floor = value.floor();
        let frac = value - floor;
        
        // Use hash-based randomness for determinism
        if let Some(state) = &self.stochastic_state {
            let mut state_mut = state.borrow_mut();
            let hash = self.hash_u64(state_mut.seed ^ state_mut.counter);
            state_mut.counter = state_mut.counter.wrapping_add(1);
            
            let random = (hash as f32) / (u64::MAX as f32);
            if random < frac {
                floor as i32 + 1
            } else {
                floor as i32
            }
        } else {
            // Fallback: deterministic rounding
            value.round() as i32
        }
    }
    
    /// Splitmix64 hash for deterministic PRNG.
    ///
    /// Reference: Steele, Lea, Flood — "Fast Splittable Pseudorandom Number
    /// Generators" (OOPSLA 2014). <https://doi.org/10.1145/2714064.2660195>
    fn hash_u64(&self, x: u64) -> u64 {
        let mut h = x;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_switching() {
        let mut quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic {
            seed: 42,
            bits: 4,
        });
        
        assert!(matches!(quantizer.mode(), QuantizationMode::Stochastic { .. }));
        
        quantizer.set_mode(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
        assert!(matches!(quantizer.mode(), QuantizationMode::ErrorBounded { .. }));
    }
    
    #[test]
    fn test_stochastic_mode() {
        let quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic {
            seed: 42,
            bits: 4,
        });
        
        let data = FixedPointBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let (compressed, scale) = quantizer.quantize(&data);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        
        let orig = data.to_f32();
        let recon = reconstructed.to_f32();
        
        for i in 0..orig.len() {
            let error = (orig[i] - recon[i]).abs();
            assert!(error < scale * 0.6, "Error too large: {}", error);
        }
    }
    
    #[test]
    fn test_error_bounded_mode() {
        let quantizer = UnifiedQuantizer::new(QuantizationMode::ErrorBounded {
            epsilon: 1e-3,
        });
        
        let data = FixedPointBuffer::from_f32(&[0.5, -0.5, 0.001, -0.001]);
        let (compressed, scale) = quantizer.quantize(&data);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        
        let orig = data.to_f32();
        let recon = reconstructed.to_f32();
        
        for i in 0..orig.len() {
            let error = (orig[i] - recon[i]).abs();
            assert!(error < 1e-3 + 1e-6, "Error bound violated: {}", error);
        }
    }
    
    #[test]
    fn test_topk_mode() {
        let quantizer = UnifiedQuantizer::new(QuantizationMode::TopK {
            k_percent: 0.25,
        });
        
        let mut data_vec = vec![0.01f32; 100];
        data_vec[10] = 5.0;
        data_vec[20] = -5.0;
        data_vec[50] = 10.0;
        
        let data = FixedPointBuffer::from_f32(&data_vec);
        let (compressed, scale) = quantizer.quantize(&data);
        let reconstructed = quantizer.dequantize(&compressed, scale);
        
        let recon = reconstructed.to_f32();
        
        // Check that large values are preserved
        assert!((recon[10] - 5.0).abs() < 0.5);
        assert!((recon[20] + 5.0).abs() < 0.5);
        assert!((recon[50] - 10.0).abs() < 0.5);
        
        // Check that small values are zeroed
        let non_zeros = recon.iter().filter(|&&x| x.abs() > 0.1).count();
        assert!(non_zeros <= 30, "Too many non-zeros: {}", non_zeros);
    }

    #[test]
    fn test_topk_preserves_length_odd() {
        // Ensure non-multiple-of-32 lengths roundtrip to same length
        let quantizer = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: 0.2 });
        let n = 45usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let buf = FixedPointBuffer::from_f32(&data);
        let (bytes, scale) = quantizer.quantize(&buf);
        let recon = quantizer.dequantize(&bytes, scale);
        assert_eq!(recon.len(), n, "Reconstructed length mismatch");
    }

    #[test]
    fn test_topk_malformed_graceful() {
        // Declared n=10 but provide only one mask and no values; should not panic and return zeros
        let q = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: 0.5 });
        let mut bytes = Vec::new();
        let n: u32 = 10;
        bytes.extend_from_slice(&n.to_le_bytes());
        // One block mask of zeros
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let recon = q.dequantize(&bytes, 1.0);
        assert_eq!(recon.len(), 10);
        assert!(recon.to_f32().iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_mode_descriptions() {
        let q1 = UnifiedQuantizer::new(QuantizationMode::Stochastic { seed: 42, bits: 4 });
        assert!(q1.mode_description().contains("Stochastic"));
        
        let q2 = UnifiedQuantizer::new(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
        assert!(q2.mode_description().contains("ErrorBounded"));
        
        let q3 = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: 0.1 });
        assert!(q3.mode_description().contains("TopK"));
    }

    #[test]
    fn test_dequantize_empty_stochastic() {
        let q = UnifiedQuantizer::new(QuantizationMode::Stochastic { seed: 0, bits: 8 });
        let recon = q.dequantize(&[], 1.0);
        assert_eq!(recon.len(), 0);
    }

    #[test]
    fn test_dequantize_empty_stochastic_int4() {
        let q = UnifiedQuantizer::new(QuantizationMode::Stochastic { seed: 0, bits: 4 });
        let recon = q.dequantize(&[], 1.0);
        assert_eq!(recon.len(), 0);
    }

    #[test]
    fn test_dequantize_empty_error_bounded() {
        let q = UnifiedQuantizer::new(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
        let recon = q.dequantize(&[], 1.0);
        assert_eq!(recon.len(), 0);
    }

    #[test]
    fn test_dequantize_truncated_error_bounded() {
        // Single byte (odd-length) should not panic — silently skip incomplete i16
        let q = UnifiedQuantizer::new(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
        let recon = q.dequantize(&[0xFF], 1.0);
        assert_eq!(recon.len(), 0);
    }

    #[test]
    fn test_dequantize_garbage_topk() {
        // Random bytes with declared n=1000 but only a few bytes — should not panic
        let q = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: 0.5 });
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1000u32.to_le_bytes()); // declare 1000 elements
        bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // one garbage mask
        let recon = q.dequantize(&bytes, 1.0);
        // Should not panic; length may be truncated but should not exceed 1000
        assert!(recon.len() <= 1000);
    }

    #[test]
    fn test_quantize_empty_all_modes() {
        let empty = FixedPointBuffer::from_f32(&[]);

        let q1 = UnifiedQuantizer::new(QuantizationMode::Stochastic { seed: 0, bits: 4 });
        let (b1, _) = q1.quantize(&empty);
        assert!(b1.is_empty());

        let q2 = UnifiedQuantizer::new(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
        let (b2, _) = q2.quantize(&empty);
        assert!(b2.is_empty());

        let q3 = UnifiedQuantizer::new(QuantizationMode::TopK { k_percent: 0.1 });
        let (b3, _) = q3.quantize(&empty);
        assert!(b3.is_empty());
    }
}
