use nangila_core::Quantizer;
use nangila_math::FixedPointBuffer;

/// Run-Length Encoded Quantizer (i16)
/// Best for data with many zero residuals (perfect prediction).
/// Format:
/// [Opcode: u8] [Data...]
/// Opcode 0x00..0x7F: Literal Run of (N+1) items. Followed by (N+1) i16 values.
/// Opcode 0x80..0xFF: Zero Run of (N-128+1) items. (0 to 127 zeroes).
pub struct RunLengthQuantizer {
    epsilon: f32,
}

impl RunLengthQuantizer {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}

impl Quantizer for RunLengthQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        let floats = residual.to_f32();
        if floats.is_empty() { return (vec![], 1.0); }
        
        let scale = 2.0 * self.epsilon;
        let mut quantized_vals = Vec::with_capacity(floats.len());
        
        // 1. Quantize first to i16
        for &val in &floats {
            let q = (val / scale).round();
            let clamped = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
            quantized_vals.push(clamped);
        }
        
        let mut compressed = Vec::new();
        let mut i = 0;
        
        while i < quantized_vals.len() {
            if quantized_vals[i] == 0 {
                // Zero Run
                let start = i;
                while i < quantized_vals.len() && quantized_vals[i] == 0 && (i - start) < 128 {
                    i += 1;
                }
                let count = i - start;
                // Encode count: 1..128 -> 0..127 + 128 = 128..255 (Opcode 0x80..0xFF)
                // Opcode = 128 + count - 1
                let opcode = (128 + count - 1) as u8;
                compressed.push(opcode);
            } else {
                // Literal Run
                let start = i;
                // Run until zero or max length 128
                while i < quantized_vals.len() && quantized_vals[i] != 0 && (i - start) < 128 {
                    i += 1;
                }
                let count = i - start;
                // Encode count: 1..128 -> 0..127 (Opcode 0x00..0x7F)
                // Opcode = count - 1
                let opcode = (count - 1) as u8;
                compressed.push(opcode);
                
                // Append literals
                for j in 0..count {
                    compressed.extend_from_slice(&quantized_vals[start + j].to_le_bytes());
                }
            }
        }
        
        (compressed, scale)
    }

    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        let mut reconstructed = Vec::new();
        let mut ptr = 0;
        
        while ptr < compressed.len() {
            let opcode = compressed[ptr];
            ptr += 1;
            
            if opcode >= 128 {
                // Zero Run
                let count = (opcode - 128 + 1) as usize;
                reconstructed.extend(std::iter::repeat(0.0).take(count));
            } else {
                // Literal Run
                let count = (opcode + 1) as usize;
                for _ in 0..count {
                    if ptr + 2 > compressed.len() { break; } // Safety
                    let bytes = [compressed[ptr], compressed[ptr+1]];
                    ptr += 2;
                    let val = i16::from_le_bytes(bytes);
                    reconstructed.push((val as f32) * scale);
                }
            }
        }
        
        FixedPointBuffer::from_f32(&reconstructed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_compression() {
        // Data: 100 zeros, 1 value, 100 zeros.
        let mut data = vec![0.0f32; 201];
        data[100] = 10.0;
        
        let buffer = FixedPointBuffer::from_f32(&data);
        let quantizer = RunLengthQuantizer::new(0.01); // scale = 0.02
        
        let (bytes, scale) = quantizer.quantize(&buffer);
        
        // Expected:
        // 100 zeros -> Run 100. (1 byte opcode)
        // 1 literal (10.0/0.02 = 500) -> Opcode (1) + 2 bytes. (3 bytes)
        // 100 zeros -> Run 100. (1 byte opcode)
        // Total: 5 bytes.
        // Original: 201 * 4 = 804 bytes.
        // Ratio: 160x!
        
        println!("Compressed Size: {}", bytes.len());
        assert!(bytes.len() <= 6, "Compression failed: {} bytes", bytes.len());
        
        let reconstructed = quantizer.dequantize(&bytes, scale);
        let recon_f32 = reconstructed.to_f32();
        
        assert_eq!(recon_f32.len(), 201);
        assert!((recon_f32[100] - 10.0).abs() < scale);
        assert_eq!(recon_f32[0], 0.0);
        assert_eq!(recon_f32[200], 0.0);
    }
}
