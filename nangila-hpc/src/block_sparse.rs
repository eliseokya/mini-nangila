use nangila_core::Quantizer;
use nangila_math::FixedPointBuffer;

/// Block-sparse quantizer: error-bounded INT16 + per-block bitmask of non-zeros.
///
/// Encoding per step (inside the `compressed` payload returned by `quantize`):
/// [u32 n_elems] [for each 32-wide block: u32 mask (LSB=first elem) then i16 values for each 1-bit]
/// Scale = 2 * epsilon, preserving |error| <= epsilon.
pub struct BlockSparseQuantizer {
    pub epsilon: f32,
    pub block_size: usize,
}

impl BlockSparseQuantizer {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon, block_size: 32 }
    }
}

impl Quantizer for BlockSparseQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        let scale = 2.0 * self.epsilon;
        let floats = residual.to_f32();
        let n = floats.len();
        let mut out = Vec::with_capacity(4 + n * 2);
        // Write original length
        out.extend_from_slice(&(n as u32).to_le_bytes());

        let bs = self.block_size;
        let mut idx = 0usize;
        while idx < n {
            let chunk_len = (n - idx).min(bs);
            // Build mask and temp buffer of non-zero quantized values
            let mut mask: u32 = 0;
            let mut qvals: Vec<i16> = Vec::with_capacity(chunk_len);
            for i in 0..chunk_len {
                let val = floats[idx + i];
                let q = (val / scale).round();
                let qi = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
                if qi != 0 {
                    mask |= 1u32 << i;
                    qvals.push(qi);
                }
            }
            // Write u32 mask
            out.extend_from_slice(&mask.to_le_bytes());
            // Write only non-zero values
            for qv in qvals {
                out.extend_from_slice(&qv.to_le_bytes());
            }
            idx += chunk_len;
        }

        (out, scale)
    }

    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        if compressed.len() < 4 {
            return FixedPointBuffer::from_f32(&[]);
        }
        let n = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
        let mut ptr = 4usize;
        let bs = self.block_size;
        let num_blocks = (n + bs - 1) / bs;
        let mut out: Vec<f32> = Vec::with_capacity(num_blocks * bs);

        for b in 0..num_blocks {
            if ptr + 4 > compressed.len() { break; }
            let mask = u32::from_le_bytes([
                compressed[ptr], compressed[ptr+1], compressed[ptr+2], compressed[ptr+3]
            ]);
            ptr += 4;
            let block_len = if b == num_blocks - 1 { n - b * bs } else { bs };
            for i in 0..block_len {
                if (mask >> i) & 1 == 1 {
                    if ptr + 2 <= compressed.len() {
                        let val = i16::from_le_bytes([compressed[ptr], compressed[ptr+1]]) as f32;
                        ptr += 2;
                        out.push(val * scale);
                    } else {
                        out.push(0.0);
                    }
                } else {
                    out.push(0.0);
                }
            }
        }

        // Truncate in case of any padding or parse errors beyond n
        if out.len() > n { out.truncate(n); }
        FixedPointBuffer::from_f32(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let eps = 1e-3;
        let q = BlockSparseQuantizer::new(eps);
        let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.01).sin()).collect();
        let buf = FixedPointBuffer::from_f32(&data);
        let (bytes, scale) = q.quantize(&buf);
        let recon = q.dequantize(&bytes, scale).to_f32();
        for i in 0..data.len() {
            assert!((data[i] - recon[i]).abs() <= eps + 1e-6);
        }
    }
}
