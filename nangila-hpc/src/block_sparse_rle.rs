use nangila_core::Quantizer;
use nangila_math::FixedPointBuffer;

/// Block-sparse + mask-RLE quantizer: error-bounded INT16 with 32-wide masks, and
/// run-length encoding of zero-mask blocks to reduce header overhead.
///
/// Format:
///   [u32 n_elems]
///   Repeated runs:
///     - Zero-mask run: 1 byte opcode in [0x80..0xFF], count = (opcode & 0x7F) + 1 blocks
///     - Literal run:   1 byte opcode in [0x00..0x7F], count = opcode + 1 blocks,
///                      followed by for each block: [u32 mask][i16 payload...]
/// Scale is returned separately (2*epsilon) and used to dequantize.
pub struct BlockSparseRleQuantizer {
    pub epsilon: f32,
    pub block_size: usize,
}

impl BlockSparseRleQuantizer {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon, block_size: 32 }
    }
}

impl Quantizer for BlockSparseRleQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        let scale = 2.0 * self.epsilon;
        let floats = residual.to_f32();
        let n = floats.len();
        let bs = self.block_size;

        // First compute masks and payloads per block
        let num_blocks = (n + bs - 1) / bs;
        let mut masks: Vec<u32> = Vec::with_capacity(num_blocks);
        let mut payloads: Vec<Vec<i16>> = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let start = b * bs;
            let end = (start + bs).min(n);
            let mut mask: u32 = 0;
            let mut vals: Vec<i16> = Vec::new();
            for i in 0..(end - start) {
                let v = floats[start + i];
                let q = (v / scale).round();
                let qi = q.max(i16::MIN as f32).min(i16::MAX as f32) as i16;
                if qi != 0 {
                    mask |= 1u32 << i;
                    vals.push(qi);
                }
            }
            masks.push(mask);
            payloads.push(vals);
        }

        // Now RLE encode masks: zero-mask runs and literal runs
        let mut out = Vec::with_capacity(4 + num_blocks * 6); // rough guess
        out.extend_from_slice(&(n as u32).to_le_bytes());

        let mut b = 0usize;
        while b < num_blocks {
            // Zero run
            if masks[b] == 0 {
                let mut count = 1usize;
                while b + count < num_blocks && masks[b + count] == 0 && count < 128 { count += 1; }
                let opcode = 0x80 | ((count - 1) as u8);
                out.push(opcode);
                b += count;
                continue;
            }
            // Literal run
            let mut count = 1usize;
            while b + count < num_blocks && masks[b + count] != 0 && count < 128 { count += 1; }
            let opcode = (count - 1) as u8; // 0..127
            out.push(opcode);
            for j in 0..count {
                let idx = b + j;
                let mask = masks[idx];
                out.extend_from_slice(&mask.to_le_bytes());
                for qv in &payloads[idx] { out.extend_from_slice(&qv.to_le_bytes()); }
            }
            b += count;
        }

        (out, scale)
    }

    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        if compressed.len() < 4 { return FixedPointBuffer::from_f32(&[]); }
        let n = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
        let bs = self.block_size;
        let num_blocks = (n + bs - 1) / bs;
        let mut out: Vec<f32> = Vec::with_capacity(num_blocks * bs);
        let mut ptr = 4usize;
        let mut b = 0usize;
        while b < num_blocks && ptr < compressed.len() {
            let opcode = compressed[ptr];
            ptr += 1;
            if opcode & 0x80 != 0 {
                // zero-mask run
                let count = ((opcode & 0x7F) as usize) + 1;
                for k in 0..count {
                    let block_len = if b + k == num_blocks - 1 { n - (b + k) * bs } else { bs };
                    for _ in 0..block_len { out.push(0.0); }
                }
                b += count;
            } else {
                let count = (opcode as usize) + 1;
                for _ in 0..count {
                    if ptr + 4 > compressed.len() { break; }
                    let mask = u32::from_le_bytes([compressed[ptr], compressed[ptr+1], compressed[ptr+2], compressed[ptr+3]]);
                    ptr += 4;
                    let block_len = if b == num_blocks - 1 { n - b * bs } else { bs };
                    for i in 0..block_len {
                        if (mask >> i) & 1 == 1 {
                            if ptr + 2 <= compressed.len() {
                                let qv = i16::from_le_bytes([compressed[ptr], compressed[ptr+1]]) as f32;
                                ptr += 2;
                                out.push(qv * scale);
                            } else {
                                out.push(0.0);
                            }
                        } else {
                            out.push(0.0);
                        }
                    }
                    b += 1;
                }
            }
        }
        if out.len() > n { out.truncate(n); }
        FixedPointBuffer::from_f32(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_epsilon() {
        let eps = 1e-3;
        let q = BlockSparseRleQuantizer::new(eps);
        // Build a smooth signal with long zero-quantized stretches
        let mut data: Vec<f32> = Vec::new();
        for t in 0..1024 { data.push(((t as f32) * 0.005).sin() * 1e-4); } // near zero
        for t in 0..1024 { data.push(((t as f32) * 0.01).sin()); }         // non-zero
        let buf = FixedPointBuffer::from_f32(&data);
        let (bytes, scale) = q.quantize(&buf);
        let recon = q.dequantize(&bytes, scale).to_f32();
        for i in 0..data.len() {
            assert!((data[i] - recon[i]).abs() <= eps + 1e-6);
        }
    }
}

