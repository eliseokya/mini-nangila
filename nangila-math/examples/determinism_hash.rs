use nangila_math::FixedPointBuffer;

fn main() {
    let vals: Vec<f32> = vec![
        0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 127.0, -128.0,
    ];
    let buf = FixedPointBuffer::from_f32(&vals);
    let mut bytes = Vec::with_capacity(buf.data.len() * 4);
    for v in &buf.data { bytes.extend_from_slice(&v.to_le_bytes()); }
    let digest = sha256(&bytes);
    println!("Q823_HASH {}", digest);
}

fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let out = hasher.finalize();
    hex::encode(out)
}

