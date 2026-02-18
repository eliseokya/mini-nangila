#[cfg(feature = "hdf5")]
mod tests {
    use nangila_checkpoint::h5;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn write_read_compressed_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_string_lossy().to_string();

        let len = 12345usize;
        let data: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 * 0.01;
                (t * 0.5).cos() + ((i % 100) as f32) * 1e-4
            })
            .collect();

        h5::write_chunked_dataset(&path, "traj", &data, 1024, 1e-3, "ErrorBoundedINT16").unwrap();
        let out = h5::read_compressed_1d(&path, "traj").unwrap();

        assert_eq!(out.len(), data.len());
        let mut max_err = 0.0f32;
        for i in 0..len {
            let e = (out[i] - data[i]).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err <= 1e-3 + 1e-6, "max_err {} exceeds epsilon", max_err);

        // Also write raw for comparison
        h5::write_dataset(&path, "traj_raw", &data, 0.0, "raw").unwrap();
        let back_raw = h5::read_dataset(&path, "traj_raw").unwrap();
        assert_eq!(back_raw.len(), data.len());
    }
}

