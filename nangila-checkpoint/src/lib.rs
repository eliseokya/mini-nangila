//! Checkpoint I/O utilities (HDF5 feature-gated)
//!
//! This crate provides convenience helpers to write/read compressed
//! datasets along with compression metadata. HDF5 support is optional.

#[cfg(feature = "hdf5")]
pub mod h5 {
    use hdf5::{File, Result as H5Result};
    use nangila_hpc::{ErrorBoundedQuantizer, LinearPredictor};
    use nangila_math::FixedPointBuffer;

    /// Writes a 1D f32 dataset and attaches basic compression metadata.
    /// Data is expected to be reconstructed values (post-dequantize).
    pub fn write_dataset(
        path: &str,
        dataset: &str,
        data: &[f32],
        epsilon: f32,
        codec: &str,
    ) -> H5Result<()> {
        let file = File::create(path)?;
        let ds = file.new_dataset_builder().with_data(data).create(dataset)?;
        ds.new_attr::<f32>().create("epsilon")?.write_scalar(&epsilon)?;
        ds.new_attr::<&str>().create("codec")?.write_scalar(&codec)?;
        Ok(())
    }

    /// Writes a chunked dataset for streaming-friendly checkpointing.
    pub fn write_chunked_dataset(
        path: &str,
        dataset: &str,
        data: &[f32],
        chunk: usize,
        epsilon: f32,
        codec: &str,
    ) -> H5Result<()> {
        let file = File::create(path)?;
        let grp = file.create_group(dataset)?;

        // Concatenate compressed bytes into a single blob with offsets and per-chunk scales.
        let mut predictor = LinearPredictor::new();
        let quant = ErrorBoundedQuantizer::new(epsilon);

        let total_len = data.len();
        let chunk = chunk.max(1);
        let num_chunks = (total_len + chunk - 1) / chunk;

        let mut blob: Vec<u8> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(num_chunks + 1);
        let mut scales: Vec<f32> = Vec::with_capacity(num_chunks);
        offsets.push(0);

        for c in 0..num_chunks {
            let start = c * chunk;
            let end = ((c + 1) * chunk).min(total_len);
            let mut buf_vec = vec![0.0f32; chunk];
            buf_vec[..(end - start)].copy_from_slice(&data[start..end]);
            let buf = FixedPointBuffer::from_f32(&buf_vec);

            let pred = predictor.predict().unwrap_or_else(|_| FixedPointBuffer::new(chunk));
            let residual = if pred.is_empty() { buf.clone() } else { buf.sub(&pred).unwrap() };
            let (bytes, scale) = quant.quantize(&residual);
            scales.push(scale);

            // Update predictor with reconstruction for next chunk
            let deq = quant.dequantize(&bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).unwrap() };
            predictor.update(&recon).unwrap();

            blob.extend_from_slice(&bytes);
            offsets.push(blob.len() as u64);
        }

        // Write datasets under group
        grp.new_dataset_builder().with_data(&blob).create("blob")?;
        grp.new_dataset_builder().with_data(&offsets).create("offsets")?;
        grp.new_dataset_builder().with_data(&scales).create("scales")?;
        grp.new_attr::<u64>().create("total_len")?.write_scalar(&(total_len as u64))?;
        grp.new_attr::<u64>().create("chunk")?.write_scalar(&(chunk as u64))?;
        grp.new_attr::<&str>().create("codec")?.write_scalar(&codec)?;
        grp.new_attr::<f32>().create("epsilon")?.write_scalar(&epsilon)?;
        Ok(())
    }

    /// Reads an f32 dataset.
    pub fn read_dataset(path: &str, dataset: &str) -> H5Result<Vec<f32>> {
        let file = File::open(path)?;
        let ds = file.dataset(dataset)?;
        let data: Vec<f32> = ds.read_raw()?;
        Ok(data)
    }

    /// Reads the compressed 1D dataset written by `write_chunked_dataset`.
    pub fn read_compressed_1d(path: &str, dataset: &str) -> H5Result<Vec<f32>> {
        let file = File::open(path)?;
        let grp = file.group(dataset)?;

        let blob_ds = grp.dataset("blob")?;
        let offsets_ds = grp.dataset("offsets")?;
        let scales_ds = grp.dataset("scales")?;

        let blob: Vec<u8> = blob_ds.read_raw()?;
        let offsets: Vec<u64> = offsets_ds.read_raw()?;
        let scales: Vec<f32> = scales_ds.read_raw()?;

        let total_len: u64 = grp.attr("total_len")?.read_scalar()?;
        let chunk: u64 = grp.attr("chunk")?.read_scalar()?;
        let epsilon: f32 = grp.attr("epsilon")?.read_scalar()?;
        let _codec: String = grp.attr("codec")?.read_scalar()?;

        let mut out = Vec::with_capacity(total_len as usize);
        let mut predictor = LinearPredictor::new();
        let quant = ErrorBoundedQuantizer::new(epsilon);

        let num_chunks = scales.len();
        for c in 0..num_chunks {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            let bytes = &blob[start..end];
            let scale = scales[c];

            let pred = predictor.predict().unwrap_or_else(|_| FixedPointBuffer::new(chunk as usize));
            let deq = quant.dequantize(bytes, scale);
            let recon = if pred.is_empty() { deq } else { pred.add(&deq).unwrap() };
            predictor.update(&recon).unwrap();

            let mut f = recon.to_f32();
            out.append(&mut f);
        }

        out.truncate(total_len as usize);
        Ok(out)
    }
}
