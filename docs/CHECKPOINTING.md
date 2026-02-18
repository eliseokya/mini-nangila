## Checkpointing (HDF5) — Mini-Nangila

This crate provides helpers to write/read f32 datasets with compression metadata.
HDF5 is optional — enable feature `hdf5` and ensure `libhdf5` is installed.

### Usage

```rust
#[cfg(feature = "hdf5")]
fn write_ckpt() -> hdf5::Result<()> {
    use nangila_checkpoint::h5;
    let data: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
    h5::write_chunked_dataset("ckpt.h5", "state", &data, 256, 1e-3, "ErrorBoundedINT16")?;
    let back = h5::read_dataset("ckpt.h5", "state")?;
    assert_eq!(back.len(), data.len());
    Ok(())
}
```

Notes:
- This stores reconstructed values with attributes `epsilon` and `codec` for provenance.
- A future HDF5 filter can store residuals directly; current API is a pragmatic first step.

### CLI Demo

Build and run the demo (requires libhdf5):

```
cd mini-nangila/examples/checkpoint_demo
cargo run --features hdf5 -- \
  --output traj_ckpt.h5 \
  --dataset traj \
  --len 50000 \
  --chunk 1024 \
  --epsilon 1e-3
```

Output includes the observed max error (should be ≤ epsilon + small rounding noise).

### Python h5py Reader

You can reconstruct the signal in Python using h5py and NumPy:

```
pip install h5py numpy
python mini-nangila/python/snippets/read_hdf5_checkpoint.py traj_ckpt.h5
```

This script reads the group layout (`blob`, `offsets`, `scales`, with attributes) and reproduces the predictor+dequantization logic.
