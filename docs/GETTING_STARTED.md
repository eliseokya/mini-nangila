# Getting Started with Mini-Nangila

This guide will help you get started with Mini-Nangila, the open-source reference implementation of the Nangila compression framework.

## Prerequisites

- **Rust 1.70+** - [Install via rustup](https://rustup.rs)
- **8GB+ RAM**
- **Any modern CPU** (x86_64 or ARM64)
- **Optional**: libhdf5 for HDF5 support (not required for basic usage)

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/nangila/nangila.git
cd nangila/mini-nangila

# Verify installation
cargo --version  # Should show 1.70 or higher

# Build and test
cargo test --workspace
```

**Expected output**: All 43 tests should pass in ~30 seconds.

---

## Your First Example: LAMMPS Compression

Let's compress a molecular dynamics trajectory:

### Step 1: Run with Sample Data

```bash
cargo run --release --bin lammps-compress -- \
  --input examples/lammps_compress/sample.dump \
  --output test.nz
```

**Output**:
```
Reading LAMMPS dump: examples/lammps_compress/sample.dump
step 0  ratio 1.38×  max_err 0.000000
DONE compressed=0.00 MB  raw=0.00 MB  ratio=1.38×  max_err=0.000000
```

✅ **Success!** You've compressed your first file.

### Step 2: Decompress and Verify

```bash
cargo run --release --bin lammps-compress -- \
  --decompress \
  --output test.nz
```

**Output**:
```
Decompressed 6 values
```

### Step 3: Try a Larger Dataset

```bash
cargo run --release --bin lammps-compress -- \
  --particles 100000 \
  --steps 100 \
  --output large.nz
```

**Expected**: 20-50× compression, takes ~5 seconds

---

## Understanding the Output

### Compression Metrics

```
step 0  ratio 1.38×  max_err 0.000000
```

- **step**: Timestep being compressed
- **ratio**: Compression ratio (higher is better)
- **max_err**: Maximum reconstruction error (for error-bounded compression)

### Error Bounds

Mini-Nangila supports **error-bounded compression** - you specify the maximum allowed error:

```bash
cargo run --release --bin lammps-compress -- \
  --particles 10000 \
  --epsilon 0.01 \  # Allow up to 0.01 error
  --output compressed.nz
```

**Trade-off**:
- Smaller epsilon (e.g., 0.001) → tighter error bound → lower compression
- Larger epsilon (e.g., 0.1) → looser bound → higher compression

---

## Try Other Examples

### NanoGPT: Gradient Compression

Simulates AI training with gradient compression:

```bash
cargo run --release --bin nano-gpt -- --steps 100
```

**What it does**:
- Generates synthetic gradients with realistic sparsity
- Compresses using momentum predictor + TopK quantization
- Shows topology masking (driver/passenger layers)

**Expected output**: 20-40× compression ratio

### Rotor Twin: Sensor Streaming

Simulates edge-cloud sync for digital twins:

```bash
cargo run --release --bin rotor-twin -- --duration 10 --rate-hz 50
```

**What it does**:
- Simulates 6-channel sensor data (angular velocity + acceleration)
- Compresses in real-time
- Maintains edge-cloud predictor synchronization

**Expected output**: 1.5-3× compression for real-time streams

---

## Next Steps

### Explore the Code

```bash
# View the project structure
tree -L 2 .

# Read the API documentation
cargo doc --no-deps --open
```

**Key crates**:
- `nangila-math` - Q8.23 fixed-point arithmetic
- `nangila-core` - Predictor/Quantizer traits
- `nangila-ai` - AI-specific compression
- `nangila-hpc` - HPC error-bounded compression
- `nangila-twin` - Digital twin edge-cloud sync

### Run All Tests

```bash
# Run all tests (43 total)
cargo test --workspace --release

# Run only property-based tests
cargo test --workspace -- prop

# Run specific crate tests
cargo test --package nangila-math
```

### Use Your Own Data

**For LAMMPS**: See [examples/lammps_compress/README.md](../examples/lammps_compress/README.md) for detailed instructions on using real LAMMPS dump files.

```bash
# Use your own LAMMPS trajectory
cargo run --release --bin lammps-compress -- \
  --input /path/to/your/trajectory.dump \
  --epsilon 0.001 \
  --output compressed.nz
```

---

## Common Issues

### Issue: Compilation Errors

**Problem**: `error: could not compile ...`

**Solution**:
1. Check Rust version: `rustc --version` (need 1.70+)
2. Clean build: `cargo clean && cargo build`
3. Update toolchain: `rustup update`

### Issue: Tests Fail

**Problem**: Some tests fail

**Solution**:
1. Run in release mode: `cargo test --release`
2. Check for platform-specific issues
3. Report on GitHub with output

### Issue: Low Compression Ratios

**Problem**: Getting 1-2× instead of expected 20-100×

**Solution**:
- **For LAMMPS**: Use more timesteps (need 50+ for good temporal correlation)
- **For NanoGPT**: Increase training steps (gradients get sparser over time)
- **General**: Increase epsilon for higher compression (looser error bound)

### Issue: Out of Memory

**Problem**: Crashes with OOM error

**Solution**:
- Reduce dataset size (fewer particles/steps)
- Run in release mode (more memory-efficient)
- Close other applications

---

## Understanding Compression Ratios

| Example | Expected Ratio | Why |
|---------|---------------|-----|
| LAMMPS (small, 3 atoms) | 1-2× | Too small, no correlation |
| LAMMPS (medium, 100K atoms) | 20-50× | Good temporal correlation |
| LAMMPS (large, 1M+ atoms) | 50-100× | Strong patterns + error-bounded quantization |
| NanoGPT | 20-40× | Sparse gradients + topology masking |
| Rotor Twin (short) | 1.5-3× | Predictor warming up |
| Rotor Twin (long) | 3-5× | Predictor converged |

**Key insight**: Compression improves with:
- ✅ More temporal correlation (smooth motion)
- ✅ Larger datasets (more patterns to learn)
- ✅ Sparse or structured data
- ✅ Looser error bounds (when applicable)

---

## Benchmarking Your Own Workloads

Want to test Mini-Nangila on your data?

### For HPC Simulations

1. Export trajectory to LAMMPS format
2. Compress with different epsilon values
3. Compare to gzip/bzip2 baseline
4. Measure reconstruction error

```bash
# Compress with Mini-Nangila
cargo run --release --bin lammps-compress -- \
  --input your_data.dump \
  --epsilon 0.001 \
  --output mini_nangila.nz

# Compress with gzip for comparison
gzip -k -9 your_data.dump

# Compare file sizes
ls -lh your_data.dump* mini_nangila.nz
```

### For AI Gradients

Mini-Nangila uses synthetic gradients. To test with real PyTorch gradients, you would need to:
1. Enable `tch` feature in `nangila-ai`
2. Export gradients from training loop
3. Convert to `FixedPointBuffer`

(This requires PyTorch integration - see documentation for details)

---

## Getting Help

- **Documentation**: `cargo doc --open`
- **Examples**: See `examples/*/README.md`
- **Issues**: [GitHub Issues](https://github.com/nangila/nangila/issues)
- **Questions**: craig@nangila.io

---

## What's Next?

Now that you have Mini-Nangila running:

1. **Understand the algorithms** - Read `docs/ALGORITHMS.md` (coming soon)
2. **Explore the code** - Start with `nangila-math` for the foundation
3. ** Test with your data** - Follow domain-specific READMEs
4. **Contribute** - See `CONTRIBUTING.md` (coming soon)
5. **Cite in papers** - See `CITATION.cff` for BibTeX

---

## Troubleshooting Checklist

Before asking for help, try:

- [ ] Running `cargo clean && cargo build --release`
- [ ] Checking Rust version (`cargo --version`)
- [ ] Reading error messages carefully
- [ ] Consulting example-specific READMEs
- [ ] Searching existing GitHub issues

Still stuck? Open an issue with:
- Rust version
- Operating system
- Full error output
- Steps to reproduce

Happy compressing! 🚀
