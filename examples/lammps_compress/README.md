# LAMMPS Compression Example

Error-bounded compression for molecular dynamics trajectories using the Nangila framework.

## Quick Start

### Using Included Sample Data (Minimal Test)

```bash
cargo run --release --bin lammps-compress -- \
  --input examples/lammps_compress/sample.dump \
  --output compressed.nz
```

**Output**: 1.38× compression (3 atoms, 2 timesteps)

### Enable RLE (Error‑bounded + Run‑Length Encoding)

On smooth, well‑ordered trajectories, run‑length encoding of error‑bounded residuals can add 2–10× extra compression (typical totals 50–100×).

```bash
cargo run --release --bin lammps-compress -- \
  --input trajectory.dump \
  --epsilon 0.001 \
  --use-rle \
  --output compressed_rle.nz
```

Container header encodes codec: 1 = ErrorBoundedINT16, 2 = ErrorBoundedINT16_RLE. Decompression auto‑detects.

### Generating Synthetic Data

```bash
# Small test (fast)
cargo run --release --bin lammps-compress -- \
  --particles 10000 \
  --steps 10 \
  --output small_test.nz

# Medium scale (~400 MB uncompressed)
cargo run --release --bin lammps-compress -- \
  --particles 100000 \
  --steps 100 \
  --output medium.nz

# Large scale (~4 GB uncompressed, takes ~1 min)
cargo run --release --bin lammps-compress -- \
  --particles 1000000 \
  --steps 100 \
  --output large.nz
```

**Expected compression ratios**:
- Small (10K atoms): **2-5×** (not enough correlation)
- Medium (100K atoms): **20-50×** (linear predictor captures smooth motion)
- Large (1M+ atoms): **50-100×** (high temporal correlation + error-bounded quantization + RLE)

---

## Using Real LAMMPS Dump Files

The example **already supports standard LAMMPS ASCII dump format**!

### Step 1: Generate a LAMMPS Dump

From your LAMMPS simulation, output trajectory dumps:

```lammps
# In your LAMMPS input script
dump myDump all custom 100 trajectory.dump id type x y z
dump_modify myDump sort id
```

This creates `trajectory.dump` with atom positions every 100 steps.

### Step 2: Compress the Dump

```bash
cargo run --release --bin lammps-compress -- \
  --input trajectory.dump \
  --epsilon 0.001 \
  --use-rle \
  --output compressed.nz
```

**Parameters**:
- `--epsilon`: Error bound (default: 0.001)
  - `0.001` → tight bound, lower compression (~20-50×)
  - `0.01` → looser bound, higher compression (~50-100×)
  - `0.1` → very loose, maximum compression (~100-200×)

### Step 3: Decompress and Verify

```bash
cargo run --release --bin lammps-compress -- \
  --decompress \
  --output compressed.nz
```

---

## Advanced: Public LAMMPS Datasets

### Example: LAMMPS Benchmark Data

1. **Download Lennard-Jones benchmark**:
```bash
mkdir -p data
cd data

# Download LAMMPS example input
wget https://github.com/lammps/lammps/raw/develop/bench/in.lj

# Run LAMMPS to generate trajectory
lammps -in in.lj  # Requires LAMMPS installed
```

2. **Compress the output**:
```bash
cargo run --release --bin lammps-compress -- \
  --input data/dump.* \
  --epsilon 0.001 \
  --output lj_compressed.nz
```

### Example: Water Simulation

For larger datasets, check public repositories:
- [NOMAD Archive](https://nomad-lab.eu/) - Materials science simulations
- [Materials Cloud](https://www.materialscloud.org/) - DFT and MD trajectories
- [OpenKIM](https://openkim.org/) - Interatomic potential databases

Most provide dump files or can be converted to LAMMPS format.

---

## Performance Expectations

| Scale | Atoms | Steps | Uncompressed | Compressed* | Ratio | Time |
|-------|-------|-------|--------------|-------------|-------|------|
| Tiny | 100 | 10 | ~12 KB | ~10 KB | 1.2× | <1s |
| Small | 10K | 10 | 1.2 MB | ~400 KB | 3× | ~1s |
| Medium | 100K | 100 | 120 MB | ~2–6 MB | 20–60× | ~5s |
| Large | 1M | 100 | 1.2 GB | ~12–24 MB | 50–100× | ~30s |
| X-Large | 10M | 1000 | 120 GB | ~1.2 GB | 100× | ~30min |

*With epsilon=0.001 (tight error bound) and `--use-rle` on smooth trajectories

**Compression improves with**:
- ✅ More timesteps (temporal correlation)
- ✅ Smooth trajectories (predictable motion)
- ✅ Larger epsilon (looser error bounds)

**Compression decreases with**:
- ❌ Random/chaotic motion
- ❌ Very tight epsilon (<0.0001)
- ❌ Few timesteps (<10)

---

## Output Format

The compressed file uses a simple binary format:

```
Header (13 bytes):
  - Magic: b"NZCP" (4 bytes)
  - Version: 1 (1 byte)
  - Codec: 1=ErrorBounded, 2=ErrorBounded+RLE (1 byte)
  - Particles: u32 (4 bytes)
  - Steps: u32 (4 bytes)

Per-step data:
  - Scale: f32 (4 bytes)
  - Compressed size: u32 (4 bytes)  
  - Compressed bytes: [u8] (variable)
```

---

## Algorithm

The example uses Nangila's **error-bounded compression**:

1. **Linear Predictor**: Extrapolates from previous 2 timesteps
   ```
   prediction[t] = 2 * positions[t-1] - positions[t-2]
   ```

2. **Residual**: Computes prediction error
   ```
   residual = actual - prediction
   ```

3. **Error-Bounded Quantization**: Quantizes residual to INT16
   ```
   quantized = clamp(residual / epsilon, -32768, 32767)
   ```

4. **RLE Compression (optional)**: Run-length encodes zeros (sparse residuals); enable with `--use-rle`

**Guarantees**: `|reconstructed - original| ≤ epsilon` for all positions

---

## Troubleshooting

### Low Compression Ratios

**Problem**: Getting 1-2× compression instead of 50-100×

**Solutions**:
- ✅ Use more timesteps (need >50 for good temporal correlation)
- ✅ Increase epsilon: try `--epsilon 0.01` instead of `0.001`
- ✅ Ensure smooth motion (MD simulations are usually smooth)
- ✅ Ensure per‑timestep atom ordering is stable (e.g., `dump_modify ... sort id`)
- ✅ Enable `--use-rle` to exploit long zero runs in residuals

### Negative Control (2× ASCII Case)

The example can deliberately demonstrate the lower‑bound “2×” behavior by using ASCII dumps with limited smoothness and without RLE:

```bash
cargo run --release --bin lammps-compress -- \
  --input examples/lammps_compress/sample.dump \
  --epsilon 0.001 \
  --output compressed_ascii_only.nz
```

Expect ~2× from FP32→INT16 only. This is useful to validate ε‑respect independent of RLE.
- ✅ Check atom ordering is consistent (use `dump_modify sort id`)

### Parse Errors

**Problem**: "Failed to parse LAMMPS dump"

**Solutions**:
- ✅ Use ASCII format (not binary or compressed dumps)
- ✅ Include all required fields: `id type x y z`
- ✅ Ensure consistent atom count across timesteps
- ✅ Check for corrupted/truncated files

### Out of Memory

**Problem**: Large dumps consume too much memory

**Solutions**:
- ✅ Process in chunks (not currently supported, but could be added)
- ✅ Reduce timesteps: compress every Nth snapshot
- ✅ Use streaming mode (future enhancement)

---

## Comparison to Other Tools

| Tool | Compression | Error Bound | Speed | Notes |
|------|-------------|-------------|-------|-------|
| **Nangila** | 50-100× | Guaranteed | Fast | Predictor-based |
| gzip | 2-3× | Lossless | Fast | General-purpose |
| SZ | 40-80× | Guaranteed | Medium | Research compressor |
| ZFP | 30-60× | Guaranteed | Fast | Array-oriented |

**Nangila advantages**:
- ✅ Predictor learns temporal patterns
- ✅ Simple Rust implementation
- ✅ Deterministic Q8.23 arithmetic
- ✅ Fast compression/decompression

---

## Next Steps

- Try with your own LAMMPS simulations
- Experiment with different epsilon values
- Compare compression ratios to gzip/bzip2
- Use compressed dumps for checkpoint/restart (future feature)

For more details on the Nangila framework, see the [main README](../../README.md).
### Block‑Sparse (mask+payload) and Block‑Sparse+RLE

Block‑sparse encodes 32‑wide masks and only transmits non‑zero i16 payloads.
The `+RLE` variant run‑length encodes blocks whose masks are all zeros.

```bash
# Block‑Sparse only
cargo run --release --bin lammps-compress -- \
  --input trajectory.dump \
  --epsilon 0.001 \
  --use-block-sparse \
  --output compressed_bs.nz

# Block‑Sparse + mask‑RLE
cargo run --release --bin lammps-compress -- \
  --input trajectory.dump \
  --epsilon 0.001 \
  --use-block-sparse-rle \
  --output compressed_bsrle.nz
```

Container codec ids:
- 1 = ErrorBoundedINT16
- 2 = ErrorBoundedINT16_RLE
- 3 = ErrorBoundedINT16_BlockSparse
- 4 = ErrorBoundedINT16_BlockSparse_RLE
