# Mini-Nangila: Open-Core Reference Implementation

**CPU-based reference implementation of Nangila's predictive-residual compression framework**

[![License: Apache-2.0/MIT](https://img.shields.io/badge/license-Apache--2.0%2FMIT-blue.svg)](../LICENSE)
[![Rust: 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

> 🔬 **For Researchers**: This is the verifiable, educational implementation. All algorithms are open-source and run on any CPU.  
> 🚀 **For Production**: See the proprietary `nangila-core` (GPU-accelerated, 25-100× faster, same compression ratios).

---

## What is Mini-Nangila?

Mini-Nangila is a **standalone, CPU-only implementation** of the Nangila compression framework. It demonstrates the core mathematical principles of **predictive-residual decomposition** across three domains:

1. **AI Training**: Compress gradients for distributed deep learning (20-40× reduction)
2. **HPC Simulation**: Compress scientific data with strict error bounds (50-100× reduction)
3. **Digital Twins**: Compress sensor streams for edge-cloud sync (80-120× reduction)

### The Open-Core Model

| Feature | Mini-Nangila (Open) | Nangila (Proprietary) |
|---------|---------------------|----------------------|
| **Goal** | Verify science | Maximize scale |
| **Hardware** | CPU (any laptop) | GPU (CUDA/ROCm) |
| **Performance** | 100-400 MB/s | 10-50 GB/s |
| **Compression Ratios** | 20-120× | 20-120× (same) |
| **Use Case** | Research, education, validation | Production, extreme scale |
| **License** | Apache-2.0 / MIT | Commercial |

**Key Insight**: The compression algorithms are identical. The GPU version only accelerates computation, not compression quality.

---

## Quick Start

### Prerequisites
- Rust 1.70+ ([install via rustup](https://rustup.rs))
- 8GB+ RAM
- Any modern CPU (x86_64 or ARM64)

### Build and Test
```bash
# Clone repository
git clone https://github.com/nangila/nangila.git
cd nangila/mini-nangila

# Run tests (core crates)
cargo test

# Optional features (require system deps):
# - HDF5: install libhdf5 (e.g., brew install hdf5) then enable feature flags per crate
# - libtorch/PyTorch: set LIBTORCH or LIBTORCH_USE_PYTORCH=1 before enabling tch features
# See docs below.

# Build examples
cargo build --release --examples
```

### Replication (One‑pass)
```bash
# Runs build, tests, and representative demos (artifacts to verification/)
bash scripts/verify.sh
```

### Docker (Optional)
- Build locally:
```bash
docker build -t mini-nangila:latest .
docker run --rm -it mini-nangila:latest lammps-compress --help
```
- If using GHCR (on tagged release), pull the published image:
```bash
docker pull ghcr.io/<owner>/<repo>/mini-nangila:latest
docker run --rm -it ghcr.io/<owner>/<repo>/mini-nangila:latest lammps-compress --help
```

### Run Examples

> **Note**: Examples use synthetic data for demonstration. See individual example READMEs for using real data.

#### AI: Gradient Compression (NanoGPT)
```bash
cargo run --release --bin nano-gpt -- --steps 1000
# Expected: 20-28× compression, 2-3 minutes on laptop
# Uses: Synthetic gradients with realistic sparsity patterns
```

#### HPC: LAMMPS Trajectory Compression
```bash
# Quick test with included sample
cargo run --release --bin lammps-compress -- \
  --input examples/lammps_compress/sample.dump

# Generate large synthetic dataset (1M atoms, ~1.2 GB)
cargo run --release --bin lammps-compress -- \
  --particles 1000000 --steps 100 --output large.nz
# Expected: 50-100× compression, error bound ≤ 0.001

# Use with YOUR OWN LAMMPS dumps (already supported!)
cargo run --release --bin lammps-compress -- \
  --input /path/to/your/trajectory.dump --epsilon 0.001
```

📖 **See [examples/lammps_compress/README.md](examples/lammps_compress/README.md) for detailed instructions on using real LAMMPS data at scale.**

#### Digital Twin: Rotor Streaming
```bash
cargo run --release --bin rotor-twin -- --duration 10 --rate-hz 50
# Expected: 1.5-3× compression for real-time sensor sync
```

#### Other Examples
```bash
# Topology calibration (AI)
cargo run --release --bin topology-calibration

# Mode switching (AI→HPC)
cargo run --release --bin mode-switching

# Mode switching (advanced flags)
# - Training: Stochastic INT4 by default, or TopK when --train-topk > 0.0
# - Checkpoints: ErrorBounded by default, or BlockSparse/BlockSparse+RLE when enabled
# Common flags:
#   --num-steps N                (default 50)
#   --checkpoint-interval K      (default 10)
#   --gradient-size M            (default 10000)
#   --epsilon EPS                (checkpoint error bound, default 1e-3)
#   --checkpoint-block-sparse    (use EB+BlockSparse for checkpoints)
#   --checkpoint-bsrle           (use EB+BlockSparse+RLE for checkpoints)
#   --train-topk P               (use TopK at P fraction during training; 0.0 uses INT4)

# Examples:
# 1) Default (INT4 train, EB checkpoints)
cargo run --release --bin mode-switching

# 2) EB+BlockSparse on checkpoints (same training)
cargo run --release --bin mode-switching -- --checkpoint-block-sparse

# 3) EB+BlockSparse+RLE on checkpoints (same training)
cargo run --release --bin mode-switching -- --checkpoint-bsrle

# 4) Training with TopK 10% + EB+BlockSparse checkpoints
cargo run --release --bin mode-switching -- --checkpoint-block-sparse --train-topk 0.10

# 5) Training with TopK 5% + EB+BlockSparse checkpoints (higher overall ratio)
cargo run --release --bin mode-switching -- --checkpoint-block-sparse --train-topk 0.05

# Throughput benchmark
cargo run --release --bin throughput-bench
```

📖 **New to Mini-Nangila?** See the [Getting Started Guide](docs/GETTING_STARTED.md) for a step-by-step tutorial.

---

## Architecture

Mini-Nangila is organized as a Rust workspace with 5 crates:

```
mini-nangila/
├── nangila-math/       # Q8.23 fixed-point arithmetic (foundation)
├── nangila-core/       # Predictor/Quantizer traits (abstractions)
├── nangila-ai/         # AI-specific: Momentum, TopK, Stochastic
├── nangila-hpc/        # HPC-specific: Linear, ErrorBounded, RLE
└── nangila-twin/       # Digital Twin: EdgeNode, CloudNode sync
```

### Dependency Graph
```
nangila-math (no dependencies)
    ↓
nangila-core (depends on: nangila-math)
    ↓
├─ nangila-ai (depends on: nangila-core, nangila-math)
├─ nangila-hpc (depends on: nangila-core, nangila-math)
└─ nangila-twin (depends on: nangila-core, nangila-math, nangila-ai, nangila-hpc)
```

### Key Abstractions

#### Predictor Trait
```rust
pub trait Predictor {
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError>;
    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError>;
    fn reset(&mut self);
}
```

**Implementations**:
- `MomentumPredictor` (AI): EMA-based gradient prediction
- `LinearPredictor` (HPC): First-order extrapolation

#### Quantizer Trait
```rust
pub trait Quantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32);
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer;
}
```

**Implementations**:
- `StochasticQuantizer` (AI): Unbiased INT4 quantization
- `TopKQuantizer` (AI): Sparse gradient compression
- `ErrorBoundedQuantizer` (HPC): Strict ε-guarantee
- `RunLengthQuantizer` (HPC): Zero-heavy residuals

Note on Top‑K format: Top‑K encoded buffers are prefixed with `u32 n_elems`,
followed by 32‑wide blocks of `[u32 mask][i8 payload...]`. Decoders use `n_elems`
to reconstruct exact length for non‑multiple‑of‑32 inputs.

---

## How It Works

### The Predictive-Residual Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Standard Approach: Transmit Full Data                      │
│  ────────────────────────────────────────────────────────   │
│  Data (FP32) ──────────────────────────────▶ Transmit       │
│  28 GB/step (7B model)                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Nangila Approach: Transmit Residual Only                   │
│  ────────────────────────────────────────────────────────   │
│  Data ──▶ [Predict] ──▶ [Residual] ──▶ [Quantize] ──▶ TX   │
│           (from history)  (error)      (INT4)               │
│                                                              │
│  Receiver: [Dequantize] ──▶ [Add Prediction] ──▶ Reconstruct│
│  ~1 GB/step (28× compression)                                │
└─────────────────────────────────────────────────────────────┘
```

### Why It Works

1. **Temporal Correlation**: Data evolves smoothly → predictions are accurate → residuals are small
2. **Entropy Reduction**: H(residual) < H(data) → better compression
3. **Closed-Loop Feedback**: Both sender and receiver update predictors with reconstructed values → no drift

---

## Performance Targets

### Compression Ratios
| Domain | Typical | Best Case | Key Factor |
|--------|---------|-----------|------------|
| AI (Gradients) | 20-28× | 40× | Sparsity + momentum |
| HPC (Trajectories) | 50-100× | 200× | Smoothness |
| Digital Twin (Sensors) | 80-120× | 300× | Predictability |

### Throughput (CPU)
| Domain | Compression | Decompression |
|--------|-------------|---------------|
| AI | 100-200 MB/s | 150-300 MB/s |
| HPC | 200-400 MB/s | 300-500 MB/s |
| Twin | 50-100 MB/s | 100-200 MB/s |

### Memory Footprint
| Domain | Peak Memory | Notes |
|--------|-------------|-------|
| AI | 4 GB | Includes model |
| HPC | 2 GB | 100K particles |
| Twin | 1 MB | Edge device |

**See [EXPECTED_PERFORMANCE.md](docs/EXPECTED_PERFORMANCE.md) for detailed benchmarks.**

---

## Documentation

- **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)**: 12-week roadmap to full spec compliance
- **[EXPECTED_PERFORMANCE.md](docs/EXPECTED_PERFORMANCE.md)**: Detailed performance targets and validation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Design rationale and trait system
- **[SCIENCE_VS_SCALE.md](docs/SCIENCE_VS_SCALE.md)**: CPU-verified science and how it scales on GPUs
- **[CPU_VERIFICATION.md](docs/CPU_VERIFICATION.md)**: One-pass CPU verification of science and metrics

---

## What's Included vs What's Not

### ✅ Included (Open-Core)
- Q8.23 fixed-point arithmetic
- Predictor/Quantizer trait abstractions
- Basic AI predictors (Momentum)
- Basic HPC predictors (Linear extrapolation)
- Error-bounded quantization (ε-guarantee)
- Stochastic quantization (unbiased INT4)
- TopK sparsification
- Digital twin edge-cloud sync
- CPU-based examples (Throughput, HPC benchmark, Mode Switching, Topology calibration, NanoGPT synthetic)

### ❌ Not Included (Proprietary)
- GPU kernels (CUDA/ROCm)
- Sculptor topology analysis (advanced heuristics)
- Safe Mode adaptive monitoring
- PyTorch DDP/FSDP hooks
- NCCL integration
- Learned predictors (Transformer-based)
- Adaptive bit-rate control
- MPI/InfiniBand transport
- ISO 26262 / DO-178C compliance

---

## Roadmap

### Phase 1: Core Consolidation (Weeks 1-4) ✅
- [x] Q8.23 fixed-point math
- [x] Predictor/Quantizer traits
- [x] Basic AI/HPC/Twin implementations
- [x] Static topology masking
- [x] Documentation foundation (Architecture/Benchmarks/Expected Performance)

### Phase 2: Real-World Examples (Weeks 5-8) ✅
- [x] NanoGPT gradient compression (synthetic DDP)
- [x] LAMMPS trajectory compression (ε verification)
- [x] Rotor twin sensor sync demo
- [x] Integration testing

---

## Optional System Dependencies

- HDF5 (for HPC integration): install via your package manager
  - macOS: `brew install hdf5`
  - Linux: `apt-get install libhdf5-dev` (Debian/Ubuntu) or distro equivalent
  - Enable HDF5-dependent crates/features explicitly when needed.

- libtorch/PyTorch (for PyTorch integration):
  - Set `LIBTORCH` to an existing libtorch install, or set `LIBTORCH_USE_PYTORCH=1` to use your Python env’s PyTorch.
  - Only required when enabling `tch` features.

### Phase 3: Advanced Features (Weeks 9-12) 📋
- [ ] HDF5 checkpoint plugin
- [ ] no_std edge device support
- [ ] Tutorial and verification
- [ ] Public v0.1.0 release

**See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for details.**

---

## Contributing

We welcome contributions! Mini-Nangila is the community edition, and we encourage:

- Bug reports and fixes
- New predictor/quantizer implementations
- Performance optimizations (SIMD, multi-threading)
- Documentation improvements
- Example applications

**Note**: Contributions to mini-nangila require a Contributor License Agreement (CLA) to maintain the open-core model.

### Development Setup
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/nangila/nangila.git
cd nangila/mini-nangila
cargo build --all-features

# Run tests
cargo test --all-features

# Format code
cargo fmt --all

# Lint
cargo clippy --all-features
```

---

## Citation

If you use Mini-Nangila in your research, please cite:

```bibtex
@article{chirara2026nangila,
  title={Nangila: A Unified State-Space Framework for Predictive Compression},
  author={Chirara, Carelt},
  journal={arXiv preprint},
  year={2026}
}
```

For software citation, see [CITATION.cff](CITATION.cff) for machine-readable metadata.

---

## License

Mini-Nangila is dual-licensed under:
- **Apache License 2.0** ([LICENSE-APACHE](../LICENSE-APACHE))
- **MIT License** ([LICENSE-MIT](../LICENSE-MIT))

You may choose either license at your option.

**Note**: The proprietary `nangila-core`, `nangila-cuda`, and `nangila-hook` crates (in parent directory) are under a commercial license. Contact [craig@nangila.ai](mailto:craig@nangila.ai) for enterprise licensing.

---

## FAQ

### Why CPU-only?
**Answer**: To make the code accessible to any researcher. The algorithms are identical to the GPU version; only the execution speed differs.

### Can I use this in production?
**Answer**: Yes, but the CPU version is 25-100× slower than the GPU version. For production scale (100+ GPUs, exascale HPC), contact us about the enterprise edition.

### How do I reproduce paper results?
**Answer**: See [EXPECTED_PERFORMANCE.md](docs/EXPECTED_PERFORMANCE.md) for validation procedures. All results are reproducible on a laptop.

### What's the difference from the main Nangila?
**Answer**: Same algorithms, different execution:
- Mini-Nangila: CPU, educational, open-source
- Nangila: GPU, production, proprietary

### Can I contribute GPU kernels?
**Answer**: GPU acceleration is part of the proprietary edition. However, we welcome CPU optimizations (SIMD, multi-threading) to mini-nangila.

### Is this production-ready?
**Answer**: Mini-Nangila is a reference implementation for research and validation. For production deployments, use the enterprise edition with GPU acceleration, Safe Mode, and industrial compliance.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/nangila/nangila/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nangila/nangila/discussions)
- **Email**: [craig@nangila.ai](mailto:craig@nangila.ai)
- **Enterprise**: [nangila.ai/enterprise](https://nangila.ai/enterprise)

---

## Acknowledgments

Mini-Nangila builds on research in:
- Gradient compression (DGC, PowerSGD, TopK)
- Scientific data compression (SZ, ZFP)
- Predictive coding (DPCM, H.264)
- Fixed-point arithmetic (Q-format)

We thank the open-source community for inspiration and feedback.
