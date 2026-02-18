# Mini-Nangila Implementation Plan
## From Current State to Full Open-Core Spec

**Target**: Complete CPU-based reference implementation that any researcher can run  
**Timeline**: 12 weeks (3 phases)  
**Hardware Requirements**: Any modern CPU (no GPU needed)

---

## Current Status: 60% Complete

### ✅ What's Working
- Q8.23 fixed-point arithmetic with deterministic rounding
- Predictor/Quantizer trait abstractions
- AI: Momentum predictor + Stochastic/TopK quantizers
- HPC: Linear predictor + Error-bounded quantizer
- Digital Twin: EdgeNode/CloudNode sync architecture
- Basic benchmarks with synthetic data

### ❌ What's Missing
- Static topology masking (Driver/Passenger detection)
- Real-world examples (NanoGPT, LAMMPS, IoT sensor)
- HDF5 checkpoint plugin
- No-std edge device support
- Documentation and tutorials
- Cross-platform determinism validation

---

## Phase 1: Core Consolidation (Weeks 1-4)

### Week 1: Topology Masking
**Goal**: Implement static layer selection for AI training

**Tasks**:
1. Implement variance-based topology detection
   ```rust
   // mini-nangila/nangila-core/src/topology.rs
   pub struct TopologyMask {
       pub driver_indices: Vec<usize>,
       pub passenger_indices: Vec<usize>,
       pub threshold: f32,
   }
   
   impl TopologyMask {
       pub fn from_variance_threshold(
           layer_variances: &[f32],
           drop_percent: f32
       ) -> Self;
       
       pub fn is_driver(&self, layer_id: usize) -> bool;
       pub fn compression_factor(&self) -> f32;
   }
   ```

2. Add calibration example
   ```rust
   // mini-nangila/examples/topology_calibration/
   // Collect gradient statistics over 100 steps
   // Generate mask that drops bottom 30% by variance
   ```

3. Unit tests for edge cases (all drivers, all passengers, empty)

**Deliverable**: Working topology mask with >20% layer reduction

**Performance Target**: 
- Calibration: <5 seconds for 1M parameters
- Mask generation: <100ms

---

### Week 2: Dual-Mode Quantizer Refactor
**Goal**: Unified interface for Stochastic vs ErrorBounded modes

**Tasks**:
1. Create enum-based quantizer
   ```rust
   // mini-nangila/nangila-core/src/quantizer.rs
   pub enum QuantizationMode {
       Stochastic { seed: u64, bits: u8 },
       ErrorBounded { epsilon: f32 },
       TopK { k_percent: f32 },
   }
   
   pub struct UnifiedQuantizer {
       mode: QuantizationMode,
   }
   ```

2. Migrate existing quantizers to use unified interface
3. Add mode-switching example (AI → HPC transition)

**Deliverable**: Single quantizer type that handles all three domains

**Performance Target**:
- No overhead vs current separate implementations
- Mode switch: <1μs

---

### Week 3-4: Documentation Foundation
**Goal**: Create "Science vs Scale" narrative

**Tasks**:
1. Write `mini-nangila/README.md`
   - Open-Core philosophy
   - Feature comparison table (from spec)
   - Quick start guide
   - Build instructions

2. Write `mini-nangila/docs/ARCHITECTURE.md`
   - Crate dependency graph
   - Trait design rationale
   - Q8.23 format explanation
   - Closed-loop prediction diagram

3. Write `mini-nangila/docs/BENCHMARKS.md`
   - Baseline results for all current examples
   - Compression ratio calculations
   - Throughput measurements (MB/s)
   - Comparison: mini-nangila (CPU) vs nangila (GPU)

**Deliverable**: Complete documentation for current codebase

**Performance Baseline** (to document):
- AI throughput: ~100-200 MB/s (CPU)
- HPC throughput: ~150-300 MB/s (CPU)
- Compression ratios: 20-40× (AI), 50-100× (HPC)

---

## Phase 2: Real-World Examples (Weeks 5-8)

### Week 5: NanoGPT Example (AI Domain)
**Goal**: Train small language model with compression

**Tasks**:
1. Create `mini-nangila/examples/nano_gpt/`
   ```
   nano_gpt/
   ├── Cargo.toml
   ├── model.rs          # Simple transformer (125M params)
   ├── trainer.rs        # Training loop with compression
   ├── data.rs           # TinyShakespeare dataset
   └── main.rs           # CLI interface
   ```

2. Implement CPU-based training loop
   - Forward pass
   - Backward pass (manual gradient computation)
   - Compressed gradient aggregation (simulate 2-worker DDP)

3. Add compression pipeline
   - Momentum predictor (β=0.9)
   - Stochastic INT4 quantizer
   - Topology mask (drop 30% of layers)

4. Measure convergence
   - Train for 1000 steps
   - Compare loss curve: compressed vs uncompressed
   - Validate <5% accuracy degradation

**Deliverable**: Working NanoGPT training with 20× compression

**Performance Target**:
- Training speed: 5-10 steps/sec (CPU)
- Compression ratio: 20-30×
- Convergence: Match baseline within 5%
- Memory: <4GB RAM

**Dataset**: TinyShakespeare (1MB, included in repo)

---

### Week 6: LAMMPS Example (HPC Domain)
**Goal**: Compress molecular dynamics trajectory

**Tasks**:
1. Create `mini-nangila/examples/lammps_compress/`
   ```
   lammps_compress/
   ├── Cargo.toml
   ├── trajectory.rs     # Load LAMMPS dump files
   ├── compressor.rs     # Predictive compression pipeline
   └── main.rs           # CLI: compress/decompress
   ```

2. Implement trajectory loader
   - Parse LAMMPS dump format (ASCII)
   - Extract position/velocity/force vectors
   - Support 10K-100K particles

3. Add compression pipeline
   - Linear extrapolator predictor
   - Error-bounded quantizer (ε=1e-3)
   - RLE for zero-heavy residuals

4. Validate error bounds
   - Max absolute error < ε
   - RMS error < ε/2
   - Energy conservation check

**Deliverable**: LAMMPS trajectory compressor with guaranteed error bounds

**Performance Target**:
- Compression ratio: 50-100×
- Throughput: 200-400 MB/s (CPU)
- Error bound: Strict ε guarantee
- Memory: <2GB for 100K particles

**Dataset**: Lennard-Jones liquid (public dataset, 10MB)

---

### Week 7: Rotor Twin Example (Digital Twin Domain)
**Goal**: Simulate edge-cloud sensor synchronization

**Tasks**:
1. Create `mini-nangila/examples/rotor_twin/`
   ```
   rotor_twin/
   ├── Cargo.toml
   ├── sensor.rs         # Simulated IMU (gyro + accel)
   ├── edge.rs           # Edge node with predictor
   ├── cloud.rs          # Cloud node (reconstruction)
   └── main.rs           # Run simulation
   ```

2. Implement sensor simulator
   - Spinning rotor physics (ω = 100 Hz)
   - 6-DOF IMU data (3-axis gyro + accel)
   - Noise injection (Gaussian, σ=0.01)

3. Add edge-cloud sync
   - Linear predictor (extrapolate rotation)
   - Error-bounded quantizer (ε=0.05)
   - TCP socket transport (localhost)

4. Measure bandwidth reduction
   - Raw: 6 floats × 100 Hz = 2.4 KB/s
   - Compressed: ~24 bytes/s (100× reduction)
   - Latency: <10ms (localhost)

**Deliverable**: Working digital twin with 100× bandwidth reduction

**Performance Target**:
- Compression ratio: 80-120×
- Latency: <10ms (edge → cloud)
- Reconstruction error: <5% RMS
- CPU usage: <5% (edge node)

**Dataset**: Synthetic (generated on-the-fly)

---

### Week 8: Integration Testing
**Goal**: Validate all three examples work together

**Tasks**:
1. Create unified test suite
   ```bash
   cd mini-nangila
   cargo test --all-features
   cargo run --example nano_gpt -- --steps 100
   cargo run --example lammps_compress -- data/lj_liquid.dump
   cargo run --example rotor_twin -- --duration 60
   ```

2. Add CI workflow
   ```yaml
   # .github/workflows/mini-nangila.yml
   - Run all examples
   - Verify compression ratios
   - Check error bounds
   - Measure throughput
   ```

3. Document results in `BENCHMARKS.md`

**Deliverable**: Automated test suite with performance validation

---

## Phase 3: Advanced Features (Weeks 9-12)

### Week 9: HDF5 Checkpoint Plugin
**Goal**: Enable drop-in compression for scientific codes

**Tasks**:
1. Create `mini-nangila/nangila-checkpoint/`
   ```rust
   // nangila-checkpoint/src/lib.rs
   pub fn register_hdf5_filter() -> Result<(), Error>;
   
   // HDF5 filter callbacks
   extern "C" fn nangila_filter_encode(...);
   extern "C" fn nangila_filter_decode(...);
   ```

2. Implement HDF5 filter plugin
   - Register custom filter ID
   - Encode: Apply predictive compression
   - Decode: Reconstruct with error bounds

3. Add Python bindings
   ```python
   # python/nangila_checkpoint.py
   import h5py
   import nangila_checkpoint
   
   nangila_checkpoint.register()
   with h5py.File('data.h5', 'w') as f:
       f.create_dataset('trajectory', data=x, 
                       compression='nangila', 
                       compression_opts={'epsilon': 1e-3})
   ```

4. Create example
   ```bash
   cd mini-nangila/examples/hdf5_plugin
   python write_compressed.py  # Create compressed HDF5
   python read_verify.py       # Verify error bounds
   ```

**Deliverable**: HDF5 plugin with Python bindings

**Performance Target**:
- Compression ratio: 30-60× (typical scientific data)
- Overhead: <10% vs uncompressed write
- Compatibility: HDF5 1.10+

---

### Week 10: Edge Device Support (no_std)
**Goal**: Enable embedded/IoT deployment

**Tasks**:
1. Create `mini-nangila/nangila-edge/`
   ```toml
   [dependencies]
   nangila-math = { path = "../nangila-math", default-features = false }
   
   [features]
   default = ["std"]
   std = ["nangila-math/std"]
   ```

2. Port core math to no_std
   - Remove Vec → use fixed-size arrays
   - Remove String → use &str
   - Remove Box → use stack allocation

3. Create minimal example
   ```rust
   // examples/edge_minimal.rs
   #![no_std]
   use nangila_edge::{LinearPredictor, ErrorBoundedQuantizer};
   
   // Compress sensor data in 2KB RAM
   ```

4. Test on embedded target (QEMU ARM Cortex-M4)

**Deliverable**: no_std compatible core with <2KB RAM footprint

**Performance Target**:
- RAM: <2KB (predictor + quantizer state)
- Flash: <10KB (code size)
- Latency: <1ms per compression (100 MHz ARM)

---

### Week 11: Tutorial & Verification
**Goal**: Prove mathematical correctness

**Tasks**:
1. Write `mini-nangila/docs/TUTORIAL.md`
   - **Section 1**: Entropy Reduction Lemma
     - Prove H(residual) < H(gradient) empirically
     - Show compression ratio vs prediction accuracy
   
   - **Section 2**: Error Bound Verification
     - Demonstrate ε-guarantee for HPC mode
     - Plot error distribution (should be < ε)
   
   - **Section 3**: Convergence Analysis
     - Compare AI training curves (compressed vs baseline)
     - Show <5% accuracy degradation

2. Add interactive Jupyter notebooks
   ```
   mini-nangila/notebooks/
   ├── 01_entropy_reduction.ipynb
   ├── 02_error_bounds.ipynb
   └── 03_convergence.ipynb
   ```

3. Create reproducibility script
   ```bash
   ./scripts/reproduce_paper_results.sh
   # Runs all examples, generates plots, compares to spec
   ```

**Deliverable**: Complete tutorial proving core claims

---

### Week 12: Release Preparation
**Goal**: Public v0.1.0 release

**Tasks**:
1. Final documentation review
   - README.md (clear quick start)
   - ARCHITECTURE.md (design rationale)
   - BENCHMARKS.md (performance data)
   - TUTORIAL.md (verification)
   - API docs (rustdoc)

2. Licensing cleanup
   - Add Apache-2.0 / MIT dual license
   - Add NOTICE file (open-core boundary)
   - Add CLA template

3. Package for distribution
   ```bash
   # Publish to crates.io
   cd mini-nangila/nangila-math && cargo publish
   cd mini-nangila/nangila-core && cargo publish
   cd mini-nangila/nangila-ai && cargo publish
   cd mini-nangila/nangila-hpc && cargo publish
   cd mini-nangila/nangila-twin && cargo publish
   ```

4. Create release announcement
   - Blog post
   - arXiv paper (optional)
   - HN/Reddit post

**Deliverable**: Public release on GitHub + crates.io

---

## Performance Targets Summary

### AI Domain (NanoGPT)
| Metric | Target | Validation |
|--------|--------|------------|
| Compression Ratio | 20-30× | Measure bytes transmitted |
| Training Speed | 5-10 steps/sec | Time 1000 steps |
| Convergence | <5% degradation | Compare final loss |
| Memory | <4GB RAM | Monitor peak usage |

### HPC Domain (LAMMPS)
| Metric | Target | Validation |
|--------|--------|------------|
| Compression Ratio | 50-100× | Measure file size |
| Throughput | 200-400 MB/s | Time compression |
| Error Bound | Strict ε guarantee | Max absolute error |
| Memory | <2GB | Monitor peak usage |

### Digital Twin (Rotor)
| Metric | Target | Validation |
|--------|--------|------------|
| Compression Ratio | 80-120× | Measure bandwidth |
| Latency | <10ms | Measure round-trip |
| Reconstruction Error | <5% RMS | Compare signals |
| CPU Usage | <5% | Monitor load |

---

## Testing Strategy

### Unit Tests (Continuous)
```bash
cargo test --all-features
```
- Fixed-point arithmetic correctness
- Predictor/quantizer interfaces
- Error bound validation
- Topology mask logic

### Integration Tests (Weekly)
```bash
./scripts/run_all_examples.sh
```
- NanoGPT training convergence
- LAMMPS error bound compliance
- Rotor twin bandwidth measurement

### Cross-Platform Tests (Pre-release)
```bash
# Test on multiple architectures
docker run --rm -v $(pwd):/work rust:latest cargo test
docker run --rm -v $(pwd):/work arm64v8/rust:latest cargo test
```
- Verify determinism (x86_64 vs ARM64)
- Check bit-exact results
- Validate no platform-specific bugs

---

## Success Criteria

### Technical
- ✅ All three domain examples working
- ✅ Compression ratios meet spec (20×, 50×, 100×)
- ✅ Error bounds verified (HPC mode)
- ✅ Convergence validated (AI mode)
- ✅ CPU-only (no GPU required)
- ✅ <5GB RAM for all examples

### Documentation
- ✅ Complete README with quick start
- ✅ Architecture documentation
- ✅ Tutorial proving core claims
- ✅ Benchmark results published
- ✅ API documentation (rustdoc)

### Community
- ✅ Public GitHub repository
- ✅ Published on crates.io
- ✅ Apache-2.0 / MIT licensed
- ✅ CLA for contributors
- ✅ Issue templates and contributing guide

---

## Risk Mitigation

### Risk: NanoGPT too slow on CPU
**Mitigation**: Use smaller model (50M params) or fewer training steps (100 instead of 1000)

### Risk: LAMMPS data too large
**Mitigation**: Use smaller trajectory (10K particles instead of 100K)

### Risk: HDF5 plugin complex
**Mitigation**: Start with simple Python wrapper, defer C plugin to Phase 4

### Risk: no_std too restrictive
**Mitigation**: Make it optional feature, not required for v0.1.0

---

## Post-Release Roadmap (Phase 4+)

### Community Feedback (Weeks 13-16)
- Address bug reports
- Add requested features
- Improve documentation based on user questions

### Advanced Examples (Weeks 17-20)
- Transformer (GPT-2 scale)
- CFD simulation (Navier-Stokes)
- Multi-sensor fusion (autonomous vehicle)

### Performance Optimization (Weeks 21-24)
- SIMD vectorization (AVX2/NEON)
- Multi-threading (Rayon)
- Memory pool allocator

---

## Resource Requirements

### Development
- 1-2 engineers (full-time)
- Access to multi-core CPU (8+ cores recommended)
- 16GB+ RAM for development
- CI/CD credits (GitHub Actions)

### Testing
- x86_64 Linux machine
- ARM64 machine (optional, can use QEMU)
- 100GB storage for datasets

### Documentation
- Technical writer (part-time, weeks 11-12)
- Graphic designer for diagrams (optional)

---

## Conclusion

This plan takes mini-nangila from 60% to 100% spec compliance in 12 weeks. The key insight is **CPU-first development**: by avoiding GPU dependencies, we make the code accessible to any researcher with a laptop.

The three real-world examples (NanoGPT, LAMMPS, Rotor) prove the core claims across all domains. The tutorial and benchmarks provide reproducible verification of the mathematical foundations.

Upon completion, mini-nangila will be a standalone, verifiable reference implementation that demonstrates the "science" while the proprietary nangila remains the "scale" solution.
