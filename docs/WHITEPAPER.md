# Mini‑Nangila Open‑Core Whitepaper (CPU‑Verified Science)

Authors: The Nangila Project

Date: 2026‑02‑11

---

## Abstract

This whitepaper presents Mini‑Nangila, an open‑core, CPU‑verified reference implementation of Nangila’s predictive‑residual compression framework spanning three domains: AI (gradient compression), HPC (ε‑bounded trajectory compression), and Digital Twins (edge‑cloud synchronization). We formalize the predictive‑residual factorization, detail quantization strategies (stochastic INT4; error‑bounded INT16), sparsity (Top‑K), and lightweight codecs (RLE) and explain why compression quality (ratios, ε‑guarantees) is hardware‑agnostic while throughput/latency scales with compute (GPUs, nodes). We report CPU baselines with reproducible artifacts, justify CPU‑first verification for determinism and accessibility, and outline the path to scale.

---

## 1. Introduction and Motivation

Large‑scale training and simulation systems are increasingly communication‑bound. Transmitting full‑precision data (e.g., FP32 gradients or trajectories) wastes bandwidth: time correlation and structure make signals predictable. Nangila reduces transmitted information by (1) predicting signals from their history and (2) quantizing only the residual error. This cuts wire bytes without compromising convergence or bounded accuracy.

Mini‑Nangila is the open‑core reference: all algorithms run on any CPU and are easy to verify. GPU acceleration is proprietary and improves speed—not compression quality. This separation makes the science auditable and the scale practical.

---

## 2. Predictive‑Residual Compression

Let x_t ∈ R^n be a signal at time t (gradient, state, sensor). A predictor P produces \hat{x}_t from history H_t (e.g., EMA or linear extrapolation):

\[ \hat{x}_t = P(H_t), \quad r_t = x_t - \hat{x}_t. \]

Only r_t is transmitted after quantization Q; the receiver reconstructs \( \tilde{x}_t = \hat{x}_t + Q^{-1}(Q(r_t)) \). Sender and receiver both update their predictors with \( \tilde{x}_t \) (closed loop), avoiding drift.

Compression factor R is multiplicative across modes (up to small overhead):

- Prediction lowers entropy (H(r_t) < H(x_t)).
- Quantization shrinks symbol space (e.g., INT4 ≈ 8×; INT16 ε‑bound preserves accuracy).
- Sparsity (Top‑K) encodes only large entries.
- Topology masks drop low‑information layers (AI).
- Light codecs (RLE) compress zero‑heavy residuals (Twin/HPC).

---

## 3. Mathematical Components

### 3.1 Predictors

- AI: Momentum (EMA) — \( m_t = \beta m_{t-1} + (1-\beta) g_{t-1}; \; \hat{g}_t = m_t \).
- HPC/Twin: Linear extrapolation — \( \hat{S}_{t+1} = 2 S_t - S_{t-1} \) (dt=1).

Predictors reduce variance of residuals (entropy reduction). In practice we log mean‑|r_t| as a proxy of predictability.

### 3.2 Quantizers

- Stochastic INT4 (AI): unbiased rounding to 4‑bit integers (−8..7) with deterministic PRNG; one scale per bucket/layer.
- Error‑bounded INT16 (HPC): \( q = \mathrm{round}(r / (2\epsilon)), \; |r - \tilde{r}| \leq \epsilon \).

### 3.3 Sparsity and Topology (AI)

- Top‑K: keep k% largest |r|; transmit indices/values. Useful to force zeros.
- Topology mask: drop a static fraction of low‑information layers; complements compression without harming convergence when chosen conservatively.

### 3.4 Lightweight Codecs (Twin/HPC)

Residuals concentrated near zero benefit from RLE: runs of zeros encoded as short opcodes; non‑zero literals stored compactly (e.g., i16 pairs). For Twin, we combine Top‑K→zeros + RLE on residual grid; HPC also uses RLE when appropriate.

---

## 4. Determinism and Reproducibility

### 4.1 Fixed‑Point Core (Q8.23)

Mini‑Nangila’s math layer uses Q8.23 fixed‑point (i32 with 23 fractional bits) for deterministic conversions and operations. Round‑to‑nearest and saturating arithmetic ensure bit‑exact results across CPUs. CI publishes a content hash of canonical encodings to confirm cross‑platform determinism.

### 4.2 Deterministic Stochastic Rounding

Stochastic quantization uses seeded PRNGs keyed by (bucket, step); we use hash‑based or per‑bucket generators so that repeated runs produce the same bytes.

### 4.3 Closed‑Loop Updates

Both sender and receiver update predictors with reconstructed states \( \tilde{x}_t \). This keeps them in lockstep, avoiding long‑term drift.

---

## 5. CPU‑First Verification

### Why CPU?

- Accessibility: any researcher can run the full suite without GPUs.
- Determinism: avoids GPU nondeterminism and vendor‑specific quirks.
- Scientific focus: compression factors and ε‑guarantees are hardware‑agnostic. CPU verifies the *algorithms*; GPUs accelerate *execution*.

### Reproducibility Pipeline

Run `./scripts/verify.sh` to build, test, generate determinism hash, and execute representative AI/HPC/Twin demos. Artifacts (CSVs, plots) are saved under `verification/`.

---

## 6. Implementation Architecture

Crates:

- `nangila‑math`: Q8.23, deterministic ops.
- `nangila‑core`: traits (Predictor, Quantizer), TopologyMask, UnifiedQuantizer.
- `nangila‑ai`: Momentum predictor; Top‑K; (Python) INT4 DDP hook.
- `nangila‑hpc`: Linear predictor; ErrorBoundedQuantizer; RLE.
- `nangila‑twin`: Edge/Cloud closed‑loop sync (Rust); gRPC demo (feature‑gated).

Design is trait‑based (zero‑cost abstractions) with clear layering: math → core → domain modules.

---

## 7. CPU Baselines (Measured)

All figures correspond to saved artifacts under `verification/`.

### 7.1 AI (2‑proc CPU DDP, Tiny configs)

- INT8 synthetic bench: 4.00×, 201 MB/s (CPU) — `ai_throughput.txt`.
- INT4 + momentum: ~8× per bucket — `ai_ddp_metrics.csv`.
- INT4 + momentum + TopK 25%: ~16× — `ai_ddp_metrics_topk.csv`.
- **Baseline (locked)**: INT4 + momentum + TopK 60% + topology drop 50% ≈ **18.8×** per bucket — `ai_20xplus_*`.

We also log residual mean‑abs to show predictor effectiveness. Convergence is tracked via loss.

### 7.2 HPC (ASCII LAMMPS, ε‑bounded)

ASCII dump (100k atoms × 100 steps, sorted IDs, 4 decimal places):

- **ε bound respected** (1e‑3): max_err ≈ 0.001003.
- **Ratio ≈ 2.00×** on the generated signal — `hpc_lammps_real.txt`, with ratio plot and ε histogram.

Notes: Real LJ trajectories (NVE, longer runs, higher smoothness) produce stronger prediction and higher R. ASCII is convenient for validation; binary I/O or HDF5 filters are better for production throughput.

### 7.3 Twin (gRPC, 60 s @ 100 Hz)

Client/server CSV logs saved. Plotting produces the ratio curve over time. Wire format supports TopK+RLE to improve ratio for predictable signals; ε can be tuned per sensor.

---

## 8. Science vs Scale (GPU and Cluster)

The algorithms and compression ratios R do not depend on hardware. Scaling changes throughput/latency:

- Communication (e.g., ring allreduce) time: \( T_{comm} ≈ (2(N−1)/N)·bytes/BW \).
- With compression: \( bytes → bytes/R \Rightarrow T'_{comm} ≈ T_{comm}/R \).
- Predictor/quantizer compute becomes negligible on GPUs; kernels achieve GB/s to tens of GB/s.

Thus CPU‑verified R directly translates to wall‑clock savings; the larger the cluster, the bigger the impact (communication dominates at scale).

---

## 9. Limitations and Practical Considerations

- **ASCII overhead** (HPC): convenient but large; use binary/HDF5 for speed at scale.
- **Mask overhead** (AI/Twin): Top‑K masks add bits; combine with RLE or block formats to mitigate.
- **Tuning**: R depends on Top‑K fraction, topology drop, ε; choose conservatively for convergence/accuracy.
- **Residual re‑injection**: always update predictors with reconstructed values to avoid drift.

---

## 10. Future Work

- Real LAMMPS end‑to‑end runs (ε distribution & ratios) on large, smooth datasets.
- HDF5 filter plugin (production‑grade) and native binary I/O paths.
- Dynamic topology and adaptive bits per layer.
- GPU kernels and NCCL/MPI transports (scale‑optimized).
- Learned predictors and Safe Mode (monitoring & fallback).

---

## 11. Reproducibility and CI

- Determinism hash (`nangila‑math/examples/determinism_hash.rs`) published for Linux/macOS.
- Bench artifacts (AI/HPC) uploaded by CI for traceability.
- `scripts/verify.sh` runs build/tests + representative demos and saves artifacts.

---

## 12. Licensing and Contributions

Open‑core crates are Apache‑2.0/MIT licensed. Contributions require CLA/DCO acknowledgement to enable dual‑licensing and enterprise add‑ons (see `docs/CLA.md`).

---

## Appendix A: Commands and Artifacts

See `docs/CPU_VERIFICATION.md` for one‑pass verification. Key commands are summarized below.

AI (2‑proc CPU DDP, baseline ~18.8×):

```
torchrun --standalone --nproc_per_node=2 -m nangila_ddp.train_nano \
  --device cpu --use_momentum --topk 0.6 --topo_drop 0.5 \
  --steps 100 --batch_size 4 --block_size 64 \
  --d_model 128 --nhead 4 --nlayers 2 \
  --metrics_path ddp_metrics.csv --train_log train_log.csv
python -m nangila_ddp.plot_train_and_compression train_log.csv ddp_metrics.csv
```

HPC (ASCII LAMMPS, ε‑bounded):

```
cargo run --release --manifest-path examples/lammps_compress/Cargo.toml -- \
  --input examples/lammps_compress/data/trajectory.lammpstrj --epsilon 1e-3
python python/snippets/plot_hpc_logs.py verification/hpc_lammps_real.txt
```

Twin (gRPC, 60 s @ 100 Hz):

```
cargo run --bin server --features net -- --listen 127.0.0.1:50051 --metrics server_metrics.csv
cargo run --bin client --features net -- --server http://127.0.0.1:50051 \
  --rate-hz 100 --duration 60 --metrics client_metrics.csv
python python/snippets/plot_rotor_net.py verification/twin_client_metrics.csv verification/twin_server_metrics.csv
```

