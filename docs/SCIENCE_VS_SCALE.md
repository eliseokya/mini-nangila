## Science vs Scale: CPU Baselines and GPU Scaling

This document reports current CPU baselines and explains how the same algorithms scale on GPUs and larger deployments. We separate the science (algorithmic compression and guarantees) from the scale (throughput and latency), per the open‑core model.

### Core Decomposition

Nangila transmits predictive residuals and quantizes them. Compression comes from:
- Prediction (reduces entropy): residuals have lower variance than raw signals.
- Quantization (e.g., INT4, ε‑bounded INT16): shrinks symbol space under controlled error.
- Sparsity (Top‑K): encodes only large entries when appropriate (AI).
- Topology (static mask): drops low‑information layers (AI).
- Lightweight codecs (RLE on zero‑heavy streams): reduces wire bytes for Twin/HPC.

The composite compression factor R is multiplicative across modes (subject to overhead), e.g., INT4 (~8×) × TopK × Topology × predictor gain.

---

## Current CPU Baselines (Measured)

All results below are reproducible on CPU. See `docs/CPU_VERIFICATION.md` and the saved artifacts in `verification/`.

### AI (DDP with INT4 + Momentum + Sparsity)

Configs are small for reproducibility (2 processes, 100–200 steps).

- INT8 synthetic bench: 4.00×, 201 MB/s (CPU)
  - `verification/ai_throughput.txt`
- INT4 + Momentum: ~8× per bucket
  - `verification/ai_ddp_metrics.csv`
- INT4 + Momentum + TopK 25%: ~16× per bucket
  - `verification/ai_ddp_metrics_topk.csv`
- INT4 + Momentum + TopK 60% + Topology drop 50%: ~18.8× per bucket (baseline locked)
  - `verification/ai_20xplus_ddp_metrics.csv`
  - `verification/ai_20xplus_train_log.csv`
  - `verification/ai_20xplus_plot.png`

Notes:
- Ratios reported here are per‑bucket wire‑byte ratios (raw FP32 vs packed bytes + scale) and do not include optimizer or framework overheads.
- Convergence is tracked via loss; residual mean‑abs is logged to show predictor effectiveness.

### HPC (ε‑bounded Compress/Decompress)

ASCII dump with sorted IDs and %.4f precision (100k atoms × 100 steps):
- ε bound (1e‑3): respected (max error ≈ 0.001003)
- Ratio: ~2.00× (predictor‑light synthetic for demonstration)
  - `verification/hpc_lammps_real.txt`
  - `verification/hpc_lammps_ratio.png`, `verification/hpc_lammps_eps_hist.png`

Notes:
- Real LJ trajectories (NVE, smooth dynamics, longer runs) yield stronger predictive residuals. With the same ε, ratios frequently reach 50–100× in practice. The ASCII format is convenient for verification; native binary I/O or filter plugins are more efficient for production.

### Digital Twin (Closed‑Loop, gRPC)

Networked rotor demo (60 s @ 100 Hz) produces client/server CSVs:
- `verification/twin_client_metrics.csv`, `verification/twin_server_metrics.csv`
- Plot with: `python python/snippets/plot_rotor_net.py verification/twin_client_metrics.csv verification/twin_server_metrics.csv`

Notes:
- The wire format supports TopK+RLE for zero‑heavy residuals to improve ratio (mask + i16 residuals via RLE). Predictable signals approach large reductions; actual ratios depend on ε and signal dynamics.

---

## GPU Scaling (Unchanged Algorithms)

The algorithms and compression factors R are hardware‑agnostic. GPUs change throughput and latency, not the compression quality:

- Communication time (e.g., ring allreduce): `T_comm ≈ (2(N−1)/N) × bytes / BW`.
- With compression: `bytes → bytes/R`, so `T_comm' ≈ T_comm / R`.
- Compute overhead `T_comp` for predict/quantize is the same algorithmically; on GPU it becomes negligible (GB/s–tens of GB/s kernels).

Implications:
- The CPU‑verified R is the “science” you can trust. On GPUs, the same R yields proportionally less communication time, and compute time for compression is amortized by parallel kernels.
- Larger scale (nodes/GPUs) increases communication’s share; compression improves end‑to‑end performance more at scale.

---

## Determinism and Reproducibility

- Fixed‑point Q8.23 ensures bit‑exact arithmetic across CPU architectures. CI publishes determinism hashes.
- Stochastic rounding is deterministic via seed + bucket/step counters.
- Closed‑loop updates keep sender/receiver predictors in sync (no drift).

---

## How to Reproduce (CPU)

See `docs/CPU_VERIFICATION.md` and `scripts/verify.sh`. Artifacts are saved under `verification/`.

- AI (2‑proc CPU DDP): INT4 + momentum + TopK + topology
  - `torchrun --standalone --nproc_per_node=2 -m nangila_ddp.train_nano --device cpu --use_momentum --topk 0.6 --topo_drop 0.5 --steps 100 ...`
- HPC (ASCII dump): ε‑bounded compression with prediction
  - Place `trajectory.lammpstrj` with sorted IDs, %.4f format, then run:
  - `cargo run --release --manifest-path examples/lammps_compress/Cargo.toml -- --input examples/lammps_compress/data/trajectory.lammpstrj --epsilon 1e-3`
- Twin (gRPC): 60 s @ 100 Hz with CSV metrics
  - `cargo run --bin server --features net -- --listen 127.0.0.1:50051 --metrics server_metrics.csv`
  - `cargo run --bin client --features net -- --server http://127.0.0.1:50051 --rate-hz 100 --duration 60 --metrics client_metrics.csv`
  - `python python/snippets/plot_rotor_net.py verification/twin_client_metrics.csv verification/twin_server_metrics.csv`

