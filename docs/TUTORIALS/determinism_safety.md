## Tutorial: Why Determinism Matters for Safety (CPU)

This tutorial demonstrates:
- Bit-exactness of Q8.23 fixed-point across platforms
- Reproducibility of stochastic rounding (deterministic PRNG)
- Closed-loop prediction avoids drift between edge/cloud

### 1) Fixed-Point Determinism

Run the determinism hash example locally:

```
cargo run -q -p nangila-math --example determinism_hash
```

Compare the `Q823_HASH` across platforms (Linux/macOS); CI uploads artifacts in `mini-nangila` workflow → `determinism-hash-*`.

### 2) Stochastic Quantization Reproducibility

The INT4 DDP hook seeds a per-bucket PRNG deterministically. Run a small CPU training and inspect `ddp_metrics.csv` to see stable ratios over time:

```
torchrun --standalone --nproc_per_node=1 -m nangila_ddp.train_nano \
  --device cpu --steps 50 --batch_size 4 --block_size 64 \
  --d_model 128 --nhead 4 --nlayers 2 \
  --metrics_path ddp_metrics.csv --train_log train_log.csv

python -m nangila_ddp.plot_train_and_compression train_log.csv ddp_metrics.csv
```

### 3) Closed-Loop Prediction (Twin)

Run the rotor twin demo and observe error stability:

```
cargo run --release --example rotor_twin -- --duration 5 --rate-hz 50 --epsilon 0.05
```

Closed-loop ensures both edge and cloud update predictors with reconstructed values, preventing drift.
