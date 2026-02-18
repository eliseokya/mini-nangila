## CPU Verification Guide (Mini‑Nangila)

Goal: enable any researcher on a commodity CPU to verify the science (predictive‑residual compression, ε bounds, determinism) without GPUs.

### 1) Build and Test (Core)

```
cd mini-nangila
cargo build
cargo test
```

### 2) Determinism Tests

Run Q8.23 determinism tests on fixed‑point arithmetic:

```
cargo test -p nangila-math
```

### 3) CPU Examples

- Throughput (AI synthetic):
```
cargo run --release --example throughput_bench
```

- HPC compress benchmark (ε bound):
```
cargo run --release --example hpc_compress_bench
```

- Topology calibration (AI):
```
cargo run --release --example topology_calibration
```

- Rotor twin (edge/cloud closed loop):
```
cargo run --release --example rotor_twin -- --duration 5 --rate-hz 50
```

### 4) HDF5 Checkpointing (Optional)

Install HDF5, then run the feature‑gated demo:

```
cd mini-nangila/examples/checkpoint_demo
cargo run --features hdf5 -- --output traj_ckpt.h5 --dataset traj --len 50000 --chunk 1024 --epsilon 1e-3
```

Read back from Python:

```
pip install h5py numpy
python ../../python/snippets/read_hdf5_checkpoint.py traj_ckpt.h5
```

### 5) CPU‑only DDP Verification (Optional)

If Python and PyTorch are available, run a 2‑process CPU DDP training to verify the INT4 comm hook logic (slow, small steps):

```
torchrun --standalone --nproc_per_node=2 \
  -m nangila_ddp.train_nano \
  --device cpu \
  --steps 100 \
  --batch_size 4 \
  --block_size 64 \
  --metrics_path ddp_metrics.csv \
  --train_log train_log.csv

python -m nangila_ddp.plot_train_and_compression train_log.csv ddp_metrics.csv
```

### 6) Reproducibility

- Random seeds are fixed in examples/tests.
- Fixed‑point math provides bit‑exact results across OS/CPU.
- Optional CI matrix (Linux/macOS) validates determinism.

