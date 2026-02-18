## NanoGPT DDP with INT4 Hook — Quickstart

This directory provides:
- `comm_hook.py`: a DDP comm hook implementing INT4 nibble packing with closed-loop predictor and metrics logging.
- `train_nano.py`: a 2‑GPU training harness using a tiny Transformer on TinyShakespeare (auto-download) or a provided text file.
- `plot_train_and_compression.py`: overlay loss vs time with compression ratio vs time.

### Prerequisites
- PyTorch with CUDA and Distributed support
- Two GPUs (for 2‑GPU run)

### Run Training (2 GPUs, CUDA)

```
torchrun --nproc_per_node=2 \
  -m nangila_ddp.train_nano \
  --steps 500 \
  --batch_size 16 \
  --block_size 128 \
  --d_model 256 --nhead 4 --nlayers 4 \
  --metrics_path ddp_metrics.csv \
  --train_log train_log.csv
```

This will:
- Auto-download TinyShakespeare (~1MB) on rank 0 if `--data` is not provided
- Register the INT4 hook (nibble-packed, all_gather) with deterministic stochastic rounding
- Log:
  - `ddp_metrics.csv`: `ts,hook_idx,bucket_elems,raw_bytes,comp_bytes,ratio`
  - `train_log.csv`: `ts,step,loss,lr,tokens_per_sec,step_ms`

### Plot Results

```
python -m nangila_ddp.plot_train_and_compression train_log.csv ddp_metrics.csv
```

Outputs `train_and_compression.png` with loss (left axis) and compression ratio (right axis) over time.

### Notes
- CPU‑only verification: you can run on CPU with Gloo backend (slower but validates science):

```
torchrun --standalone --nproc_per_node=2 \
  -m nangila_ddp.train_nano \
  --device cpu \
  --steps 100 \
  --batch_size 4 \
  --block_size 64 \
  --metrics_path ddp_metrics.csv \
  --train_log train_log.csv
```

- Hook returns reduced average gradients reconstructed from gathered compressed residuals; optimizer step proceeds normally.
- `metrics_path` is written by rank 0 from inside the hook; `train_log.csv` is written by rank 0 from the training loop.
- For larger models or more steps, adjust batch size / model dims for your GPU memory.
