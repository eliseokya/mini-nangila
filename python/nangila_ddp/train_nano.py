import os
import io
import time
import math
import argparse
import urllib.request
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from nangila_ddp import register_nangila_hook


URL_TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        self.block = block_size
        self.chars = sorted(list(set(text)))
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.vocab = len(self.chars)
        self.data = torch.tensor([self.c2i[c] for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, self.data.numel() - self.block - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.block]
        y = self.data[idx + 1: idx + 1 + self.block]
        return x, y


class TinyGPT(nn.Module):
    def __init__(self, d_model=256, nhead=4, nlayers=4, vocab=5000, block_size=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, block_size, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.lm = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, : x.size(1), :]
        h = self.enc(h)
        return self.lm(h)


def get_rank_world():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def main():
    ap = argparse.ArgumentParser(description="NanoGPT DDP with INT4 hook")
    ap.add_argument("--data", type=str, default="", help="Path to text file (TinyShakespeare). If empty, auto-download.")
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--metrics_path", type=str, default="ddp_metrics.csv")
    ap.add_argument("--train_log", type=str, default="train_log.csv")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--use_momentum", action="store_true", help="Enable per-bucket EMA predictor in hook")
    ap.add_argument("--topk", type=float, default=0.0, help="TopK fraction (0.0..1.0) for static mask in hook")
    ap.add_argument("--topo_drop", type=float, default=0.0, help="Static topology drop fraction (0.0..1.0)")
    args = ap.parse_args()

    # Backend/device selection
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "nccl" if dev == "cuda" else "gloo"

    dist.init_process_group(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if dev == "cuda":
        torch.cuda.set_device(local_rank)
    rank, world = get_rank_world()

    torch.manual_seed(args.seed + rank)
    if dev == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    # Load data
    if rank == 0:
        if args.data and os.path.exists(args.data):
            text = open(args.data, "r", encoding="utf-8").read()
        else:
            try:
                print("Downloading TinyShakespeare...")
                text = download_text(URL_TINY_SHAKESPEARE)
            except Exception:
                print("Download failed; using synthetic text")
                text = ("abc def ghi ") * 100000
    else:
        text = ""
    # broadcast text length then content
    lens = torch.tensor([len(text)], dtype=torch.long, device=dev) if rank == 0 else torch.zeros(1, dtype=torch.long, device=dev)
    dist.broadcast(lens, src=0)
    buf = torch.empty(int(lens.item()), dtype=torch.uint8, device=dev)
    if rank == 0 and text:
        buf.copy_(torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8, device=dev))
    dist.broadcast(buf, src=0)
    text = bytes(buf.cpu().tolist()).decode("utf-8")

    dataset = CharDataset(text, args.block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    model = TinyGPT(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, vocab=dataset.vocab, block_size=args.block_size)
    if dev == "cuda":
        model = model.cuda()
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)
    register_nangila_hook(
        model,
        beta=0.9,
        bits=4,
        stochastic=True,
        use_fp16_transport=False,
        metrics_path=args.metrics_path,
        use_momentum=args.use_momentum,
        topk_percent=args.topk,
        topo_drop_percent=args.topo_drop,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    if rank == 0:
        with open(args.train_log, "w") as fh:
            fh.write("ts,step,loss,lr,tokens_per_sec,step_ms\n")

    model.train()
    step = 0
    tok_per_step = args.batch_size * args.block_size
    t0 = time.time()
    for epoch in range(1000000):
        sampler.set_epoch(epoch)
        for xb, yb in loader:
            step += 1
            if dev == "cuda":
                xb = xb.cuda(non_blocking=True)
                yb = yb.cuda(non_blocking=True)

            t_step = time.time()
            logits = model(xb)
            loss = loss_fn(logits.view(-1, dataset.vocab), yb.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_ms = (time.time() - t_step) * 1000.0

            if rank == 0:
                elapsed = time.time() - t0
                tps = tok_per_step / max(1e-6, (step_ms / 1000.0))
                with open(args.train_log, "a") as fh:
                    fh.write(f"{time.time():.6f},{step},{loss.item():.6f},{args.lr},{tps:.2f},{step_ms:.2f}\n")
                if step % 10 == 0:
                    print(f"step {step} loss {loss.item():.4f} tps {tps:.1f} tokens/s")
            if step >= args.steps:
                break
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
