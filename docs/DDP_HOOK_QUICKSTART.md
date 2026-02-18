## DDP Hook Quickstart (Mini-Nangila)

This guide shows how to register the compression-aware DDP communication hook for PyTorch.

### Install

- Ensure PyTorch Distributed is available and you can launch multi-GPU runs.
- The hook lives under `python/nangila_ddp/comm_hook.py`.

### Usage

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila_ddp import register_nangila_hook

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

register_nangila_hook(model, beta=0.9, bits=4, stochastic=True, use_fp16_transport=True)

# Proceed with training; hook logs estimated compression per bucket
```

Run with torchrun:

```bash
torchrun --nproc_per_node=2 python -m nangila_ddp.examples.minigpt_train
```

Notes:
- Current hook uses FP16 allreduce to achieve ~2× traffic reduction and logs estimated INT4 compression.
- For true INT4 transport, integrate a byte-tensor allreduce or custom backend.
