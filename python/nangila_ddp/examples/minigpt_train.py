import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila_ddp import register_nangila_hook


class TinyGPT(nn.Module):
    def __init__(self, d_model=256, nhead=4, nlayers=4, vocab=5000):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.lm = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.emb(x)
        h = self.enc(h)
        return self.lm(h)


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = TinyGPT().cuda()
    model = DDP(model, device_ids=[local_rank])
    register_nangila_hook(model, beta=0.9, bits=4, stochastic=True, use_fp16_transport=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    vocab = 5000
    for step in range(100):
        # random toy batch
        x = torch.randint(0, vocab, (8, 128), device="cuda")
        y = torch.randint(0, vocab, (8, 128), device="cuda")
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

