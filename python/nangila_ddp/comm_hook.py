"""
PyTorch DDP communication hook skeleton using Nangila-like compression logic.

This hook demonstrates:
  - Momentum predictor on gradient buckets
  - Stochastic INT4 quantization (simulated)
  - Closed-loop predictor update (using dequantized residual)

Note: For transport integration, this skeleton currently executes the standard
allreduce on the original bucket (no bandwidth reduction). It logs estimated
compressed bytes and can optionally use FP16 allreduce to reduce traffic by ~2x.
Integrating real INT4 transport requires a custom backend or byte-tensor allgather.
"""
import math
import time
import torch
from typing import Callable, Optional
from torch.distributed import GradBucket


def _absmax_scale(x: torch.Tensor, qmax: int) -> torch.Tensor:
    # Scale so that max(abs(x)) maps to qmax; avoid NaN/Inf
    maxval = torch.nan_to_num(x.abs().max(), nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.where(maxval > 1e-9, maxval / float(qmax), torch.ones_like(maxval))
    return scale


class _MomentumPredictor:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.state: Optional[torch.Tensor] = None

    @torch.no_grad()
    def predict(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.state is None:
            return torch.zeros(shape, device=device, dtype=dtype)
        return self.state

    @torch.no_grad()
    def update(self, observation: torch.Tensor) -> None:
        if self.state is None:
            self.state = observation.clone()
        else:
            self.state.mul_(self.beta).add_(observation, alpha=(1.0 - self.beta))


def register_nangila_hook(
    model: torch.nn.Module,
    *,
    beta: float = 0.9,
    bits: int = 4,
    stochastic: bool = True,
    use_fp16_transport: bool = True,
    seed: int = 12345,
    metrics_path: str | None = None,
    use_momentum: bool = False,
    topk_percent: float = 0.0,
    topo_drop_percent: float = 0.0,
    topo_seed: int = 424242,
) -> None:
    """Register a compression-aware DDP comm hook.

    Args:
        model: a DDP-wrapped module
        beta: momentum coefficient for predictor
        bits: 4 or 8 (quantization resolution for estimation)
        stochastic: enable stochastic rounding estimate (currently simulated)
        use_fp16_transport: if True, cast buckets to float16 before allreduce
    Notes:
        - This hook logs estimated compressed sizes and applies closed-loop predictor
          updates based on dequantized residuals.
        - Actual byte-level INT4 transport is backend-dependent and not implemented here.
    """

    # Optional per-bucket EMA predictor; defaults to stateless zeros for CPU simplicity.
    states: dict[int, torch.Tensor] = {}
    topo_masks: dict[int, torch.Tensor] = {}
    step_counter = {"n": 0}
    metrics = {
        "path": metrics_path,
    }

    def pack_int4(q: torch.Tensor) -> torch.Tensor:
        """Pack int8 values in [-8,7] into uint8 nibbles on-device."""
        n = q.numel()
        if n % 2 != 0:
            q = torch.cat([q, torch.zeros(1, dtype=q.dtype, device=q.device)])
        q = q.view(-1, 2)
        # two's complement nibble
        hi = (q[:, 0] & 0x0F).to(torch.uint8)
        lo = (q[:, 1] & 0x0F).to(torch.uint8)
        byte = (hi << 4) | lo
        return byte.contiguous()

    def unpack_int4(packed: torch.Tensor, length: int) -> torch.Tensor:
        """Unpack uint8 bytes into int8 values in [-8,7] on-device."""
        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        hi = (hi.to(torch.int8) << 4) >> 4
        lo = (lo.to(torch.int8) << 4) >> 4
        out = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
        out[0::2] = hi
        out[1::2] = lo
        return out[:length]

    def pack_mask(mask: torch.Tensor) -> torch.Tensor:
        n = mask.numel()
        pad = (8 - (n % 8)) % 8
        if pad:
            mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype, device=mask.device)])
        mask = mask.view(-1, 8).to(torch.uint8)
        byte = (mask[:, 0] << 0) | (mask[:, 1] << 1) | (mask[:, 2] << 2) | (mask[:, 3] << 3) | \
               (mask[:, 4] << 4) | (mask[:, 5] << 5) | (mask[:, 6] << 6) | (mask[:, 7] << 7)
        return byte.contiguous()

    def unpack_mask(packed: torch.Tensor, length: int) -> torch.Tensor:
        if packed.numel() == 0:
            return torch.zeros(length, dtype=torch.bool, device=packed.device)
        b = packed
        bits = torch.stack([((b >> i) & 1) for i in range(8)], dim=1)  # [B,8]
        bits = bits.reshape(-1)
        return bits[:length].to(torch.bool)

    @torch.no_grad()
    def _hook(state, bucket: GradBucket):
        buf = bucket.buffer()  # flattened tensor
        device = buf.device
        dtype = buf.dtype
        world_size = torch.distributed.get_world_size()

        # Closed-loop predictor predicts the reduced average gradient per bucket
        if use_momentum:
            try:
                bidx = bucket.index()
            except Exception:
                bidx = 0
            prev = states.get(bidx)
            if prev is not None and tuple(prev.shape) == tuple(buf.shape):
                pred = prev
            else:
                pred = torch.zeros(buf.shape, device=device, dtype=dtype)
        else:
            pred = torch.zeros(buf.shape, device=device, dtype=dtype)

        # Residual per rank
        residual = buf - pred

        # Static topology mask (drop a fixed fraction deterministically per bucket)
        nn = residual.numel()
        use_topology = topo_drop_percent > 0.0
        if use_topology:
            try:
                bidx = bucket.index()
            except Exception:
                bidx = 0
            topo = topo_masks.get(bidx)
            if topo is None or int(topo.numel()) != nn:
                # Deterministic mask: keep (1 - topo_drop_percent) fraction
                keep = max(1, int(nn * (1.0 - topo_drop_percent)))
                idx = torch.arange(nn, device=device)
                # shuffle deterministically by hashing seed and bidx
                g = torch.Generator(device=device)
                g.manual_seed((topo_seed ^ (bidx << 16)) & 0xFFFFFFFFFFFFFFFF)
                perm = idx[torch.randperm(nn, generator=g, device=device)]
                keep_idx = perm[:keep]
                topo_mask = torch.zeros(nn, dtype=torch.bool, device=device)
                topo_mask[keep_idx] = True
                topo_masks[bidx] = topo_mask
                topo = topo_mask
            # Apply mask
            residual_topo = residual[topo]
        else:
            topo = None
            residual_topo = residual

        # Quantize residual to INT4 (with optional TopK sparsity)
        qmax = 7 if bits == 4 else 127
        res_mean_abs = torch.mean(torch.abs(residual)).item()
        use_topk = topk_percent > 0.0
        if use_topk:
            n_base = residual_topo.numel()
            k = max(1, int(n_base * topk_percent))
            top_vals, top_idx = torch.topk(residual_topo.abs(), k, sorted=False)
            mask_topk = torch.zeros(n_base, dtype=torch.uint8, device=device)
            mask_topk[top_idx] = 1
            residual_kept = residual_topo[top_idx]
            scale = _absmax_scale(residual_kept, qmax)
            scaled = residual_kept / scale
        else:
            scale = _absmax_scale(residual_topo, qmax)
            scaled = residual_topo / scale
        if stochastic:
            step_counter["n"] += 1
            gen = torch.Generator(device=device)
            if use_momentum:
                try:
                    bidx = bucket.index()
                except Exception:
                    bidx = 0
                seed_mix = seed ^ (bidx << 16) ^ (step_counter["n"] << 32)
            else:
                seed_mix = seed ^ (step_counter["n"] << 32)
            gen.manual_seed(seed_mix & 0xFFFFFFFFFFFFFFFF)
            noise = torch.rand(scaled.shape, device=device, generator=gen, dtype=scaled.dtype) - 0.5
            q = torch.clamp((scaled + noise).floor(), min=-qmax - 1, max=qmax).to(torch.int8)
        else:
            q = torch.clamp(torch.round(scaled), min=-qmax - 1, max=qmax).to(torch.int8)

        if bits == 4:
            packed_vals = pack_int4(q)
        else:
            packed_vals = q.to(torch.uint8)
        if use_topk:
            packed_mask = pack_mask(mask_topk.to(torch.bool))
        else:
            packed_mask = torch.empty(0, dtype=torch.uint8, device=device)

        # Gather variable-length packed tensors and scales
        val_len = torch.tensor([packed_vals.numel()], device=device, dtype=torch.int32)
        mask_len = torch.tensor([packed_mask.numel()], device=device, dtype=torch.int32)
        val_lens = [torch.empty_like(val_len) for _ in range(world_size)]
        mask_lens = [torch.empty_like(mask_len) for _ in range(world_size)]
        torch.distributed.all_gather(val_lens, val_len)
        torch.distributed.all_gather(mask_lens, mask_len)
        val_lens = torch.stack(val_lens).view(-1)
        mask_lens = torch.stack(mask_lens).view(-1)
        max_val_len = int(val_lens.max().item())
        max_mask_len = int(mask_lens.max().item())

        # Pad to max lengths
        if packed_vals.numel() < max_val_len:
            pad = torch.zeros(max_val_len - packed_vals.numel(), dtype=packed_vals.dtype, device=device)
            vals_padded = torch.cat([packed_vals, pad])
        else:
            vals_padded = packed_vals
        if max_mask_len > 0:
            if packed_mask.numel() < max_mask_len:
                pad = torch.zeros(max_mask_len - packed_mask.numel(), dtype=packed_mask.dtype, device=device)
                mask_padded = torch.cat([packed_mask, pad])
            else:
                mask_padded = packed_mask
            gathered_masks = [torch.empty(max_mask_len, dtype=packed_mask.dtype, device=device) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_masks, mask_padded)
        else:
            gathered_masks = [torch.empty(0, dtype=torch.uint8, device=device) for _ in range(world_size)]

        gathered_vals = [torch.empty(max_val_len, dtype=packed_vals.dtype, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_vals, vals_padded)

        scale_vec = torch.tensor([scale.item()], device=device, dtype=torch.float32)
        scales = [torch.empty_like(scale_vec) for _ in range(world_size)]
        torch.distributed.all_gather(scales, scale_vec)
        scales = torch.stack(scales).view(-1)

        # Dequantize, sum residuals
        res_sum = torch.zeros_like(buf, dtype=torch.float32)
        nn = buf.numel()
        for r in range(world_size):
            vlen = int(val_lens[r].item())
            mlen = int(mask_lens[r].item())
            pv = gathered_vals[r][:vlen]
            pm = gathered_masks[r][:mlen]
            if mlen > 0:
                # Reconstruct relative to topology-kept positions
                base_len = nn if (topo is None) else int(topo.sum().item())
                mask_r = unpack_mask(pm, base_len)
                keep_count = int(mask_r.sum().item())
                if bits == 4:
                    qi_all = unpack_int4(pv, keep_count).to(torch.float32)
                else:
                    qi_all = pv.to(torch.int8).to(torch.float32)[:keep_count]
                if topo is not None:
                    # Fill within topology-kept positions
                    res = torch.zeros(nn, dtype=torch.float32, device=device)
                    res_topo = res[topo]
                    res_topo[mask_r] = qi_all * scales[r]
                    res[topo] = res_topo
                else:
                    res = torch.zeros(nn, dtype=torch.float32, device=device)
                    res[mask_r] = qi_all * scales[r]
            else:
                if bits == 4:
                    qi = unpack_int4(pv, nn).to(torch.float32)
                else:
                    qi = pv.to(torch.int8).to(torch.float32)[:nn]
                res = qi * scales[r]
            res_sum.add_(res)

        # Compute reduced average: pred + sum(residuals)/world
        reduced_avg = (pred.to(torch.float32) + (res_sum / float(world_size))).to(dtype)

        # Update predictor with reconstructed average
        # Update EMA state if enabled
        if use_momentum:
            prev = states.get(bidx)
            if prev is not None and tuple(prev.shape) == tuple(reduced_avg.shape):
                prev.mul_(beta).add_(reduced_avg, alpha=(1.0 - beta))
            else:
                states[bidx] = reduced_avg.clone()

        # Log on rank 0
        if torch.distributed.get_rank() == 0:
            raw = buf.numel() * 4
            est_bytes = int(val_len.item() + mask_len.item() + 4)
            ratio = raw / max(est_bytes, 1)
            ts = time.time()
            line = (
                f"ts,{ts:.6f},hook_idx,{step_counter['n']},bucket_elems,{buf.numel()},raw_bytes,{raw},comp_bytes,{est_bytes},"
                f"ratio,{ratio:.6f},res_mean_abs,{res_mean_abs:.6e}\n"
            )
            print(
                f"[nangila-ddp-int4] bucket={buf.numel()} raw={raw/1e6:.3f}MB comp_local={est_bytes/1e6:.3f}MB "
                f"ratio={ratio:.2f}x pred_mode=EMA bits={bits}")
            if metrics["path"]:
                try:
                    with open(metrics["path"], "a") as fh:
                        fh.write(line)
                except Exception:
                    pass

        # Return Future with reduced tensor
        fut = torch.futures.Future()
        fut.set_result(reduced_avg)
        return fut

    if not hasattr(model, "register_comm_hook"):
        raise RuntimeError("Model is not DDP-wrapped; call torch.nn.parallel.DistributedDataParallel first.")

    # state can be None; predictor kept in closure
    model.register_comm_hook(state=None, hook=_hook)
