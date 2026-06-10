#!/usr/bin/env python
"""Scaling profile: how self-attention time grows with token count.

Demonstrates the O(n^2) self-attention cost that motivates the Phase-2 Mamba
(O(n)) replacement. Runs the real Wan DiT at increasing latent token counts
and reports self-attention time per forward.

Memory note: token count is frames * (H/2) * (W/2); large counts can OOM on a
16 GB machine. Defaults stay <= 768 tokens. Loads only the DiT (bfloat16).

Usage:
    HF_TOKEN=... python scripts/scaling_profile.py
"""
import gc
import time
from collections import defaultdict

import psutil
import torch
from diffusers import WanTransformer3DModel

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FRAME_COUNTS = [1, 2, 3]  # latent frames -> 256, 512, 768 tokens at 32x32 latent


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).eval()

    attn_t = defaultdict(float)
    starts = {}
    for name, mod in model.named_modules():
        if name.endswith(".attn1"):  # self-attention only (the O(n^2) part)
            mod.register_forward_pre_hook(
                lambda m, i, n=name: starts.__setitem__(n, time.perf_counter()))
            mod.register_forward_hook(
                lambda m, i, o, n=name: attn_t.__setitem__(
                    "self", attn_t["self"] + time.perf_counter() - starts[n]))

    txt = torch.randn(1, 226, 4096, dtype=torch.bfloat16)
    ts = torch.tensor([500])
    print(f"{'tokens':>7}{'total ms':>10}{'self-attn ms':>14}{'attn/tok (us)':>15}")
    rows = []
    for frames in FRAME_COUNTS:
        tokens = frames * 16 * 16
        hs = torch.randn(1, 16, frames, 32, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            model(hs, ts, txt, return_dict=False)  # warmup
            attn_t["self"] = 0.0
            t0 = time.perf_counter()
            model(hs, ts, txt, return_dict=False)
            total = (time.perf_counter() - t0) * 1000
        a = attn_t["self"] * 1000
        print(f"{tokens:>7}{total:>10.0f}{a:>14.0f}{a / tokens * 1000:>15.1f}")
        rows.append((tokens, total, a))
        del hs
        gc.collect()

    if len(rows) >= 2:
        (t1, _, a1), (t2, _, a2) = rows[0], rows[-1]
        rt = t2 / t1
        print(f"\ntokens x{rt:.1f} -> self-attn time x{a2 / a1:.1f}  "
              f"(linear=x{rt:.1f}, O(n^2)=x{rt ** 2:.1f})")


if __name__ == "__main__":
    main()
