# Phase 1 Results — Wan 1.3B DiT Compute Profile

**First real measurements from the actual model**, produced by the `wan-profiler`
tool on the Wan 2.1 T2V 1.3B diffusion transformer (DiT), CPU-only, bfloat16.

- Machine: Apple M4 (ARM64), CPU-only, bfloat16
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, transformer subfolder only (1.419 B params)
- Load: ~7 s, ~3.9 GB RAM; one forward (256 tokens): ~5.4 s, peak ~6.3 GB
- Note: only the **DiT** is loaded — the ~11 GB T5 text encoder is replaced by a
  dummy 4096-dim embedding, so profiling fits comfortably in 16 GB.

## Table 1 — where the compute goes (256 latent tokens, 40 layers)

Exclusive ("self") time per category, sums to ~100%:

| Component       | % of forward |
|-----------------|-------------:|
| Attention (self + cross) | **60.1%** |
| FFN (feed-forward)       | **38.5%** |
| Embedding                |  0.8% |
| Other (norm/patch/rope)  |  0.4% |
| Normalization            |  0.2% |
| Activation               |  0.0% |

At a finer split (separate run), attention divides into **~22.6% self-attention**
and **~36.8% cross-attention** (text conditioning). This matters for Phase 2: a
plain Mamba block can replace self-attention cleanly, but cross-attention needs a
text-conditioned design.

**Implications:** attention (~60%) justifies Phase 2 (Mamba SSM); FFN (~38%)
justifies Phase 4 (1-bit quantization).

## Figure 2 — self-attention scales O(n²)

Self-attention time vs. latent token count (`scripts/scaling_profile.py`):

| tokens | total fwd (ms) | self-attn (ms) | µs / token |
|-------:|---------------:|---------------:|-----------:|
| 256 | 4,210 | 1,157 | 4,518 |
| 512 | 10,342 | 3,623 | 7,076 |
| 768 | 18,684 | 7,367 | 9,592 |

**tokens ×3 → self-attention time ×6.4** (linear would be ×3, pure O(n²) ×9).
Per-token cost more than doubles — clearly super-linear, trending quadratic.
This is the bottleneck the O(n) Mamba replacement (Phase 2) targets.

At only 768 tokens, one denoising step already takes **~18.7 s** on CPU —
the headline motivation for the whole optimization stack.

## Reproduce

```bash
pip install -e .
HF_TOKEN=... python scripts/run_profile.py --model wan-1.3b --output results/ --frames 4
HF_TOKEN=... python scripts/scaling_profile.py
```
