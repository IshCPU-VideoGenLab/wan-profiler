# Profiling Methodology

## Objective

Produce a precise breakdown of where Wan 1.3B spends its compute during a
single forward pass. This data informs every architectural decision in
Phases 2–5 of the CPU-native video generation project.

## What We Measure

### 1. Wall-Clock Time Per Module
We use PyTorch forward pre-hooks and post-hooks to measure elapsed time for
each named module. This captures the actual time spent, including memory
allocation overhead, cache effects, and OS scheduling.

**Method:** `time.perf_counter()` in pre-hook and post-hook. We subtract
start from end to get elapsed time in milliseconds.

**Why not `torch.profiler`?** The built-in profiler is CUDA-oriented and
adds significant overhead on CPU. Our hook-based approach is lighter and
gives us exactly the data we need.

### 2. FLOPs Per Module (Analytical)
We estimate FLOPs analytically based on layer type and tensor dimensions,
not by counting actual operations. This gives us the theoretical compute
cost independent of hardware-specific execution patterns.

**Counted operations:**
- Linear layers: 2 * in_features * out_features (multiply-accumulate)
- Attention: QKV projection + score computation + softmax + output projection
- Layer normalization: mean + variance + normalize + affine
- Activations: varies by type (ReLU=1, GELU≈8 ops per element)

### 3. Memory Per Module
We track process-level RSS (Resident Set Size) before and after each module's
forward pass. The delta gives an approximation of memory allocated by that module.

**Limitation:** On CPU, PyTorch doesn't track allocations as precisely as
CUDA's `memory_allocated()`. Process RSS includes all allocations, not just
PyTorch tensors. This is an approximation.

## Experimental Setup

### Hardware
- Intel Pentium Gold 7505 (2 cores, 4 threads, 3.5 GHz)
- 16 GB DDR4 3200 MHz (single channel)
- No discrete GPU

### Software
- Python 3.9
- PyTorch (CPU-only build)
- Model: Wan 1.3B (float16)

### Input Configuration
- Batch size: 1
- Frames: 8
- Resolution: 256×256
- Latent space: 4 channels, 32×32 spatial (after 8× VAE compression)
- Timestep: 500 (middle of diffusion schedule)

### Protocol
1. Load model in float16 with `low_cpu_mem_usage=True`
2. Create dummy input tensors
3. Run N warmup iterations (default: 2) — discard these
4. Run M profiling iterations (default: 5) — collect measurements
5. Report mean and standard deviation across M iterations

### Controls
- System is idle during profiling (no other CPU-heavy processes)
- `gc.collect()` called between profiling iterations
- `torch.no_grad()` context for all measurements
- Same random seed for dummy inputs across runs

## Output Format

Results are saved as structured JSON with three levels of detail:

1. **Summary:** Total time, total FLOPs, peak memory
2. **Category breakdown:** Time aggregated by operation type (attention, FFN, etc.)
3. **Per-module breakdown:** Individual module metrics sorted by time

## What We're Looking For

The profiling data answers these specific questions for Phase 2+:

| Question | Informs |
|----------|---------|
| What % of time is spent in attention? | Phase 2 (Mamba replacement priority) |
| What % of time is spent in FFN layers? | Phase 4 (BitNet quantization targets) |
| How much temporal redundancy exists? | Phase 3 (codec design opportunity) |
| What's the memory bottleneck? | Phase 5 (AVX2 kernel memory layout) |
| What operations are trivially CPU-friendly? | What we can keep as-is |

## Reproducibility

All profiling code, configuration, and raw results are committed to this
repository. To reproduce:

```bash
git clone https://github.com/IshCPU-VideoGenLab/wan-profiler
cd wan-profiler
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_profile.py --model wan-1.3b --output results/ --full
```
