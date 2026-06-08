<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# wan-profiler

**Precise compute profiling of Wan 1.3B for CPU-native video generation research.**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab) — building the first video generation model that trains and runs entirely on commodity CPUs.

---

## Why This Exists

Video generation models are assumed to require GPUs. But *where* exactly does the compute go? Which operations truly need GPU parallelism, and which could run on a CPU if the architecture were redesigned?

**wan-profiler** answers these questions by instrumenting [Wan 1.3B](https://github.com/Wan-Video/Wan) and producing a precise breakdown of:

- **Time** — wall-clock time per layer, per operation type
- **FLOPs** — floating-point operations distributed across attention, FFN, normalization, and temporal blocks
- **Memory** — peak allocation, per-module memory footprint, allocation patterns
- **Data movement** — memory bandwidth utilization, identifying bottleneck operations

These measurements directly inform the architectural decisions in our CPU-native video generation pipeline.

---

## The Bigger Picture

This is **Phase 1** of a 7-phase research project:

| Phase | Repo | Goal |
|-------|------|------|
| **1** | **wan-profiler** (this repo) | **Profile where Wan 1.3B spends compute** |
| 2 | mamba-video | Replace attention with Mamba/SSM blocks |
| 3 | codec-video-gen | Codec-inspired temporal design (I-frames + P-frame deltas) |
| 4 | bitnet-video | 1-bit quantization (BitNet) |
| 5 | simd-kernels | Portable SIMD execution engine (AVX2 + NEON) |
| 6 | (distributed) | Zeroth-order distributed CPU training |
| 7 | cpu-video-gen | Flagship paper repo with full pipeline |

The thesis: by combining **BitNet quantization**, **Mamba/SSM architecture**, **codec-inspired temporal compression**, and **portable SIMD kernels (AVX2 + NEON)**, video generation becomes feasible on commodity hardware — across both x86 and ARM, no CUDA required.

---

## Hardware Target

CPU-native means **no GPU**, on **commodity hardware**, across **both architectures** (x86 and ARM):

| Role | Machine | Spec |
|------|---------|------|
| **Primary** — development + benchmarking | Apple MacBook Air M4 | ARM64 / NEON, 10 cores, 16–24 GB unified memory, no discrete GPU |
| **Supported** — verified in CI | Commodity x86 with AVX2 | any modern Intel/AMD CPU |
| **Origin** — proof-of-concept (retired) | Intel Pentium Gold 7505 | x86-64 / AVX2, 2C/4T, 16 GB DDR4, no GPU |

The portable SIMD kernels (Phase 5) run natively on x86 (AVX2) and ARM (NEON), so the
"runs on commodity hardware, no CUDA" claim holds **across architectures** — not just one machine.
The project began on the Pentium Gold to prove the thesis on the weakest plausible hardware.

---

## Installation

### Prerequisites

- Python 3.9+
- ~10 GB disk space (for model weights)
- 16 GB RAM minimum

### Setup

```bash
# Clone the repo
git clone https://github.com/IshCPU-VideoGenLab/wan-profiler.git
cd wan-profiler

# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install wan-profiler as editable package
pip install -e .
```

### Downloading Model Weights

```bash
# Option 1: HuggingFace (recommended)
python -c "from wan_profiler.loader import download_model; download_model('wan-1.3b')"

# Option 2: Manual download
# Download from https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
# Place weights in ./model_weights/wan-1.3b/
```

---

## Usage

### Quick Profile (Recommended Start)

```bash
# Basic profiling — measures time per module
python scripts/run_profile.py --model wan-1.3b --output results/

# Memory-constrained mode (for 16GB machines)
python scripts/run_profile.py --model wan-1.3b --output results/ --low-memory

# Full profiling suite (time + FLOPs + memory)
python scripts/run_profile.py --model wan-1.3b --output results/ --full
```

### Visualize Results

```bash
# Generate charts from profiling data
python scripts/visualize.py --input results/profile_results.json --output results/
```

### Python API

```python
from wan_profiler.config import ProfileConfig
from wan_profiler.profiler import profile_model
from wan_profiler.report import generate_report

# Configure
config = ProfileConfig(
    model_name="wan-1.3b",
    low_memory=True,
    profile_flops=True,
    profile_memory=True,
    num_warmup_steps=2,
    num_profile_steps=5,
)

# Run profiling
results = profile_model(config)

# Generate report
generate_report(results, output_dir="results/")
```

---

## Output Format

Results are saved as structured JSON:

```json
{
  "model": "wan-1.3b",
  "hardware": {
    "cpu": "Intel Pentium Gold 7505",
    "cores": 2,
    "ram_gb": 16
  },
  "timestamp": "2026-05-01T14:30:00",
  "summary": {
    "total_time_ms": 45000,
    "total_flops": 1.2e12,
    "peak_memory_mb": 8500
  },
  "per_module": [
    {
      "name": "transformer.blocks.0.attention",
      "type": "attention",
      "time_ms": 1200,
      "time_pct": 15.3,
      "flops": 2.4e10,
      "memory_mb": 450
    }
  ]
}
```

---

## Project Structure

```
wan-profiler/
├── CLAUDE.md              # Claude Code project context
├── README.md              # You are here
├── LICENSE                # MIT License
├── requirements.txt       # Dependencies
├── setup.py               # Package installation
├── lessons.md             # Accumulated learnings
├── tasks/
│   └── todo.md            # Phase 1 task tracker
├── src/
│   └── wan_profiler/
│       ├── __init__.py    # Package init
│       ├── config.py      # Profiling configuration
│       ├── loader.py      # Memory-efficient model loading
│       ├── profiler.py    # Core profiling engine
│       ├── flops.py       # FLOPs counting
│       ├── memory.py      # Memory tracking
│       ├── report.py      # Report generation
│       └── cli.py         # CLI entry point
├── scripts/
│   ├── run_profile.py     # Main profiling script
│   └── visualize.py       # Chart generation
├── tests/
│   ├── test_config.py
│   ├── test_profiler.py
│   └── test_report.py
├── results/               # Output directory
└── docs/
    └── profiling_plan.md  # Methodology documentation
```

---

## Contributing

This is a solo research project by [Ishmael Affum Kwakye](https://github.com/calyxish) at the University of Ghana, Legon. Contributions, suggestions, and discussions are welcome via Issues.

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## Citation

If you use wan-profiler or reference its findings:

```bibtex
@software{kwakye2026wanprofiler,
  author = {Kwakye, Ishmael Affum},
  title = {wan-profiler: Compute Profiling of Wan 1.3B for CPU-Native Video Generation},
  year = {2026},
  url = {https://github.com/IshCPU-VideoGenLab/wan-profiler},
  institution = {University of Ghana, Legon}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Related Work

- [Wan Video](https://github.com/Wan-Video/Wan) — The base model being profiled
- [BitNet](https://arxiv.org/abs/2310.11453) — 1-bit weight quantization
- [Mamba](https://arxiv.org/abs/2312.00752) — Selective state space models
- [TeachDiffusion](https://github.com/calyxish) — Calyx's math teaching video generation project

---

*Phase 1 of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab). The first step toward CPU-native video generation.*
