# CLAUDE.md — wan-profiler

> This file is read by Claude Code at the start of every session.
> It provides full context so you never have to re-explain the project.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** wan-profiler
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon
- **Phase:** 1 of 7

---

## What This Project Is

This is the **profiling and analysis** phase of a larger research effort to build the
first-ever CPU-native video generation model. Before we redesign anything, we need to
know exactly where Wan 1.3B spends its compute.

**wan-profiler** instruments the Wan 1.3B video generation model and produces precise
measurements of:

- Time spent in each layer/module (attention, FFN, normalization, etc.)
- Memory allocation patterns per operation
- FLOPs distribution across the forward pass
- Data movement bottlenecks (memory bandwidth usage)
- Which operations are GPU-dependent vs. CPU-feasible

The output of this repo directly informs Phase 2 (architecture surgery).

---

## The Bigger Picture — Why This Exists

The parent project (`cpu-video-gen`) aims to build a video generation model that:

1. Uses **BitNet 1-bit quantization** (XNOR + popcount instead of float matmul)
2. Uses **Mamba/SSM architecture** (linear O(n), no attention)
3. Uses **codec-inspired temporal design** (I-frames + P-frame deltas, like H.264)
4. Runs on **portable SIMD kernels** — AVX2 (x86) + NEON (ARM) + scalar fallback, no CUDA anywhere

Each pillar eliminates a GPU dependency. Combined, they make CPU-native video gen viable.

---

## Hardware — READ THIS CAREFULLY

- **Primary (development + benchmarking):** MacBook Air M4 — ARM64 / NEON, no GPU.
- **Supported, CI-verified:** commodity x86 with AVX2 (any modern Intel/AMD CPU).
- **Origin, proof-of-concept (retired):** Intel Pentium Gold 7505 — x86-64 / AVX2, 2C/4T, 16 GB.

CPU-native, no GPU, across **both** architectures. We develop on the M4, but **all code must stay
within the commodity-hardware design budget** — assume **2–4 cores, 16 GB RAM (~12 GB usable),
no GPU**, and it must run on x86 (AVX2) **and** ARM (NEON). Never write code that assumes a specific
ISA, many cores, large RAM, or a GPU. The Pentium Gold proved the weakest-hardware case; the paper
reports CPU-native results on both architectures. Python 3.9, venv (no conda).

### What This Means For Code:

- **Never assume GPU/CUDA is available.** All profiling must work in CPU-only mode.
- **Memory is tight.** 16 GB total. Model loading must be memory-efficient.
  Use `torch.float16` or `torch.bfloat16` where possible. Consider layer-by-layer loading.
- **2 cores only.** Don't write code that assumes parallel speedup from many cores.
- **No heavy dependencies.** Keep the dependency tree minimal and pip-installable.
- **Disk is limited.** Results should be compact (JSON/CSV, not giant binary dumps).
- **Wan 1.3B is the target.** Not 2.2, not 5B. 1.3B fits in 16 GB RAM with care.

---

## Code Conventions

- **Python 3.9** — no walrus operators in complex expressions, no `match` statements
- **Type hints** on all function signatures
- **Docstrings** on all public functions (Google style)
- **No classes unless necessary** — prefer functions and dataclasses
- **Logging** via `logging` module, never `print()` for anything that runs in production
- **Config** via dataclasses or simple dicts, not YAML/TOML parsers (keep deps minimal)
- **Tests** in `tests/` using `pytest`
- **Results** output as JSON (machine-readable) with optional human-readable summary

---

## File Structure

```
wan-profiler/
├── CLAUDE.md              ← You are here
├── README.md              ← Public-facing project description
├── LICENSE                 ← MIT License
├── requirements.txt       ← Minimal dependencies
├── setup.py               ← Package setup
├── lessons.md             ← Mistakes & learnings (grows over time)
├── tasks/
│   └── todo.md            ← Phase 1 task roadmap
├── .claude/
│   ├── settings.json      ← Claude Code guardrails
│   ├── commands/
│   │   ├── review.md      ← /project:review command
│   │   └── progress.md    ← /project:progress command
│   └── rules/
│       └── python.md      ← Python style enforcement
├── src/
│   └── wan_profiler/
│       ├── __init__.py
│       ├── config.py       ← Profiling configuration
│       ├── loader.py       ← Memory-efficient model loading
│       ├── profiler.py     ← Core profiling engine
│       ├── flops.py        ← FLOPs counting per module
│       ├── memory.py       ← Memory tracking utilities
│       ├── report.py       ← Report generation (JSON + summary)
│       └── cli.py          ← Command-line entry point
├── scripts/
│   ├── run_profile.py      ← Main profiling script
│   └── visualize.py        ← Generate charts from results
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_profiler.py
│   └── test_report.py
├── results/                ← Profiling output (gitignored except .gitkeep)
│   └── .gitkeep
└── docs/
    └── profiling_plan.md   ← Detailed profiling methodology
```

---

## Key Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run profiling (basic)
python scripts/run_profile.py --model wan-1.3b --output results/

# Run profiling (memory-constrained mode)
python scripts/run_profile.py --model wan-1.3b --output results/ --low-memory

# Run tests
pytest tests/ -v

# Generate visualization from results
python scripts/visualize.py --input results/profile_results.json --output results/
```

---

## Task Management

Check `tasks/todo.md` before starting any work session.
Mark tasks complete as you go. This is the single source of truth for project progress.

---

## Lessons Learned

Check `lessons.md` before writing new code. If a mistake was made before,
there's a rule there to prevent it from happening again.

---

## Research Context

This profiling data will appear in the final paper as Table 1 or Figure 1 —
the "where does the compute actually go?" breakdown that motivates every
architectural decision in Phases 2–5.

The numbers we produce here are not throwaway. They are evidence.
