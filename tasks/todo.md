# Phase 1 — wan-profiler Task Roadmap

> Claude Code: check this file at the start of every session.
> Mark tasks `[x]` when complete. Add subtasks as needed.

---

## Milestone 1: Environment & Model Loading
- [ ] Verify Python 3.9 + venv setup works on target machine
- [ ] Install PyTorch CPU-only build (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- [ ] Download Wan 1.3B weights from HuggingFace
- [ ] Implement `loader.py` — memory-efficient model loading
- [ ] Test: model loads without OOM on 16GB machine
- [ ] Measure baseline memory footprint of loaded model

## Milestone 2: Time Profiling
- [ ] Implement `profiler.py` — hook-based per-module timing
- [ ] Register forward hooks on every named module
- [ ] Run dummy forward pass and collect per-layer wall-clock times
- [ ] Handle warmup runs (discard first N iterations)
- [ ] Aggregate timing data across multiple runs for stability
- [ ] Identify top-10 most time-consuming modules

## Milestone 3: FLOPs Counting
- [ ] Implement `flops.py` — FLOPs estimation per module
- [ ] Count FLOPs for: attention, linear layers, normalization, activations
- [ ] Validate FLOPs count against known estimates for transformer models
- [ ] Produce FLOPs breakdown by operation type (attention vs FFN vs other)

## Milestone 4: Memory Profiling
- [ ] Implement `memory.py` — per-module memory tracking
- [ ] Track peak memory during forward pass
- [ ] Track memory allocated per module (activations + parameters)
- [ ] Identify memory bottleneck layers
- [ ] Test memory profiling doesn't itself cause OOM

## Milestone 5: Report Generation
- [ ] Implement `report.py` — structured JSON output
- [ ] Include hardware metadata in report
- [ ] Include per-module breakdown (time, FLOPs, memory)
- [ ] Generate human-readable summary table to stdout
- [ ] Implement `visualize.py` — bar charts for time/FLOPs/memory distribution

## Milestone 6: Analysis & Documentation
- [ ] Run full profiling suite on target hardware
- [ ] Identify which modules are attention-heavy (targets for Mamba replacement)
- [ ] Identify which modules have redundant temporal computation (targets for codec design)
- [ ] Identify which operations can trivially run as 1-bit (targets for BitNet)
- [ ] Write `docs/profiling_plan.md` with methodology
- [ ] Update README with actual results
- [ ] Write findings summary for Phase 2 handoff

## Milestone 7: Testing & Polish
- [ ] Unit tests for config, profiler, report modules
- [ ] Integration test: full pipeline from load → profile → report
- [ ] CI-ready test suite (pytest passes cleanly)
- [ ] Clean up code, docstrings, type hints
- [ ] Final commit and tag v0.1.0

---

## Notes
- Priority: Milestones 1–2 are the critical path. Get timing data first.
- FLOPs and memory profiling are valuable but secondary.
- The report format matters — these numbers go into the paper.
