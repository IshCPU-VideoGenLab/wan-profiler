# /project:review

Perform a full code review of the wan-profiler project.

## Steps

1. Read `CLAUDE.md` to refresh project context and constraints.
2. Read `lessons.md` to check for known issues.
3. Review all Python files in `src/wan_profiler/` for:
   - Type hint completeness on all function signatures
   - Google-style docstrings on all public functions
   - No use of `print()` — must use `logging` module
   - Python 3.9 compatibility (no `match` statements, no complex walrus operators)
   - Memory safety (no operations that could OOM on 16GB)
   - CPU-only compatibility (no CUDA assumptions)
   - Line length ≤ 100 characters
4. Review `scripts/` for consistency with `src/` conventions.
5. Run `pytest tests/ -v` and report results.
6. Summarize findings as:
   - **Critical** — will break on target hardware
   - **Important** — violates project conventions
   - **Suggestions** — improvements for clarity or performance
