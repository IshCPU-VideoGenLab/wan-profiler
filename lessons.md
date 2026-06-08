# Lessons Learned

> This file accumulates rules and corrections discovered during development.
> Claude Code checks this before writing new code to avoid repeating mistakes.
> Each entry includes the date, what went wrong, and the rule to follow.

---

<!-- Example format:
## 2026-05-15 — Memory overflow when loading full model
**What happened:** Loaded all Wan 1.3B weights into RAM at once. OOM on 16GB machine.
**Rule:** Always use `loader.load_model_layerwise()` on machines with ≤16GB RAM.
-->
