# /project:progress

Report the current progress of the wan-profiler project.

## Steps

1. Read `tasks/todo.md` and count completed vs total tasks per milestone.
2. Check which source files in `src/wan_profiler/` have substantive code (not just stubs).
3. Check if any results exist in `results/`.
4. Check `lessons.md` for number of entries.
5. Report:

```
=== wan-profiler Progress Report ===

Milestone 1 (Environment & Loading):  X/Y complete
Milestone 2 (Time Profiling):         X/Y complete
Milestone 3 (FLOPs Counting):         X/Y complete
Milestone 4 (Memory Profiling):       X/Y complete
Milestone 5 (Report Generation):      X/Y complete
Milestone 6 (Analysis & Docs):        X/Y complete
Milestone 7 (Testing & Polish):       X/Y complete

Overall: X/Y tasks complete (Z%)

Source files with code: [list]
Results files: [list or "none yet"]
Lessons learned: N entries

Next recommended task: [first unchecked item]
```
