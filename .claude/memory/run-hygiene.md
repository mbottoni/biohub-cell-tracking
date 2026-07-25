---
name: run-hygiene
description: "User's execution rules — estimate time before runs, cap at 1h, verbose scripts, commit changes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T17:51:26.928Z
---

Execution rules the user set for this project:
- **Estimate time-to-complete before every run; never trigger a run expected to exceed ~1 hour.**
  Long training goes to a cloud GPU (Kaggle T4 kernel), not local MPS. Local MPS scripts have a
  --calibrate mode + --max-minutes budget guard for exactly this.
- **Keep scripts EXTREMELY verbose** — per-step heartbeats (loss, frames/s, it/s, mem, epoch-ETA),
  explicit ETA lines, and monitor many things so progress is visible at a glance.
- **Commit changes** as work progresses (git), with clear messages explaining the why.

**Why**: the user actively monitors runs and wants no surprise long/expensive jobs, plus a clean
history to reason about experiments.

**How to apply**: before any train/eval, print the estimated duration and confirm it's under
budget; back long jobs with a time guard or run them on the cloud. Prefer background tasks with a
completion poll for anything slow. Commit after each meaningful change. Keep local vs cloud code
cleanly separated (library in src/ctrack, local runs in scripts/local, Kaggle kernels in kernels/).
See [[detector-status]].
