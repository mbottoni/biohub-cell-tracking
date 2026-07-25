---
name: competition-objective
description: "The Biohub cell-tracking Kaggle competition goal, constraints, and current strategy"
metadata: 
  node_type: memory
  type: project
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T17:50:43.921Z
---

**Kaggle "Biohub - Cell Tracking During Development"** ($60k, deadline 2026-09-29,
kernels-only). Goal: **surpass leaderboard 0.932** on the custom tracking metric.

Task = 3D point detection + graph linking (NOT segmentation). Submission is a directed-forest
point graph CSV. Data: 199 train + 4 test volumes, all (100,64,256,256) uint16, two embryos
(6bba = ~85% of edges, 44b6). GT is SPARSELY annotated (0.4-10% of the ~20-30k real cells per
volume are labeled), but the metric expects you to predict ~all cells. See [[metric-and-contamination]].

Current strategy (chosen path): a **fully-owned pipeline** (our detector + linker + ILP) scored
locally in seconds on a clean held-out split, submitting only when it clears the bar. This
sidesteps the contamination that inflates pretrained-pipeline local scores. The public E005
notebook scores LB 0.888; leaders at 0.932 are beyond it. Believed gaps to 0.932: temporal
detection context, a LEARNED edge/link model (highest ROI — dominant metric term), and explicit
division handling (0.1-weighted term). Detector work in progress — see [[detector-status]].
