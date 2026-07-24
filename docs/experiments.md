# Experiment log

One row per run. If it isn't here, it didn't happen. Config hash comes from the file in
`configs/`; keep it committed so any row can be reproduced.

| # | Date | Phase | Description | Config | Local score | Public LB | Notes |
|---|------|-------|-------------|--------|-------------|-----------|-------|
| 1 | 07-24 | 0 | blob + Hungarian + gap close | thr 0.1 | 0.632 | — | 4 `44b6` only (sparse embryo). |
| 2 | 07-24 | 0 | same, both embryos | thr 0.1 | **0.644** | — | 4 `6bba` + 4 `44b6`, representative. Metric now **verified** vs reference. |
| — | | | #2 threshold sweep | 0.1→0.6 | 0.644→0.359 | — | Monotone; optimum at max recall. Node-budget lever small ([metric.md](metric.md)). |

| 3 | 07-24 | 3 | blob + **ILP** (dist) vs Hungarian | thr 0.3, cap 12µm | edge_J **0.660** vs 0.633 | — | 4 volumes; ILP wins 3/4, +0.027 aggregate, no learned affinity yet. |

**Reference points:** public LB ~0.897, strong rule-based ~0.857. Blob baseline 0.644 is the
floor. ILP linking (global, handles divisions/appearance) beats greedy Hungarian and is the
new default linker — the win grows once learned edge affinities replace distance.

## CV ↔ LB tracking

Plot/track these two columns together. A divergence means the split is leaking or the LB
sequence differs from ours — investigate before trusting either number.

## Golden holdout

One sequence is reserved and **never** used for any tuning decision until Week 9. Record here
when it is chosen, and every time it is touched:

- Chosen: _TBD_
- Touched: _never_
