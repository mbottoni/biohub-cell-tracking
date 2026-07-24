# Experiment log

One row per run. If it isn't here, it didn't happen. Config hash comes from the file in
`configs/`; keep it committed so any row can be reproduced.

| # | Date | Phase | Description | Config | Local score | Public LB | Notes |
|---|------|-------|-------------|--------|-------------|-----------|-------|
| 1 | 07-24 | 0 | blob detect + Hungarian + gap close | thr 0.1, disp 10µm | **0.632** | — | 4 `44b6` volumes only (sparse embryo). Local metric unverified vs LB. |
| — | | | same, threshold sweep | thr 0.1→0.7 | 0.632→0.348 | — | Monotone: optimum at max recall on this detector. Node-budget lever is small (see [metric.md](metric.md)). |

**Caveat on #1:** scored on four `44b6` volumes — the sparse embryo (0.99% density, 15% of all
edges). The real score is dominated by `6bba`. Re-run across both before drawing conclusions.

## CV ↔ LB tracking

Plot/track these two columns together. A divergence means the split is leaking or the LB
sequence differs from ours — investigate before trusting either number.

## Golden holdout

One sequence is reserved and **never** used for any tuning decision until Week 9. Record here
when it is chosen, and every time it is touched:

- Chosen: _TBD_
- Touched: _never_
