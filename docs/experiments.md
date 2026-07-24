# Experiment log

One row per run. If it isn't here, it didn't happen. Config hash comes from the file in
`configs/`; keep it committed so any row can be reproduced.

| # | Date | Phase | Description | Config | CV (holdout seq) | Public LB | Runtime | Notes |
|---|------|-------|-------------|--------|------------------|-----------|---------|-------|
| — | — | 0 | *(pending data access — Kaggle token 401s)* | — | — | — | — | — |

## CV ↔ LB tracking

Plot/track these two columns together. A divergence means the split is leaking or the LB
sequence differs from ours — investigate before trusting either number.

## Golden holdout

One sequence is reserved and **never** used for any tuning decision until Week 9. Record here
when it is chosen, and every time it is touched:

- Chosen: _TBD_
- Touched: _never_
