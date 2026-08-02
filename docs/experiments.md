# Experiment log

One row per run. If it isn't here, it didn't happen. Config hash comes from the file in
`configs/`; keep it committed so any row can be reproduced.

| # | Date | Phase | Description | Config | Local score | Public LB | Notes |
|---|------|-------|-------------|--------|-------------|-----------|-------|
| 1 | 07-24 | 0 | blob + Hungarian + gap close | thr 0.1 | 0.632 | — | 4 `44b6` only (sparse embryo). |
| 2 | 07-24 | 0 | same, both embryos | thr 0.1 | **0.644** | — | 4 `6bba` + 4 `44b6`, representative. Metric now **verified** vs reference. |
| — | | | #2 threshold sweep | 0.1→0.6 | 0.644→0.359 | — | Monotone; optimum at max recall. Node-budget lever small ([metric.md](metric.md)). |

| 3 | 07-24 | 3 | blob + **ILP** (dist) vs Hungarian | thr 0.3, cap 12µm | edge_J **0.660** vs 0.633 | — | 4 volumes; ILP wins 3/4, +0.027 aggregate, no learned affinity yet. |

| 4 | 07-24 | 3 | **UNet detector** + ILP, thr 0.5 | fold 0, base 16 | edge_J 0.521, **adj 0.441** | — | 3 held-out volumes. **98.7% peak recall** but massively over-detects (34k-60k nodes vs ~25k est), so the node-budget multiplier tanks adj below blob. |

**Key finding (exp 4):** the learned detector has excellent recall but over-detects at low
threshold, and the node-budget multiplier punishes it. This is exactly the metric lever from
docs/metric.md — a *confident* detector wants a HIGH threshold (unlike blob, which loses real
cells). Explains the public POINT_THRESHOLD=0.955. Threshold sweep in progress.

| 5 | 07-24 | 3 | UNet base-16, **local MPS**, 24 vols | fold 0, stride 4/10, 5 ep | val peak-recall **0.920** | — | 23 min, budget-capped. Curve 82.8→91.1→91.1→92.0→91.7. Subset ceiling ~92% (vs 98.7% on full data). |

| 6 | 07-24 | 3 | UNet base-16, local MPS, **60 vols** | fold 0, stride 4/12, 5 ep | val peak-recall **0.964** | — | 48.8 min, budget-guard stopped before ep 5. 94.1→95.4→95.5→96.4→96.3. More data breaks the 24-vol ceiling. |

| 7 | 07-25 | repro | **E005 public pipeline** (detector+edge+ILP+searched divisions) | thr 0.96875, disap 1.575 | — | **LB 0.888** | Reproduced buaaauto E005 on T4. Jump from 0.601 baseline to the public frontier. |

| 8 | 07-25 | 3 | **owned pipeline**: base-16 detector + distance-ILP | 5 held-out val, thr sweep | edge_J **0.546** (best @0.95) | — | Monotonic 0.95:0.546→0.99:0.096. Detector over-detects (29k nodes) & isn't confidence-calibrated; distance linking hangs edge FPs. Far below 0.888 frontier -- detector needs GPU-scale training (base-32 TemporalUNet). |

| 9 | 08-01 | LB | **probe: MIN_TRACK_LEN 6→20** (node-ratio multiplier, §10.5) | probe_len20 | — | **0.886** | Nodes 122.8k→96.1k (−22%), very uneven: 44b6_0b24845f −57%, 44b6_0113de3b −13%. Kernel had run 07-26 but was never submitted. |
| 10 | 08-01 | LB | **probe: pad +0.5·N_pred isolated far-field nodes** | probe_pad | — | **0.868** | Isolates the multiplier term: added nodes carry no edges, so only `1−0.1·ratio` moves. |
| 11 | 08-01 | LB | **IDEA 1: D4 shared-point edge TTA** (pilkwang 0.916 line) | d4edge | — | **0.913** | +0.001 vs base = **within noise** (≥0.005 readable). Our best config but not proven better. 8 inverse-aligned encoder views → transformer per view → logits averaged before sigmoid→ILP. Nodes 119.0k→120.3k (+1.0%, unexplained — expected 0). Predict 11.05 min vs 9.23 base. |
| 12 | 08-01 | LB | **IDEA 2: harmonic detection fusion** (yusuketogashi 0.915 line) | harmonic | — | **0.889** | **−0.023, dead.** Nodes −2.9% bought +0.3% multiplier and cost 2.8% edge_J. Single-lever verified. |
| 13 | 08-02 | local | **Annotation structure over 199 GT graphs** | `scripts/local/annotation_structure.py` | — | n/a | Kills §10.5's ranker. Annotated components: median **21 nodes**, span **22%** of movie, bounding box **71%** of volume, centroid (0.49,0.49,0.51)±0.08. `min_len=20` keeps only 54.7% of annotated tracks / 82.8% of annotated nodes → explains exp 9's edge_J loss. Annotation ≈ random 3.6% sample. |
| 14 | 08-02 | LB | **Division probe: SAFE_DIV rate caps ×5** (geometry fixed) | divprobe | — | _running_ | Off the 0.913 d4edge best. Measures the only term worth up to 0.1 that has never been read on real test. |

### Decomposition of our 0.913 (from exp 10, the pad probe)

`probe_pad` adds edgeless far-field nodes, so only the multiplier can move. Two scores ⇒ two
equations: `edge_J·k = 0.88` and `1.1·edge_J + 0.1·div_J = 1.00`. Under every assumption about the
unknown division term, **k = N_pred/est_n_nodes ≈ 0.97–1.02 and the multiplier ≈ 1.00** — the node
budget is full, not 75% full as §10.5 assumed. `edge_J ≈ 0.909` carries ~99% of the score.

**Reference points:** public LB ~0.897, strong rule-based ~0.857. Blob baseline 0.644. Learned
detector recall is there; the work is converting it to score via threshold + node-count control.

## Runtime — two different numbers, do not conflate them (measured 08-01)

**Commit run (what we watch): ~21 min end-to-end.** Setup 4.5 min, prediction 9.2 min, post-proc +
nbconvert 7.5 min. But this runs over the **4-volume dev sample** in `test/`, not the real test set.

**Scoring rerun (what sets the LB): ≥2h, not yet measured.** At submission Kaggle swaps in the
hidden test set (`docs/data.md`) and reruns the whole notebook. Four submissions on 08-01 were still
`pending` at 122 min / 91 min with no error. If the hidden set holds ~10× the dev sample's volumes,
prediction alone scales from 9 min to ~1.5h, which would make STRATEGY §10.2's "~6h/run" a
description of *this* number and broadly right.

⚠ An earlier revision of this section claimed the ~21 min figure superseded §10.2 and that compute
is not the throughput limit. That was wrong — it compared a dev-sample run against a hidden-test
rerun. Iteration on configs is cheap; **getting a score is not**. Measure the true rerun latency
before planning around it.

**Local MPS timing:** ~1.6 s/step warm (base-16, 64³), ~3.5 min/epoch on 24 vols. `--calibrate`
over-estimates ~2× (measures cold steps before MPS kernel warmup) — trust the first real
`>>> ETA` line, not the calibration, for the true rate.

## CV ↔ LB tracking

Plot/track these two columns together. A divergence means the split is leaking or the LB
sequence differs from ours — investigate before trusting either number.

## Golden holdout

One sequence is reserved and **never** used for any tuning decision until Week 9. Record here
when it is chosen, and every time it is touched:

- Chosen: _TBD_
- Touched: _never_
