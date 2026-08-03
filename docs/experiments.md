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
| 14 | 08-02 | — | **Division probe: SAFE_DIV rate caps ×5** (geometry fixed) | divprobe | — | **not submitted** | **No-op — caps were never binding.** candidates 355→355, added 323→**350**, `safe_division_skipped_cap`=0 in *both* runs. A +27-division delta on 120k nodes is unreadable. Not worth a submission; kept as the record of why. |
| 15 | 08-02 | local | **GT division census** (`docs/gt_stats.csv`) | — | — | n/a | The 4 visible test volumes are byte-identical to train (docs/data.md) so their GT is known: **3 annotated divisions total** (0/0/0/3). Across all 199 volumes: median **0** per volume, mean 0.8, max 5, **112 volumes have zero**. We insert **350**. ⇒ the division lever points at *removal*. |
| 16 | 08-02 | LB | **Divisions OFF** (`OUTPUT_SAFE_DIVISIONS=0`) | divoff | — | _queued_ | Each inserted division also adds a second-child edge = plain FP against edge_J. Tests what 350 spurious divisions cost. |
| 17 | 08-02 | LB | **Swap primary/secondary checkpoints** | swapw | — | _queued_ | Reinterpreted after §11.5: both are 400-epoch models, so this measures **seed variance**, not a training-quality gap. **Ran; the seeds disagree per-volume far more than in total** — nodes 120,276→120,617 (+0.28%) but per-volume Δ = −658, **+2,252 (+12%)**, −296, −957. Totals cancel; the models genuinely see different cells. |
| 18 | 08-02 | LB | **SECONDARY_EDGE_WEIGHT 0.15 → 0.50** | secw | — | _running_ | Two equal-quality 400ep seeds deserve an even blend; 0.85/0.15 under-uses the second model. Row 17's per-volume divergence makes this the **stronger of the two ensembling bets** — it perturbs edge logits substantially, where swapw's totals barely move. |

| 19 | 08-02 | smoke | **Training smoke run 1** | trainsmoke v1 | — | **FAIL** | `weights/` is a symlink into read-only `/kaggle/input` ⇒ `OSError [Errno 30]`, no checkpoint. Both arms died in ~8 s. Confirmed the train mount: **199 zarr + 199 geff, all usable**. |
| 20 | 08-02 | smoke | **Training smoke run 2** (symlink fixed) | trainsmoke v2 | — | **PASS** | T4, batch 4, 8 iters. **window2 3.90 s/iter → 5.34 h/epoch**; **window3 11.18 s/iter → 15.13 h/epoch**. Both checkpoints (8.4 MB) reload via inference `load_model`. `max_nodes=11`. ⇒ T4 is ~8× slower than the author's hardware; from-scratch is infeasible (534 GPU-h @100ep vs ~240 h quota). Caveat: 8 iters includes warmup; validation returned 0/0 (untrained model) so the early-stop metric is **untested**. |

**Timing reference (measured 08-02).** Kernel setup ~8.3 min · load 199 volumes ~30 min ·
re-smoke ~15 min · inference run ~21 min · LB rerun ~4–6 h · **training change → LB number ~16 h**.
See STRATEGY §12.5 for the full table and §12.4 for the speedup plan with per-idea verification cost.

| 19b | 08-02 | LB | **secw: SECONDARY_EDGE_WEIGHT 0.15 → 0.50** | secw | — | **0.913** | **Flat vs base.** Even-weight blending of the two 400ep seeds changes nothing, despite 12%/volume node divergence. Retires the ensemble-weight lever. |
| 21 | 08-02 | train | **Edge-head fine-tune, 10,000 steps** | edgeft | holdout recall 0.961 | **not submitted** | 194 min, 0.69 s/iter. **Epoch 0 never beaten**: test_loss flat at 0.0002 for all 20 epochs, recall drifts 0.9619→0.9608. The association head was already at a local optimum. Not submitted — functionally identical to the model already scored 0.913. |
| 22 | 08-03 | LB | **DET_THRESHOLD 0.96875 → 0.96** | thr096 | — | **0.914** | nodes +1.1%; per-volume k 0.991/0.564/0.977/1.003 → 0.995/0.589/0.986/1.009. Movement concentrated on the under-detected volume, as §13.5 predicted. **+0.001 — our first score above the 0.913 wall, but at the noise floor.** |
| 23 | 08-03 | LB | **DET_THRESHOLD → 0.95** | thr095 | — | **0.914** | nodes +1.9%; k → 0.997/0.607/0.989/1.012. **Same 0.914 as thr096 despite ~1.7× the added nodes** — the curve is flat between 0.96 and 0.95. See §13.6. |
| 24 | 08-03 | — | **OR-fusion (dual of harmonic)** | orfuse | — | **BUG, not submitted** | Must add nodes; instead nodes −0.1% and volume 3 **lost 23%** (k 0.977→0.750). Cause: harmonic-on-complement saturates to p≈1 when either seed is confident, and the clamp at 1−1e−6 **flattens the logit field**. Peaks come from local-max pooling, so a plateau yields *no* maxima — detections destroyed exactly where confidence is highest. Needs a formulation that raises p without saturating. |
| 25 | 08-03 | local | **Division geometry over 199 GT graphs** | `scripts/local/division_geometry.py` | — | n/a | Corrects §14. 151 annotated divisions. Real sister separation median **10.57 µm**; our `SAFE_DIV_SISTER_MAX_UM` is **8.5** — below the median of the truth. Exact three-gate admission of real divisions = **0.265**; at (12/12/14.4) it is **0.894**. Ranking `parent_dist + 0.15·sister` sorts **ascending**, so it prefers the least division-like candidates (real divisions score median 5.74, ordinary motion 1.82). ⇒ we are structurally blind to ~3/4 of divisions. Revised EV **+0.01..+0.03** (was +0.02..+0.09). |
| 26 | 08-03 | local | **divgeom local test** (patched fn from the built notebook, 199 GT graphs) | `scripts/local/test_divgeom.py` | recall **0.126 → 0.881** | n/a | Delete one child edge per real division (the *farther* daughter) and ask if the heuristic restores it. 0.881 **independently confirms** the 0.894 predicted from gate geometry in §15.2. ⚠ The **re-rank is NOT exercised** — wide-gates+old-rank and wide-gates+new-rank tie (93/136, 125/209 vs 124/209); GT is too sparse for candidates to compete per frame. |
| 27 | 08-03 | smoke | **divgeom v1 on Kaggle** | divgeom v1 | — | **FAIL** | `KeyError: safe_division_frame_cap_hit` after ~17 min. `stats` in `filter_output_graph` is a **plain pre-seeded dict**, not a defaultdict; a new counter must be declared in that literal. Our fixture used `defaultdict(int)` and masked it — the test was more permissive than the caller. Also: Kaggle slugs the **title**, not the metadata `id`, so v1 landed at a different kernel ref. |
| 28 | 08-03 | LB | **divgeom v2: division geometry re-centred** | divgeom | see row 26 | _submitted 23:24 UTC, ref 55223260, pending_ | COMPLETE, 11.25 predict-min. candidates **355→13,495** (38×), inserted **350→119**, `safe_division_skipped_cap` 276, **`safe_division_frame_cap_hit` 80 ⇒ the frame cap DOES bind** (kills §11.6's "caps never bind"). Nodes 120,276→120,118 (**−0.13%**) so the node budget is untouched and any LB move is attributable to divisions. Submission verified: 235,932 rows, exactly **119 forks**. Ranking now selects 119/13,495 (0.88%) and is the untested component. Read: **flat ⇒ ranking failed; +0.02..+0.04 ⇒ it worked.** |

**edge_J ≈ recall² now has two independent confirmations (08-03).**
(1) Harmonic fusion cut nodes 2.9%; the model predicted −0.025, measured −0.023.
(2) `edgeft`'s honest holdout gives recall **0.961** → predicted edge_J 0.923, while the LB
edge_J of 0.909 inverts to recall **0.953**. Two unrelated routes — a local held-out measurement
and an LB inversion — agree within ~1%. `edgeft`'s flat loss curve independently confirms the
negative half: if association were the bottleneck, held-out edge loss would have room to fall.

**Seed variance is the live lever (08-02).** Row 17 is the first evidence that anything in this
pipeline still has slack: two independently-seeded 400-epoch models produce up to 12% different node
counts on the same volume. Detection recall is 97.5% and `edge_J` is stuck at 0.909 across the whole
public field, so if two seeds disagree that much, an ensemble of them is under-exploited at
0.85/0.15. Ordering for the next submissions: `secw` (0.50 blend) → `divoff` → `swapw`.

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
