# Strategy & Findings — Biohub Cell Tracking (living doc)

**Updated:** 2026-07-25 (v2 — two-seeds scored 0.912; pivot to ensembling). Supersedes the
strategic sections of `PLAN.md` where they conflict.
Goal: **surpass public LB 0.932** on the custom tracking metric ($60k, deadline 2026-09-29,
kernels-only). See also `docs/metric.md`, `docs/data.md`, `docs/detector.md`, `docs/experiments.md`.

---

## 1. Current standing (verified)

| Submission | LB (public) | Notes |
|---|---|---|
| **two-seeds ensemble + DeepCenter** | **0.912** | **our best CONFIRMED LB** (genuine, not exploit) |
| unet_transformer repro (sdeogade/pilkwang fork) | pending | expected ~0.94; scoring |
| E005 repro (detector+edge+ILP+searched divisions) | 0.888 | |
| Baseline blob + Hungarian | 0.601 | local 0.644 → confirms **~0.04 local>LB offset** |
| Owned pipeline (greedy+multi-hyp), golden-8 local | 0.489 (~0.45 LB proj.) | our measurement lab, not competitive |

**Where we stand:** best confirmed **0.912**, target **0.932** — gap **~0.02**, closable with
genuine improvements. Our from-scratch owned pipeline (~0.45) is not competitive on its own; it is
the clean **measurement lab** (§3). → **strategy pivot** (§5): build on the strong public pipelines.

### ⚠ Public LB is partly gamed — target the PRIVATE board
A public solution (xiaoleilian mix-divaug) ships a **division `metric exploit`**: it appends
fabricated out-of-volume nodes (t=−1000, coords −10000) forming fake division forks to inflate
`division_jaccard` (claimed +0.08). Its own author excludes it from **final-2** (the private-scored
submissions that award the prize). Implications:
- Some public ~0.94 scores are inflated by such exploits → don't treat public numbers as the real bar.
- **We will NOT use the exploit.** The prize is won on the private board with genuine tracking quality,
  where our clean golden-8 harness (measuring real quality) is a structural advantage.
- Final-2 picks are reserved for genuine, durable submissions only.

## 2. Verified findings

- **Metric** (verified field-for-field vs reference `biohub_tracking.metrics`):
  `score = adj_edge_jaccard + 0.1·division_jaccard`,
  `adj = edge_jaccard·(1 − 0.1·total_node_ratio)`,
  `total_node_ratio = (N_pred − N_total)/N_total` where **N_total = `estimated_number_of_nodes`**
  (~20–30k/vol, NOT the sparse annotated count). Only edges touching ANNOTATED GT nodes count;
  node match ≤7µm (anisotropy-scaled). ⇒ ~90% of score is edge_jaccard on annotated cells;
  over-detection is taxed via the node-ratio multiplier.
- **Contamination:** the 4 visible test volumes are byte-identical to 4 train volumes. Pretrained
  pipelines read inflated LOCALLY (E005 0.969 local vs 0.888 LB). ⇒ trust only a proper held-out
  split. We use `hold_out_golden(n=8, seed=0)` — the **golden-8** harness. **This is our edge:**
  forkers of public notebooks cannot honestly measure whether a change helps; we can.
- **Local→LB offset ≈ 0.04** (local higher). Budget for it before submitting.
- **Detector over-detection (fixed):** selecting checkpoints by val peak-recall made the detector
  over-detect WORSE as it trained (golden-8 score 0.40→0.26, nodes 42k→93k). Fix (committed):
  select/early-stop on `sel = recall·exp(−0.1·|node_ratio|)` anchored to `est_nodes/100`, and
  `pos_weight` 20→4. New epoch-0 detector responds to threshold instead of saturating.

## 3. Owned pipeline — our measurement lab (built + tested)

Not currently competitive on the LB, but the fast clean loop we use to decide what actually
helps. Module map in memory `pipeline-architecture`; key modules:
- `detect/unet3d.py` — UNet3D(+optional offset-to-parent head, linajea movement vectors).
- `detect/infer.py` — flip-TTA + heatmap ensembling.
- `link/greedy.py` — fast divisions-aware greedy linker (no ILP → no SCIP-speed wall).
- `link/candidates.py` — multi-hypothesis candidates + confidence + singleton pruning (Ultrack-style).
- `link/edge_model.py` — `motion_affinity` (label-free, from predicted offsets), `train_edge_model`.
- `link/ilp.py` — tracksdata ILP with confidence-weighted node selection.
- `scripts/local/eval_full.py` — integrated golden-8 scorer (detector→multi-hyp→greedy/ILP→metric).

**Measured (golden-8, same epoch-0 detector):** distance-ILP 0.429 → **greedy + multi-hyp 0.489**
(edge_J 0.450→0.506). The GT-trained pairwise edge model HURT (sparse-label problem) → use
`motion_affinity` instead. Offset-detector training in progress to enable motion linking.

## 4. Reference solutions (the frontier is public)

### sdeogade / `unet_transformer` (~0.94, name "selected_94")
Thin driver over the **pilkwang support pack** (`biohub-tracking-support-pack-50ep-v1`: repo +
50-epoch weights). Method = **temporal-UNet + transformer edge predictor (`simple_node_transformer`)
+ tracksdata ILP**, plus heavy engineering:
- dual-seed detection **consensus**; 8-view **D4 TTA** on detection logits;
- ILP weights tuned (appearance 0.0, disappearance 1.5, division); det threshold 0.96875;
- post-proc: single-frame + `t→t+3` **gap closing**, density-adaptive gap distance, **safe-division
  geometry filters** (sister/child max-µm, per-frame & global division-rate caps), short-track
  filter (min len 6) + adaptive rescue, optional **DeepCenter veto** model.
- Attaches 4 public datasets (support pack, temporal-UNet seed314159, DeepCenter, postprocess pack).

### xiaoleilian / `mix-divaug` (close to OUR pipeline)
**2× 3D-UNet heatmap ensemble** (one plain, one **tophat** preproc) → **two-pass µm-gated Hungarian
+ line-fit trajectory linking** → gap repair. Same family as our owned pipeline. Techniques:
- **tophat** morphological preprocessing; **2-model ensemble** (averaged heatmaps, own preproc each);
- **line-fit** linking — fit recent track direction (window 2, weight 0.8), refine/gate links by it
  (a simpler motion cue than the offset head);
- velocity gap-close, short-track filter (min 4), graph repair;
- a **division "metric hack"** final cell that augments division-like structure to boost the
  `0.1·division_jaccard` term (legit metric-optimization; weights not attached so not reproducible).

**Both validate our component choices** (learned edge model, divisions, TTA, gap closing). The gap
to 0.94 is engineering depth + tuning, not concept.

## 5. Strategy: reproduce the frontier, then differentiate on our harness

**Do NOT try to make the owned pipeline leap from 0.45 to 0.93.** Instead:

1. **Reproduce sdeogade's `unet_transformer` (~0.94)** as our submittable baseline under our account
   (fork: `kernels/unet_transformer_repro/`, running). Confirm the score.
2. **Differentiate on golden-8** — A/B each candidate upgrade on our clean harness (the edge nobody
   forking the notebook has), keep only what measurably helps, port winners into the pipeline.
3. **Submit past the field.** Since many entrants run the same public notebook, the score itself is
   not differentiating — our improvements + honest measurement are.

**Why this wins, not just matches:** (a) golden-8 lets us measure real gains where forkers can't;
(b) contamination cuts our way; (c) we have a catalog of tested components from BOTH solutions to
stack.

## 6. Candidate upgrades to A/B on golden-8 (the differentiation toolbox)

From sdeogade: transformer edge predictor, D4 TTA, ILP weight tuning, safe-division geometry
filters, density-adaptive gap closing, dual-seed consensus, DeepCenter veto.
From xiaoleilian: tophat preproc, multi-model ensemble, line-fit motion linking, velocity gap-close,
division metric-augmentation.
Ours (owned, tested): node-ratio-aware detector selection, greedy divisions linker, multi-hypothesis
selection, label-free motion affinity from an offset head.

Prioritize by metric leverage: edge_jaccard (linking + detection of annotated cells) dominates;
`total_node_ratio` is a multiplier; `0.1·division_jaccard` is cheap bolt-on points.

## 7. Roadmap / next steps

- [x] Build owned pipeline (detector+offset head, greedy, multi-hyp, motion affinity, TTA/ensemble).
- [x] Identify + fetch the two public reference solutions.
- [x] **P1** Reproduce `unet_transformer` under our account + submit. *(LB score pending)*
- [x] **P2** Map plug-in points in `predict_unet_transformer.py` (PredictConfig knobs + `--evaluate` CLI).
- [x] **P3** Confirm golden-8 harness == support-pack `metrics.py` (measurements trustworthy).
- [~] **P4** Differentiate — GENUINE, private-durable only:
    - [~] Inference-knob tuning: golden-8 det-threshold × ILP-weight sweep *(kernel running)*.
    - [ ] **Ensemble the two strong pipelines** (two-seeds 0.912 + unet_transformer) — highest-EV
          legitimate move; different architectures → likely complementary → best shot at >0.932.
    - [ ] Safe-division geometry tuning (real divisions), more seeds/TTA.
    - [ ] Confirm each candidate on the real LB (5 subs/day); reserve final-2 for durable winners.
- Offset-detector run retired (recall stuck ~0.45; motion A/B showed no benefit at low recall).
- Deferred: temporal 2-frame detector input (redundant with offset head).

**Submission budget:** 5/day. As of 2026-07-25: 3 used (E005, two-seeds, unet_transformer repro).

---

## 9. Update (2026-07-25 late) — overfit finding, dead levers, CV skew, the winning playbook

### 9.1 The post-processing is overfit → LEAN generalizes better (the money-board edge)
Ran the FULL pipeline vs a LEAN variant (additive post-proc OFF: gap-close, gap2, motion-relink,
safe-divisions, linefit) on the golden-8 harness:
- **FULL selected_94: 0.862** golden-8 vs **0.912 LB**  (~0.05 gap)
- **LEAN: 0.882** golden-8 (edge_J 0.870→0.893) — the additive post-proc *hurts* held-out edge_J.
- Monotonic: less post-proc → better generalization. The `selected_94` post-proc is tuned to the
  contaminated LB test and costs ~0.02–0.04 on unseen data. **On the private board, overfit configs
  drop and LEAN rises.** LEAN submitted (`54985941`). This is our strongest genuine play.
- MINIMAL (all post-proc off incl. subtractive) ERRORED — some cleanup steps are structurally
  required; LEAN is the best *working* lean config.

### 9.2 Dead levers (verified locally, so we don't waste submissions)
- **Division specialist — DEAD.** Metric TP needs a predicted division at a matched GT-division
  parent; precision is hopeless (geometry gates give 0–3 TP for 23–119 FP), and the wrong 2nd-child
  edges also drop edge_J. Every div-adding config scored WORSE. Ceiling of ~4 TP (6 GT divs, only 4
  have 2 detected children) is unreachable without flooding FP. This is why the field uses the fake-
  division exploit — real divisions are unwinnable. (Our local div metric is also an approximation;
  do not tune against it.)
- **Detector-ensembling-to-ADD — DEAD.** Metric is `adj = edge_J·(1 − 0.1·ratio)`, ratio =
  (N_pred−N_total)/N_total. The pipeline UNDER-detects (ratio ≈ −0.2 to −0.3) and **negative ratio is
  a BONUS**. Adding detections shrinks the bonus → lowers score. (Also why LEAN>FULL: fewer added nodes.)
- **Node pruning — tapped out.** LEAN already at the node-count optimum; pruning harder loses
  annotated cells and edge_J collapses.

### 9.3 Adversarial validation — our CV is SKEWED (important)
golden-8 (seed 0) is NOT representative of train:

| | golden-8 | rest-of-train |
|---|---|---|
| est. total cells/vol | 36,976 | 17,637 |
| annotation density | 0.017 | 0.039 |
| embryo mix | 50/50 | 35/65 (6bba-heavy) |

golden-8 is ~2× denser (harder) and embryo-balanced. So part of the FULL 0.862-vs-LB-0.912 gap is
*difficulty*, not only overfit (the LEAN>FULL comparison is same-volume so still valid, but absolute
numbers/node-count tuning may not transfer). **Fix:** REP-VAL — a representative 12-vol holdout
(8 6bba/4 44b6, density ≈ train median). Re-running LEAN vs FULL on REP-VAL to confirm the
generalization finding before trusting it further. Kernels: `rep_full`, `rep_lean`.

### 9.4 The Kaggle-winning playbook, ordered, with our status
1. **Trust CV over LB** — have it; #4 showed it's skewed → fixing with REP-VAL. Our #1 asset given contamination.
2. **Diverse ensembling** — UNTAPPED, highest-EV remaining absolute lever. Blend the
   `local-association-ranker` (diverse edge model) + DeepCenter/Spotiflow at the EDGE-probability level
   (not detection-union, which hurts via the node bonus). Targets the dominant edge_jaccard term.
3. **Retrain + pseudo-labeling** — UNTAPPED, highest ceiling, multi-day GPU. Hold out REP-VAL for clean
   CV; pseudo-label to densify the ~0.4–10% sparse GT.
4. **Adversarial validation** — DONE (found the skew).
5. **Metric-aware (legit)** — mostly done (under-detection bonus, avoid overfit post-proc).
6. **Seed/fold avg + TTA** — mostly done in-pipeline (dual-seed + D4 TTA).
7. **Final-2 selection** — {LEAN → private bet, selected_94 → LB hedge}, near the 2026-09-29 deadline.

### 9.5 Next steps
1. **REP-VAL LEAN-vs-FULL** (running) — gate: does LEAN win on a representative holdout? If yes, lock LEAN.
2. **#2 diverse edge-ensembling** — blend the local-association-ranker; measure on REP-VAL.
3. **#3 retrain holding out REP-VAL + pseudo-label** — the real absolute-score investment.
4. Watch pending LB scores (LEAN `54985941`, repro `54983506`) to cross-check overfit story on the real test.

### 9.6 Honest bottom line
The fast/local absolute-score levers are exhausted — divisions, detection-ensembling-to-add, and
pruning are all dead. Our genuine edge is **generalization discipline** (a verified CV + the lean
config that wins held-out where the field overfits the contaminated LB). Bigger absolute gains need
**diverse edge-ensembling (#2)** or **retraining+pseudo-labeling (#3)**. #1 is not guaranteed, but the
strategy targets the private board that pays, which most of the field is not optimizing for.

## 8. Operational notes

- GPU: Kaggle caps **2 concurrent GPU sessions**; kernels default to P100 (sm_60, torch-incompatible)
  → always set `machine_shape=NvidiaTeslaT4`. No CLI "cancel"; `kernels_delete` frees a slot (source
  is local, re-pushable).
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`. `ctrack` shipped to kernels via the
  `ctrack-src` dataset; detector weights via `biohub-detector-weights`.
- 5 submissions/day. Calibrate against the ~0.04 offset before spending one.
