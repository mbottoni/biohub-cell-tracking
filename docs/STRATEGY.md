# Strategy & Findings — Biohub Cell Tracking (living doc)

**Updated:** 2026-08-02 (v4 — six LB measurements landed. The public "metric hack" is a **net
negative** on the real board, and the §10.5 node-ratio thesis is **dead**: we sit at ~97% of the
node budget, not 75%, and annotated cells are not identifiable from track structure. See **§11**,
which supersedes §10.5/§10.6 wherever they conflict.) Supersedes the strategic sections of
`PLAN.md` where they conflict.
Goal: **surpass public LB 0.932** on the custom tracking metric ($60k, deadline 2026-09-29,
kernels-only). See also `docs/metric.md`, `docs/data.md`, `docs/detector.md`, `docs/experiments.md`.

---

## 1. Current standing (verified)

| Submission | LB (public) | golden-8 | Notes |
|---|---|---|---|
| **two-seeds ensemble + DeepCenter** (`54976907`) | **0.912** | — | **our best CONFIRMED LB** (genuine, not exploit) |
| unet_transformer repro / FULL (`54983506`) | 0.911 | 0.862 | expected ~0.94 from the author — **0.03 repro gap, unexplained** |
| LEAN (additive post-proc OFF) (`54985941`) | 0.890 | 0.882 | **CV said LEAN>FULL by +0.020; LB says −0.021 → rank inverted (§10)** |
| E005 repro (detector+edge+ILP+searched divisions) | 0.888 | — | |
| Baseline blob + Hungarian | 0.601 | 0.644 | early "~0.04 local>LB offset" reading |
| Owned pipeline (greedy+multi-hyp), golden-8 local | — | 0.489 | our measurement lab, not competitive |

**Where we stand:** best confirmed **0.912**, target **0.932** — gap **~0.02**, closable with
genuine improvements. Our from-scratch owned pipeline (~0.45) is not competitive on its own; it is
the clean **measurement lab** (§3). → **strategy pivot** (§5): build on the strong public pipelines.

### ⚠ Public LB is partly gamed — but it is NOT contaminated (corrected 2026-07-26)
A public solution (xiaoleilian mix-divaug) ships a **division `metric exploit`**: it appends
fabricated out-of-volume nodes (t=−1000, coords −10000) forming fake division forks to inflate
`division_jaccard` (claimed +0.08). Its own author excludes it from **final-2** (the private-scored
submissions that award the prize). Implications:
- Some public ~0.94 scores are inflated by such exploits → don't treat public *rankings* as the real bar.
- **We will NOT use the exploit** unless explicitly decided otherwise. Note it is a *metric* exploit,
  not a public-only trick — it would inflate the private board too.
- Final-2 picks are reserved for genuine, durable submissions only.

**What this does NOT mean (we had this backwards until 2026-07-26):** exploit-inflation is a
*ranking* problem, not a *signal* problem. Our own LB score is measured on real hidden data —
kernels-only, `test/` is swapped at rerun (`docs/data.md`). The contaminated measurement is the
LOCAL one. See §10.1.

## 2. Verified findings

- **Metric** (verified field-for-field vs reference `biohub_tracking.metrics`):
  `score = adj_edge_jaccard + 0.1·division_jaccard`,
  `adj = edge_jaccard·(1 − 0.1·total_node_ratio)`,
  `total_node_ratio = (N_pred − N_total)/N_total` where **N_total = `estimated_number_of_nodes`**
  (~20–30k/vol, NOT the sparse annotated count). Only edges touching ANNOTATED GT nodes count;
  node match ≤7µm (anisotropy-scaled). ⇒ ~90% of score is edge_jaccard on annotated cells;
  over-detection is taxed via the node-ratio multiplier.
- **Contamination — direction corrected (2026-07-26):** the 4 visible test volumes are byte-identical
  to 4 train volumes, but that dir is a *dev sample*; Kaggle swaps it for real hidden data at rerun
  (kernels-only, `docs/data.md`). **The LB is clean; our LOCAL harness is the contaminated one** —
  the pilkwang weights were trained on all 199 train volumes, golden-8 included, so golden-8 scores
  of the pretrained pipeline are inflated *and* mis-rank configs (§10.1). golden-8 remains valid for
  models WE train with it held out; it is not a trustworthy instrument for tuning pretrained-pipeline
  knobs.
- **Local→LB offset ≈ 0.04** (local higher) — a rough level offset only. It says nothing about
  ranking, and §10.1 shows the ranking can invert.
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
    - [~] Inference-knob tuning: **moved to the LB** as the measuring instrument (§10.2); the
          golden-8 sweep is no longer trusted for pretrained-pipeline knobs.
    - [ ] **Ensemble the two strong pipelines** (two-seeds 0.912 + unet_transformer) — highest-EV
          legitimate move; different architectures → likely complementary → best shot at >0.932.
    - [ ] Safe-division geometry tuning (real divisions), more seeds/TTA.
    - [ ] Confirm each candidate on the real LB (5 subs/day); reserve final-2 for durable winners.
- Offset-detector run retired (recall stuck ~0.45; motion A/B showed no benefit at low recall).
- Deferred: temporal 2-frame detector input (redundant with offset head).

**Submission budget:** 5/day, deadline 2026-09-29 → **~300+ remaining, 6 used.** Not the scarce
resource; compute is (~6h/run, 2 GPU slots → ~2–4 configs/day). See §10.2.

---

## 9. Update (2026-07-25 late) — overfit finding, dead levers, CV skew, the winning playbook

> ⚠ **SUPERSEDED IN PART by §10 (2026-07-26).** Everything in §9 was measured on golden-8, which
> we now know mis-ranks configs for the pretrained pipeline: its headline call (LEAN > FULL) was
> **inverted on the real LB**. Treat §9.1 as falsified and §9.2's "dead levers" as *unverified*
> (killed by the same instrument), not settled. The §9.3 skew analysis and §9.4 playbook still hold.

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

## 10. Update (2026-07-26) — the CV inverted on the real LB; pivot to LB-driven search

Goal restated by the user: **optimize the actual test-set score.** This section is the operative plan.

### 10.1 The falsification (the important one)
Both pending submissions landed:

| config | golden-8 | LB | |
|---|---|---|---|
| FULL (`selected_94`, `54983506`) | 0.862 | **0.911** | |
| LEAN (additive post-proc off, `54985941`) | 0.882 | **0.890** | |
| Δ (LEAN − FULL) | **+0.020** | **−0.021** | **rank inverted** |

§9.1 predicted LEAN would win on unseen data. It lost by 0.021. Not compression — inversion.

**Root cause: we had the contamination backwards.** The premise was "the LB test is contaminated,
golden-8 is clean." The opposite is true. `test/` is a dev sample swapped at rerun (`docs/data.md`),
so the LB has always been scored on real hidden crops. Meanwhile the pilkwang weights saw all 199
train volumes — golden-8 included — so golden-8 is the leaky measurement. We were using the
inflated instrument to overrule the clean one.

**Consequences:**
- The **LB is our best available signal** for tuning this pipeline. Local golden-8/REP-VAL numbers
  are for models we train ourselves with the split honestly held out.
- §9.2's dead levers (division specialist, detection-ensembling-to-add, node pruning) were all
  killed on golden-8 → **re-open them**, each with one LB probe. STRATEGY itself already warned the
  local division metric is an approximation not to be tuned against.
- §9.6's "our edge is generalization discipline" no longer follows from the evidence.

### 10.2 The unspent resource: submissions
Deadline **2026-09-29**, 5 subs/day → **~300+ submissions remaining**; we have used 6. We were
rationing submissions to protect a CV that does not predict the target. Real throughput limit is
compute (~6h/run, 2 concurrent GPU sessions) → **~2–4 configs/day**, still ~150+ measurements.

Noise floor: two 99.4%-identical pipelines scored 0.911 vs 0.912 ⇒ effects **≥0.005 are readable**.

Overfitting-to-public risk is real but low: public and private are both hidden crops from the same
two embryos, so public is a fair proxy. Mitigation = keep final-2 hedged, don't just submit the two
best public scores.

### 10.3 Plan, ordered by EV
1. **Close the repro gap (no compute, no submissions).** Author reports ~0.94 for
   `unet_transformer`; our repro got **0.911**. That 0.03 dwarfs any knob. Diff our kernel against
   the published notebook line-by-line: config preset, checkpoints, TTA, attached datasets, and
   whether their 0.94 includes division metric-augmentation.
2. **LB coordinate descent from the 0.912 config.** One knob at a time, ordered by metric leverage:
   `DET_THRESHOLD` (trades edge_J against the negative-node-ratio bonus) → `ILP_APPEARANCE/
   DISAPPEARANCE_WEIGHT` → `DUAL_SEED_EDGE_THRESHOLD` → `SECONDARY_EDGE_WEIGHT` → safe-division
   geometry. Record every (config → LB) pair in `docs/experiments.md`.
3. **Re-probe the "dead" levers on the LB**, one submission each — divisions first
   (`0.1·division_jaccard` is up to 0.1 of the score and has never been measured on real test data).
4. **Then** diverse edge-ensembling (playbook #2). Deprioritize retrain+pseudo-label (#3): highest
   ceiling but multi-day GPU steered by the instrument we just caught mis-ranking.

### 10.5 Where the remaining headroom actually is — the node-ratio multiplier (thesis)

> ☠ **DEAD — falsified 2026-08-02 by direct LB measurement. See §11.2.** The premise below (that we
> sit at ~75% of the node budget with a 1.025 multiplier) is wrong: the `probe_pad` diagnostic pins
> us at **~97%**, multiplier **~1.003**. And the closing proposal — rank tracks by P(annotated) —
> is not buildable from the features named: annotated cells are structurally indistinguishable from
> unannotated ones (§11.3). Kept for the metric algebra, which is still correct.
Decompose the field's ~0.93: `edge_J ≈ 0.89` × `multiplier ≈ 1.03` + a small division term. Everyone
is optimizing `edge_J`, which is near its ceiling. **The multiplier is the untouched term.**

Two facts, both verified in-repo, combine into a big free lunch:
- **Only edges touching ANNOTATED GT nodes are scored** (`docs/metric.md`). A predicted edge between
  two unannotated cells is neither TP nor FP — **invisible to `edge_J`**.
- **Annotation is ~3.6% of the node budget.** Over 199 train volumes (`docs/gt_stats.csv`):
  median `nodes/est_n_nodes` = **0.0356** (mean 0.061, p90 0.151); median est 17,909 vs 659 annotated.

So ~95% of what we predict can only ever affect the score through `1 − 0.1·ratio`, where
`ratio = (N_pred − N_total)/N_total` and **negative ratio is a bonus**:

| N_pred as % of `est_n_nodes` | ratio | multiplier |
|---|---|---|
| 75% (≈ where the pipeline sits today) | −0.25 | 1.025 |
| 50% | −0.50 | 1.050 |
| 30% | −0.70 | 1.070 |
| 10% | −0.90 | **1.090** |

Pruning from ~75% → ~10% of the budget is **+6% relative ≈ +0.055 absolute** — the whole 0.912→0.932
gap and more — *at zero `edge_J` cost, provided the pruned nodes are ones that were never scored.*
Selective pruning may even *raise* `edge_J` by removing FP edges near annotated cells.

**The problem this reduces to: predict which cells the annotators traced.** That is a supervised
problem with 199 labelled volumes — model the *annotation process*, not the biology. Rank predicted
tracks by P(annotated) from features (track length, lifespan, t-start, intensity, local density,
position), keep the top-K, tune K on the LB. Needs high recall on annotated cells; precision is free.

Note §9.2 called node pruning "tapped out" — that was *unselective* pruning (min-track-length) scored
on the leaky harness. A learned annotation-likelihood ranker is a different lever.

**First experiment is free** (no GPU, no submission): on train GT, check whether annotated cells are
predictable from track features. If AUC is high, this is the winning move.

### 10.6 The public "metric hack" decoded — and it contradicts our division metric

> ⚠ **The central inference below is WRONG (corrected 2026-08-02, §11.1).** This section read
> kirneo's LB hillclimbing comments as proof that "on the real scorer, this pays." Matching every
> hack-publishing author to their leaderboard row shows the opposite: they all sit *below* the clean
> line (kirneo 0.908, xiaoleilian 0.889, outwrest 0.881). `src/ctrack/eval/biohub.py:170` was right
> — the fake forks are a no-op and the hub edges cost `edge_J`. What survives: the observation that
> our division metric differs from the official one, and that divisions remain unmeasured by us.
Pulled the three top public notebooks (2026-07-26). Two are literally titled *metric hack*.

`kirneo/metric-hack-last-call-update` (101 votes) post-processes a clean submission with
`augment_dataset()`:
- one **hub node** at `t=−1000, (z,y,x)=(−10000,−10000,−10000)`, with an edge to the **root of each
  of the top `MAX_COMPONENTS=3400` connected components** — collapsing the predicted forest into a
  single tree;
- a chain of **`FORKS=32` fabricated 2-child divisions** hanging off that hub, also out-of-volume.

The author's own comments show LB hillclimbing on both knobs (`20→30→40` forks;
`2600→3400→3800` components, "Хочу 3800" = *I want 3800*). **So on the real scorer, this pays.**

**But our metric implementation says it should be a no-op.** `src/ctrack/eval/biohub.py:170` states
outright that unmatched predicted divisions are excluded, "which is exactly why the public fake
forks augmentation is a free no-op here" — and the hub edges should if anything *hurt* `edge_J` as
FPs on matched roots. Our docstring already flagged the gap: we implement the simple out-degree
form, while the reference `score_divisions` "verifies daughter-lineage coverage via grandchildren
and a bipartite matching."

⇒ **Our division metric does not match the official one**, and the field has empirically found the
difference. This independently undermines §9.2's "division specialist is DEAD" — that verdict was
produced by the approximate division term *and* the leaky harness. Divisions are worth up to 0.1 of
the score and remain unmeasured by us on real test data.

Position unchanged: we do not ship the exploit. But we should stop treating our local division
number as evidence of anything.

Also noted: `outwrest/metric-hack-minimal-baseline-tta-2gpu` (79 votes) is a **bare** pipeline —
UNet + transformer edge + ILP + TTA, *zero* post-processing, straight to CSV. That a stripped
baseline is competitive is weak evidence that the elaborate post-proc stack is worth little.

### 10.4 Honest bottom line (replaces §9.6)
We spent a session building measurement infrastructure around a leaky harness and drew a confident,
wrong conclusion from it. The correction is cheap and the position is fine: best is 0.912, target
0.932, and we have ~300 unspent measurements on the only instrument that is actually clean. The work
now is disciplined LB search, explaining the 0.03 repro gap, and testing the §10.5 node-ratio thesis
— which, if it holds, is worth more than every knob in §10.3 combined.

## 11. Update (2026-08-02) — six LB reads: the hack is negative, the multiplier is tapped

Best is now **0.913** (`kernels/d4edge`), rank **137 / 1881** (was 267/1858). Supersedes §10.5/§10.6.

### 11.1 The public Code section is misleading — the metric hack LOSES points
Matching every notebook author to their leaderboard row (the leaderboard CSV carries usernames):

| author | notebook | rank | LB |
|---|---|---|---|
| xiaoleilian | mix-divaug, claimed +0.08 | 888 | 0.889 |
| outwrest | metric-hack-minimal | 1021 | 0.881 |
| kirneo | metric-hack-last-call (108 votes) | 417 | **0.908** |
| kaiwalyaatulraut | biohub-cell-tracking-solution | 599 | 0.901 |
| boristown | V5–V9 ladder, 113 subs | 349 | 0.911 |

Every hack publisher is *below* the clean line. Sorting the Code section by score ranks the hack
notebooks top because that sort is not the leaderboard. **Do not ship the exploit** — not for
ethics here, but because it costs points, exactly as `src/ctrack/eval/biohub.py:170` predicted.

### 11.2 There is a hard wall at 0.913, and the node budget is nearly full
~10 teams forking the clean `selected_101` notebook all land on **0.913** regardless of effort
(prvsiyan 4 subs, nikitagajbhiye30 13, saitejabandaruin 16, indarkarhana 53, amanatar 69). Only
two public authors clear it: **pilkwang 0.916** (D4 edge TTA) and **yusuketogashi 0.915**
(harmonic fusion + bidirectional edges).

`probe_pad` (append 0.5·N_pred edgeless far-field nodes; only the multiplier can move) gives two
equations that solve the decomposition. With `D = 0.1·division_jaccard`:

    base − pad = edge_J · 0.05k = 0.044   →   edge_J · k = 0.88
    base = edge_J(1.1 − 0.1k) + D         →   1.1·edge_J + D = 1.00

| if D = | edge_J | k = N_pred/est_n_nodes | multiplier |
|---|---|---|---|
| 0.00 | 0.909 | 0.968 | 1.003 |
| 0.02 | 0.891 | 0.988 | 1.001 |
| 0.05 | 0.864 | 1.019 | 0.998 |

**Under every assumption we sit at ~97–102% of the node budget with a multiplier of ~1.00** — not
the 75% / 1.025 of §10.5. §9.2's "the pipeline UNDER-detects (ratio ≈ −0.2 to −0.3) and negative
ratio is a BONUS" is therefore also wrong; the ratio is ≈ −0.03. (Note the *marginal* trade is
unchanged — the multiplier is linear in k — so this does not by itself re-open threshold tuning.)

### 11.3 Selective pruning is not buildable — annotated cells are not marked
§10.5's proposal was to rank tracks by P(annotated) and keep the top-K. Measured on all 199 train
GEFF graphs (`scripts/local/annotation_structure.py`, 4,435 annotated tracks / 133,318 nodes):

- **Length does not separate.** Annotated components are *short*: median 21 nodes, p25 = 12.
  A `min_len=20` rule keeps only **54.7%** of annotated tracks and **82.8%** of annotated nodes —
  i.e. it discards 17% of the scored material, which is why `probe_len20` lost 4.8% of `edge_J`.
- **Span does not separate.** Median annotated component spans **22%** of the movie, p25 = 13%.
- **Position does not separate.** The annotated bounding box covers a median **71%** of the volume;
  centroids sit at (0.49, 0.49, 0.51) with sd ≈ 0.08 — dead centre by symmetry, i.e. scattered.

Annotation behaves like a **random ~3.6% sample** (median `nodes/est_n_nodes` = 0.0356), which is
what an unbiased evaluation set should look like. So there is no cheap feature to rank on, and
every prune we ran lost more `edge_J` than it gained:

| | nodes | multiplier gain | edge_J cost | net LB |
|---|---|---|---|---|
| harmonic fusion | −2.9% | +0.3% | −2.8% | 0.889 (−0.023) |
| min_track_len 20 | −21.7% | +2.1% | −4.8% | 0.886 (−0.026) |

Break-even needs the `edge_J` loss under `0.0966·p`. **Treat the multiplier as tapped out.**

### 11.4 What the six reads say about the remaining levers

| lever | LB | verdict |
|---|---|---|
| D4 edge TTA (idea 1) | 0.913 (+0.001) | within noise (≥0.005 readable); our best config, not proven better |
| harmonic fusion (idea 2) | 0.889 (−0.023) | dead |
| node pruning (idea 3) | 0.886 / 0.868 | dead as a lever, invaluable as calibration |
| bidirectional edges (idea 4) | built, unsubmitted | likely subsumed by 8-view edge TTA |
| **divisions** | probe running | **the only untested term worth up to 0.1** |
| **retraining** | not started | **highest ceiling** — see below |

`edge_J ≈ 0.909` is where the entire public field is frozen, and everyone shares the same public
checkpoints.

### 11.5 The weights are NOT undertrained — "50ep" is a stale slug (2026-08-02)
The dataset `pilkwang/biohub-tracking-support-pack-**50ep**-v1` ships an `ARTIFACT_MANIFEST.json`
that reads `biohub-tracking-support-pack-**400ep**-snapshot-v1`, and every kernel we have ever run
printed it on the `Artifact name:` line. The primary is a 400-epoch model, not a 50-epoch one.

The two attached checkpoints are therefore **two independently-seeded 400-epoch models** (primary
sha `12f6881e…`, secondary `9bac2fa0…`), not a weak/strong pair. The secondary's shipped curve
(`history.csv`: 0.946 @ ep50 → 0.978 @ ep381, gains decelerating — 200-300 mean 0.9685,
300-400 mean 0.9721) describes both.

**Two consequences:**
1. "We run the weaker model as primary" is **false**. `kernels/swapw` is a seed-A/seed-B swap and
   measures seed variance, not a training-quality gap.
2. **"Retrain for more epochs" is a weak lever.** The public weights are already at 400/500 epochs
   on all 199 volumes with decelerating validation gains; pushing to 500 looks worth ≈ +0.003 on
   *validation detection*, which is not where `edge_J` is losing. The leaders' +0.03 is therefore
   probably **not** more epochs of this architecture.

`kernels/secw` survives the correction with a *better* rationale than it was built on: two
equal-quality models should be blended more evenly than 0.85/0.15, so raising
`SECONDARY_EDGE_WEIGHT` to 0.50 is a proper 2-model ensemble rather than a fix for a bad primary.

### 11.6 The division term is a liability, not an opportunity (2026-08-02)
The 4 visible test volumes are byte-identical to train volumes, so their ground truth is already in
`docs/gt_stats.csv`. It says:

| volume | GT divisions |
|---|---|
| 44b6_0113de3b | 0 |
| 44b6_0b24845f | 0 |
| 6bba_05b6850b | 0 |
| 6bba_05db0fb1 | **3** |

**Three annotated divisions in total — and our pipeline inserts 350.** Across all 199 train volumes
the median is **0** divisions per volume (mean 0.8, max 5; 112 volumes have none).

`divprobe` (rate caps ×5) confirmed the caps were never the constraint: candidates 355→355, added
323→350, `safe_division_skipped_cap` = 0 in *both* the probe and the baseline. It was not submitted
— a +27-division delta on 120k nodes cannot be read.

So the term cannot be farmed by adding more. Worse, each inserted division carries a second-child
edge, which is a plain FP against `edge_J` — the term holding ~99% of the score. `kernels/divoff`
(`OUTPUT_SAFE_DIVISIONS=0`) asks what those 350 spurious divisions cost. This also explains §9.2's
"every div-adding config scored WORSE" without needing the leaky harness to explain it: we were
adding ~90 divisions per volume where ~1 exists.

⇒ Priority **(SUPERSEDED by §13 — see below)**: (1) `divoff`; (2) even-weight seed ensembling
(`secw`, `swapw`); (3) cross-family diversity — unverified, since `cellmot-baseline-artifacts` is
*attached but never referenced* in both notebooks that ship it; (4) no more post-processing knobs.

**§13 replaces the ordering above.** The EDA shows `edge_J ≈ recall²`, so **detection recall is the
lever** and the association head is the smaller term. Current priority:
1. **Lower `DET_THRESHOLD`** (0.96, then 0.95) — quantified break-even, one submission each (§13.4).
2. `divoff`, `swapw` — already built, cheap confirmations.
3. `edgeft` — still worth submitting as a blendable diverse member, but it targets the smaller term.

**Method note:** the `Artifact name:` line was in our logs from the first run. A diagnostic that
greps for the wrong filename is worse than none — it produced a confident answer from an absent
file. Check `ARTIFACT_MANIFEST.json` before trusting any dataset slug.

## 12. Training our own model (2026-08-02) — do NOT copy the public recipe

### 12.1 The public model's validation is fully leaky — 400 epochs is not a target
`secondary_seed_weights/.../split_manifest.json` reports **train=199, validation=40, overlap=40**.
Every validation volume is also a training volume (`dataset_splits_alltrain.json`, method
`unet_transformer_alltrain_seed314159_v1`). Two of them — `44b6_0113de3b`, `44b6_0b24845f` — are
the visible test volumes.

Consequences:
- Their `validation_score` is a **training-set score**. It cannot detect overfitting, so "best
  epoch 381" was selected on a metric that cannot tell memorisation from generalisation, and the
  +0.032 gain from epoch 50 → 381 is not evidence that 400 epochs are needed.
- Their selection metric compounds it: `score = acc · recall` with `acc = 0.9998`, so selection is
  effectively **maximise node recall** — on a pipeline whose detection recall is already 97.5%,
  whose bottleneck is `edge_J`, and whose metric *penalises* over-detection via the node ratio.

⇒ Train against a clean holdout and early-stop on it. Expect convergence far below 400 epochs.
⇒ 138 GPU-h (400 epochs at their 1217 s/epoch) was never the right budget to plan against.

### 12.2 The honest split (`configs/honest_split.json`)
`scripts/local/make_honest_split.py` — 169 train / 30 holdout, **overlap 0**, stratified by embryo
× density tercile so the holdout matches the corpus (median density 0.029 vs corpus 0.0356), unlike
the §9.3 golden-8 which was ~2× denser than train. The 4 dev-test volumes are forced into the
holdout so validation is measured on exactly what the visible LB scores.

### 12.3 Smoke test first — and it already earned its keep
`kernels/trainsmoke` (built off d4edge cells 0-7, so the environment matches inference) proves the
loop before any GPU-days are committed, and measures s/iter on a **T4** rather than trusting the
author's hardware.

Run 1 **FAILED**, as intended: `dataspec.WEIGHTS_PATH = <repo>/weights`, but the setup cell
materialises `weights/` as a **symlink into the read-only `/kaggle/input` mount**
(`link_or_copy_tree` prefers `os.symlink`), so training cannot write a checkpoint —
`OSError: [Errno 30] Read-only file system`. Both arms died in ~8 s. Fix: replace the symlink with
a real copy before training, then write-probe the directory.

Confirmed by run 1: the competition `train` mount carries **199 zarr + 199 geff, all usable**.

Run 2 **PASSED**. Measured on a Kaggle T4 (8 iterations, batch 4, 6 train / 2 val volumes):

| config | s/iter | h/epoch (199 vols) | h @100 epochs | checkpoint | `load_model` |
|---|---|---|---|---|---|
| window 2 | **3.90** | 5.34 | 534 | 8.4 MB ✓ | ✓ `window_size=2` |
| window 3 | **11.18** | 15.13 | 1513 | 8.4 MB ✓ | ✓ `window_size=3` |

Both checkpoints wrote with a correct `config.json` and reloaded through the *inference* loader.

**A T4 is ~8× slower than the author's hardware** (their 1217 s/epoch over ~2,463 iters implies
≈0.49 s/iter). ⇒ **From-scratch training is infeasible here**: 100 epochs = 534 GPU-h against a
remaining quota of ~240 GPU-h (30/week × 8 weeks).

Caveats on the measurement: 8 iterations includes CUDA/cuDNN warmup, so 3.90 s/iter is probably
pessimistic — re-measure over ~50 iterations before committing. And validation returned
`acc=0.0000 recall=0.0000`, which is expected for a randomly-initialised model that detects nothing
but means **the early-stopping metric is still untested**; warm-start must be shown to produce
non-zero validation before it is trusted to stop a long run.

### 12.4 Making iterations fast without sacrificing performance
Audited the shipped trainer: `autocast` 0, `GradScaler` 0, `cudnn.benchmark` 0, `channels_last` 0,
no window caching. Everything below is therefore additive. Speed factors are **estimates** unless
marked measured; each row gives what it costs to verify.

| # | change | why it does not cost accuracy | est. speedup | time to verify |
|---|---|---|---|---|
| 1 | **Warm-start** `--unet-weights` from the 400ep checkpoint | starts from a converged model; we need displacement, not convergence | **~20× fewer epochs** (400 → ~15) | one re-smoke, **~15 min** |
| 2 | **Freeze the UNet, train only the edge transformer** | detection recall is already 97.5% and saturated; `edge_J` is the bottleneck and 99% of the score. `max_nodes=11` (measured) confirms the transformer is tiny — the volumetric encoder is the cost | 2–4× | one re-smoke, **~15 min** |
| 3 | **AMP fp16 + `GradScaler`**, `cudnn.benchmark=True` | tensor-core path on 3D convs; fixed 64³ shapes make autotuning pay back immediately | 2–3× | one re-smoke, **~15 min** |
| 4 | **Cache preprocessed windows as a Kaggle dataset** | pure I/O, no model change | removes **~5–12 min/session** (corrected: the 71 s figure was whole-run wall including train+val; actual load is ~12–28 s for 8 volumes ⇒ ~5–12 min for 199) | build dataset ~40 min, then **~15 min** to verify |
| 5 | **Batch 8** (the public config's value) | standard; better GPU utilisation | ~1.3× | folded into any re-smoke |
| 6 | **Drop window-3** | measured 2.9× slower *and* the dual-seed blender rejects mismatched inference grids, so it needs its own pipeline | avoids a 2.9× tax | already measured |

Stacked (1+2+3+5), 5.34 h/epoch should fall to roughly **0.5–1.0 h/epoch**.

### 12.6 Fast-iteration smoke — MEASURED: batch 8 + cudnn.benchmark is a 3x REGRESSION

`kernels/fasttrain`, T4, 24 iters, window 2, three arms sharing batch 8 + `cudnn.benchmark`:

| run | s/iter | h/epoch | vs 3.90 s/iter baseline | val_acc | val_recall |
|---|---|---|---|---|---|
| A cold, UNet trainable | 28.95 | 19.80 | **0.27×** | 0.9987 | 0.9051 |
| B + warm start | 27.91 | 19.09 | **0.28×** | 0.9998 | 0.9743 |
| C + frozen UNet | 22.23 | 15.21 | **0.35×** | 0.9998 | **0.9842** |

**Every arm is ~3× SLOWER than the batch-4 baseline.** Ideas 5 (batch 8) and cudnn.benchmark are
rejected. Cause not isolated — both were changed globally across all three arms, which was a
methodology error: this file had just argued for one change per smoke.

The strongest clue is **validation**, which is forward-only and therefore untouched by freezing:
it costs **513–519 s in all three arms** vs **12.4 s** in the batch-4 smoke (§12.3) — 42× worse,
~21 s/batch vs 0.26 s/batch. An 80× cost for a 2× batch is not normal scaling. Suspect batch 8
exceeding T4 VRAM (first UNet layer alone is ~4.3 GB at 8×2×32×64³) and thrashing the allocator,
or `cudnn.benchmark` re-autotuning per shape. Untangle with one arm each.

**What did work, and is now verified:**
- **Warm start (idea 1) — confirmed by construction.** `matched 136/136 tensors; 0 left at init`,
  and the quality shows it: after 24 iterations, cold reaches recall 0.9051 (edge loss 0.0055) while
  warm reaches **0.9743** (edge loss 0.0001). The `--unet-weights` trap is real and avoided.
- **Freezing (idea 2) is only 1.25× on training**, not the estimated 2–4× — the UNet *forward* still
  runs to produce features; only its backward is skipped. It does **not** speed up validation at all.
  But it gave the **best validation recall of the three (0.9842)** at 27.9% of the parameters.
- **The early-stop metric works** — closes the §12.3 caveat that it had only ever returned zeros.

### 12.7 The reframe: count STEPS, not epochs
"Epoch" is an arbitrary unit here, and planning against it is what made training look infeasible.
With a warm start we do not need passes over all 199 volumes — we need enough optimizer steps to
displace the model. At the corrected batch-4 rate with freezing (≈3.1 s/iter est.), **5,000 steps
≈ 4.3 GPU-h** and **10,000 steps ≈ 8.6 GPU-h** — both inside one 12 h session and one week's quota,
against the 534 GPU-h that a naive 100-epoch plan implied.

Use `--max-iters` to define a short epoch (e.g. 500 steps) so validation and checkpointing happen
often enough for early stopping on `configs/honest_split.json` to bite.

Corrected recipe: **batch 4, no cudnn.benchmark, warm start ON, freeze ON, `--max-iters 500`,
early-stop on the honest split.** Next smoke must vary exactly one thing to confirm the rate
returns to ~3.1 s/iter.

### 12.8 Edge-head fine-tune — smoke PASS, and freezing is worth 4.3x not 1.25x

`kernels/edgeft_smoke` (batch 4, no cudnn.benchmark, warm start + frozen UNet):

| measure | value |
|---|---|
| **s/iter** | **0.90** (baseline 3.90; fasttrain batch-8 22.23) |
| full run 10,000 steps | **2.5 GPU-h** (was estimated 4–9) |
| warm start | matched **136/136** tensors |
| frozen | 580,386 trainable / 2,076,706 (**27.9%**) |
| grid vs primary | window 2, downsample (1,4,4) — **BLENDABLE: True** |

**This isolates the §12.6 regression.** Freezing measured only 1.25× at batch 8 with
`cudnn.benchmark`; at batch 4 without it the same change yields **3.90 → 0.90 s/iter**. The batch-8
regime was masking the benefit, not merely adding overhead. (Part of the 4.3× is better warmup
amortisation over 20 iters vs 8, so treat 4.3× as the config-level gain, not freezing alone.)

Consequence: the whole "training is infeasible on Kaggle" conclusion was an artifact of measuring
the wrong configuration. 10,000 steps at 0.90 s/iter is **2.5 GPU-h** — under one tenth of a week's
quota, against the 534 GPU-h a 100-epoch batch-8 plan implied.

Two cosmetic issues left in the log, neither blocking: the summary line still prints
"Best score (acc*recall)" though selection now uses `-test_loss`, and `test_loss` prints at 4 dp so
it reads as 0.0000 — the selection uses the float, and it did discriminate (epoch 0 best, epoch 1
not), but the log cannot show the margin. Worth widening if early stopping ever looks indecisive.

### 12.9 Local M1 / MPS — works, but ~14x slower than a T4. Use it for debugging, not training.

Ran the real trainer on the M1 (`/tmp/localtrain`, one volume, 10 iters, batch 4). It needed two
small patches and then trained and checkpointed successfully:
  * device selection is hardcoded `cuda`-or-`cpu` → added an MPS branch;
  * three unconditional `torch.cuda.synchronize()` timing calls → guarded.

**Measured: 12.3 s/iter** (unfrozen), against **0.90 s/iter** on a T4 — **13.7× slower**. Internal
split: data 22%, forward 56%, backward 23%. Freezing removes most of the backward share, so a frozen
run lands ≈9.5 s/iter — still ~10× the T4. A 10,000-step fine-tune would be **~26 h locally vs
2.5 GPU-h on Kaggle**.

**A synthetic conv3d proxy badly over-promised.** It measured MPS forward at 58 ms/step and MPS
1.5× *faster* than CPU forward-only, suggesting MPS might beat the T4. The real UNet forward is
~3 s/iter — roughly 50× the proxy. A 3-layer conv stack is not a UNet; do not size hardware from a
toy graph. (Third time this session an extrapolation from a small proxy misled: also the 8-iteration
s/iter figures and the "71 s ⇒ 30 min" load estimate.)

**Verdict: stay on Kaggle for training.** At 2.5 GPU-h per fine-tune against ~30 GPU-h/week, quota
allows ~12 runs/week — so quota is *not* the binding constraint for this workload, and local's only
real advantage does not apply.

**Local is still worth keeping** as a debug environment: it catches code bugs (like the `os`
NameError that cost a 6-minute kernel cycle) in seconds, with no setup, queue, or GPU slot. Keep
`/tmp/localtrain` reproducible via the two patches above.

### 12.5 End-to-end timings — what each step costs before you see a number

| step | measured / estimated | notes |
|---|---|---|
| kernel setup (wheels, repo, weights copy) | **4.5 min** measured (fasttrain) | of which **~3.7 min is copying the secondary seed checkpoint, which TRAINING DOES NOT USE** — drop that dataset from training kernels |
| load 199 train volumes | **~5–12 min** est. from ~12–28 s / 8 vols | → ~0 with idea 4. ⚠ an earlier revision said ~30 min, from misreading whole-run wall as load time |
| re-smoke a speed change | **~15 min** measured (11.6 min run 2) | the unit of iteration for §12.4 |
| full warm-start training, 15 epochs | **~10–15 GPU-h** est. after 1+2+3+5 | fits one 12 h session; 534 h without them |
| inference with the new checkpoint | **~21 min** measured | dev-sample commit run |
| LB rerun after submitting | **~4–6 h** measured (08-02 batch) | hidden-test rerun, not the commit run |
| **training change → LB number** | **~16 h** | 9 h train + 0.4 h infer + ~6 h rerun |

Budget context: ~30 GPU-h/week, 12 h session cap, deadline 2026-09-29, **5 submissions/day**.
A single training experiment costs roughly half a week of GPU quota, so verify each §12.4 change
with a 15-minute re-smoke *before* spending it — the smoke has already caught one fatal bug
(§12.3) that would otherwise have surfaced hours into a long run.

## 13. EDA (2026-08-02) — edge_J is detection-limited, not linking-limited

Measured over all 199 GT graphs (`docs/gt_stats.csv`, volume 104x104x104 um, T=100).

### 13.1 Association is geometrically trivial
| quantity | value |
|---|---|
| GT displacement per frame | **1.82 um** median (p90 3.63, and p90 never exceeds 10 um in any volume) |
| cells per frame (`est_n_nodes`/100) | 179 median (p90 576) |
| mean cell spacing | 18.5 um median (p10 12.5) |
| **displacement / spacing** | **0.103 median**, p90 0.164, **max 0.221** |

A cell moves ~10% of the way to its nearest neighbour between frames; >0.5 is where linking turns
ambiguous. Even the hardest volume sits at 0.22. Our gates (`MOTION_RELINK_TIGHT` 6.0,
`RELAXED` 10.0, `OUTPUT_EDGE_MAX` 14.0) are 3-8x larger than typical motion, so they are not binding.

### 13.2 GT is strictly consecutive
`edge_dt_max = 1.0` and `edge_dt_skip_frac = 0` in **all 199 volumes**. So
`OUTPUT_ENFORCE_NEXT_FRAME=1` costs nothing, and there is no hidden pool of skip-edges to recover.

### 13.3 edge_J ~ recall^2 — and that closes the whole gap
An edge scores only if BOTH endpoints are detected, so with linking near-perfect (13.1),
`edge_J ~ r^2`:

| node recall r | ceiling edge_J |
|---|---|
| 0.990 | 0.980 |
| 0.975 | 0.951 |
| **0.953** | **0.908** |
| 0.940 | 0.884 |

Our measured `edge_J = 0.909` implies **effective recall ~0.953**. Detection recall alone explains
essentially the entire missing 0.09. The public model's reported 0.975 is on the leaky split (12.1).

⇒ **The lever is detection recall on annotated cells, not the association head.** This supersedes
the 12.x framing that the edge transformer is where the score is lost.

### 13.4 The threshold trade, quantified — and 9.2 retired
Adding nodes costs `0.1*dk` on the multiplier; recall gain multiplies `edge_J` by `(r'/r)^2`.
Net positive when **`dr/r > 0.05*dk`**.

Adding 10% more nodes (`k` 0.97 -> 1.07, multiplier 1.003 -> 0.993, -1.0%) pays if recall rises
>0.5% *relative* (0.953 -> 0.958). A 1-point absolute recall gain is worth **~+0.010**.

This retires 9.2's "detector-ensembling-to-ADD is DEAD", whose premise was a large negative-ratio
bonus to protect; the pad probe (11.2) showed the multiplier is ~1.00, so there is almost no bonus
at stake. It also explains why the tuned public forks converged on `DET_THRESHOLD = 0.965`, below
our 0.96875.

⇒ Next experiment: **lower `DET_THRESHOLD`** (0.96875 -> 0.96 and 0.95), one submission each.
Read `nodes` from run_stats to get `dk` and back out the realised recall gain.

### 13.5 The node budget is wildly uneven ACROSS volumes — and the multiplier is per-volume

`src/ctrack/eval/biohub.py:247` — `adj_edge_jaccard = np.mean(per-volume adj)`. The multiplier is
applied **per volume and macro-averaged**, while `edge_jaccard` is micro-averaged (summed counts).
So node-budget calibration matters per volume, not in aggregate.

At our fixed `DET_THRESHOLD=0.96875` (d4edge, LB 0.913), using `run_stats` nodes against
`gt_stats.est_n_nodes` for the four dev-test volumes:

| volume | N_pred | est_n_nodes | k | ratio | multiplier |
|---|---|---|---|---|---|
| 44b6_0113de3b | 25,535 | 25,755 | 0.991 | −0.009 | 1.0009 |
| **44b6_0b24845f** | 18,503 | 32,795 | **0.564** | −0.436 | **1.0436** |
| 6bba_05b6850b | 6,217 | 6,362 | 0.977 | −0.023 | 1.0023 |
| 6bba_05db0fb1 | 70,021 | 69,800 | 1.003 | +0.003 | 0.9997 |
| aggregate | 120,276 | 134,712 | 0.893 | −0.107 | 1.0107 |

**Three volumes sit at k ≈ 0.98–1.00; one detects only 56% of its estimated cells.** A 1.78× spread
from a single global threshold.

`44b6_0b24845f` is a persistent outlier across three independent measurements: k = 0.564 here, the
worst loss under `probe_len20` (−57% of nodes), and the largest seed disagreement in `swapw` (+12%).
It is the volume the model is least sure about, and it is *earning* a 4.4% multiplier bonus by
under-detecting — almost certainly while destroying its own `edge_J`, since `edge_J ≈ recall²`.

**Prediction for `thr096`/`thr095`:** a global threshold drop is three small effects plus one large
one. The three saturated volumes cross k = 1 into multiplier penalty for modest recall gain; the
under-detected volume gains a lot of recall while giving up bonus it should never have had. If the
scores come back muted, that mixture is why — and it argues for adaptivity rather than a lower
global constant.

**Caveat on §11.2.** The pad-probe solve (`k ≈ 0.968`) assumed a *single global* k. With the
multiplier applied per volume over a 1.78× spread, that figure is an effective aggregate, not a
per-volume truth. The qualitative conclusion (multiplier ≈ 1.00, no large bonus to protect) stands;
the precise number should not be over-read.

**Blocked follow-up:** per-volume adaptive thresholding targeting k ≈ 1 would fix this directly, but
`estimated_number_of_nodes` lives in the `.geff` `extra` metadata and `test/` ships **`.zarr` only**
(`docs/data.md`). It is not available at inference. A proxy would have to be estimated from the
image itself, or from detection counts at a reference threshold.

### 13.6 The threshold lever is real in sign but ~10× weaker than 13.4's arithmetic (2026-08-03)

Both reads landed. `DET_THRESHOLD` 0.96875 (base, 0.913) → **0.96: 0.914** → **0.95: 0.914**.

**This is our first score above the 0.913 wall.** It is also +0.001, at the noise floor (§11.4 puts
readability at ≥0.005). Two independent configs moving the same direction is weak-but-consistent
evidence that the sign is right; it is not proof either beats base.

**The shape of the curve is the informative part.** `thr095` adds ~1.7× the nodes `thr096` does and
returns the same 0.914. Inverting through §11.2's decomposition, +0.001 of score is
`Δedge_J ≈ +0.001`, i.e. recall 0.9534 → **0.9539 — a tenth of a point**. §13.4's illustration
("~10% more nodes should buy ~0.5% relative recall, worth +0.005") over-predicted the yield by
roughly an order of magnitude, and the second half of the node addition bought nothing readable.

**Why: threshold can only admit peaks that already exist.** Detections come from local-max pooling
over the heat map; the confidence cut then filters those peaks. Lowering the cut recovers cells
whose peak is *present but low*. It cannot recover a cell that produced **no local maximum at all**
— merged with a neighbour, or absent from the field. The flatness says the missing ~4.7 recall
points are mostly of the second kind.

`orfuse` (exp 24) is the same physics observed from the other side: saturating the probability
field flattened it, local-max pooling then found *no* maxima, and detections were destroyed exactly
where confidence was highest. Peak *existence*, not peak *height*, is what gates our recall.

⇒ **Retire global-threshold tuning.** Do not spend further submissions on 0.94/0.93; the curve is
flat where it should be steepest. §13.3's conclusion survives intact — recall is still the lever,
and `edge_J ≈ recall²` still holds — but the reachable knob is **candidate generation**, not the
confidence cut. Candidates, in rough EV order:

1. **Peak extraction geometry** — the NMS / local-max window radius. Two cells inside one window
   yield one detection no matter the threshold, and §13.1 measured mean spacing at 18.5 µm with
   p10 = 12.5 µm, so crowded volumes are exactly where merging should bite. Note `44b6_0b24845f`,
   the k = 0.564 outlier of §13.5, is the densest of the four. This parameter is inside the
   installed inference package, not exposed as a `BIOHUB_*` env var — check whether it is reachable
   before planning around it.
2. **`OUTPUT_MIN_TRACK_LEN` 6 → 3 or 2, in the *additive* direction.** Every prune we have run
   raised it (`len20`, −0.026). The additive direction has never been tested, and §11.3 measured
   annotated components at p25 = 12 nodes — short tracks are not disproportionately unannotated.
3. **A detector that finds different cells** — the only route left with a large ceiling, and the
   one the 0.947 leader is presumably on. Note §11.5: more epochs of *this* architecture is not it.

**Method note.** §13.4 produced a quantitative prediction that was wrong by ~10× and it was still
the most valuable thing in §13 — it made the flat curve legible as a fact about peak extraction
rather than as an uninformative null. Keep writing the arithmetic down before the submission.

## 14. The division term is a PRECISION play worth up to +0.1 (2026-08-03)

⚠ **This partially corrects §11.6.** That section was right that adding divisions cannot help and
that our 350 insertions are mostly wrong. It then concluded "divisions are a liability,
deprioritise" — and never did the arithmetic on what *precision* is worth. The arithmetic says the
division term is the largest unclaimed pool of score on the board.

### 14.1 The metric makes division FPs nearly free to remove and TPs enormously valuable

From `src/ctrack/eval/biohub.py:161-196`, `division_jaccard` is micro-summed over volumes with

    TP: GT dividing node whose *matched* pred node also divides
    FN: GT dividing node not recovered
    FP: pred division at a *matched* node whose GT node isn't dividing

**FP is counted only at matched nodes**, and GT annotation is a sparse sample. Over the four dev
volumes (`docs/gt_stats.csv`): 2,193 annotated nodes against 134,712 estimated and 120,276
predicted — an annotation rate of **1.6%**. So of our ~350 inserted divisions, only

    350 x (2193 / 120276) ~ 6.4

are visible as FPs. GT offers **3** divisions on those volumes. With `D = 0.1 * div_J`:

| scenario | TP | FP | FN | div_J | score contribution |
|---|---|---|---|---|---|
| current (spray 350) | 0 | 6.4 | 3 | 0.00 | 0.000 |
| current, lucky | 1 | 6.4 | 2 | 0.106 | +0.011 |
| current, all three hit | 3 | 6.4 | 0 | 0.319 | +0.032 |
| **precise, one hit** | 1 | 0 | 2 | 0.333 | **+0.033** |
| **precise, all three** | 3 | 0 | 0 | 1.000 | **+0.100** |

The swing between spraying and being precise is **+0.07 to +0.10**. Nothing else on our board is
within an order of magnitude of that.

### 14.2 The leaderboard shape corroborates it

68 teams sit above us; the distribution is a dense shelf (18 at 0.915, 12 at 0.916, 7 at 0.917)
then a thin tail to 0.947. The shelf is what shared checkpoints with `edge_J ~ 0.909` and
`div_J ~ 0` produce. The gap from the shelf to the leader is **+0.033** — squarely inside the
division term's range and hard to explain from `edge_J`, which §13.6 just showed is stuck.

This is a hypothesis consistent with the LB shape, not a proof. But it is testable for one
submission.

### 14.3 We do not know our own `div_J` — and `divoff` measures it

§11.2's decomposition never pinned `D`; it tabulated D = 0.00 / 0.02 / 0.05 and could not
distinguish them. So we do not know whether we are already collecting ~0.03 of division score or
none of it, and that determines whether precision work is worth +0.03 or +0.09.

`kernels/divoff` (`OUTPUT_SAFE_DIVISIONS=0`, built 08-02, **never submitted**) drives TP to 0, hence
`div_J` to 0. Then

    D_base ~ (score_base - score_divoff) + edge_J gain from dropping ~350 FP child edges (~+0.003)

- `divoff` ~ 0.910  =>  D ~ 0.006. We collect almost nothing; the full +0.09 is open.
- `divoff` ~ 0.885  =>  D ~ 0.03. We already hit the TPs; precision is worth up to +0.07 on top.

Either outcome is decisive, and it costs one already-built submission. **Treat `divoff` as a probe,
not a candidate** — like `probe_pad`, it is expected to score below base and is run for the number.

### 14.4 Two precision knobs are already exposed, both untuned

1. `BIOHUB_DEEPCENTER_SAFE_DIV_VETO` is currently **`'0'`** in our best config, while the analogous
   `BIOHUB_DEEPCENTER_GAP_VETO` is `1`. Turning it on requires the independent center model to
   confirm each division — precision, exactly the direction 14.1 wants, for a one-line change.
2. `BIOHUB_ILP_DIVISION_WEIGHT` (default 1.0) is the ILP's per-division *cost*. Raising it
   suppresses divisions at the source. Note this is a **second, independent** division path:
   `divoff` disables only the post-processing insertion, so it does not zero ILP divisions. Read
   `run_stats` to confirm how many survive before over-reading the probe.

Also tighten `SAFE_DIV_MAX_UM` (4.66), `SAFE_DIV_SISTER_MAX_UM` (8.5),
`SAFE_DIV_FRAME_FRAC_CAP` (0.0076) — but §11.6 measured `safe_division_skipped_cap = 0`, so the
caps are not currently binding and tightening the *rate* caps alone will be a no-op. The distance
gates and the veto are the live ones.

**Caveat carried forward.** `_evaluate_divisions` is our simplified out-degree reading; the
reference additionally verifies daughter lineages via grandchildren and a bipartite match
(`biohub.py:173-177`). That makes real TPs *harder* than modelled, which pushes further toward
precision, not less. Do not tune divisions hard against the local harness — use the LB.

### 14.5 Ranked plan (supersedes §13.6's ordering)

| # | lever | est. impact | cost | status |
|---|---|---|---|---|
| 1 | **`divoff` as a `D`-probe** | information only | 1 sub, built | do first |
| 2 | **Division precision** (DeepCenter veto on, tighter gates) | **+0.02 .. +0.09** | 1-2 subs, ~1 line | gated on 1 |
| 3 | **Peak-pooling radius** (`pool_kernel_um`) | ~+0.01 per recall point | 1 sub, patchable | ready |
| 4 | **Logit-max seed fusion** (fix `orfuse`) | +0.005 .. +0.015 | 1 sub, fix built kernel | ready |
| 5 | Per-volume adaptive detection (k -> 1) | +0.01 .. +0.02 | needs an `est_n_nodes` proxy | blocked |
| 6 | A genuinely different detector | highest ceiling | weeks, ~240 GPU-h | last resort |

**On 3:** §13.6 flagged peak extraction as unreachable pending a check. It is reachable —
`pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)` sits inside the `_merged`
`predict_video` string in cell 9 that we already replace wholesale (`build_recall_kernels.py`
patches the detection blend the same way). Override `pool_k` directly; print the default first.

**On 4:** `orfuse` failed on a saturation bug (exp 24), not on a bad thesis. Harmonic-on-complement
clamps to 1-1e-6 and flattens the logit field, and local-max pooling then finds no maxima. A max or
log-sum-exp **in logit space** unions the two seeds' peaks without ever saturating. `swapw` measured
up to 12% per-volume disagreement between the seeds, so the union is materially larger than either.

**Retired — do not spend submissions here:** global `DET_THRESHOLD` (§13.6), node pruning /
`MIN_TRACK_LEN` up (§11.3), ensemble blend weights (exp 19b), more training epochs (§11.5), the
public metric hack (§11.1).

## 15. Division geometry, measured — §14's thesis survives but its numbers were wrong (2026-08-03)

⚠ **Corrects §14.1 and §14.3.** §14 was arithmetic over an unknown TP. This section measures the
missing quantity on all 199 GT graphs (`scripts/local/division_geometry.py`). The structural
argument holds; the expected value is **+0.01 .. +0.03, not +0.02 .. +0.09**, and `divoff` is
**not** the decisive probe §14.3 claimed.

### 15.1 The hole in §14: `div_J = TP/(TP+FP+FN)` is 0 when TP is 0

Removing false positives buys nothing unless we ever hit a real division. §14 never established
that we do. So: are our gates capable of admitting a real division at all?

`add_safe_divisions_postlink` (cell 11) applies three distance gates — verified by reading the
code, not the names. A source with exactly one existing child, plus an *orphan* node at t+1:

    child_dist  = d(parent, existing child)   <= SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65
    parent_dist = d(parent, candidate)        <= SAFE_DIV_MAX_UM                = 4.66
    sister_dist = d(existing child, candidate)<= SAFE_DIV_SISTER_MAX_UM         = 8.5

### 15.2 Real divisions look nothing like what the gates select for

151 annotated divisions across 199 volumes (87 volumes have at least one), against 128,581
ordinary single-child steps:

| quantity | median | p90 |
|---|---|---|
| division parent -> daughter (nearer) | **4.08 um** | — |
| division parent -> daughter (farther) | **7.13 um** | 10.05 |
| **division sister <-> sister** | **10.57 um** | 14.36 |
| ordinary (non-dividing) motion | **1.82 um** | 4.14 |

The geometry is self-consistent and physical: a parent splits and the daughters land roughly one
cell diameter apart (10.6 um), each about half that from the parent's last position (5.3-5.75 um).
Compare §13.1's mean cell spacing of 18.5 um — sisters really are anomalously close, so the signal
exists.

**But `SAFE_DIV_SISTER_MAX_UM = 8.5` sits below the median of the truth.** Admission rates for real
divisions, exact three-gate test allowing either daughter to be the "existing child":

| EXISTING | CAND | SISTER | real divisions admitted |
|---|---|---|---|
| **7.65** | **4.66** | **8.5** (current) | **0.265** |
| 10 | 10 | 12 | 0.616 |
| 12 | 12 | 14.4 | **0.894** |
| 12 | 12 | 18 | 0.954 |
| 99 | 99 | 8.5 | 0.291 |

The sister gate is the binding one: it alone caps us at 0.291. **Our configuration is structurally
incapable of seeing ~3/4 of all real divisions.**

### 15.3 The ranking is inverted

Within a frame, proposals are sorted by `score = parent_dist + 0.15 * sister_dist`, **ascending** —
smallest distances win — and truncated at
`frame_cap = round(n_sources * SAFE_DIV_FRAME_FRAC_CAP)`.

Real divisions score a median 5.74 on that function. Ordinary motion (median step 1.82) scores far
lower. So the sort actively prefers the *least* division-like candidates, and real divisions —
which sit in the upper tail of the admitted range — are the first dropped when the cap binds.

**And we do not know whether the cap binds.** §11.6 reported `safe_division_skipped_cap = 0`, but
that stat is only incremented by the *global* cap; the `frame_cap` break increments nothing. With
~0.76% of sources per frame, `frame_cap` is order 10 while candidates are plausibly many more.
**Instrument this before tuning it** — add a counter on the frame-cap break.

### 15.4 Revised expected value, and a falsifiable prediction

Our four dev volumes carry **3** GT divisions and 26.5% gate admission, so expected TP <= 0.8 before
any detection or linking loss — realistically ~0. With FP ~ 6.4 (§14.1) and FN ~ 3:

    div_J ~ 0.03   =>   contribution ~ 0.003

**Prediction: `divoff` comes back flat (0.913-0.916), not down.** It should gain ~0.003 of `edge_J`
from dropping ~350 false child edges and lose ~0.003 of division term. §14.3 called it "decisive";
it is not — the expected result is a null, and a null cannot distinguish "we farm nothing" from
"two small effects cancelled". **Demote it.** It is still the cheapest bound on `D`, but it is no
longer the first submission.

**The actual play is the opposite of what the config does now:** widen the gates so real divisions
are *admissible*, then cut the count hard with a *correct* ranking. Concretely —

1. `SAFE_DIV_SISTER_MAX_UM` 8.5 -> **14.4**, `SAFE_DIV_MAX_UM` 4.66 -> **12**,
   `SAFE_DIV_EXISTING_CHILD_MAX_UM` 7.65 -> **12**. Admission 0.265 -> 0.894.
2. Replace the ascending `parent_dist + 0.15*sister_dist` sort with a **likelihood-shaped** score,
   e.g. `|sister - 10.6| + |parent - 5.3|`, which peaks on the measured division geometry instead
   of on stationary cells.
3. Turn on `BIOHUB_DEEPCENTER_SAFE_DIV_VETO` (currently `'0'`, while the analogous gap veto is `1`)
   and tighten `SAFE_DIV_GLOBAL_FRAC_CAP` so the total insert count *falls* even as the gates widen.

Estimated outcome: TP from ~0 to perhaps 40-60% of GT divisions, FP held near current levels by the
tighter caps => `div_J ~ 0.1-0.26` => **+0.01 .. +0.03**. The +0.1 ceiling needs near-perfect
division detection and is not reachable by gate tuning alone.

### 15.5 What survives from §14 unchanged

- FP is counted **only at matched nodes**, and annotation over the dev volumes is 2,193 nodes
  against 120,276 predicted (**1.6%**). Our ~350 insertions therefore cost only ~6 visible FPs.
  This is why divisions are cheap to get wrong and valuable to get right.
- The division term is still the largest unclaimed pool on the board, and still the only plausible
  account of the LB's dense shelf at 0.913-0.917 versus a thin tail to 0.947 (§14.2).
- §13.6's conclusion is untouched: `edge_J` is stuck and the threshold is spent.

**Method note.** §14 was written from arithmetic over an unmeasured quantity and got the magnitude
wrong by ~3x within a single session. The measurement that fixed it took ~10 minutes on data we
already had. Measure before ranking.

### 15.6 `divgeom` built and run end-to-end — the mechanism works as designed (2026-08-03)

`kernels/divgeom` implements 15.4. Local test (`scripts/local/test_divgeom.py`) lifts the **actual
patched function** out of the built notebook, with that notebook's own config, and replays it over
all 199 GT graphs — deleting one child edge per real division (the harder assignment: the *farther*
daughter) and asking whether the heuristic puts it back:

| config | real divisions recovered |
|---|---|
| `d4edge` (shipped gates) | 19/151 = **0.126** |
| `divgeom` (re-centred, caps lifted) | 133/151 = **0.881** |

**0.881 independently confirms the 0.894 predicted from the gate distributions in 15.2** — a
functional test of shipped code against a distribution calculation, two unrelated routes.

**Kaggle commit run (v2, COMPLETE, 11.25 predict-min):**

| | candidates | inserted | global-cap breaks | frame-cap breaks | nodes |
|---|---|---|---|---|---|
| `d4edge` | 355 | 350 | 0 | *unmeasured* | 120,276 |
| **`divgeom`** | **13,495** | **119** | 276 | **80** | 120,118 |

Four things this establishes:

1. **The widened gates expand the pool 38×** (355 -> 13,495), as intended.
2. **The tightened cap lands where designed** — 119 insertions against a ~120 target.
3. **`frame_cap` DOES bind — 80 breaks.** §11.6 concluded from `safe_division_skipped_cap = 0` that
   the caps never bind; that stat only ever covered the *global* cap and the frame cap had been
   truncating silently since the beginning. 15.3 suspected it; this measures it.
4. **The node budget is untouched** (−0.13%), so this is a clean single-lever change against
   `edge_J` — whatever the LB says is attributable to divisions.

Submission verified: 235,932 rows, 115,814 edges, **exactly 119 forks**, matching the counter.

**The ranking now carries the experiment, and it is the part we could not test.** 119 of 13,495
proposals survive — a 0.88% selection rate — so the score function decides essentially everything.
The local fixture cannot exercise it: with wide gates, old rank and new rank score *identically*
(93/136 at budget 2; 125/209 vs 124/209 at budget 5), because a GT graph is a ~1.6-3.6% sample and
too few candidates ever compete within a frame. At inference density they compete 113:1. **If
`divgeom` moves the LB, the follow-up that isolates the re-rank is wide-gates + old-rank.**

**Expected LB.** With 119 inserted, visible FP ~ 119 x 0.0182 ~ 2.2 against 3 GT divisions:
`div_J = TP/(TP + 2.2 + (3-TP))`. TP=1 -> +0.019; TP=2 -> +0.038; TP=0 -> 0.000 and we land flat.
The base case (350 inserted, 26.5% admission) is `div_J ~ 0.03` -> +0.003. So the read is roughly
**flat if the ranking failed, +0.02..+0.04 if it worked.**

**Method note — the fixture must never be more permissive than the caller.** v1 died on Kaggle
after ~17 min with `KeyError: safe_division_frame_cap_hit`: `stats` in `filter_output_graph` is a
plain pre-seeded dict, and a new counter has to be declared in that literal. Our test used
`defaultdict(int)` and absorbed the missing key silently. `seed_stats()` now parses the real literal
out of the notebook. Related: Kaggle derives a kernel's slug from its **title**, not the id in
`kernel-metadata.json` — v1 was silently created at a different slug, which broke `kernels status`.

## 16. ☠ `divgeom` LOST 0.006 — the EDA answered the wrong question (2026-08-04)

**LB 0.908 vs 0.914 base.** Not the predicted flat, and not the predicted gain: a *readable* loss
(≥0.005). §15.4's intervention was wrong, and §15.6's "flat ⇒ ranking failed / up ⇒ it worked"
framing did not admit the outcome that actually occurred.

### 16.1 What the run diff shows

`run_stats`, d4edge vs divgeom, same four dev volumes:

| stat | d4edge | divgeom | delta |
|---|---|---|---|
| `safe_division_candidates` | 355 | 13,495 | **+13,140** |
| `safe_divisions_added` | 323 | 119 | −204 |
| `edges` | 116,134 | 115,814 | **−320** |
| `nodes` | 120,276 | 120,118 | **−158** |
| `short_track_components_removed` | 1,133 | 1,171 | +38 |
| `short_track_nodes_removed` | 4,431 | 4,585 | **+154** |
| `short_track_edges_removed` | 3,298 | 3,414 | +116 |

Edges fell by 320 while only 204 division edges were removed. The other 116 is exactly the extra
short-track deletion.

### 16.2 Error 1 — the EDA measured recall and I read it as precision

§15.2 measured **P(geometry | division)**: real divisions have sisters at 10.57 um, so a gate at
8.5 um rejects most of them. True, and it is why admission rose 0.265 -> 0.894.

It never measured **P(division | geometry)** — the quantity that decides whether a *candidate* is
real. Widening the gates raised the proposal pool **38x, 355 -> 13,495**, against a base rate of
**0.76 real divisions per volume**. At that base rate, matching the *typical* geometry of a division
is not discriminative: thousands of coincidental orphan-near-parent pairs land in the same window,
and a score centred on the division mean cannot separate them. We then kept 119 of 13,495 — a 0.88%
cut driven entirely by an untested ranking — and apparently kept ~none of the real ones, where the
old tight-gate rule was catching some.

**The tight gates were doing precision work.** `div_J = TP/(TP+FP+FN)` punishes volume; a narrow,
specific window is the right shape for it. §15.2 read "below the median of the truth" as a defect.
It is a recall/precision trade, and the LB says the shipped trade was better than mine.

**The warning was already in our own notes and I submitted anyway.** §15.6 recorded that the local
fixture's precision (0.526) was "optimistic and must not be read as a false-positive rate", and that
the re-rank was unexercised. Those were the two load-bearing unknowns, and both broke.

### 16.3 Error 2 — safe divisions are not a pure division lever

Adopting an orphan splices it into a track component. Insert fewer divisions and 204 orphans stay
orphaned, fall under `MIN_TRACK_LEN = 6`, and `filter_short_track_components` deletes them: +154
nodes and +116 edges removed. So `safe_divisions_added` is **also a node-retention knob**, and any
change to it moves `edge_J` independently of `div_J`.

This means the −0.006 is a *mixture* — lost division TPs plus lost annotated nodes — and one LB read
cannot separate them. It also retro-taints §11.6's framing of `divoff` as a clean division probe:
turning divisions off would strand ~323 orphans and delete ~245 nodes, confounding the very term it
was built to measure. **`divoff` is not a clean probe. Do not run it as one.**

### 16.4 What this establishes, and what is next

- **The division term is probably live.** You cannot lose 0.006 to a division change unless there
  was division score to lose. d4edge is plausibly earning `div_J ~ 0.06` (TP ~ 5 of ~30 GT
  divisions on the hidden set, consistent with the 26.5% gate admission).
- **☠ Widening the division gates is dead.** Do not revisit. The pool explodes faster than any
  geometric ranking can filter it.
- **The live question is precision on the EXISTING pool.** `divgeom_veto` should be rebuilt on the
  **original** gates (4.66 / 8.5 / 7.65) plus `DEEPCENTER_SAFE_DIV_VETO=1` — an *independent* signal
  (the center model) rather than more of the same geometry. That is the one untried precision
  mechanism, and it does not touch the pool size, so it also does not disturb node retention.
- Any future division experiment must report `short_track_nodes_removed` alongside
  `safe_divisions_added`, or the two levers stay confounded.

**Method note.** Three estimates in two days: +0.02..+0.09 (arithmetic over an unmeasured TP) ->
+0.01..+0.03 (measured the geometry) -> bimodal (isolated the untested ranking) -> **−0.006
measured**. Every revision came from measuring something already in hand, and the final one came
from the LB. The recurring failure is not arithmetic, it is measuring the conditional in the
convenient direction: `P(evidence | hypothesis)` is easy to get from GT, `P(hypothesis | evidence)`
is the one that decides. §10.1 and §11.1 are the same mistake in different clothes.

## 17. We can score locally, and it changes almost everything (2026-08-04)

`scripts/local/score_submission.py`. The four `test/` volumes are byte-identical to train volumes,
so their GT is on disk, and a commit run's `submission.csv` is a complete predicted graph over
exactly those four. Every quantity we have spent two weeks *inferring from LB deltas* is directly
computable, for free, before spending a submission.

### 17.1 It is validated against the board

| | local | LB |
|---|---|---|
| `d4edge` | 0.9216 | 0.914 |
| `divgeom` | 0.9092 | 0.908 |
| **delta** | **−0.0124** | **−0.006** |

Right sign, right order of magnitude, absolute level optimistic by ~0.008. **New workflow: commit
run (~21 min, costs no submission) -> download `submission.csv` -> score locally -> submit only if
local says up.** That is the loop that was missing for the whole project.

Caveat: four volumes hold only 3 annotated divisions, so `division_jaccard` is quantised in steps of
~0.125 and is a coarse instrument. `edge_jaccard` rests on ~2,100 GT edges and is stable.

### 17.2 The `divgeom` loss was 100% divisions — §16.3 was wrong

| | edge_J | adj | div_J | score |
|---|---|---|---|---|
| `d4edge` | 0.8895 | 0.9091 | **0.1250** | 0.9216 |
| `divgeom` | 0.8898 | 0.9092 | **0.0000** | 0.9092 |

`edge_J` moved **+0.0003**. ⚠ **§16.3's "error 2" is refuted** — the short-track/node-retention side
effect is mechanically real but costs nothing measurable. The entire −0.0124 is the division term.

**And the division term is live and quantified for the first time:** `d4edge` scores
**TP=1, FP=5, FN=2 -> div_J = 0.125**, worth **+0.0125** of score. `divgeom` scores TP=0 -> 0.
So §16.2 was right: widening the gates destroyed the one real division the tight config was
catching. ☠ Confirmed dead.

### 17.3 The score is a MACRO average — small volumes count as much as big ones

`edge_jaccard` is micro (summed counts) but **`adj_edge_jaccard` is the mean of per-volume values**,
and the score uses the macro one:

| volume | GT edges | edge_J | adj | node recall |
|---|---|---|---|---|
| 44b6_0113de3b | 50 | 0.9038 | 0.9046 | **0.9423** |
| 44b6_0b24845f | 49 | 0.8846 | 0.9232 | 0.9804 |
| 6bba_05b6850b | 845 | 0.9709 | 0.9731 | 0.9988 |
| 6bba_05db0fb1 | 1,183 | **0.8359** | 0.8356 | 0.9910 |

A volume with **50** annotated edges carries the same 25% of the score as one with 1,183. Any
tuning that optimises aggregate counts is optimising the wrong average.

### 17.4 ⚠ `edge_J ~ recall²` is FALSE where it matters — §13.3 and §13.6 are wrong

§13.3 concluded "detection-limited, not linking-limited" from an aggregate inversion. Measured
per volume:

| volume | node recall | recall² (predicted edge_J) | actual edge_J | gap |
|---|---|---|---|---|
| 6bba_05b6850b | 0.9988 | 0.998 | 0.971 | −0.027 |
| **6bba_05db0fb1** | **0.9910** | **0.982** | **0.836** | **−0.146** |

On the volume that dominates micro `edge_J` and carries 25% of the macro score, detection is
essentially perfect (99.1%) and `edge_J` is 0.836: **133 FP and 83 FN edges built on correct nodes.**
That is ~15 points of association loss, not detection loss. §13.6's "retire everything except
candidate generation" followed from the wrong premise.

**This also explains §11.2's decomposition drift.** `1.1·edge_J + D = 1.00` with D = 0.0125 gives
edge_J = 0.897 — close to the measured 0.8895, so the algebra was fine; the *interpretation*
(recall² ⇒ detection) was not.

### 17.5 The learned association model is a ~1 µm tiebreaker — which explains `edgeft`

`run_stats`: `motion_relink_replaced_raw_edges` equals `raw_edges` on every volume. The relinker
**replaces 100% of the transformer's edges** (`edges = motion_edges`) and re-derives them from a
Hungarian assignment whose cost is

    cost = motion_dist + 0.05 * raw_dist - MOTION_RELINK_LEARNED_BONUS * prob

`motion_dist` is in µm and competing assignments inside `MOTION_RELINK_RELAXED_UM = 10` differ by
several µm; `prob ∈ [0,1]` with bonus 1.0. **The learned signal is worth at most 1 µm of tiebreak
against a distance-dominated objective.**

⇒ **§21 (`edgeft`) is explained.** We fine-tuned an association head for 10,000 steps and nothing
moved because its output is discarded and re-derived geometrically. The flat loss curve was not
evidence that association is solved — it is evidence that we were tuning a component that barely
reaches the objective. §12.x was closer to right than §13 gave it credit for.

### 17.6 Ranked plan — every item now has a local pre-check

| # | lever | grounding | est. | local test first? |
|---|---|---|---|---|
| 1 | **`MOTION_RELINK_LEARNED_BONUS` sweep** (1.0 -> 3, 6) | 17.4 + 17.5: the dominant volume is linking-limited and the learned term is a rounding error in the cost | +0.005 .. +0.02 | yes |
| 2 | **Division precision on the ORIGINAL gates** (`DEEPCENTER_SAFE_DIV_VETO=1`) | 17.2: TP=1, FP=5. Cut FP to 0 at constant TP -> div_J 0.125 -> 0.33 | **+0.02** | yes |
| 3 | **Per-volume detection threshold** | 17.3: `44b6_0113de3b` has the worst recall (0.9423) and carries 25% | +0.005 .. +0.01 | yes |
| 4 | `MOTION_RELINK_RELAXED_UM` / velocity weight | same premise as 1, adjacent knob | small | yes |

☠ Dead, do not revisit: widened division gates (§16), global `DET_THRESHOLD` (§13.6), node pruning
(§11.3), ensemble blend weights (exp 19b), more epochs (§11.5), the metric hack (§11.1).

**Method note.** The instrument that settled all of this was buildable on day one — the GT and the
submission files were always on disk. §16's method note blamed measuring `P(evidence|hypothesis)`;
the deeper fault was inferring from LB deltas at 4-6 h per read when a direct measurement was
available for free. Build the instrument before the hypothesis.

## 18. The oracle ceiling: association is 76% of what is left (2026-08-04)

`scripts/local/link_ceiling.py`. For a fixed node set the best achievable edge_J is
`(# GT edges with BOTH endpoints matched) / (# GT edges)` — no linker can beat it, a perfect one
reaches it. That is a **pure detection** quantity, so the gap to our measured `edge_J` is exactly
what association could still buy. This replaces §13.3's `recall²` heuristic with a measurement.

| volume | GT edges | **oracle** | actual edge_J | gap | detFN | assocFN | swaps | FP |
|---|---|---|---|---|---|---|---|---|
| 44b6_0113de3b | 50 | 0.9600 | 0.9038 | +0.056 | 2 | 1 | 1 | 2 |
| 44b6_0b24845f | 49 | 0.9592 | 0.8846 | +0.075 | 2 | 1 | 1 | 3 |
| 6bba_05b6850b | 845 | 0.9976 | 0.9709 | +0.027 | 2 | 8 | 6 | 15 |
| 6bba_05db0fb1 | 1,183 | 0.9848 | 0.8359 | **+0.149** | 18 | 65 | 57 | 133 |
| **macro** | | **0.9754** | **0.8988** | **+0.0766** | 24 | 75 | 65 | 153 |

Counts reconcile exactly with `evaluate`. **Detection costs us 24 edges; association costs 75 FN and
153 FP. Association is 76% of all remaining loss** — and the ceiling says a perfect linker on our
current nodes would be worth **+0.077** of score. ☠ §13.3 and §13.6 are dead: detection was never
the lever.

### 18.1 The swaps are real, not a matching artifact

The obvious objection: the matcher tolerates 7.0 µm and the dense volume has ~11.7 µm spacing, so
maybe GT nodes match the wrong detections. Tested — **no**:

| for the 65 swaps | median |
|---|---|
| dist(node **we linked to**, GT target) | **7.28 µm** |
| dist(node the **metric matched**, GT target) | **2.87 µm** |
| our link lands closer than the matched node | **0 / 65** |

The metric's matching is right and our links are wrong. We linked the source to a genuinely
different physical cell.

### 18.2 They are fast cells, and geometry provably cannot fix them

| | correct links (n=2,038) | swaps (n=65) |
|---|---|---|
| GT displacement | 1.46 µm | **5.20 µm** |
| pred-pred distance | 1.15 µm | **8.76 µm** |
| detection match error | 1.72 µm | 3.45 µm |

Swapped cells move 3.6x faster than average, and their detections are twice as poorly localised.
**In 100% of swaps the link we took was shorter than the correct one** (median margin 6.93 µm), and
simulating the relinker's own cost at other velocity weights flips **0 of 65 at every setting
tested (0.5 / 0.75 / 1.0 / 1.25)**. ☠ `MOTION_RELINK_VELOCITY_WEIGHT` is dead — killed for free,
before a submission.

⇒ No distance, gate, or motion-extrapolation tuning can recover these. The **learned edge
probability is the only term in the cost that can**, which is the strongest possible statement of
§17.5.

### 18.3 What that implies for `relinkw` — and a warning

`cost = motion + 0.05·raw − BONUS·prob`, and `prob ∈ [0,1]`, so a swap can only flip if
`BONUS > margin`:

| BONUS | swaps it could flip, at best |
|---|---|
| 1.0 (current) | 1.5% |
| **3.0** (`relinkw3`) | **7.7%** |
| **6.0** (`relinkw6`) | **33.8%** |
| 10.0 | 92.3% |

**Pre-registered: `relinkw3` should be flat and `relinkw6` small at best.** The thesis needs
BONUS ≈ 10 to be tested at all. (Caveat: the margins are measured on raw distance while the cost
uses motion-compensated distance, so these are indicative bounds, not exact.)

**⚠ The asymmetry is the real risk.** There are **2,038 correct edges against 65 swaps** — a 31:1
ratio. A bonus large enough to override geometry on the swaps can equally override it on links that
are currently right. Corrupting 2% of correct edges (41) would wipe out a perfect fix of all 65
swaps. So this lever is not "free upside"; it is a bet on the transformer being sharply
discriminative exactly where geometry is misleading.

### 18.4 The next instrument, before betting on the bonus

We cannot currently see the edge probabilities — they are internal to the run. **Dump them from a
commit run** (one free run) and measure offline, on these 65 known-wrong cases, whether the
transformer ranks the correct target above the distractor. That converts §18.3's bet into a
measurement:

- if it ranks correct > distractor on most swaps ⇒ raise BONUS, upside up to +0.077;
- if it does not ⇒ the shipped association model cannot solve this, and the lever is a **new
  association model**, not a reweighting — which is very likely where the 0.947 leader is.

**Method note.** Every conclusion in this section came from files already on disk and cost zero
submissions. Two candidate levers (`VELOCITY_WEIGHT`, and `relinkw3` as a meaningful test) were
killed before spending anything, and the headline lever was re-scoped from "+0.02" to "up to +0.077
but structurally risky". This is the loop §17 was built for, working as intended.

## 19. `relinkw3` measured — the margin model held, the headline number was noise (2026-08-04)

Commit run, scored locally. **Not submitted.**

| | d4edge | relinkw3 | delta |
|---|---|---|---|
| edge_jaccard | 0.8895 | 0.8961 | +0.0066 |
| adj_edge_jaccard | 0.9091 | 0.9212 | **+0.0121** |
| division_jaccard | 0.1250 | 0.1111 | −0.0139 |
| **score** | 0.9216 | 0.9323 | **+0.0107** |

At face value our first config to beat `d4edge`. It is mostly an artifact.

### 19.1 §18.3's mechanism prediction was accurate

Predicted: `BONUS = 3.0` can flip at most 7.7% of 65 swaps ≈ 5. Measured: swaps **65 → 62**,
`edge_tp` +6, `edge_fp` −8, `edge_fn` −6 on `6bba_05db0fb1`. The margin model held.

### 19.2 But the headline is 80% one edge on a 49-edge volume

`adj_edge_jaccard` is a MACRO mean over 4 volumes (§17.3), so per-volume contributions are:

| volume | GT edges | adj | contributes |
|---|---|---|---|
| **44b6_0b24845f** | **49** | 0.9232 → 0.9618 | **+0.0096** ⚠ thin |
| 6bba_05db0fb1 | 1,183 | 0.8356 → 0.8453 | +0.0024 |
| 6bba_05b6850b | 845 | 0.9731 → 0.9732 | +0.0000 |
| 44b6_0113de3b | 50 | 0.9046 → 0.9046 | +0.0000 ⚠ thin |

**One edge on a 49-edge volume is worth +0.0096 of reported score.** The hidden test has ~10x the
volumes, so that noise averages away and does not transfer.

    ROBUST delta (volumes with >=200 GT edges):  +0.0024
    headline delta                               +0.0121

⇒ `relinkw3`'s real expected LB movement is **~+0.002 — below the 0.005 readable threshold.**
§18.3's call of "flat" was right in substance; I then mis-scored my own prediction by reading the
headline instead of the decomposition, which is §17.3's finding applied against itself.

**`score_submission.py` now prints this decomposition and flags any volume under 200 GT edges.**
The instrument had the same overread available to it as the LB does; now it defends against it.

### 19.3 Divisions moved too — this was not a clean single lever

`division_jaccard` fell 0.125 → 0.111 (TP held at 1, one extra FP). Relinking changes which nodes
are orphans and which have out-degree 1, so it feeds the safe-division candidate pool. Any future
association experiment must report `div_J` alongside `edge_J`; they are coupled through the
post-processing.

### 19.4 Standing read

- `relinkw6` becomes the real test: §18.3 says it can flip ≤33.8% of swaps, ~4x `relinkw3`, so the
  **robust** component should land near +0.010 if the mechanism scales — that would be readable.
- The 31:1 correct-to-swap asymmetry (§18.3) is still the risk, and `relinkw6` is where it starts
  to bite.
- `probdump` remains the decisive instrument: if the correct edge is not even a candidate,
  `learned_prob` is 0 and no bonus can ever reach it.

## 20. ☠ STOP TUNING DIVISIONS — three attempts, three ways to destroy the same TP (2026-08-04)

> **⚠ PARTLY SUPERSEDED by §23.3 (same day).** Everything below about the three *knob*
> interventions is correct and they stay dead. But the conclusion "divisions are closed" does
> not follow from it. `division_jaccard` is **micro**-pooled, so the term's headroom is
> **+0.0875** — larger than §18's association oracle — and every number here was measured
> against **3** GT divisions when 151 are on disk. This section established that *the
> instrument cannot see divisions*, and then filed it as *the term has no headroom*. The
> untried lever is a fitted selector over the wide candidate pool; the hand-written ranking of
> §16 is not evidence against it.

`divveto` (`DEEPCENTER_SAFE_DIV_VETO=1`, original gates), commit run, scored locally, **not
submitted**:

| | d4edge | divveto | delta |
|---|---|---|---|
| edge_jaccard | 0.8895 | 0.8910 | +0.0015 |
| adj (robust part) | | | **+0.0008** |
| **division_jaccard** | **0.1250** | **0.0000** | **−0.1250** |
| **score** | 0.9216 | 0.9100 | **−0.0116** |

### 20.1 The veto is not a filter, it is an off-switch

    deepcenter_safe_div_checked     0 -> 355
    deepcenter_safe_div_accepted    0 ->   0
    deepcenter_safe_div_rejected    0 -> 355     <-- 100%
    safe_divisions_added          323 ->   0

At `DEEPCENTER_SAFE_DIV_THRESHOLD = 0.12` the centre model scores **every** candidate below
threshold. So `divveto` is functionally identical to `divoff` — which means **§14.3's "decisive
probe" is now answered for free**: turning divisions off costs **−0.0116 locally**, and by §17.1's
~2x scaling roughly **−0.006 on the LB**, the same as `divgeom`. ☠ `divoff` no longer needs a
submission; do not spend one on it.

### 20.2 The single TP is the whole term, and everything breaks it

| config | div TP | div FP | div FN | div_J |
|---|---|---|---|---|
| **d4edge (shipped)** | **1** | 5 | 2 | **0.125** |
| divgeom (widen gates + re-rank) | 0 | 3 | 3 | 0.000 |
| divveto (centre-model veto) | 0 | 0 | 3 | 0.000 |

Three independent interventions, three routes to `TP = 0`. `divveto` even did the "right" thing —
it removed all 5 false positives — and still lost, because Jaccard with `TP = 0` is 0 regardless.
**The shipped configuration is at a local optimum we cannot knob-turn our way out of.**

### 20.3 And the local instrument cannot guide us here

The four dev volumes hold **3** annotated divisions. `division_jaccard` is therefore a 3-sample
statistic quantised in steps of ~0.125, and any threshold tuned against it is fitted to a single
true positive. §17.1 flagged this; §20.2 is what it looks like in practice.

⇒ **Divisions are closed.** The term is worth **+0.0125** today and we have no instrument capable of
improving it safely, while §18's association headroom is **+0.077** — six times larger, measured on
~2,100 GT edges rather than 3. Every remaining submission and commit run goes to association.

☠ Dead, do not revisit: `divgeom`/widened gates (§16), `divveto`/centre-model veto (§20),
`divoff` (answered here for free), safe-division rate caps (§11.6), division geometry re-ranking
(§16). The division post-processing stays exactly as shipped.

**Method note — the loop is now paying for itself.** In one session the local instrument has killed
`MOTION_RELINK_VELOCITY_WEIGHT` (§18.2), correctly deflated `relinkw3` from a headline +0.0107 to a
robust +0.0024 (§19.2), and retired the entire division programme including a submission we had
queued (`divoff`). **Cost: zero submissions.** Compare §16, where the same class of question cost a
readable 0.006 on the board.

## 21. The association model is GOOD — the candidate gate is what loses us the score (2026-08-04)

`relinkw6` (BONUS 6.0) and `probdump` both landed. Together they close §18.3's bet.

### 21.1 The bonus saturates, and `probdump` says exactly why

| config | headline adj | **robust adj** |
|---|---|---|
| `relinkw3` (BONUS 3) | +0.0121 | **+0.0024** |
| `relinkw6` (BONUS 6) | +0.0125 | **+0.0028** |

Doubling the bonus bought **+0.0004**. §18.3 predicted the reachable fraction should rise 7.7% ->
33.8%; it did not. The dump explains it:

| of the 65 swaps | |
|---|---|
| correct edge is **NOT a candidate** (`prob` identically 0) | **47 / 65 = 72%** |
| correct edge **is** a candidate | 18 / 65 |
| ...and the transformer ranks it **above** the taken edge | **18 / 18 = 100%** |
| median `prob(correct)` vs `prob(taken)` | **0.838 vs 0.000** |

`learned_prob` returns 0.0 for non-candidates, so on 72% of swaps the bonus term is *identically
zero* and no value of `MOTION_RELINK_LEARNED_BONUS` can ever move them. **The lever has a hard
ceiling of 28% of swaps** — that is the saturation, measured.

### 21.2 The headline result: the transformer is right every time it is allowed to speak

**18/18.** Where the correct edge survives to become a candidate, the association model prefers it
over the distractor 100% of the time, by a median margin of 0.838 probability. ⚠ This overturns the
pessimistic branch of §18.4 and inverts §13's framing completely: **the shipped association model is
not the weak component. The candidate gate in front of it is.**

It also fully closes §21/`edgeft` (exp 21): fine-tuning that head was always going to do nothing,
because the head is right already and its correct answers are being filtered out upstream.

### 21.3 What is filtering them — measured, not guessed

| | median | max |
|---|---|---|
| all 112,876 candidate edges | 1.72 µm | 15.33 |
| swap-correct edges **present** | 8.50 µm | 13.99 |
| swap-correct edges **missing** | 8.76 µm | 14.56 |

- **0%** of the missing edges exceed the longest observed candidate ⇒ **not a distance cap**.
- **79%** sit beyond the candidate p99 (7.5 µm) ⇒ they are in the far tail of an otherwise
  short-edge distribution.
- The correct target is typically the **2nd nearest** node at t+1 (median rank 2; it is among the
  5 nearest 89% of the time) ⇒ **not a neighbourhood/kNN restriction either.**

⇒ By elimination the gate is the **probability threshold**, `BIOHUB_DUAL_SEED_EDGE_THRESHOLD =
0.48`. These are genuine, geometrically close successors that the transformer scored under 0.48, so
they never entered the candidate set.

### 21.4 Next: `edgethr` — lower the candidate threshold

`BIOHUB_DUAL_SEED_EDGE_THRESHOLD` **0.48 -> 0.15**, on top of BONUS 6. This is the only change that
addresses the 72%, and the two knobs are coupled: admitting a candidate does nothing unless the
bonus can lift it over the motion margin, and the bonus does nothing for a candidate that is absent.
They ship together for the same reason §16's gates and caps did.

**And with the same risk, explicitly noted.** `divgeom` failed by flooding a candidate pool and
trusting a ranking. The differences now: the diagnosis is measured (72%, not inferred), the ranking
is validated at 18/18 rather than assumed, and **`score_submission.py` will grade it locally before
any submission**. If the robust delta does not clear ~0.005, it does not go to the board.

**Ceiling.** Even perfect recovery of all 65 swaps is bounded by §18's oracle: macro association
headroom **+0.077**, of which the two thick volumes contribute ~+0.044 robust. Realistic target for
`edgethr` is **+0.005 .. +0.015 robust**.

### 21.5 Standing

- LB best **0.914**, rank **86 / 1966**. Leader 0.947. **5 submissions available, 0 used today.**
- Best local config: `relinkw6`, robust **+0.0028** — below the 0.005 readable floor, so **not worth
  a submission yet**. Keep iterating locally; that is what the instrument bought us.

## 22. `edgethr` — right answer, wrong mechanism; and an instrument bug (2026-08-04)

### 22.1 ⚠ The "robust delta" in §19.2 and §21.5 was understated 2x

It summed the per-volume contributions of the thick volumes but still divided by **all 4**. What
transfers to a ~40-volume hidden set is the **mean per-volume adj delta**, estimated over the
volumes thick enough to measure — so the sum must be divided by the number *retained*, not by `n`.
Fixed in `score_submission.py`. Corrected table:

| config | transferable adj | div_J delta | **net expected score** |
|---|---|---|---|
| `relinkw3` (BONUS 3) | +0.0049 | −0.014 | +0.0035 |
| `relinkw6` (BONUS 6) | +0.0056 | −0.014 | **+0.0042** |
| **`edgethr30`** (thr 0.30, BONUS 6) | **+0.0062** | −0.025 | **+0.0037** |
| `edgethr15` (thr 0.15, BONUS 6) | +0.0026 | −0.034 | +0.0000 |
| `divveto` | +0.0017 | −0.125 | −0.0108 |

§21.5's "relinkw6 robust +0.0028, not worth submitting" was half the true figure.

### 22.2 The gate hypothesis is refuted — it helped for a reason I did not predict

| config | edge_J | **detFN** | assocFN | swaps | FP |
|---|---|---|---|---|---|
| d4edge | 0.8988 | 24 | 75 | 65 | 153 |
| relinkw6 | 0.9109 | 22 | 70 | 61 | 143 |
| **edgethr30** | **0.9172** | **14** | 72 | 62 | 139 |
| edgethr15 | 0.9157 | 12 | **76** | **65** | 146 |

§21.4 predicted opening the gate would convert swaps. **It did not** — swaps went 65 → 62 → 65.
What it actually did was cut **detection-FN 24 → 14**: more candidate edges keep more nodes out of
the short-track filter, so more GT nodes get matched. That is §16.3's node-retention mechanism,
which §17.2 found cost nothing there, mattering here in the opposite direction.

### 22.3 ⚠ The 18/18 was selection-biased, and the arithmetic was available beforehand

§21.2 measured the transformer at 18/18 **on edges that had already passed the 0.48 gate** (median
prob 0.838) and I generalised it to the 47 that had not. Those are, by construction, the model's
*low-confidence* cases. At threshold 0.30 a newly admitted edge has `prob ∈ [0.30, 0.48]`, so
`BONUS × prob ≤ 6 × 0.48 = 2.9 µm` against §18.2's median margin of **6.93 µm** — never enough.
Flipping them would need `BONUS ≈ 17`, and at `thr = 0.15` the admitted probs are lower still, which
is why `edgethr15` is *worse* than `edgethr30`.

**This is §16's error repeated**: measuring a conditional on the convenient subset and generalising
it to the inconvenient one. The margin arithmetic that refutes it needed no run at all.

### 22.4 Association by reweighting is done — the plateau is real

| | transferable adj |
|---|---|
| relinkw3 / relinkw6 / edgethr30 | +0.0049 / +0.0056 / **+0.0062** |
| §18 oracle headroom (thick volumes) | **+0.044** |

Four configurations spanning a 4x range of bonus and a 3x range of candidate threshold all land
between +0.005 and +0.006. **~87% of the association headroom is unreachable by reweighting the
existing model.** §18.2 said why: geometry points at the wrong cell and the correct one is the
model's low-confidence case. Closing that gap needs a *better association model*, not a knob —
which is very likely what separates 0.947 from the 0.913 shelf.

### 22.5 Recommendation

> **⚠ ORDERING SUPERSEDED by §23.6.** The read below — that `edgethr30` is our best measured
> config and worth ~+0.003..+0.006 — still stands, and it is submitted. What is superseded is
> the implied next step of ranking more configs in the same space: §23.1 shows that space is
> ~0.006 wide across eight LB reads, while the leader is +0.033 away. Fix the instrument
> (§23.2), then fit models to the 128,883 labelled edges and 151 labelled divisions we own
> rather than tuning the next scalar.

`edgethr30` is our best measured config at **+0.0062 transferable adj / ~+0.0037 net**, marginally
ahead of `relinkw6`. Both sit at the edge of the 0.005 readable floor. `edgethr30` is the better
pick on mechanism — its gain is detection-FN recovery (24 → 14), which is a real structural
improvement, whereas its `div_J` penalty is a 3-sample artifact (§20.3) that should shrink on a
~30-division hidden set.

⚠ Neither is a large win. The honest expectation is **+0.003 .. +0.006**, i.e. 0.914 → ~0.918-0.920.

## 23. The instrument is the bottleneck, not the pipeline — critique and re-ranked plan (2026-08-04)

§22 ended with "closing that gap needs a *better association model*, not a knob" and a
recommendation to keep spending on `edgethr`-class configs. This section argues that read is
half right and its ordering is wrong: the binding constraint is not which knob we turn, it is
that **we have been measuring on 1.65% of the ground truth we already own**, and one of the
terms we retired on that evidence is the largest single block of score still on the table.

Supersedes §22.5's recommendation and §20's "divisions are closed".

### 23.1 The knob space is exhausted and the leader was never in it

Every score from 0.888 → 0.914 came from `BIOHUB_*` env vars over pilkwang's frozen weights.
The measured diameter of that space, over eight LB reads:

| config | LB delta vs base |
|---|---|
| `thr096` / `thr095` | **+0.001** |
| `d4edge` | +0.001 (within noise) |
| `secw` | 0.000 |
| `divgeom` | −0.006 |
| `harmonic` | −0.023 |
| `probe_len20` / `probe_pad` | −0.028 / −0.046 |

**The best knob we have ever found is worth +0.001.** The public field clusters at 0.913
because everyone is searching this same space over the same public artifact — 0.913 is that
artifact's fixed point, not a frontier. The leader is at **0.947, +0.033**, which is 33x the
largest knob effect ever measured here. They are not running a tuned version of our pipeline.

⇒ Continuing to rank knobs is searching a space we have proven, eight times, is ~0.006 wide.

### 23.2 ⚠ Our instrument is 1.65% of the GT we own — and it is a biased 1.65%

The four `test/` volumes are byte-identical to train volumes, which is why §17 could score
locally at all. But they are a tiny and unrepresentative slice of what is on disk:

| | 4 dev volumes | 199 train volumes |
|---|---|---|
| GT edges | 2,127 | **128,883** |
| GT divisions | 3 | **151** |
| median `est_n_nodes` | 32,795 | **17,909** |
| median annotation coverage | 1.76% | **3.56%** |
| volumes with <200 GT edges | **2 of 4** | ~25% |

Denser in cells and thinner in annotation is the worst combination a fixture can have, and
**half of it is statistically unusable**. Every thin-volume pathology in this log is a
consequence: §19.2's "80% of the headline is one edge on a 49-edge volume", §20.3's
3-sample `division_jaccard`, and §22.1's 2x robustness bug — which exists only because a
thin-volume correction had to be invented in the first place.

**The shipped kernel has always supported the fix.** Cell 5:

    _test_dir_override = os.environ.get("BIOHUB_TEST_DIR", "").strip()
    TEST_DIR = Path(_test_dir_override) if _test_dir_override else COMP_DIR / "test"

`list_test_stems()` globs `TEST_DIR/*.zarr` and `read_test_frame` reads
`TEST_DIR/{dataset}.zarr` — both read-only, so a symlink farm into the read-only train mount
works. (Exp 19's Errno 30 was a *write* through such a symlink, not a read; that precedent was
misread as "symlinks don't work here".)

`configs/evalset.json` (`scripts/local/select_evalset.py`) picks 24 volumes stratified on
`est_n_nodes` x embryo, force-including the dev 4:

| | dev | evalset | gain |
|---|---|---|---|
| volumes | 4 | 24 | 6x |
| GT edges | 2,127 | **16,649** | **7.8x** |
| GT divisions | 3 | **25** | **8.3x** |
| thick volumes (≥200 edges) | 2 | **18** | 9x |

Cost: one commit run (~2 h GPU), **zero submissions**. Deliberately *not* enriched for
divisions — a representative draw already yields ~0.76/volume, and enrichment would bias
`division_jaccard`, which is micro-pooled (§23.3). Division *model fitting* uses all 199 GT
graphs offline and needs no inference, so it has no claim on this budget.

The contamination objection does not bite: pilkwang's weights saw all 199 train volumes, but
they also saw the dev 4, so contamination is a constant across any A/B we run. We strictly gain.

**Method note.** §17's own closing line was "build the instrument before the hypothesis". That
lesson was applied once and then not carried far enough — the instrument was built at the size
that happened to be free, not the size the question needed.

### 23.3 ⚠ `division_jaccard` is MICRO-pooled — §20's "divisions are closed" is an instrument conclusion

From `ctrack.eval.biohub.summarise`, confirmed by reading the code:

* `adj_edge_jaccard` — **macro**, `np.mean` of per-volume values (as §17.3 found).
* `division_jaccard` — **micro**, `_jaccard(sum tp, sum fp, sum fn)` over the whole set, with
  the whole term dropped when no volume has a GT division.

So the division term is a single pooled ratio, and today it is:

    div_J = TP/(TP+FP+FN) = 1/(1+5+2) = 0.125   ->  contributes 0.0125
    perfect                                      ->  contributes 0.1000
    HEADROOM                                                     +0.0875

**That is larger than §18's association oracle of +0.077.** §20 concluded "divisions are
closed" after three interventions all drove TP → 0 — but every one of those was measured
against **3** GT divisions, where `div_J` is quantised in steps of ~0.125 and a single TP is
the entire term. What §20 actually established is that *the instrument cannot see divisions*.
That is an argument for fixing the instrument, not for retiring the term. We own **151**
labelled divisions across 199 volumes.

Two further facts make the term softer than §14 assumed:

* Division FP counts only at **matched annotated** nodes. We insert ~350 divisions per 4
  volumes and only **5** became FP — a ~1.4% annotated-landing rate. The FP budget is far
  cheaper than "350 spurious divisions" suggests.
* §15 measured that the shipped gates admit **26.5%** of real divisions, and gates at
  (12/12/14.4) admit **89.4%**. The candidate geometry is not the blocker; selection is.

☠ What stays dead: `divgeom`'s widened gates *with a hand-written ranking* (§16), the
centre-model veto (§20), `divoff`. What is **not** dead, and was never tried: a fitted
selector over the wide candidate pool.

### 23.4 The pattern behind every dead lever

§16.2 diagnosed its own failure exactly right — "the EDA measured `P(geometry|division)`
(recall), not `P(division|geometry)` (precision)" — and then never built the thing that
estimates a precision. It widened gates by hand and ranked 13,495 candidates by a hand-written
formula, `|parent−5.75| + |sister−10.57|`. That is the signature of a problem that needs a
classifier and got a threshold.

Generalising across the whole log: **we hold 128,883 labelled edges and 151 labelled divisions,
and every lever ever attempted has been a hand-set scalar constant tuned against 2,127 edges
and 3 divisions.** `MOTION_RELINK_LEARNED_BONUS`, `DET_THRESHOLD`, `SAFE_DIV_*`,
`DUAL_SEED_EDGE_THRESHOLD`, `SECONDARY_EDGE_WEIGHT`, `MIN_TRACK_LEN` — every one a scalar. Not
one model has been fitted to the supervision sitting on disk.

§22.4's "association by reweighting is done" is true of *one scalar inside a fixed linear
cost*. It is not a statement about the cost function, which has never been learned.

### 23.5 Two facts from the corpus EDA worth having on record

**GT is 100% `dt = 1`.** Across all 199 volumes `edge_dt_max = 1` and `edge_dt_skip_frac = 0`,
no exceptions. Any predicted edge with `dt ≠ 1` is structurally incapable of being a TP.
Gap-closing inserts synthetic midpoint nodes so it *should* be compliant — but this has never
been audited on an actual `submission.csv`.

**`BIOHUB_RUN_OUTPUT_DIAGNOSTICS` is dead in the notebook.** It is read into a variable in
cell 5 that no cell ever references. `build_edgethr_kernel.py` flips it 0 → 1 as one of its two
"changes"; that half was a no-op. (It may still be read by the external support-pack code, so
this is "no effect in the notebook", not "no effect anywhere".)

### 23.6 Ranked plan

Ranked by (impact x probability) / cost. Every item has a local pre-check, and #1 is what makes
the pre-checks trustworthy.

| # | lever | grounding | est. | status |
|---|---|---|---|---|
| **1** | **Widen the eval set** (`BIOHUB_TEST_DIR`) | §23.2 — 1.65% → 12.9% of GT, 2 → 18 thick volumes | enabler | **running** |
| **2** | **Division classifier** on wide candidates | §23.3 — micro headroom +0.0875; 151 labels; §15's 89.4% admission; §16 failed on a *hand* ranking | **+0.02..+0.04** | next |
| 3 | **Learned edge re-ranker** on dumped features | §22.4 plateau is one scalar, not the cost fn; 128,883 labelled edges; `probdump` proves the dump works | +0.01..+0.03 | queued |
| 4 | **Sub-voxel localisation refinement** | §18.2 — swaps have 3.45 µm match error vs 1.72 baseline; ~3.5 µm noise per endpoint against 5.20 µm true motion | +0.005..+0.02 | queued |
| 5 | **Per-volume detection threshold** | §17.3 macro + coverage spans 150x + per-volume k spans 0.564..1.003; global sweep is spent | +0.005..+0.015 | needs #1 |
| 6 | **Cheap audits** (dt≠1 edges, division-FP siting) | §23.5 | 0..+0.01 | queued |
| 7 | A genuinely new association model | §18 oracle +0.077 | blocked: §12's 534 GPU-h vs ~240 h quota | — |

**Honest ceiling statement.** Our oracle on the current node set is 0.9754 local (~0.967
LB-equivalent under §17.1's offset). The leader at 0.947 sits *inside* it, which means they
could be running comparable detections with a much better linker. #3 is the affordable ~20% of
that; the other 80% needs a compute answer we do not have.

⚠ Both #2's +0.0875 and its FP arithmetic assume the hidden test's division rate matches
train's 0.76/volume and that the ~1.4% annotated-landing rate holds. #1 checks both before
anything is committed.

## 24. Division lever measured — the selector wins, but it is not what blocks us (2026-08-04)

Executing §23.6 item #2. Three measurements, all local, zero submissions. The headline is that
§23.3's ranking of this lever was **too optimistic and I am revising it down** before spending
anything on it — but the sub-lever that survives is real, and the mechanism is now known.

### 24.1 A fitted selector beats the hand-written ranking 2.4x on identical data

`scripts/local/division_separability.py`, 199 GT graphs, wide gates (12/14.4/12), 7,027
candidate rows / 135 true second-daughters, 5-fold **grouped by volume** (out-of-fold):

| ranking | AP | TP at top-151 |
|---|---|---|
| random (positive rate) | 0.0192 | — |
| shipped `parent + 0.15·sister` | 0.0889 | **21 / 135** |
| `divgeom` `|p−5.75| + |s−10.57|` (§16) | 0.1228 | 25 / 135 |
| **fitted gradient boosting** | **0.2827** | **50 / 135** |

**3.2x the shipped AP and 2.4x its true positives at the same insertion budget.** §16's
hand-written ranking was barely better than the shipped one; the fit is a different regime.

Two robustness checks, because this is exactly the shape of claim that has burned us before:

* **Construction bias refuted.** Positives are built by designating the *nearer* daughter as
  the already-linked child, forcing `parent_dist ≥ child_dist` for every positive. But **98.7%
  of negatives satisfy it too**, so it is not exploitable. Ablating the biased pair
  (`child_dist`, `asym`) still gives AP 0.2554 and **52/135** at top-151 vs the shipped 21.
* **Permutation importance** says three of the top five features are ones the shipped heuristic
  **does not have at all**:

      parent_dist          +0.2137
      parent_over_spacing  +0.1127   <-- density-normalised; NEW
      sister_dist          +0.0587
      cos_angle            +0.0544   <-- daughters on opposite sides; NEW
      cand_fwd_len         +0.0375   <-- does the candidate continue as a track; NEW

  `parent_over_spacing` being second is the mechanistic reason the gates could never be tuned:
  a fixed 4.66 µm gate means completely different things in a 3,780-node volume and a
  78,600-node one, and annotation coverage spans 150x across the corpus.

⚠ This pool is annotated cells only — **47 candidates/volume against a real run's ~3,374**
(§16: 13,495 over 4 volumes), so ~70x too little competition. Absolute precision here is
meaningless. Only the relative ordering on identical rows transfers.

### 24.2 The realistic pool cannot be reconstructed from `submission.csv`

`scripts/local/division_candidates.py` rebuilds the candidate pool from the predicted graph
(~30,000 nodes/volume — the right density). It found **135 labelled candidates and 0
positives**, which is a property of the reconstruction, not of the data:

`submission.csv` is **post**-insertion. A source that successfully became a division now has
out-degree 2, so it is excluded from the out-degree-1 source set; and an orphan consumed as a
second child now has in-degree 1, so it is excluded from the candidate set. **The successful
insertions are exactly the rows the reconstruction deletes.** Verified directly: of the 3 GT
divisions in `6bba_05db0fb1`, the one we score as TP maps to a predicted node with out-degree
2, and the two FNs map to out-degree-1 nodes.

⇒ Fitting on the real distribution needs the pool dumped **at the call site**, i.e. a `divdump`
kernel in the shape of `probdump`. Do not try to recover it offline.

### 24.3 ⚠ Both division FNs are unreachable by ANY selector — n=2, provisional

The decisive diagnostic. For the two GT divisions we miss, where is the second daughter?

| GT division | second daughter | admissible as a candidate? |
|---|---|---|
| 25000381 (t=24) | **not detected at all** (no matched predicted node) | **no** — detection failure |
| 63001217 (t=62) | detected as pred 50328, but **in-degree 1** | **no** — already linked to another parent |

In both cases there were **0 orphans within 12 µm of the source**. No ranking function over an
empty candidate set can recover either one.

**This is §21.1's pattern again** — there, 72% of swap-correct edges were never candidates, so
no `BONUS` could reach them. Here, 2/2 division FNs are inadmissible, so no selector can reach
them. Candidate admissibility, not ranking, is the binding constraint for the second time.

⚠ **n = 2.** This is precisely the 3-division instrument that §20.3 warned about and that §23.3
criticised §20 for trusting. I am not concluding from it. It is a *mechanism hypothesis* with a
pre-registered test: `evalset_d4edge` carries **25** GT divisions, and re-running this
diagnostic there either confirms "FNs are inadmissible" or refutes it.

### 24.4 Revised EV for the division lever — down, not up

If §24.3 holds at n=25, the selector cannot add TPs; it can only remove FPs. §20.2's arithmetic
then bounds it exactly:

    now         TP 1, FP 5, FN 2  ->  div_J = 0.125  ->  +0.0125 of score
    FP -> 0     TP 1, FP 0, FN 2  ->  div_J = 0.333  ->  +0.0333 of score
    gain                                                 +0.021 local, ~+0.010 LB (§17.1)

So **§23.6's "+0.02..+0.04" for lever #2 is revised to ~+0.021 local / ~+0.010 LB**, and the
"+0.0875 headroom" of §23.3 remains real but is **not reachable by selection** — it is gated by
detection and by linking, which is where §18's association work already points.

What survives, and it is not nothing: §24.1's model is precisely the tool for cutting FPs at
constant TP, and unlike `divveto` (which removed all 5 FPs *and* the TP, §20.1) a ranked
selector can be operated at a budget that keeps the TP.

### 24.5 Audit: no structural leak, and the FP budget confirmed

`scripts/local/audit_submission.py` on `d4edge`:

* **0 edges with `dt ≠ 1`** out of 116,134, in every volume. GT is 100% `dt = 1`, so this class
  of guaranteed-non-TP edge does not exist in our output. Gap-closing is compliant. ☑ closed.
* Max out-degree 2 (no impossible ≥3), all edge lengths ≤ 7.3 µm against a 14.0 µm cap.
* **323 predicted divisions, 6 on annotated nodes = 1.86%** — which reconciles exactly with the
  scorer's TP 1 + FP 5 = 6. Confirms §23.3's ~1.4% estimate and independently validates both
  the audit and `ctrack.eval.biohub`.

### 24.6 The linker is a strict 1-1 matching, so it structurally cannot emit a division

Measured over all 116,134 edges of `d4edge`:

| volume | in-degree histogram | out-degree histogram |
|---|---|---|
| 44b6_0113de3b | `{1: 24,795}` | `{1: 24,717, 2: 39}` |
| 44b6_0b24845f | `{1: 17,347}` | `{1: 17,215, 2: 66}` |
| 6bba_05b6850b | `{1: 6,015}` | `{1: 5,995, 2: 10}` |
| 6bba_05db0fb1 | `{1: 67,977}` | `{1: 67,561, 2: 208}` |

**Every node that has a parent has exactly one.** No node anywhere has in-degree ≥ 2. And the
out-degree-2 counts (39/66/10/208 = **323**) match `safe_divisions_added` exactly, while
`run_stats` shows `division_like_sources` equal to the same figure.

⇒ **Every division in our output comes from the post-hoc safe-division scavenge; not one comes
from the ILP.** `motion_relink_replaced_raw_edges` equals `raw_edges` on every volume (§17.5),
so the relinker discards the ILP's solution — including its division variables `v_i`, which
cell 3's objective explicitly prices at 1.0 — and re-derives a strict bipartite 1-1 assignment.
A Hungarian cannot give a node two children, so divisions can only be added afterwards, and
only from nodes the assignment left orphaned.

**That is the mechanism behind §24.3 case (b).** The daughter of GT division 63001217 was
detected, but the assignment had already consumed it as some other cell's child, giving it
in-degree 1 and permanently disqualifying it as a division candidate. This is not a threshold;
it is the shape of the algorithm.

### 24.7 ⚠ At the shipped gates there is almost no selection left to improve

`run_stats` again:

    safe_division_candidates    39 /  79 /  10 / 227   = 355
    safe_divisions_added        39 /  66 /  10 / 208   = 323   (91%)

**We accept 91% of what the gates admit.** The "ranking" that §16 tried to improve, and that
§24.1 shows a fitted model would improve 2.4x, is barely operating at the shipped configuration
— it is choosing 323 of 355.

This closes the loop on §16 and on my own §23.3 framing:

* A better *ranking* at the shipped gates has ~nothing to rank. To give the model room you must
  widen the gates, which is exactly `divgeom` (355 → 13,495 candidates).
* But widening cannot reach the actual FNs, because §24.3 found **0 orphans within 12 µm** of
  both missed sources. Their daughters are undetected or already linked. Wider gates admit only
  *more wrong candidates*.

⇒ That is precisely why `divgeom` lost 0.006, and it means **§23.3's "+0.0875 headroom" is real
but almost entirely unreachable through the division post-processing.** What remains for lever
#2 is only FP-trimming within the existing 355: drop the 5 FPs while keeping the 1 TP, worth
`div_J 0.125 → 0.333` = **+0.021 local / ~+0.010 LB**, as §24.4 already revised it. The
classifier is still the right tool for that trim — `divveto` failed (§20.1) precisely because it
was an off-switch rather than a ranked budget — but the lever is a trim, not a recovery.

**The real division lever is upstream**: make division-awareness part of the assignment
(let a likely-dividing source claim a second child during matching) instead of scavenging
orphans afterwards. That is a pipeline change, not a knob, and it shares a root cause with
§18's association work — both are blocked by what the 1-1 assignment decides.

## 25. ☠ Sub-voxel localisation refinement is dead — quantisation is 5% of the error (2026-08-04)

§23.6 ranked lever #4 at +0.005..+0.02 on §18.2's observation that swapped cells have 3.45 µm
detection match error against a 1.72 µm baseline, i.e. the linker's distance costs are fed
noisy inputs. Killed locally, for free, before any GPU spend.

**Predicted coordinates are pure integer voxels.** Every `z`, `y`, `x` in `submission.csv` is
integer-valued — there is no sub-voxel refinement anywhere in the pipeline. So the lever looked
open. It is not, because quantisation is not what dominates the error:

| residual (matched pred node − its GT node), n = 2,177 | z | y | x |
|---|---|---|---|
| systematic bias (mean) | −0.639 | +0.007 | −0.239 |
| **random scatter (std)** | **1.707** | **0.943** | **1.039** |

    quantisation RMS (uniform within one voxel)   0.498 um
    observed total RMS                            2.210 um
    variance share                                   5.1%
    RMS after perfectly removing it               2.153 um  =>  2.6% reduction

**A 2.6% reduction in localisation RMS cannot move a linker whose competing assignments differ
by several µm** (§17.5: the relink cost is `motion_dist + 0.05·raw_dist − BONUS·prob`, and
§18.2's swap margins are a median 6.93 µm). The residual is genuine detector scatter — in z it
is 1.05 voxels, i.e. the detector is uncertain by about one voxel, not misrounded by one.

**And the systematic component is not correctable either.** It looks tempting at 39.6% of the
median error, but it is not a global convention offset — per-volume z bias runs −0.07, −0.91,
−1.52, −0.03 µm. Two volumes near zero, two strongly negative. A global correction would help
two and hurt two, and since `adj_edge_jaccard` is a macro mean (§17.3) that is a wash at best.
Measuring the bias per volume requires the GT we do not have at inference.

☠ Lever #4 struck from §23.6. Cost to kill: one local script, zero GPU, zero submissions.

## 26. The widened instrument landed — and it overturns the division read (2026-08-04)

`evalset_d4edge` completed: 24 volumes, 477,264 nodes, 457,727 edges, ~4 h GPU, **zero
submissions**. This is §23.6 lever #1 delivered, and it immediately did the job it was built for.

### 26.1 ☑ Acceptance test PASSED — the instrument is valid

The four dev volumes, scored inside the 24-volume run, reproduce `d4edge` **digit for digit**:

| volume | edge_J | adj | ratio | nrecall | e_tp/fp/fn | d_tp/fp/fn |
|---|---|---|---|---|---|---|
| 44b6_0113de3b | 0.9038 | 0.9046 | −0.009 | 0.9423 | 47/2/3 | 0/0/0 |
| 44b6_0b24845f | 0.8846 | 0.9232 | −0.436 | 0.9804 | 46/3/3 | 0/0/0 |
| 6bba_05b6850b | 0.9709 | 0.9731 | −0.023 | 0.9988 | 835/15/10 | 0/1/0 |
| 6bba_05db0fb1 | 0.8359 | 0.8356 | +0.003 | 0.9910 | 1100/133/83 | 1/4/2 |

Every figure matches §17.1's record exactly. `BIOHUB_TEST_DIR` redirection changes nothing about
how a volume is processed. The instrument can be trusted.

### 26.2 ⚠ The dev-4 instrument was optimistic by +0.027 — and **4x** on divisions

| | dev 4 | **evalset 24** |
|---|---|---|
| edge_jaccard (micro) | 0.8895 | **0.9039** |
| adj_edge_jaccard (macro) | 0.9091 | **0.8916** |
| division_jaccard (micro) | **0.1250** | **0.0308** |
| **SCORE** | **0.9216** | **0.8947** |

    divisions:  TP 2    FP 40    FN 23     (TP+FN = 25, exactly the GT count)

**Division recall is 2/25 = 8%, not the 1/3 the dev set implied, and `div_J` is 4x lower.**
§20.3 warned this statistic was quantised into steps of ~0.125 on 3 samples; it was, and it was
wrong in the optimistic direction. The division term contributes **0.0031** to our score today,
not 0.0125 — so its headroom is **+0.0969**, even larger than §23.3 computed.

⚠ **The absolute level does NOT transfer to the LB and must not be used as a predictor.** Dev-4
local 0.9216 sat *above* LB 0.914; evalset local 0.8947 sits *below* it. Both are contaminated
train volumes, so this is a composition effect, not a contamination one — the evalset includes
hard volumes the dev 4 excluded. Use this instrument for **deltas and decomposition on a large
sample**, which is what it was built for. §17.1's "+0.008 optimistic" calibration is specific to
the dev 4 and does not carry over.

### 26.3 The pre-registered test fired — §24.3 confirmed at the gate, refuted as a lever claim

`division_fn_diagnosis.py` on 25 GT divisions, at two gates:

| bucket | **gate 4.66 (shipped)** | gate 12.0 (wide) |
|---|---|---|
| TP (already a division) | 2 (8%) | 2 |
| **FN-REACHABLE** (orphan daughter in gate) | **0 (0%)** | **9 (36%)** |
| FN-consumed (daughter taken by another parent) | 8 (32%) | 8 |
| FN-daughter (daughter never detected) | 5 (20%) | 5 |
| FN out-of-gate | 10 (40%) | 1 |

Read carefully, because the two halves point opposite ways:

* **§24.3's hypothesis is CONFIRMED at the shipped configuration.** At the 4.66 µm parent gate,
  **0 of 25** division FNs are reachable by better selection. A fitted selector bolted onto the
  shipped gates cannot recover a single division. My n=2 reading was right about *this*.
* **But it is REFUTED as a statement about the lever.** Widening the parent gate to 12 µm makes
  **9 of 25 (36%)** reachable — their daughters are genuine orphans, admissible in principle,
  sitting between 4.66 and 12 µm from the parent. The n=2 dev sample contained zero such cases
  purely by chance.

**The binding constraint is the gate, and it is far too tight** — consistent with §15, which
measured real sister separation at a median 10.57 µm against a gate of 8.5, and with §25's
finding that the geometry was never centred on the truth.

⇒ The combination that has **never been tried** is *wide gates + a fitted selector*. §16
(`divgeom`) tried wide gates + a hand-written ranking and lost 0.006 by flooding 355 → 13,495
candidates. The shipped config runs tight gates, where §24.7's 91%/76.8% acceptance shows the
ranking barely operates. §24.1's model is 2.4x better at ranking on identical rows, and FP
control is exactly what `divgeom` lacked.

### 26.4 Revised EV for lever #2 — third revision, now on 25 samples not 3

    now                         TP  2, FP 40, FN 23  ->  div_J 0.031  ->  +0.0031
    wide gates, recover all 9, FP held at 40    TP 11, FN 14  ->  0.169  ->  +0.0138
    ...and FP trimmed to 20                     TP 11, FN 14  ->  0.244  ->  +0.0213
    ...and FP trimmed to 10                     TP 11, FN 14  ->  0.314  ->  +0.0283

**Realistic band +0.014 .. +0.028 local**, contingent entirely on the selector holding FPs down
while the gate is open — the precise failure mode of `divgeom`. My estimate history, kept
visible: §23.3 said +0.02..+0.04 (arithmetic), §24.4 revised **down** to +0.021 (n=2), this
revises to +0.014..+0.028 (n=25). The n=2 revision was the unreliable one, which is why it
shipped as a hypothesis with a pre-registered test rather than a conclusion.

### 26.5 `FN-consumed` at 32% is the largest single bucket — §24.6 confirmed at scale

Eight of 25 GT divisions (32%) fail because the 1-1 assignment already gave the daughter to
another parent. That was 1 case at n=2; it holds at n=25. **No gate and no selector can reach
these** — the node is not an orphan. Only a division-aware assignment (§24.7) can, which
remains the largest structural lever on this term and shares its root cause with §18's
association work.

Detection accounts for a further 5/25 (20%).

### 26.6 New failure modes that were invisible on 4 volumes

Per-volume `adj` across the 24 spans **0.5713 to 0.9983**. Two volumes are catastrophic and
have no analogue in the dev set:

| volume | edge_J | adj | node recall |
|---|---|---|---|
| **44b6_87bba6c4** | **0.5743** | **0.5713** | 0.9739 |
| **6bba_786893ac** | **0.7245** | **0.7260** | 0.9625 |

Node recall is 97%+ on both, so **detection is fine and linking collapses** — 53 FP/33 FN edges
on one, 253 FP/201 FN on the other. This is §17.4's linking-limited pattern, far more extreme
than anything the dev 4 showed, and it is why macro `adj` (0.8916) sits *below* micro `edge_J`
(0.9039). Since the score is a macro mean (§17.3), **these two volumes cost more than the entire
division term**, and they are a new, unexplored target.

## 27. Oracle at scale: association is 80% of the loss, and §22.4 closed a door it never opened

`link_ceiling.py` on the validated 24-volume set. §18 computed this on 4 volumes and 65 swaps;
this is 24 volumes and **379 swaps**.

    MACRO oracle (detection ceiling)  0.9764
    MACRO actual  edge_J              0.8810
    ASSOCIATION HEADROOM             +0.0954   = 80% of all remaining loss
    detection-FN 288   association-FN 497 (swaps 379)   FP 902
    swaps where the WRONG link is SHORTER than the correct one:  377/379 = 99%

Every §18 conclusion survives the 6x sample and strengthens:

* Association headroom rises **+0.0766 → +0.0954**, and its share of remaining loss **76% → 80%**.
* Detection loss is `1 − 0.9764 = 0.0236` macro. **Association is 4x detection.**
* §18.2's "100% of swaps took a shorter link" was 65/65; at scale it is **377/379 = 99%**. This
  is the most robust single fact we have about our errors.

And it dwarfs everything else on the board: association **+0.0954** against the division term's
realistically-reachable **+0.014..+0.028** (§26.4).

### 27.1 §26.6's collapsed volumes are association failures, not detection failures

| volume | GT edges | oracle | actual | **gap** | detFN | assFN | swaps | FP |
|---|---|---|---|---|---|---|---|---|
| **44b6_87bba6c4** | 149 | **0.9597** | **0.5743** | **+0.3855** | 6 | 27 | 23 | 53 |
| **6bba_786893ac** | 1,395 | 0.9462 | 0.7245 | +0.2217 | 75 | 126 | 105 | 253 |
| 6bba_0e7c0d07 | 198 | 0.9798 | 0.7479 | +0.2319 | 4 | 19 | 15 | 36 |
| 6bba_05db0fb1 | 1,183 | 0.9848 | 0.8359 | +0.1489 | 18 | 65 | 57 | 133 |

`44b6_87bba6c4` is the cleanest case in the corpus: **detection could support 0.96 and we deliver
0.57.** Six detection FNs against 27 association FNs and 53 FPs. Nothing about the detector is
wrong on this volume — the linker simply fails on it. Under a macro mean this one volume costs
`0.3855/24 = 0.016` of score, more than the entire division term contributes today (0.0031).

### 27.2 ⚠ §22.4 concluded "reweighting is done" without testing the value its own arithmetic prescribed

§22.3 derived, correctly, that flipping a swap needs `BONUS × prob > margin`, and that with
newly-admitted candidates at `prob ≤ 0.48` against a median 6.93 µm margin **`BONUS ≈ 17`**
would be required. §22.4 then declared association-by-reweighting finished — on evidence from
`relinkw3` (BONUS 3), `relinkw6` (BONUS 6), `edgethr30` and `edgethr15`.

**The prescribed value was never run.** Concluding a lever is exhausted from settings at 3 and 6
when your own model says 17 is the threshold is not a null result; it is an untested hypothesis.
This is the same shape of error as §16.2 and §22.3 themselves diagnose.

⚠ The counter-argument is real and must ship with it: §18.3's asymmetry is now **~40:1** correct
edges to swaps at this scale, so a bonus large enough to override geometry on 379 swaps can
corrupt far more than 379 currently-correct links. That is *why it needs measuring* rather than
asserting in either direction — and the 24-volume instrument can now measure it for the cost of
one commit run and zero submissions.

### 27.3 Standing plan

| # | lever | headroom | reachable est. | next action |
|---|---|---|---|---|
| **1** | **Association** | **+0.0954 (80%)** | unknown — §22.4's door is not actually closed | `evalset_relinkw17`: test the prescribed BONUS ≈ 17 |
| 2 | Divisions | +0.0969 | **+0.014..+0.028** | `divdump` → fit selector → wide gates + selector (§26.3) |
| 3 | The 1-1 assignment | 32% of div FNs + shares the association root cause | unknown | `evalset_norelink` (built, §24.6) |
| 4 | Detection | +0.0236 | small | — |

⚠ **Re-baseline everything.** `edgethr30`'s "+0.0062 transferable adj" and every other config
delta on record were measured on the dev 4, which §26.2 shows is optimistic by +0.027 overall and
4x on divisions. Those rankings are not trustworthy and should be re-measured on the 24-volume
set before any further submission is spent.

## 28. BONUS=17 run — §22.4's conclusion survives its own missing experiment (2026-08-04)

§27.2 charged that §22.4 declared association-by-reweighting exhausted while never testing the
`BONUS ≈ 17` its own arithmetic prescribed. That charge was methodologically correct. The
experiment has now been run, and **it confirms §22.4 rather than overturning it.**

Three configs, same 24 volumes, same instrument:

| config | edge_J | adj (macro) | div_J | **score** | transferable adj |
|---|---|---|---|---|---|
| `d4edge` | 0.9039 | 0.8916 | 0.0308 | 0.8947 | — |
| **`edgethr30`** | 0.9090 | 0.8972 | 0.0143 | **0.8986** | **+0.0053** |
| `bonus17` (BONUS 17) | 0.9093 | 0.8978 | 0.0141 | 0.8992 | **+0.0004** |

### 28.1 Tripling the bonus does essentially nothing

`bonus17 − edgethr30`: edge_J **+0.0003**, adj +0.0006, score +0.0006, **transferable adj
+0.0004**. Most volumes are bit-identical. And swaps did not fall — **379 → 385**.

The mechanism is visible in `run_stats`, and it is decisive:

    motion_relink_tight_edges     465,311  ->  465,311    IDENTICAL
    motion_relink_relaxed_edges    19,266  ->   19,315    +49

**Raising the learned bonus from 6 to 17 — an 11 µm swing in a cost measured in µm — left the
tight-radius assignment completely unchanged.** Only 49 of 19,266 relaxed edges moved.

Two reasons, both already on record:

* Most links are unambiguous. Median GT displacement is 1.82 µm against ~11.7 µm spacing, so the
  argmin is robust to large cost perturbations. Swaps are 385 of ~478,000 edges = **0.08%**.
* On the contested 0.08%, §21.1 measured that the correct edge is **not a candidate in 72%** of
  swaps, so `learned_prob` is identically 0 and `BONUS × 0 = 0` at any bonus. No value of the
  knob can reach them.

### 28.2 The honest scoring of §27.2

§27.2's criticism was valid as method — concluding a lever is dead from settings below your own
model's threshold *is* an untested hypothesis. But the test came back negative, and **§22.4's
conclusion stands on evidence now instead of extrapolation**: association by reweighting the
shipped model is finished. Between BONUS 1, 3, 6 and 17, and candidate thresholds 0.48, 0.30 and
0.15, every configuration lands within +0.006 of the others.

⚠ Note also that §18.3's feared 40:1 corruption never materialised — edge_J went *up*
marginally, not down. The bonus is not dangerous; it is **inert**. That is a different failure
mode from the one we pre-registered, and it is worth more than the null result itself: the
learned term is not competing with geometry at all.

### 28.3 `edgethr30` re-baselined — and the dev-4 delta was roughly right

`edgethr30` is confirmed as our best config at **+0.0053 transferable adj / +0.0039 net score**,
now measured across **18 thick volumes instead of 2**. Its dev-4 estimate was +0.0062 adj /
+0.0037 net (§22.1).

**So the dev-4 instrument was badly wrong about LEVELS (score 0.9216 vs 0.8947, `div_J` 4x) but
approximately right about this DELTA.** That is a useful and non-obvious calibration: config
deltas on the edge term were transferring reasonably even from 2 thick volumes; the division
term never was. §26.2's warning should be read as level-specific, not delta-specific.

### 28.4 `edgethr30`'s real mechanism, confirmed at scale

The oracle moves with it: detection-FN **288 → 196** (−32%) and the detection ceiling rises
**0.9764 → 0.9833**. §22.2 found this on 4 volumes — opening the candidate gate retains nodes
through the short-track filter — and it holds on 24 (`short_track_nodes_removed` 22,773 →
20,183). `edgethr30` is a *detection-retention* change that was mislabelled an association
change, exactly as §22.2 said.

Association headroom after it: **+0.0937 = 85% of all remaining loss** (was 80%). Fixing
detection made association's share larger, not smaller.

### 28.5 Standing plan, revised

| # | lever | status |
|---|---|---|
| ☠ | association by reweighting (BONUS / candidate threshold) | **DEAD, now on evidence** (§28.1) — do not revisit |
| ☠ | sub-voxel localisation | DEAD (§25) |
| ☠ | division knobs (gates, veto, caps, rate) | DEAD (§16, §20) |
| **1** | **a better association model** | +0.0937 (85%) — compute-blocked (§12), the only large lever left |
| 2 | wide gates + **fitted** division selector | +0.014..+0.028 (§26.4), needs `divdump` |
| 3 | division-aware assignment | 32% of div FNs (§26.5) + shares the association root cause |
| 4 | the collapsed volumes (`44b6_87bba6c4` 0.57 vs oracle 0.96) | unexplained; costs more than the whole division term |

`edgethr30` remains the config to submit. `bonus17` is not worth a submission.

## 29. The collapsed volumes: fixed µm thresholds on a corpus with no fixed scale (2026-08-04)

§27.1 flagged `44b6_87bba6c4` (oracle 0.9597, actual **0.5743**) as the largest unexplained
deficit in the corpus. Investigated locally, free.

### 29.1 What predicts the oracle-actual gap, over the 24 volumes

| predictor | corr with gap | corr with swap-rate | corr with FP-rate |
|---|---|---|---|
| **annotation density** | **−0.577** | −0.510 | −0.546 |
| **disp_p90** (motion) | **+0.460** | +0.516 | +0.459 |
| est_n_nodes (cell count) | +0.451 | +0.364 | +0.428 |

⚠ These are moderate — `disp_p90` gives R² ≈ 0.21 — on n = 24. Directionally consistent across
three different response variables, which is what makes them worth acting on, but not a law.

### 29.2 The two worst volumes have **different** mechanisms

|  | gap | oracle | detFN | assFN | swap | FP | GT edges | disp_p90 | density | est_n |
|---|---|---|---|---|---|---|---|---|---|---|
| `44b6_87bba6c4` | 0.3855 | **0.9597** | **6** | 27 | 23 | 53 | 149 | **8.00** | 0.0054 | 28,102 |
| `6bba_786893ac` | 0.2217 | 0.9462 | **75** | 126 | 105 | 253 | 1,395 | 3.37 | 0.0248 | 58,979 |
| *corpus-of-24 median* | | | | | | | | *3.55* | *0.0358* | *18,712* |

My first read — "the collapsed volumes are fast-motion" — is **wrong as a general claim** and I am
recording it because it was nearly written up. `6bba_786893ac`, the second worst, has
*below-median* motion (3.37). Its problem is detection: 75 detection-FNs and the lowest oracle in
the set, in the densest volume (3.2x the median cell count).

`44b6_87bba6c4` is the fast-motion case: detection is essentially fine (6 detFN, oracle 0.96) and
the linker destroys it, at **0.36 FP per GT edge — twice the next worst volume in the corpus.**
Its `disp_p90` of 8.00 µm sits *above* `MOTION_RELINK_TIGHT_UM = 6.0`, so more than 10% of its
true links fall outside the tight pass, into the contested 6–10 µm relaxed regime.

**The common thread is not speed, it is scale**: both are large volumes with below-median
annotation coverage. The metric counts an edge as FP when *either* endpoint matched an annotated
node, so in a big, thinly-annotated volume every annotated cell sits among thousands of
unannotated distractors and each mis-link lands "in the dark" as a pure FP. That is exactly what
the FP-rate correlations say (−0.546 with density, +0.428 with cell count).

### 29.3 The unifying diagnosis — and a lever that is not on the dead list

**The pipeline uses fixed µm thresholds throughout** — `MOTION_RELINK_TIGHT_UM` 6.0,
`RELAXED_UM` 10.0, `SAFE_DIV_MAX_UM` 4.66, `OUTPUT_EDGE_MAX_UM` 14.0 — on a corpus where the
natural scale is not fixed at all: annotation coverage spans **150x** (0.00127 → 0.202), cell
count **21x** (3,780 → 78,600), and `disp_p90` roughly **5x** (1.72 → 8.00 µm).

This is the same finding twice, arrived at independently:

* §24.1's permutation importance ranked **`parent_over_spacing` second** of fourteen features for
  division selection — density-normalised distance beat almost everything.
* §29.1 shows the linking gap tracks motion and density, not absolute distance.

**And the pipeline's own author already conceded the point in one place**: gap-closing has
`GAP_DENSITY_ADAPTIVE = 1` with a reference distance, gain, and neighbour count — a full
density-adaptive scheme. The motion relinker, which §17.5 shows rewrites **100%** of edges, has
no such mechanism. That asymmetry looks like an oversight rather than a decision.

⇒ **Scale-adaptive relink radii** is a real, measured, untried lever, and it is not on the dead
list. ⚠ It is *not* a single env var: widening `MOTION_RELINK_TIGHT_UM` globally would help
`87bba6c4` and hurt the dense volumes, and with `adj` a macro mean (§17.3) that trade is likely a
wash. It needs per-volume adaptation keyed to something measurable at inference (median
nearest-neighbour spacing, or observed frame-to-frame displacement), which is a code change of
the same shape as `GAP_DENSITY_ADAPTIVE`.

**Cheap falsification before building it:** run one commit-run pair with
`MOTION_RELINK_TIGHT_UM` at 6.0 vs ~9.0 on the 24-volume set. If `87bba6c4` improves markedly
while the dense volumes degrade, adaptation is worth building; if `87bba6c4` does not move, the
radius is not the constraint and this dies for one run.

## 30. ⚠ `edgethr30` scored 0.913 — FLAT. Both instruments over-predicted by ~0.004 (2026-08-05)

| config | local (dev 4) | local (evalset 24) | **public LB** |
|---|---|---|---|
| `d4edge` | 0.9216 | 0.8947 | **0.913** |
| `thr096` / `thr095` | — | — | **0.914** ← still our best |
| **`edgethr30`** | 0.9253 (+0.0037 net) | 0.8986 (**+0.0039 net**) | **0.913 (+0.000)** |

§28.5 said "`edgethr30` remains the config to submit" and §22.5 predicted 0.914 → ~0.918-0.920.
**Both were wrong.** The board says flat against `d4edge` and **0.001 below** our standing best.
The LB quantises to 0.001, so a real +0.004 would have read 0.917; it read 0.913. The true delta
is ~0.000.

### 30.1 The widened instrument did NOT fix delta prediction for this class of change

This is the uncomfortable part. §26 established that the dev-4 instrument was wrong about levels
and §28.3 noted it was *approximately right about this delta* (+0.0062 dev-4 vs +0.0053 on 24
volumes) — I treated that agreement as corroboration. **Both agreed with each other and both
disagreed with the board.** Two instruments concurring is not evidence when they share a bias.

The shared bias is contamination, and it is specific to what `edgethr30` does. The pilkwang
weights were trained on **all 199 train volumes**, and every local volume we can score — dev 4 or
evalset 24 — is a train volume. `edgethr30` lowers `DUAL_SEED_EDGE_THRESHOLD` 0.48 → 0.30, i.e. it
admits **lower-confidence** edges. On volumes the model memorised, a low-confidence edge is far
more likely to be correct than on genuinely unseen data. So the measured gain is an artifact of
scoring a memorising model on its own training set, and it evaporates on the hidden test.

That also explains the mechanism cleanly: §28.4 attributed the gain to detection-FN recovery
(288 → 196) via node retention. Those extra retained nodes are real *on train volumes*. On hidden
data the same threshold admits noise instead.

### 30.2 The hard structural limit this exposes

**There is no uncontaminated local data for this model.** The weights saw all 199 train volumes;
the only held-out data in existence is the hidden test set, reachable at 5 submissions/day and
~4-6 h per read. So:

| question type | local instrument | trustworthy? |
|---|---|---|
| structural / decomposition (oracle ceiling, division FN buckets, in-degree, dt=1 audit, gap predictors) | evalset 24 | **yes** — these do not depend on the model being calibrated |
| deltas from changes that do **not** touch model confidence (geometry, gates, radii, post-proc caps) | evalset 24 | probably — untested |
| deltas from changes that **exploit model confidence** (`DUAL_SEED_EDGE_THRESHOLD`, `DET_THRESHOLD`, ensemble weights) | evalset 24 | **NO — measured to over-predict by ~0.004** |

⚠ This retroactively re-frames §10.1's CV inversion, which was recorded as a mystery. It is the
same effect: local validation on train volumes flatters exactly the knobs that trade confidence
for recall.

⇒ §23's core critique — that we were measuring on 1.65% of our GT — was correct and the fix was
worth building; §26's structural findings all stand. But **widening the eval set fixed the sample
size, not the contamination.** I over-claimed in §26/§28 by treating the widened instrument as
generally predictive. It is not, for this class of change.

### 30.3 What this does to the standing plan

* **Our best remains 0.914** (`thr096`/`thr095`), unchanged since 08-03. `edgethr30`, `bonus17`,
  `relinkw3/6`, `edgethr15` are all now measured or inferred flat-to-negative.
* §28's conclusion that association-by-reweighting is dead **strengthens** — `edgethr30` was the
  last member of that family with a positive local reading, and the board says it is flat too.
* The remaining levers (§29's scale-adaptive radii, §26.3's wide-gate fitted selector, §26.5's
  division-aware assignment) are all **geometry/structure** changes, not confidence changes, so
  they sit in the middle row of the table above — plausibly measurable locally, but that
  assumption is now itself untested and should be stated whenever it is relied on.
* ⚠ Before spending more submissions, note the base rate: **11 LB reads, best +0.001 over the
  0.913 shelf.** Nothing in the knob space has moved this board.

## 31. ★ The relink RADIUS is the binding constraint on association (2026-08-05)

Two free local experiments. The first died; the second is the strongest result since §18.

### 31.1 ☠ Collective motion does not explain the swaps

Hypothesis: embryonic cells move as tissue, `MOTION_RELINK_VELOCITY_WEIGHT` failed because it uses
the cell's *own* history (which a swap corrupts), and a *neighbourhood* flow field would be robust.
`scripts/local/tissue_flow.py`, flow = median displacement of the 12 nearest linked cells:

| criterion | picks the CORRECT successor |
|---|---|
| raw distance | **2 / 379 = 0.5%** |
| flow-compensated distance | **5 / 379 = 1.3%** |

Local flow magnitude is a median **1.22 µm** against a ~7.5 µm discrepancy — far too small to
matter. ☠ Dead, for free, no GPU.

### 31.2 ★ But it exposed the real mechanism: the correct successor is OUT OF RANGE

The distances in that null result are the finding:

    distance from source to the CORRECT successor (predicted coords), n=379 swaps
        p10   6.19    p25   7.67    p50   9.37    p75  11.47    p90  13.75

        beyond MOTION_RELINK_TIGHT_UM   (6.0):  343/379 = 90.5%
        beyond MOTION_RELINK_RELAXED_UM (10.0): 156/379 = 41.2%   <-- CANNOT BE FORMED
        beyond OUTPUT_EDGE_MAX_UM       (14.0):  30/379 =  7.9%

        the link we actually TOOK: median 1.86 um, 0.0% beyond TIGHT

**These are not ambiguous decisions.** The relinker takes a node 1.9 µm away while the correct
answer sits 9.4 µm away — and for **41% of swaps the correct edge is beyond the relinker's maximum
radius, so no cost function, weight or bonus can ever produce it.** Only 0.5% of swaps are cases
where the correct successor was even the nearer option.

**This is the fourth independent appearance of the same theme.** §21.1: 72% of swap-correct edges
are not candidates. §24.3/§26.3: 0/25 divisions reachable at the shipped gate. §28: the bonus is
inert because `prob ≡ 0` on unreachable edges. Now: 41% of correct successors are outside the
relink radius. **Every association lever we have tried failed because the correct answer was never
in the consideration set — not because the ranking was wrong.**

### 31.3 Why the distances are so large — and it is only half real motion

    GT displacement of these same edges:   median 4.82 um   (corpus median 1.82 => 2.7x)
    predicted distance for the same edges: median 9.37 um

The cells genuinely move 2.7x faster than typical, **and our detections roughly double the apparent
displacement on top of that** — consistent with §18.2's 3.45 µm match error on swaps against a 1.72
µm baseline. Note only **7.1%** of these edges have a true GT displacement beyond 10 µm, while
**41.2%** of the predicted distances do. So the radius is not mainly too small for real biology; it
is too small for real biology *plus our own localisation error*.

⚠ §25 killed sub-voxel refinement and that stands — the error is genuine detector scatter, not
quantisation, so it cannot be refined away. The radius has to absorb it instead.

### 31.4 The lever, and why it may be nearly free

Raise `MOTION_RELINK_TIGHT_UM` (6.0) and especially `MOTION_RELINK_RELAXED_UM` (10.0). To reach the
median correct successor needs TIGHT ≈ 9.4; to reach p75, ≈ 11.5; RELAXED at 14.0 would make ~92%
of correct successors *formable at all*.

**The asymmetry that makes this promising:** the cost is distance-dominated, and 100% of the wrong
links we take are already inside 6 µm. Adding *far* candidates cannot outcompete a close correct
option on any cell that already has one — so widening should be close to inert where we are right,
and only bite where every close option is wrong, which is exactly the swap population.

⚠ Honest counterweights, both real:
* This is necessary but not sufficient. Making the edge *formable* does not make it *chosen* —
  §28 showed the learned term cannot tilt a distance-dominated cost, and at 9.4 µm vs 1.9 µm the
  geometry argues loudly for the wrong answer. Widening alone may change nothing.
* §29 found the gap correlates with density, and wider radii cost most in dense volumes. Under a
  macro mean, hurting the dense majority to help the fast minority can easily net negative.
* It **cannot be simulated offline** — widening ADDS edges, and `postproc_sim.py` only models
  removals. This one needs a kernel run.

⇒ This supersedes §29's radius test as the highest-value experiment, and sharpens it: sweep
`RELAXED_UM` 10 → 14 (the hard ceiling, 41% → 8% unformable) rather than nudging TIGHT to 9.
It is pure geometry, so it sits in §30.2's trustworthy row.

### 31.5 A new instrument: post-processing optimised offline, exactly, in minutes

`scripts/local/postproc_sim.py`. `submission.csv` is the complete final graph, so **any
post-processing rule that only REMOVES nodes or edges can be applied to it and scored exactly with
the real metric on all 24 volumes** — no kernel, no GPU, no submission. A class of experiment that
has been costing ~4 h each is now a ~10 min local sweep.

| variant | edge_J | adj | div_J | SCORE | delta |
|---|---|---|---|---|---|
| **BASELINE (as submitted)** | 0.9039 | 0.8916 | 0.0308 | **0.8947** | — |
| minlen ≥ 7 | 0.9017 | 0.8900 | 0.0308 | 0.8931 | −0.0015 |
| minlen ≥ 8 | 0.8992 | 0.8895 | 0.0308 | 0.8926 | −0.0020 |
| minlen ≥ 10 | 0.8950 | 0.8886 | 0.0308 | 0.8917 | −0.0030 |
| minlen ≥ 12 | 0.8846 | 0.8776 | 0.0303 | 0.8806 | −0.0140 |
| minlen ≥ 15 | 0.8708 | 0.8680 | 0.0303 | 0.8711 | −0.0236 |
| minlen ≥ 20 | 0.8236 | 0.8325 | 0.0154 | 0.8340 | −0.0606 |
| maxedge ≤ 10.0 µm | 0.9039 | 0.8916 | 0.0308 | 0.8947 | **+0.0000** |
| maxedge ≤ 8.0 µm | 0.9039 | 0.8916 | 0.0308 | 0.8947 | **+0.0000** |
| maxedge ≤ 6.0 µm | 0.9034 | 0.8912 | 0.0308 | 0.8943 | −0.0004 |
| **divtrim ORACLE (uses GT)** | 0.9042 | 0.8915 | **0.0800** | 0.8995 | **+0.0049** |

**☠ `MIN_TRACK_LEN` is monotonically bad above 6.** No shallow optimum hides between the shipped 6
and exp 9's 20 — every step down the ladder loses. Exp 9's LB loss of −0.027 at L=20 corresponds to
−0.0606 locally, a ~2.2x scaling consistent with §26.2. (Only L ≥ 7 is testable offline; whether
L < 6 would be better cannot be answered by removal.) Killed for free.

**`maxedge` is inert**, exactly as predicted — the shipped 14.0 µm cap never binds and no scoring
edge exceeds 8 µm. Included as a control, and it behaved as a control should, which is mild
evidence the simulator is faithful.

**★ `divtrim` puts a hard ceiling on lever #2's FP half: +0.0049.** This deletes *exactly* the
false-positive divisions using ground truth, so **no real selector can beat it.** §24.4 estimated
this branch at +0.021 from the dev-4 numbers and §26.4 quoted +0.014..+0.028 for the combined
lever; the FP-removal half alone is now measured at **+0.0049 maximum**, because with FN = 23 the
Jaccard is dominated by misses, not false alarms. Any realistic classifier lands well under that.

⇒ **The division programme is worth pursuing only through the RECALL half** (§26.3's 9/25 reachable
at a 12 µm gate). Trimming false positives, however cleverly, is capped at half a point.

### 31.6 ★ Reading the relinker reverses §31.4's experiment — the TIGHT pass pre-commits

Before spending a run, `motion_relink_edges` was read. It is a **two-stage greedy** assignment,
not one global one:

    for pass_name, gate_um in (("tight", TIGHT_UM), ("relaxed", RELAXED_UM)):
        pass_sources = [s for s in source_ids if s in unmatched_sources]
        pass_targets = [t for t in target_ids if t in unmatched_targets]
        matches = assign_pass(pass_sources, pass_targets, gate_um)

and `assign_pass` excludes any pair with `raw > gate_um` before `linear_sum_assignment`.

**So a source whose correct successor is 9.37 µm away (§31.2's median) sees only the wrong node
1.86 µm away inside the 6 µm tight gate, is matched to it, and is struck from `unmatched_sources`
before the relaxed pass ever runs.** The tight pass pre-commits.

⇒ **§31.4's proposed `RELAXED_UM` 10 → 14 sweep would have measured nothing**, because these
sources never reach the relaxed pass. That experiment is abandoned unrun; it was designed from the
distance histogram without checking the control flow, which is §16's error in a new costume.

**The lever is `TIGHT_UM`.** At 6.0 it excludes **90.5%** of correct successors from the only
assignment that matters. Raising it puts the far-correct and near-wrong targets into the *same*
Hungarian, where they can be weighed jointly — and where the wrong target's own true parent
(usually an unannotated cell, invisible to our diagnostics, which is why §31's "93.4% unowned"
split was uninformative) can claim it and push our source onto its real successor. **That
competition does not happen today at any setting of any other knob.**

Two arms launched, each ONE change against `d4edge`, to get a curve not a point:

| arm | `MOTION_RELINK_TIGHT_UM` | covers |
|---|---|---|
| `tight9` | 6.0 → **9.0** | the p50 correct-successor distance (9.37) |
| `tight12` | 6.0 → **12.0** | the p75 (11.47) |

**Pre-registered read:** swaps down + adj up ⇒ pre-commitment was the mechanism, and the real fix
is a single-stage assignment over the full radius. Swaps down but FP up / adj flat ⇒ widening
trades one error for another and the cost, not the gate, is the problem. Nothing moves ⇒ the cost
is so distance-dominated that even in a joint assignment the near-wrong target wins, and
association is closed to everything but a new model.

⚠ Genuine two-sided risk: at 12 µm in a volume with ~11.7 µm spacing every cell has several
admissible targets. §29 found the oracle gap correlates with density and wide gates cost most
exactly there; under a macro mean, hurting the dense majority to help the fast minority can net
negative.

## 32. ★★ The recall cliff sits EXACTLY at the relink gates (2026-08-05)

`scripts/local/error_anatomy.py`, 16,361 association-testable GT edges (both endpoints detected,
so detection failures are excluded and only linking is on trial).

### 32.1 ★★ P(correct link) vs GT displacement — the cliff is the gate

| GT displacement (µm) | edges | % of all | **P(correct)** | lost |
|---|---|---|---|---|
| 0–1 | 4,731 | 28.9% | 98.9% | 50 |
| 1–2 | 4,554 | 27.8% | 98.2% | 82 |
| 2–3 | 3,607 | 22.0% | 98.5% | 53 |
| 3–4 | 1,583 | 9.7% | 97.9% | 34 |
| 4–5 | 868 | 5.3% | 96.4% | 31 |
| 5–6 | 442 | 2.7% | **91.2%** | 39 |
| **6–8** | 374 | 2.3% | **76.7%** | 87 |
| **8–10** | 154 | 0.9% | **51.3%** | 75 |
| **>10** | 48 | 0.3% | **4.2%** | 46 |

**Read the boundaries.** Recall is a flat 98% up to 4 µm, holds 91% to 6 µm, then falls off a
cliff — and the cliff edges land on **6.0 µm = `MOTION_RELINK_TIGHT_UM`** and **10.0 µm =
`MOTION_RELINK_RELAXED_UM`**. Beyond the relaxed gate we get **4.2%** of edges right, which is
consistent with chance.

**This is the cleanest evidence in the whole project that our association performance is set by
the GATES, not by the cost function.** It independently confirms §31.6's control-flow reading:
the tight pass pre-commits at 6 µm, and that is precisely where recall breaks. Four prior
findings said admissibility was the constraint (§21.1, §24.3, §28, §31.2); this one shows the
constraint's *location* directly, and it coincides with a configured constant to the µm.

**The prize, bounded:** edges with displacement ≥ 4 µm are only **11.5%** of testable edges but
carry **55.9%** of all association loss (278 of 497). Recovering just the three post-cliff bins to
the 91% of the 5–6 µm bin would return ~114 edges ≈ 23% of association loss ≈ **+0.021**;
recovering them fully would be ~+0.037. That is the realistic band for `tight9`/`tight12`.

⚠ Necessary, not sufficient — §31.6's caveat stands. Admitting the edge lets the Hungarian weigh
it; it does not guarantee the Hungarian picks it against a 5x-closer wrong option.

### 32.2 Errors are BURSTY in time — median VMR 1.93 against a Poisson 1.0

| volume | swaps | mean/frame | **VMR** | |
|---|---|---|---|---|
| 6bba_786893ac | 126 | 1.260 | 1.96 | BURSTY |
| 6bba_05db0fb1 | 65 | 0.650 | 1.89 | BURSTY |
| 6bba_d3da753b | 43 | 0.430 | **4.29** | BURSTY |
| 6bba_debd7bfa | 34 | 0.340 | 2.60 | BURSTY |
| 6bba_2819ca14 | 23 | 0.230 | 3.64 | BURSTY |
| 44b6_87bba6c4 | 27 | 0.270 | 0.95 | uniform |

Errors **cluster in time**. Something episodic drives them — a division wave, a fast tissue
movement phase, an imaging artifact. This matters because it means a **per-FRAME** adaptive gate
is well-targeted: widen only on the frames that need it, paying the density cost (§29, §31.6)
only where it buys something, instead of globally.

Note `44b6_87bba6c4` is uniform — its problem is a property of the whole volume (fast throughout,
§29.2), not an episode. Two different regimes again.

### 32.3 Localisation error IS systematically aligned with motion — but it is second-order

A motion-blur test: project the residual `pred − gt` of each source onto its direction of travel.

| | along motion | perpendicular | n |
|---|---|---|---|
| correct links | **+0.469 µm** | 1.382 µm | 14,510 |
| mis-linked | **−0.866 µm** | **2.546 µm** | 488 |

Overall mean projection **+0.426 µm, z = +38.9** — overwhelmingly systematic, not scatter. §25
ruled out *quantisation* as a fixable component; it never tested for a motion-aligned one, and
this is one.

⚠ But do not over-read it. The sign **flips** between correct (+0.47) and mis-linked (−0.87), so a
single global shift cannot fix both, and the magnitudes (~0.4–0.9 µm) are small against the
multi-µm gate effect in §32.1. Perpendicular scatter on mis-linked cells is ~1.8x that of correct
ones, which is ordinary noise, not bias. Filed as real, understood, and **not currently
actionable** — second-order behind the gates.

### 32.4 ☠ The relaxed-pass fraction is too weak to key an adaptive policy on

Tested as an *inference-time* hardness proxy (needs no GT), from `run_stats`:
`relaxed / (tight + relaxed)` relinked edges.

    corr(relaxed_frac, oracle gap) = +0.317      corr(relaxed_frac, swap rate) = +0.324
    top half by relaxed%: mean gap 0.1132        bottom half: 0.0776

Only ~1.5x separation, and it **misses the second-worst volume entirely** — `6bba_786893ac` has
gap 0.2217 on a relaxed fraction of just 2.83%, below the median. ☠ Not a usable detector on its
own. If a per-frame adaptive gate (§32.2) is built, it needs a better trigger than this.

## 33. ☠☠ The gate is PROTECTION, not obstruction — §32.1's causality was backwards (2026-08-05)

`tight9` and `tight12`, both scored on the 24-volume set against `evalset_d4edge`.

| `MOTION_RELINK_TIGHT_UM` | edge_J | adj | div_J | **score** | Δ | transferable adj |
|---|---|---|---|---|---|---|
| **6.0 (shipped)** | 0.9039 | 0.8916 | 0.0308 | **0.8947** | — | — |
| 9.0 (`tight9`) | 0.8805 | 0.8700 | 0.0370 | 0.8737 | **−0.0209** | **−0.0256** |
| 12.0 (`tight12`) | 0.8121 | 0.7692 | 0.0000 | **0.7692** | **−0.1255** | **−0.0969** |

**Monotonically catastrophic.** Not flat, not marginal — widening the gate destroys the score, and
the damage accelerates.

### 33.1 The mechanism, measured

| | TIGHT 6 | TIGHT 9 |
|---|---|---|
| swaps | 379 | **557 (+47%)** |
| edge FP | 902 | **1,221 (+35%)** |
| association-FN | 497 | 645 |
| detection-FN | 288 | 269 |
| relaxed-pass edges | 16,651 (3.57%) | 665 (0.14%) |

The intervention did exactly what it was designed to do mechanically — the tight pass absorbed the
6–9 µm range and the relaxed pass emptied out (at TIGHT 12 it is **0.00%**, the second stage is
dead). And the result was **47% more swaps.** Admitting the far-but-correct target also admits
many more far-but-wrong ones, and the Hungarian takes the wrong ones more often than the right.

### 33.2 ⚠ I read §32.1's coincidence backwards

§32.1 found the recall cliff lands exactly on 6.0 µm and 10.0 µm and I read it causally: *the gate
cuts off good links, so widen it.* The causality runs the other way. **The gate sits at 6.0 because
that is where the assignment problem stops being solvable by geometry**, and the author tuned it
there. Past that distance the correct target is not merely far — it is indistinguishable from
several wrong ones, and a bigger candidate set makes the guess worse, not better.

The observation in §32.1 stands as a fact; only its causal interpretation was wrong. Recall does
collapse past 6 µm, and edges ≥4 µm do carry 55.9% of association loss. But that loss is **not
recoverable by admission** — which is the specific thing this experiment was built to find out, and
it found it out for ~8 GPU-h and zero submissions.

**Method note.** §31.6 caught one bad experiment by reading the control flow before running it. This
one could not have been caught that way — the mechanism reasoning was sound and the code confirmed
it. The failure was in the *sign* of an inference from a correlation, which only a run could settle.
That is the correct use of a run.

### 33.3 ☠ Association is now closed to knobs entirely

Every dimension of the relink cost and its candidate set has been swept and the shipped value wins
or ties in every one:

| dimension | values tested | best |
|---|---|---|
| `MOTION_RELINK_LEARNED_BONUS` | 1, 3, 6, 17 | all within +0.006 (§28) |
| `DUAL_SEED_EDGE_THRESHOLD` | 0.48, 0.30, 0.15 | shipped, on the board (§30) |
| `MOTION_RELINK_VELOCITY_WEIGHT` | 0.5, 0.75, 1.0, 1.25 | flips 0/65 (§18.2) |
| **`MOTION_RELINK_TIGHT_UM`** | **6, 9, 12** | **shipped, by a wide margin** |
| neighbourhood flow term | — | 1.3% vs 0.5% (§31.1) |

⇒ **The +0.0954 association headroom is real and is reachable by none of them.** §22.4's conclusion
now holds across ranking *and* admission, which were the only two families of knob available. What
remains is a better association model, which §12 measured as compute-infeasible (534 GPU-h against
a ~240 h quota).

This also closes §23.1's argument empirically: the public pipeline is at a **well-tuned local
optimum**, we have now probed it along five independent axes, and the 0.913–0.914 shelf *is* that
optimum. The leader's +0.033 is not hiding in this configuration space.

### 33.4 Standing

* Best LB **0.914** (`thr096`/`thr095`), unchanged. Nothing today moved it.
* ☠ Newly closed on evidence: gate widening (§33), collective motion (§31.1), `MIN_TRACK_LEN`
  above 6 (§31.5), division FP-trimming capped at +0.0049 (§31.5).
* Still open, in order of measured size: a new association model (+0.0954, compute-blocked);
  wide-gate **division recall** via a fitted selector (§26.3, 9/25 reachable — note this is
  *division* admission, which §33 did NOT test and which failed for a different reason in §16);
  division-aware assignment (§26.5, 32% of division FNs).

## 34. Is the association headroom INFORMATION-blocked or compute-blocked? (2026-08-06)

§33 closed association to every knob, leaving "train a better association model" — priced by §12
at 534 GPU-h against a ~240 h quota. Before treating that as the plan, a prior question that is
free to ask: **do the images even contain information that distinguishes the correct successor
from the one we wrongly take?** If not, the +0.0954 is unreachable rather than expensive.

### 34.1 ⚠ The first version of this test was CONFOUNDED, and I nearly published it

Compared normalised cross-correlation of ~6×6×8 µm patches: source vs correct target, source vs
the taken distractor. Result: the correct target was **less** similar, 37.1% at **z = −4.8**.

I was one step from writing "appearance is anti-informative ⇒ the headroom is information-blocked."
**That would have been wrong.** The distractor sits a median **1.86 µm** from the source and the
correct target **9.37 µm** (§31.2) — so the distractor's patch physically *overlaps* the source's
and the correct target's does not. NCC was measuring spatial overlap, not cell identity.

Confound confirmed directly: `corr(patch NCC, distance between the two cells) = −0.171`.

This is §32.1's error again — reading a measurement whose confound had not been controlled — and
it is recorded rather than quietly fixed because it is the third instance of the same failure mode
in three days.

### 34.2 The corrected test: no detectable signal, but it vetoes nothing

Cell-level descriptors with no spatial term (peak, integrated intensity, local contrast, radial
falloff), compared as vectors:

    correct target's descriptor is closer to the source:  52.8%   z = +0.8   (n = 235)

Chance. Four hand-made scalars do not separate the correct successor from the distractor.

⚠ **Three reasons this is weak evidence and must not be used as a veto:**

1. Four hand-designed scalars failing says little about what a trained network would find. The
   whole premise of learned association is that such structure is not hand-specifiable.
2. **The test is underpowered.** At n = 235 the standard error is 3.3 points, so a real 55% effect
   would land at z ≈ 1.5 — indistinguishable from this result. It cannot rule out a modest signal.
3. It uses GT-matched identities, so it is an upper bound on the *convenience* of the signal, not
   on its existence.

**What it does change:** I went looking for encouragement for option 1 and did not find it. The
informational premise behind "train a better association model" is **unsupported — neither
confirmed nor refuted.** That is a materially weaker position than "compute-blocked but sound",
which is how §33.4 left it.

### 34.3 Standing on the three remaining options, with evidence quality stated

| option | ceiling | ceiling evidence | capture evidence | precedent |
|---|---|---|---|---|
| **1. new association model** | **+0.0954** | **strong** — oracle bound, 16,649 edges | **none**; architecturally discarded (§17.5/§24.6), informationally unsupported (§34.2) | `edgeft` flat loss; BONUS inert 1→17 |
| 2. division selector (recall) | +0.0118 | moderate — 9/25, n=25 | **weak** — fitted on a pool 70× too sparse (§24.1), real pool not reconstructable (§24.2) | **§16 lost 0.006 on the board** |
| 3. division-aware assignment | +0.0123 | measured — 8/25 | **none** — requires rewriting the assignment | **§33: a 3 µm change cost 0.021** |

**The asymmetry is the honest answer.** Options 2 and 3 together, both succeeding perfectly, total
~+0.024 — still short of the leader's +0.033 — and both belong to intervention families with
losing board precedents. Option 1 is the only one sized to close the gap and is the one with the
least support.

⇒ **Cheapest next fact: `evalset_norelink`** (built, unrun). The relinker discards 100% of learned
edges, so a better association model has nowhere to plug in unless the ILP path is viable on its
own. That run answers a prerequisite for option 1 for ~4 GPU-h and zero submissions, and it is
worth knowing *before* anyone contemplates the 534 GPU-h.

## 35. PLAN — a new association model (2026-08-06)

### 35.0 ⚠ Correction: association is NOT compute-blocked

§33.4, §34.3 and several commits describe option 1 as "compute-blocked, 534 GPU-h against a ~240 h
quota." **That figure is for the wrong thing.** Exp 20 measured the **TemporalUNet detector**
from scratch (window2 5.34 h/epoch, window3 15.13 h/epoch). Association is separable:

    exp 21 (edgeft): edge head only, 10,000 steps, 194 min at 0.69 s/iter

and a cost model over *precomputed node positions* is minutes on the M1. We never need to retrain
the detector — we keep pilkwang's detections and replace only the linking decision. The compute
wall is real for detection and does not apply here.

### 35.1 What the model must do — and why "a better edge model" is not enough

`motion_relink_replaced_raw_edges == raw_edges` on every volume (§17.5, §24.6): the relinker
**discards 100% of learned edges** and re-derives them from

    cost = motion_dist + 0.05*raw_dist - BONUS * learned_prob

§28 proved the `BONUS` channel is inert across 1/3/6/17. So a better *edge probability* has no way
in. **The model must replace the cost function itself and own the assignment**, not advise it.

Injection point: `motion_relink_edges` → `assign_pass` in cell 11. One function.

### 35.2 ★ The enabler: a LOCAL REPLAY of the relinker

`assign_pass` consumes only node positions (plus the inert prob), so it can be reimplemented
locally, run over the node set in a downloaded `submission.csv`, and scored exactly with the real
metric on 24 volumes. **That turns every model evaluation from ~4 GPU-h into seconds** — the same
trick as §31.5's `postproc_sim`, applied to the linking stage.

⚠ It is a *testbed*, not a simulator of the shipped run: the relinker executes mid-pipeline on a
different node set than the final submission's (before short-track filtering, after gap close), so
absolute numbers will not match the kernel exactly. What it gives faithfully is the **relative**
comparison between cost functions on an identical node set, which is exactly the question.

**Validation gate (hard):** replaying the *shipped* cost on the shipped nodes must reproduce the
shipped edges to high agreement. If it does not, we do not understand the stage well enough to
replace it, and the plan stops here.

### 35.3 Evaluation discipline — the three ways this can lie to us

1. **Contamination (§30).** Every locally-scorable volume is a pilkwang training volume. §30
   measured a ~0.004 over-prediction for confidence-exploiting changes. Detections are
   unrealistically good here, so association is *easier* locally than on hidden test. Use
   `configs/honest_split.json` (169/30, embryo-balanced) and treat absolute gains as inflated.
2. **Pairwise accuracy ≠ assignment quality (§33).** Widening the gate improved candidate
   *availability* and cost 0.021 because the assignment got worse. **Every model is scored
   end-to-end through the Hungarian with the real metric**, never on AUC.
3. **The ~40:1 asymmetry (§18.3).** 379 swaps against ~15,000 correct links. A model that fixes
   every swap and corrupts 2% of correct links nets zero. **Track edge FP and per-volume macro
   adj, not swap count.**

### 35.4 Phases, with go/no-go gates

| phase | work | cost | gate to proceed |
|---|---|---|---|
| **0** | Reimplement `assign_pass` locally; replay shipped cost on evalset nodes | free, hours | **reproduces shipped edges** at high agreement |
| **1** | Wire replay → real metric; confirm baseline scores match §26's 0.8947 | free | baseline reproduces |
| **2** | Feature set + train a cost model on honest-split TRAIN volumes | free, minutes | — |
| **3** | Score end-to-end on honest-split TEST volumes, macro adj + edge FP | free | **beats geometric cost on held-out volumes** |
| **4** | Patch the kernel, run `evalset_learned`, score on 24 volumes | ~4 GPU-h | **transferable adj > +0.005** |
| **5** | Submit | 1/day | only if phase 4 clears |
| **5b** | *If phase 3 plateaus:* add learned image features (needs GPU, ~3 h by exp 21's rate) | ~3 GPU-h | phase 3 gate again |

### 35.5 What the model gets that geometry does not

Geometry has been exhausted (§33: five axes). The features it never sees:

* **Competition structure** — how many other sources want this target, and their distances. The
  Hungarian resolves competition globally but the *cost* is blind to it.
* **Multi-frame consistency** — the cost is strictly pairwise on one frame pair; a track's history
  over 3+ frames disambiguates a fast cell from a wrong-but-close one.
* **Local scale** — §24.1 ranked `parent_over_spacing` **2nd of 14** features for divisions, and
  §29 found fixed µm thresholds fail across a corpus spanning 150× in coverage. Density-normalised
  distance is the single most under-used quantity in this pipeline.
* **Learned appearance** (phase 5b) — §34 found no signal in 4 hand-made scalars, but that test was
  underpowered (SE 3.3 pts) and hand-designed, so it neither supports nor refutes this.

### 35.6 Honest expectation

The oracle says +0.0954 (§27). Realistic capture is far less: 98% of links are already correct and
trivially so, and the contested 11.5% of edges carry 56% of the loss (§32.1). **A good outcome is
+0.01..+0.02 transferable adj**; anything above that would be the largest single gain in the
project's history and should be disbelieved until the board confirms it.

☠ And the standing risk: five independent knob families have now been swept with the shipped value
winning every time (§33.3). The prior that this configuration is at a well-tuned optimum is strong,
and a learned cost is the first intervention that is not a knob — which is the reason to try it and
also the reason it may break the assignment in ways knobs could not.

## 36. ★ Phase 0 PASSED — the relinker replays locally at 99.7% (2026-08-06)

`scripts/local/relink_replay.py` reimplements `motion_relink_edges` / `assign_pass` (two-stage
greedy Hungarian, gates 6/10, `cost = motion + 0.05·raw`, velocity weight 0.5, predecessor chaining)
and replays it over the node set in `evalset_d4edge/submission.csv`.

    TOTAL   shipped 457,727   replay 456,601   matched 456,368   =  99.7%

Every one of the 24 volumes is **≥99.6%**, several at 99.9–100%.

This is markedly better than §35.2 predicted, and the reason matters: the replay runs on
**post-processed** nodes — `linefit_smooth_output_graph` has moved essentially every coordinate,
gap-closing added nodes, the short-track filter removed 22,773, and `learned_edge_probs` are
unavailable so `BONUS·prob` is dropped entirely. Agreement survives all of that because **the
assignment is distance-dominated and overwhelmingly unambiguous** — the same fact §32.1 measured
as 98% recall below 4 µm, and the same fact that makes `BONUS` inert (§28). The perturbations do
not move the argmin.

**Consequence: association experiments now cost ~3 minutes instead of ~4 GPU-h.** This is the
linking-stage analogue of §31.5's `postproc_sim`, and it is the piece that makes §35 tractable
at all.

⚠ It varies **edges only**. The real `tight9` run also changed the node set (480,663 vs 477,264)
because different edges feed different short-track filtering, so the replay isolates the linking
effect and will not reproduce a kernel run exactly.

### 36.1 ★★ And it PREDICTS two experiments we actually paid for

Replaying gates 6/9/12 on the shipped nodes and scoring with the real metric, against the
board-side runs from §33:

| `TIGHT_UM` | replay edge_J | replay adj | replay Δscore | **actual Kaggle Δ** | error |
|---|---|---|---|---|---|
| 6.0 | 0.9046 | 0.8930 | +0.0000 | +0.0000 | — |
| 9.0 | 0.8838 | 0.8698 | **−0.0233** | **−0.0209** | 13% |
| 12.0 | 0.8240 | 0.7843 | **−0.1087** | **−0.1255** | 13% |

Correct sign, correct ordering, magnitude within **13%** on both, and the baseline reproduces
0.8930 against the real 0.8947 (−0.0017). The residual is expected and explained: the replay holds
the node set fixed while the real runs changed it (tight9 produced 480,663 nodes vs 477,264),
because different edges feed different short-track filtering.

**The testbed predicts real experiments.** Phases 0 and 1 of §35.4 are cleared, and phase 2 may
proceed.

⚠ **Retrospective, and it stings:** §33's `tight9`/`tight12` cost ~8 GPU-h and a day of waiting.
This replay answers the same question in **3 minutes** and would have given the same conclusion.
The instrument was buildable before the experiment, exactly as §17's method note said after the
last time this happened — and exactly as §23.2 said about `BIOHUB_TEST_DIR`. That is three
occasions now where the cheap instrument existed and was built only after paying for the expensive
answer. **Standing rule for this project: before any kernel run, ask whether the question can be
answered by replaying files already on disk.**

## 37. ☠ Phase 3 FAILED — a learned cost does not beat geometry out of sample (2026-08-06)

§35's plan executed end to end. Phases 0–1 passed (§36); **phase 3, the real gate, failed.**

Trained on the 16 evalset volumes on the honest-split TRAIN side, scored on the 8 held-out ones,
end-to-end through the Hungarian with the real metric:

| cost function | edge_J | adj | score | edge FP | edge FN | Δ |
|---|---|---|---|---|---|---|
| **geometric (shipped)** | 0.8927 | 0.8844 | **0.8844** | 238 | 194 | — |
| learned `−log(p)` | 0.8934 | 0.8842 | 0.8842 | 236 | 193 | **−0.0002** |
| `geo − 0.5·logit(p)` | 0.8929 | 0.8836 | 0.8836 | 237 | 194 | −0.0008 |
| `geo − 2.0·logit(p)` | 0.8934 | 0.8842 | 0.8842 | 236 | 193 | −0.0002 |
| `geo − 8.0·logit(p)` | 0.8934 | 0.8842 | 0.8842 | 236 | 193 | −0.0002 |

Nothing moves. And **three of the four formulations produce bit-identical output** across a 16×
range of blend weight — the signature of a model that has simply relearned the distance ranking,
so every monotone transformation of it yields the same argmin.

### 37.1 Why: the training distribution is dominated by cases geometry already solves

    20,935 labelled pairs, 12,477 positive = 59.6% positive

A 59.6% positive rate means the median source has **fewer than two candidates within 10 µm** —
there is almost nothing to choose between. This is §32.1 restated as a training-set property: 98%
of links sit below 4 µm and are unambiguous, so a model fitted to all labelled pairs spends its
capacity on examples where distance is already the right answer.

⚠ And the deeper constraint, which no reweighting fixes: **labels only exist for annotated cells.**
Only ~3.6% of cells are annotated, so supervision comes from ~13k GT edges rather than the 457k
predicted edges the linker actually decides. The same sparsity that capped the division work
(§24.2) caps this.

### 37.2 What this closes

Association has now been attacked along every available axis and the shipped geometry wins each
time:

| axis | result |
|---|---|
| `MOTION_RELINK_LEARNED_BONUS` 1/3/6/17 | inert (§28) |
| `DUAL_SEED_EDGE_THRESHOLD` 0.48/0.30/0.15 | shipped best, confirmed on the board (§30) |
| `MOTION_RELINK_VELOCITY_WEIGHT` ×4 | 0/65 flips (§18.2) |
| `MOTION_RELINK_TIGHT_UM` 6/9/12 | shipped best by a wide margin (§33) |
| neighbourhood flow term | 1.3% vs 0.5% (§31.1) |
| hand-made appearance descriptors | chance, underpowered (§34) |
| **learned cost over geometry + competition structure** | **−0.0002 out of sample (§37)** |

⇒ The +0.0954 oracle headroom is real, and **is not recoverable from geometry, competition
structure, or the supervision available.** The oracle proves a perfect assignment *exists*; it does
not prove the information to find it is present in what we can feed a model.

**This is a genuine result, not a failed experiment.** It cost zero GPU and zero submissions, and
it says the 0.913–0.914 shelf is not a tuning problem — consistent with §33.3's finding that five
independent knob families all favour the shipped configuration.

### 37.3 ★ The model DOES learn — and it still buys nothing. AUC would have lied.

The obvious objection to §37.1 is that the model drowned in easy examples. Tested by retraining on
contested subsets only, and separately measuring held-out **AUC** on those pairs — which
distinguishes "the model cannot learn" from "it learned but the assignment does not change":

| trained on | pairs | held-out AUC (model) | AUC (raw distance only) | **lift** | end-to-end Δ |
|---|---|---|---|---|---|
| contested (`n_cand ≥ 2`) | 13,048 | 0.9827 | 0.9777 | +0.0050 | **−0.0002** |
| **hard (`raw ≥ 4 µm`)** | 8,890 | **0.8756** | **0.8190** | **+0.0567** | **−0.0002** |

**On the hard pairs the model beats pure distance by +0.057 AUC — a large, real, out-of-sample
discriminative gain — and the end-to-end score does not move at all.** Edge FP goes 238 → 236: two
edges out of 238.

So the first hypothesis is refuted and the second confirmed: **the model learns genuine signal
beyond distance, and that signal does not convert into a better assignment.** The reason is that
its probability is still overwhelmingly a function of distance (distance alone already scores 0.819
AUC), so the argmin of the assignment is essentially unchanged.

⚠ **This is the single most important methodological result of the session.** Had phase 3 been
gated on AUC — the natural, conventional choice — a **+0.057 lift** would have been declared a
success and shipped something worth **−0.0002**. §35.3 pre-registered "score end-to-end, never on
AUC" because §33 had shown availability improving while the score fell; that discipline just caught
a false positive that would otherwise have cost a submission and a wrong conclusion in the log.

**Conversion rate, for future reference: a +0.057 AUC lift on the hard subset moved ~1% of the
errors.** Pairwise metrics on this problem are close to meaningless as a proxy for score.

### 37.4 What would still be untested after this

Honest list, so nobody re-derives it:

* **Better detections.** The oracle is computed *on our node set*. A different detector changes the
  ceiling itself, and §27 shows detection loss is only 0.0236 — but the leader's +0.033 has to come
  from somewhere, and detection is the one stage we have never varied.
* **Denser supervision.** Every model here learned from ~3.6% of cells. Pseudo-labelling the other
  96% from confident tracks was on the Kaggle playbook (§9.4 item #3) and was never attempted.
* **Image features inside the association model** (§35.4 phase 5b). §34's null used four hand-made
  scalars and was underpowered, so it neither supports nor refutes this.

## 38. ⚠ The node-budget multiplier — SEE §39 BEFORE TRUSTING ANY OF THIS (2026-08-06)

> **⚠ SUBSTANTIALLY UNDERMINED by §39, same day.** Every number below assumes the multiplier is
> **unclamped above**, i.e. that under-predicting nodes *pays a bonus*. That property comes from our
> own reimplementation, `docs/metric.md` explicitly flags the reference as a **community**
> reimplementation rather than the official Kaggle scorer, and **every LB probe we have ever run
> tested only the penalty side.** §39 shows a capped multiplier predicts the known LB values better,
> and that under a cap the +0.0712 oracle becomes **−0.0070**. Read §39 first.

### 38.1 ⚠ §11.2's "the node budget is full" was a 4-volume artifact

`adj = edge_J · (1 − 0.1·ratio)`, `ratio = (N_pred − est_n_nodes)/est_n_nodes`, and the multiplier
is **unclamped above**. §11.2 concluded from the dev 4 (ratios −0.009, −0.436, −0.023, +0.003) that
`k ≈ 0.97–1.02` and "the node budget is full, not 75% full as §10.5 assumed."

On 24 volumes that is wrong:

    ratio spans  -0.710 .. +0.386   (span 1.096)
    18/24 volumes UNDER-predict and collect a bonus;  6/24 over-predict
    44b6_66f9292d scores adj = 1.0132  -- ABOVE 1.0, because the multiplier is uncapped
    the multiplier is currently worth  +0.0106  of our score

`44b6_66f9292d` predicts **71% fewer nodes than there are cells** and still scores edge_J 0.9461 —
because edge_J counts only edges touching the ~3.6% of cells that are annotated. **A predicted
node carrying no scored edge is pure cost in the multiplier and free to delete.**

And `est_n_nodes` is *not needed at inference* to exploit this: any node removal improves the
ratio whatever the denominator is.

### 38.2 The oracle: +0.0712, and edge_J does not move

Dropping every node that carries no scored edge (uses GT; upper bound on all node-budget rules):

| | edge_J | adj | score | Δ |
|---|---|---|---|---|
| baseline | 0.9039 | 0.8916 | 0.8947 | — |
| **ORACLE** | **0.9031** | **0.9628** | **0.9659** | **+0.0712** |

**edge_J changes by −0.0008.** The entire +0.0712 is the multiplier. This is the second-largest
oracle in the project after association's +0.0954, and unlike that one it had never been measured.

### 38.3 ☠ But every rule that does not know the annotation fails badly

| rule | Δscore |
|---|---|
| shell_xy keep 0.95 | **−0.0194** |
| shell_all keep 0.95 | −0.0243 |
| shell_xy keep 0.90 | −0.0855 |
| random keep 0.95 (control) | **−0.0800** |
| random keep 0.90 | −0.1665 |

Removing 5% of nodes at random costs 0.08 — because an edge dies if *either* endpoint goes, so a
5% node cut breaks ~10% of edges. §13's centre-bias (annotated bbox = 71% of volume) is far too
weak to separate: the outer shell still contains annotated cells.

This also explains `probe_len20` (−0.027 for a 22% node cut) and is the mirror image of
`probe_pad` (+50% nodes, −0.045) — both are the same trade seen from opposite sides.

⇒ The lever reduces entirely to one question: **is "carries a scored edge" predictable from
inference-time features?** §13 concluded annotation is ≈ a random 3.6% sample, which would make it
unpredictable — but that was never tested directly, and the payoff justifies testing it.

⚠ **Framing note.** This is metric-directed optimisation, not tracking quality. It is a different
kind of thing from the fabricated-fork exploit the project explicitly declined (§SESSION_LOG:
"target the private board with genuine quality") — pruning genuine low-value detections is the same
category as `DET_THRESHOLD` tuning. But it optimises the scoring rule rather than the biology, and
§11.1 already recorded one case where the obvious metric hack lost points on the board. If this
proves reachable it is a decision to take deliberately.

### 38.4 ☠ And it does not convert: AUC 0.775 → +0.0008

The target *is* predictable (§38.3's open question answered: held-out **AUC 0.7750**, not the coin
flip §13's "random 3.6% sample" implied). Model-ranked removal, scored end-to-end on held-out
volumes:

| drop fraction | edge_J | adj | score | Δ |
|---|---|---|---|---|
| 0 (base) | 0.8917 | 0.8822 | 0.8872 | — |
| **5%** | 0.8867 | 0.8827 | 0.8880 | **+0.0008** |
| 10% | 0.8728 | 0.8763 | 0.8816 | −0.0056 |
| 20% | 0.8535 | 0.8798 | 0.8857 | −0.0015 |
| 50% | 0.6573 | 0.6874 | 0.6874 | −0.1999 |

**Best: +0.0008 against a +0.0712 oracle — 1% of it, and far below the 0.005 readable floor.**

The arithmetic of why: the multiplier gain is **linear** in nodes dropped, while the edge_J loss is
**quadratic** (an edge dies if *either* endpoint goes). At AUC 0.775 the classifier destroys 18.5%
of scored nodes to drop 50% of all nodes, and quadratic beats linear long before the multiplier
pays. ☠ Node-budget optimisation is closed.

### 38.5 ★ The pattern across the session: discriminative power does not convert

This is now the **second** time in one session that a model with genuine, out-of-sample
discriminative power bought nothing:

| lever | model's signal | end-to-end Δ |
|---|---|---|
| learned relink cost (§37.3) | **+0.057 AUC** over distance on hard pairs | **−0.0002** |
| node-budget classifier (§38.4) | **AUC 0.775** (vs 0.5 chance) | **+0.0008** |

Both fail for structurally similar reasons: the metric's error surface is **not** a sum of
independent per-item decisions. An assignment argmin is robust to large cost perturbations (§36's
99.7% replay agreement across smoothing and dropped terms is the same fact), and an edge requires
*both* endpoints. Local accuracy therefore decouples from score.

⇒ **Standing rule for this project: never rank an idea by AUC, accuracy, or any per-item metric.
Only end-to-end score on held-out volumes counts.** Three oracles are now measured and all three
are unreachable with the information available:

| oracle | size | status |
|---|---|---|
| association (§27) | +0.0954 | unreachable — 7 axes tried (§37) |
| **node budget (§38)** | **+0.0712** | **unreachable — 1% captured (§38.4)** |
| division (§23.3) | +0.0875 headroom | ~+0.012 reachable at best (§26.4, §31.5) |

**The oracles measure what a perfect decision would earn. They do not measure whether the
information to make that decision exists — and on this problem, three times out of three, it does
not.** That is the real explanation for the 0.913–0.914 shelf, and it is a stronger statement than
"we ran out of knobs."

## 39. ⚠ Critique of §38 — the whole lever rests on an unverified property of the scorer

Self-review of §38 found four problems. The first is disqualifying.

### 39.1 ★ The multiplier may be CAPPED at 1.0 in the real scorer, and then the lever INVERTS

§38 rests entirely on `adj = edge_J·(1 − 0.1·ratio)` being unclamped above, so that
under-prediction earns a bonus. Two facts should have been checked first:

* `docs/metric.md` states plainly that our reference is a **community reimplementation, not the
  official Kaggle scorer**, and that "one link remains unconfirmed".
* **Every LB probe we have run tested the PENALTY side only.** `probe_pad` added 50% nodes
  (ratio → +0.5, multiplier → 0.95) and scored 0.868 against a predicted 0.913 × 0.95 = **0.867** —
  an excellent match that confirms the penalty side and says *nothing* about the bonus side.

Scoring with and without a cap at 1.0, against known LB values:

| config | volumes | unclamped | **capped** | actual LB | closer |
|---|---|---|---|---|---|
| `d4edge` | dev 4 | 0.9216 | **0.9112** | **0.913** | **capped** (0.0018 vs 0.0086) |
| `edgethr30` | dev 4 | 0.9356 | **0.9265** | **0.913** | **capped** (0.0135 vs 0.0226) |
| `d4edge` | eval 24 | 0.8947 | 0.8821 | — | cap removes **0.0126** |

And the consequence for §38, on the same 8 held-out volumes:

    baseline                 0.8872
    ORACLE, unclamped        0.9636   = +0.0763
    ORACLE, capped           0.8803   = -0.0070      <-- the lever becomes NEGATIVE

⚠ **Not proof.** Contamination inflates local scores independently, so *any* downward correction
would fit the LB better — the cap and contamination are confounded, and two points cannot separate
them. But that is the point: **§38's headline was contingent on an unverified scorer property, and
I presented it with two stars as "an oracle nobody has measured" without checking.** The correct
framing is that the bonus side of the multiplier is **unverified**, and the lever is worth
+0.0712 or −0.0070 depending on a fact we do not have.

**Cheapest resolution:** one submission of a config that differs *only* in node count on the bonus
side. Do not build anything on the bonus side until then.

### 39.2 The oracle and the conversion test used different volume sets

§38.4 compared "+0.0008 captured" against "+0.0712 oracle" — but the oracle was computed on all 24
volumes and the conversion on the 8 held-out ones. Recomputed on the same 8, the oracle is
**+0.0763**, so the capture fraction is 1.0% rather than 1.1%. The conclusion is unchanged, but the
comparison as published was invalid.

### 39.3 The +0.0008 is not a number, it is noise

**3 of the 8 held-out volumes are thin** (`44b6_0113de3b` 50 GT edges, `44b6_0b24845f` 49,
`6bba_0e7c0d07` 198). §19.2 established that a *single edge* on a 49-edge volume moves the macro
mean by **+0.0096**. So +0.0008 is roughly one twelfth of one edge — far below the resolution of
this instrument. The sweep is also non-monotone (10% → −0.0056 but 20% → −0.0015), which is the
same noise showing through. I reported "+0.0008, below the readable floor" without doing the
thin-volume decomposition that §19.2/§32 made mandatory.

### 39.4 Two claims in the recommendation were wrong

* **"Better detections is the only lever sized to matter."** §27's own oracle says detection-limited
  loss is `1 − 0.9764 = 0.0236` macro. The gap to the leader is **0.033**. So perfect detection with
  the current linker **cannot close the gap** — the opposite of what I asserted.
* **"Detection is the one stage never varied."** `DET_THRESHOLD` was swept (`thr096`, `thr095`),
  plus `harmonic` and `orfuse` fusion. What was never successfully done is retraining the detector
  *weights*. The narrower claim is true; the one I made is not.

### 39.5 And the unifying theory was over-generalised

§38.5 proposed that all three oracles are unreachable because "the information does not exist".
Reachability was tested for **two**, not three, and the failure modes differ in kind: the node
budget fails because annotation is arbitrary from the model's viewpoint (genuinely no information),
whereas association may fail because the right information was never fed in — §34's appearance test
was underpowered by its own admission. One law over two data points with different mechanisms is
rhetoric, not evidence.

**What survives §39:** the node-budget *observation* (ratio spans 1.096, 18/24 volumes on the bonus
side under our scorer, `adj > 1` occurs) is real and worth knowing. Everything downstream of it is
contingent on §39.1.

## 40. ★★ LOCALISATION is the hidden half of the association loss — and a revised plan (2026-08-06)

**Supersedes the standing read of §37.2, §38.5 and §39.5** that "the +0.0954 association headroom is
unreachable because the information does not exist". It is partly reachable, and the information is
present; it is being destroyed upstream of the linker. §33.3's "association is closed to knobs" and
§37's "a learned cost buys nothing" both stand — the correction is that the linker was never the
place to look.

### 40.1 ★ Why seven association levers in a row failed

`link_ceiling.py` defines the oracle as *GT edges with both endpoints matched*, and `match_nodes`
matches within **7 µm**. So a node localised 3 µm from truth still counts as "detected", and every
error its bad coordinate causes is booked as **association loss**. §27's decomposition — association
80–85% of remaining loss, detection only 0.0236 — is arithmetically correct and **structurally
misleading**: coordinate error is invisible in it and reappears as the linker's fault.

That is why BONUS (§28), edge threshold (§30), velocity weight (§18.2), TIGHT_UM (§33), a
neighbourhood flow term (§31.1), hand-made appearance descriptors (§34) and a learned cost (§37) all
returned null. **Each asked the linker to overcome an error it cannot see.** §31.3 had the number in
hand a day earlier and it was read as a curiosity: over the 379 swaps the predicted source→correct-
successor distance is **9.37 µm** while the GT displacement of those same edges is **4.82 µm**. The
missing ~4.5 µm is our own coordinate error, and it is exactly what pushes correct successors past
`MOTION_RELINK_TIGHT_UM` = 6.0.

### 40.2 ★★ The causal measurement: perfect localisation is worth +0.0259 transferable

`scripts/local/localisation_doseresponse.py`, on the 24 evalset volumes through the §36 replay.
Perturbed coordinates drive the **linking decision only**; scoring always uses the **original**
positions, so node recall, node matching and the node-budget ratio are held byte-fixed and the only
thing that can move is which edges the assignment picks.

| arm | RMS change | edge_J | adj | Δ |
|---|---|---|---|---|
| replay baseline | — | 0.9046 | 0.8930 | — |
| noise ×0.25 | +0.552 µm | 0.9045 | 0.8929 | −0.0002 |
| noise ×0.5 | +1.106 µm | 0.9041 | 0.8918 | −0.0013 |
| noise ×1.0 | +2.207 µm | 0.8844 | 0.8685 | −0.0245 |
| **snap λ=0.5** | −0.251 µm | 0.9110 | 0.9010 | **+0.0080** |
| **snap λ=1.0** | −0.501 µm | **0.9274** | **0.9236** | **+0.0305** |

`snap` shrinks matched nodes toward their GT position by λ; `noise` adds anisotropic Gaussian error
to **every** node with §25's measured per-axis scatter (z 1.707 / y 0.943 / x 1.039 µm).

**Snap λ=1 moves edge_J 0.9046 → 0.9274 against the §27 oracle of 0.9764 — it captures 31.8% of the
association headroom.** Every previous attempt captured 0–1%.

☑ **Thin-volume decomposition done before publishing** (§19.2/§39.3 make this mandatory;
`snap_decompose.py`):

    headline macro adj delta (all 24)            +0.0305
    TRANSFERABLE (mean over 18 thick volumes)    +0.0259
    improved 19/24, worsened 2/24 (both -0.002, noise)
    edge FP  876 -> 633   (-27.7%)

It survives. The gain is broad, not one edge on a thin volume, and the FP collapse rules out a
matching artifact. Three volumes moved exactly 0.0000 — verified as real, not a bug: their edge sets
*do* change (15 and 209 edges) but only between **unannotated** cells, which the metric cannot see.

☑ **Placebo control.** The obvious objection is that *any* large displacement of annotated nodes
might decouple them from unannotated distractors and help for a reason unrelated to accuracy. Moving
each matched node **the same distance** but in a **random direction**, three seeds:

| arm | RMS moved | adj | Δ |
|---|---|---|---|
| snap λ=1 (toward truth) | 0.501 µm | 0.9236 | **+0.0305** |
| placebo (random direction) | 0.501 µm | 0.8786 / 0.8804 / 0.8799 | **−0.0140** |

A **0.045 spread at identical displacement magnitude**, attributable purely to direction. The gain
is accuracy, not perturbation.

⚠ **Direction of the bound, and it is the opposite of what it looks like.** Snapping corrects only
matched (≈annotated) nodes, so a correct target becomes exact while unannotated distractors stay
noisy — which reads like an unfair advantage. It is not: a real detector improvement sharpens
distractors *too*, and distractor noise can only ever move a wrong target **closer** to the source by
chance, manufacturing false competition. Removing it helps. So uniform perfect localisation is
**≥** snap-matched-only, and **+0.0259 is a lower bound on the ceiling, not an upper one.**

### 40.3 ★ It explains §26.6's collapsed volumes, which §29 could not

`localisation_vs_gap.py` measures per-volume mean residual on matched nodes (independent of the
linker — `match_nodes` uses position only).

    corr(residual, adj)               -0.648  (24 vols)   -0.523  (18 thick)
    corr(residual, association gap)   +0.623  (24 vols)   +0.371  (18 thick)

**The two worst volumes by residual are exactly §26.6's two collapsed volumes** — `44b6_87bba6c4`
3.137 µm and `6bba_786893ac` 2.697 µm against a corpus mean ~2.0 µm — and snapping recovers
**+0.0854** and **+0.0877** of their adj. This beats every predictor in §29.1 (annotation density
−0.577, disp_p90 +0.460), and unlike those it comes with a causal intervention rather than a
correlation.

⚠ Residual is partly a proxy for density, which §29 already flagged; the correlation alone would not
settle causation. The snap intervention is what does.

### 40.4 ☠ Post-hoc image refinement is DEAD — and §25 was struck for the wrong reason

Predicted coordinates are pure integer voxels and the pipeline never refines them, so centre-of-mass
refinement on the raw image looked open. §25 had struck "sub-voxel refinement" off on the grounds
that *quantisation* is only 5.1% of the error variance — true, but a narrower claim than it was used
for, since CoM re-estimates the centroid rather than merely undoing rounding.

Tested properly (`localisation_refine.py`, 6 volumes, 4 configurations). **All four make it worse:**

| window (vox) | peak frac | RMS 2.186 µm → | macro adj Δ |
|---|---|---|---|
| ±2,8,8 | 0.5 | 2.504 (−14.5%) | −0.0018 |
| ±1,4,4 | 0.5 | 2.316 (−5.9%) | −0.0028 |
| ±1,3,3 | 0.7 | 2.422 (−10.8%) | −0.0039 |
| ±2,6,6 | 0.8 | 2.909 (−33.0%) | −0.0053 |

**The detector's argmax is already a better centroid estimate than a raw-intensity centroid.** At
3–8 µm cell diameters and ~11.7 µm spacing, neighbouring nuclei pull the CoM; only the sparsest
volume (`44b6_0113de3b`, 49 matched nodes) improved, by 51%.

⇒ Localisation must be improved **inside the detector**, or not at all. This does **not** rule out
CoM on the detection **heatmap**, which is trained to peak at cell centres where raw intensity peaks
at bright chromatin — an untested and materially different signal.

### 40.5 ⚠ The response is CONVEX — this is the go/no-go criterion

λ=0.5 buys +0.0080 but λ=1.0 buys +0.0305. Fitting the noise arms gives loss ∝ RMS^~1.7. So:

| RMS reduction | expected Δ (local instrument) |
|---|---|
| 10% | +0.001 .. +0.005 |
| 20% | +0.002 .. +0.010 |
| 30% | +0.003 .. +0.014 |
| 50% | **+0.008 .. +0.021** |

**A small localisation improvement buys nothing measurable.** This is the same robustness that made
BONUS inert (§28) and the learned cost null (§37) — the assignment argmin does not move for small
perturbations, which is also why §36's replay agreed at 99.7% across smoothing and dropped terms.

⇒ **PRE-REGISTERED GATE: no localisation intervention proceeds to a submission unless it first
reduces matched-node RMS by ≥30% on the evalset.** That screen is free, local, and needs only
coordinates plus GT. It is the cheapest go/no-go in the project and it must be run before any GPU.

### 40.6 ⚠ The §39.1 cap question — two tests now disagree, and it no longer matters

§39.1 said `probe_pad` "confirms the penalty side and says nothing about the bonus side". That is
wrong: the baseline's own multiplier appears in the ratio of the two scores, so the delta *is*
informative. Using the evalset-24 node-ratio distribution as the proxy for the hidden set:

| hypothesis | predicted `probe_pad` LB | actual | miss |
|---|---|---|---|
| **uncapped** | 0.8731 | **0.868** | **0.005** |
| capped | 0.8800 | 0.868 | 0.012 |

**The delta test favours UNCAPPED; §39.1's level test favoured CAPPED.** They disagree, and each has
its own confound — the level test by contamination (which inflates local edge_J), the delta test by
the hidden set's unknown node-ratio distribution. Neither settles it.

☠ **And it is not worth settling.** §38.4 already showed node-budget optimisation captures 1% of its
oracle *under the favourable hypothesis*; under the cap it is negative. The lever is dead either
way. A discriminating probe would need a ≥30% pad for only 0.005 of separation — **do not spend a
submission on this.** Recorded so nobody re-derives it.

### 40.7 The revised plan, ranked by (validated ceiling × demonstrated reachability)

Everything below is screened locally, free, before any GPU. Standing rule from §36 applies: *before
any kernel run, ask whether replaying files on disk answers it.*

| # | lever | ceiling | evidence | cost | gate |
|---|---|---|---|---|---|
| **1** | **detector-seed coordinate averaging** | ≤+0.026 | §40.2 causal; row 17 measured two 400ep seeds disagreeing 12%/volume | free screen, then 1 kernel run, 0 subs | RMS −30% (§40.5) |
| **2** | **detector TTA (D4 on the heatmap)** | ≤+0.026 | §40.2; averaging reduces peak jitter. §11 did TTA on the *edge* head, never the detector | 1 kernel run, 0 subs | RMS −30% |
| **3** | **heatmap sub-voxel CoM** | ≤+0.026 | §40.4 rules out *raw-intensity* CoM only | 1 kernel run, 0 subs | RMS −30% |
| **4** | **divisions: wide gates + FITTED selector** | +0.014..+0.028 | §26.3 9/25 FNs reachable at 12 µm; §24.1 fitted selector beats hand ranking 2.4× | `divdump` run, 0 subs | held-out div_J ↑ |
| **5** | **division-aware assignment** | +0.0123 | §24.6 the 1-1 Hungarian structurally cannot emit a division; 32% of div FNs are "FN-consumed" | free — replay owns `assign_pass` | end-to-end adj ↑ |
| ☠ | node budget, relink knobs, learned cost, post-hoc CoM | — | §38.4, §33.3, §37, §40.4 | — | closed |

**Why 1 and 2 lead.** Both attack the largest validated cause of the largest validated oracle, both
need **zero training**, and both use assets already in the kernel. Seed averaging is the strongest
prior: two independently-seeded 400-epoch models make *independent* localisation errors, which is
precisely the component averaging removes — and row 19b showed blending their edge *logits* is
flat, so their disagreement has never been harvested where it actually lives.

★ **Pre-registration for lever 1, derivable before spending anything.** If two seeds' localisation
errors are of equal variance and **fully independent**, averaging the matched pair gives
`RMS → σ/√2`, a **29.3% reduction** — landing almost exactly on §40.5's 30% gate. So the experiment
is sharply falsifiable in advance:

* errors independent ⇒ ~29% reduction ⇒ **+0.003 .. +0.014**, marginally clears the gate;
* errors correlated (same architecture, same data, differing only in seed — plausible) ⇒ **well
  under 29%** ⇒ fails the gate outright and lever 1 dies for free.

The measurement that decides it needs no score at all: dump both seeds' pre-fusion detections once,
match them, and compare per-seed residual against averaged residual. **If that ratio is not ≤0.71,
stop.** Same shape as §32.4's trigger test — establish the mechanism is strong enough *before*
building on it.

⚠ Note the honest tension: this prior says lever 1 is a **coin flip**, not a favourite. It leads the
ranking because it is the cheapest test of the highest-ceiling lever, not because it is likely to
work. Lever 2 (TTA over 8 flip views) has more headroom in principle but the views share a model and
input, so their correlation is higher still and the same gate applies.

**Item 4 is the best-validated unexecuted lever.** §26.3 pre-registered "wide gates + fitted
selector" as the untried combination and it was never run; §16 lost 0.006 doing wide gates + *hand*
ranking, which is the arm we already know fails.

### 40.9 Implementation reconnaissance — where the patch goes, and one lever re-priced

Read before building anything, so the next session does not re-derive it.

**The injection point is one function.** `tracking_repo/scripts/predict_unet_transformer.py`,
`_detect_cells_pooled` (line ~254):

    pooled   = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak  = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])
    coords   = peak_idx.float().cpu().numpy()          # <-- integer voxel argmax
    ... .astype(np.int16)                              # <-- and then cast to int16

Coordinates are the **integer argmax of the detection heatmap**, and `logits` is in scope — so
heatmap CoM (lever 3) is a ~10-line change right here. The D4 TTA machinery already exists in the
same file (`_xy_tta_names` / `_apply_xy_tta` / `_invert_xy_tta`, the `SHARED_POINT_TTA_HELPERS`
block), so lever 2 reuses it rather than writing it.

⚠ **The detection lattice is NOT readable from `submission.csv` — my first attempt at this was
wrong and is left here as the correction.** I checked the coordinate grid in
`evalset_d4edge/submission.csv`, found z/y/x integral on the *fine* grid with every residue mod 4
present, and concluded §25's 0.498 µm quantisation figure was confirmed. **That inference is
invalid.** The submission's coordinates are **post-`linefit_smooth_output_graph`**, which §36
established moves essentially every coordinate, and post-gap-closing, which inserts midpoints. Their
lattice therefore says nothing about the detector's.

The real path is `predict_video` lines 783-788: `_detect_cells_pooled` returns int16 in
**downsampled** space, coords are then scaled by `ds_arr` and **cast to int16 a second time**. If
`downsample = (1,4,4)` the detector lattice is isotropic **1.625 µm**, giving quantisation RMS
`sqrt(3·1.625²/12)` = **0.813 µm** — 13.8% of variance, not 5.1%. `downsample` is read from the
checkpoint config and is not in the run log, so **we do not know it locally**; the run below prints
it.

⚠ And it matters more than it looks, because the **relinker runs BEFORE the smoothing**. The
coordinates the assignment actually sees are the raw doubly-quantised ones, not the 2.186 µm
smoothed ones we measured. **True localisation error at link time is worse than measured, and the
quantisation share of it is larger** — both of which price lever 3 UP, not down.

### 40.8 Honest expectation, and what this does not solve

Local numbers are contaminated (§30): every scorable volume is a pilkwang training volume, and
confidence-exploiting changes over-predicted by ~0.004 there. Localisation is less
confidence-exploiting than `edgethr30` was — the mechanism is structural, not a threshold on
memorised confidence — but **halve the local delta before believing it.**

    levers 1-3, if RMS falls 30-50%:   +0.005 .. +0.013 on the board
    lever 4, FP-controlled:            +0.005 .. +0.010 on the board
    gap to the leader (0.949 vs 0.914): 0.035

⇒ **Both succeeding still leaves us short.** That is the honest arithmetic and it should be stated
before any of it is run. Closing 0.035 on the current node set is not supported by any oracle we
have measured; §27's detection ceiling is 0.9764, so even a perfect linker on today's detections
leaves 0.0236, and the localisation lever is a *fraction* of that plus a fraction of association.
**The remainder has to come from a better detector**, which §39.4 correctly noted has never been
retrained — and which §40.2 now gives a *mechanistic* reason to pursue rather than a hunch.

⚠ §35.0's correction applies to a new target: exp 20's 534 GPU-h is for the **TemporalUNet from
scratch**. Fine-tuning an existing detector for localisation (soft-argmax or an offset-regression
head) is the scale of exp 21's edge-head run — **194 min**, not 534 h. That is the phase-2 bet, and
it should be taken only after levers 1–3 have measured how much of the residual is reducible at all.

## 41. ☠☠ Levers 1 and 2 are ALREADY SHIPPED — read the config before building (2026-08-06)

**§40.7 ranked detector-seed averaging first and detector TTA second. Both are already in the
running configuration.** Found by reading `kernels/evalset_d4edge/` and its run log before writing
any code — the run whose coordinates produced §40.2's 2.186 µm residual.

From the shipped config and confirmed in the log of that exact run:

    Merged edge-TTA + dual-seed patch applied | det=d4 edge=d4 identity_weight=0.125
    Secondary model: ... | edge weight=0.150 | detection weight=0.475 | link mode=low_margin_consensus

* **`det=d4`** — the primary detector's heatmap is averaged over **8 D4 views** (`_xy_tta_names`
  → `det_logits[f] / len(det_tta_names)`). ☠ **Lever 2 is dead: already applied.**
* **`detection weight = 0.475`** — the secondary seed's heatmap is *itself* D4-averaged (its own
  flip/rot/transpose block, `/ _secondary_nv`) and then blended into the primary at **near-even
  weight** after mean/scale alignment. ☠ **Lever 1 is dead: already applied**, and at a far better
  operating point than the naive coordinate averaging §40.7 proposed.

**The shipped detection heatmap is an average over 2 seeds × 8 views = 16 forward passes.** The
2.186 µm residual is what survives *after* 16-way averaging.

### 41.1 What this does to §40's programme

The §40.7 pre-registration asked whether seed averaging clears a 30% RMS reduction. **It has already
been taken and the 2.186 µm is the post-averaging number** — so the reducible *variance* component
is largely spent. What remains is bias shared across seeds and views, plus genuine ambiguity, and
neither yields to more averaging.

⇒ Two of the three cheap localisation levers die **for free, before any GPU**. This is the fourth
time in the log that the cheap check existed before the expensive answer (§17, §23.2, §36) — except
this time it was made first. **Standing rule, extended: before building any lever, grep the shipped
config and the run log for it. "Never trust a name" (§CLAUDE.md) generalises to "never trust that an
idea is new".**

### 41.2 ★ But lever 3 is PROMOTED, and for a better reason than §40.7 gave it

Sub-voxel refinement on the heatmap was ranked third and provisionally demoted. It is now the **only
surviving cheap localisation lever**, and the 16-way averaging is precisely what makes it credible:

* Averaging 16 views produces a **smooth heatmap whose shape encodes where those 16 views collectively
  think the cell is**. Taking `torch.nonzero(logits == maxpool(logits))` — an **integer argmax** —
  throws that distribution away and keeps only its mode, on a lattice.
* So the ensemble information we already paid 16 forward passes for **is present in the heatmap and
  discarded at readout**. A centroid/parabolic estimator recovers it at **zero additional inference
  cost**.
* And §40.9's correction says the quantisation being discarded is up to **0.813 µm** on a
  doubly-quantised path, against a 2.186 µm total — a larger share than §25's 5.1%.

This is a materially stronger hypothesis than "CoM might help", and it is falsifiable by the same
§40.5 gate. ⚠ It remains true that §40.4 measured raw-**intensity** CoM as harmful; that is a
different signal from the trained heatmap and does not transfer.

### 41.3 The revised ordering

| # | lever | status |
|---|---|---|
| ☠ | detector-seed coordinate averaging | **already shipped** (detection weight 0.475) |
| ☠ | detector D4 TTA | **already shipped** (`det=d4`) |
| **1** | **heatmap sub-voxel readout** (`evalset_locref`) | the only cheap localisation lever left |
| 2 | divisions: wide gates + fitted selector | §26.3's untried combination, needs `divdump` |
| 3 | detector fine-tune for localisation | expensive branch, exp 21 scale (~194 min) |

### 41.4 ⚠ PRE-REGISTERED before running `locref` — the arithmetic predicts it FAILS

Written before the kernel is pushed, because the numbers are available beforehand and this project
has repeatedly published a result whose prediction was only constructed afterwards.

`scripts/local/test_locref.py` smoke, on synthetic heatmaps:

| arm | argmax error | parabolic | reduction |
|---|---|---|---|
| exact paraboloid (correctness check only) | 0.4335 vox | 0.0000 | +100% |
| **gaussian logit + noise (the honest arm)** | ~0.42 vox | ~0.17 | **+59%** |

⚠ The paraboloid arm is **exact by construction** — parabolic interpolation inverts a parabola —
so it validates the implementation and says nothing about real heatmaps. The realistic arm removes
**59% of the QUANTISATION error**, which is not the same as 59% of the total.

**The pricing, stated in advance.** With `downsample = (1,4,4)` the detection lattice is isotropic
1.625 µm, so quantisation RMS is `sqrt(3·1.625²/12)` = **0.813 µm** against a measured total of
**2.186 µm**. Removing 59% of the quantisation component gives

    new RMS = sqrt(2.186² - 0.813² + (0.41·0.813)²) = 2.056 um      = a 5.9% total reduction

and §40.5's convexity table prices a 5.9% reduction at **well under +0.005** — **below the
pre-registered 30% gate and below the readable floor.**

⇒ **PREDICTION: `locref` fails the §40.5 gate.** Recorded now so the run cannot be reinterpreted
afterwards.

**Why run it anyway.** Three unknowns that only the detector can resolve, all answered by the same
run at zero submissions:

1. **`downsample` is read from the checkpoint config and we do not know it** — `[1,4,4]` is only the
   library default. The run prints it. If the lattice is coarser, quantisation's share is larger and
   the pricing above is too pessimistic.
2. **The relinker runs BEFORE `linefit_smooth_output_graph`**, so the coordinates the assignment
   actually consumes are the raw doubly-quantised ones, not the 2.186 µm smoothed ones we measured.
   The true error at link time is unmeasured and larger.
3. **The parabola may capture more than quantisation.** If the 16-way-averaged peak is systematically
   offset from the true centre in a way the peak's *shape* encodes, the estimator tracks it. That is
   speculative and is the only route by which the prediction is wrong.

**Run it on the dev 4 first (~21 min), not the evalset 24 (~4 h).** The readout is matched-node RMS,
a per-node quantity with thousands of samples on 4 volumes; §23.2's warning about the dev 4 is about
*score* composition, not per-node residuals. Only if RMS surprises us does the 24-volume score run
become worth 4 GPU-h.

## 42. ☠ `locref` — the run BROKE, and the lever is capped structurally anyway (2026-08-06)

Two separate conclusions, and they must not be conflated. **The run is not a valid test of the
lever** — it had an unintended side effect that destroyed 45% of the graph. **The lever is dead
regardless**, on an argument that needs no run at all and that I should have made before spending
one.

### 42.1 The patch applied correctly and was not a no-op

    STRATEGY 41.2 sub-voxel readout applied (4/4 edits matched)
    downsample=(1, 4, 4) | locref=parabolic
    [41.2] locref peaks=100,629 mean|delta|=0.2301 voxel/axis

☑ `downsample = (1,4,4)` **confirmed** — §41.4's assumed isotropic 1.625 µm detection lattice was
right, so that unknown is closed. ☑ Mean offset 0.230 voxel/axis against 0.25 for a uniform
distribution over ±0.5, so the estimator produced well-spread, non-degenerate offsets. The
silent-no-op guard did its job: this patch demonstrably ran.

### 42.2 ☠ And it destroyed 45% of the graph

| | `d4edge` | `locref` | Δ |
|---|---|---|---|
| raw_nodes | 124,085 | **73,635** | **−40%** |
| raw_edges | 116,343 | **63,793** | **−45%** |
| nodes | 120,276 | 66,731 | −45% |

Per-volume the loss is wildly uneven (ratio 0.80 / **0.39** / 0.93 / 0.55), which rules out anything
uniform like a threshold shift.

**`raw_nodes` is the ILP's output**, and the peak counter shows detection still produced ≥100,629
peaks — so ~30% of *detected* nodes were dropped by the solver, against ~0% for `d4edge`.

**Diagnosis (leading hypothesis, not isolated):** the refined coordinates were fed into
`predict_edges` — `p_coords_src * ds_arr_t` — and the node transformer was trained on **integer**
downsampled coordinates. Fractional inputs are out of distribution for its positional encoding, edge
probabilities fell, and the ILP then found it cheaper not to select nodes at all. ⚠ I have not
isolated this against alternatives (changed `edge_dist`, solver numerics); the magnitude and the
node/edge co-collapse are what point at it.

**The design error is mine and it is instructive: I refined the coordinates everywhere, when the
refinement was only ever wanted in the OUTPUT.** The fix is one edit — round before
`predict_edges`, keep sub-voxel only at the end. I am not spending a run on it, because of §42.3.

### 42.3 ★ The submission format is integer-voxel, which caps the lever below the gate

Cell 11's writer, verified in the built notebook:

    "z": max(0, int(round(float(node["z"])))),
    "y": max(0, int(round(float(node["y"])))),
    "x": max(0, int(round(float(node["x"])))),

**Coordinates are rounded to integers on the FINE voxel grid, and GT coordinates are integers on
that same grid** (verified directly: `all_integer=True` on both a 52-node and a 1,229-node GT
graph). So sub-voxel precision cannot be *expressed* in a submission at all.

What survives is only the change of *lattice*:

| axis | downsample | detection lattice | after refinement + rounding |
|---|---|---|---|
| z | 1 | 1.625 µm | **1.625 µm — refinement entirely destroyed** |
| y, x | 4 | 1.625 µm (4 fine voxels) | 0.40625 µm (1 fine voxel) |

    quantisation RMS  0.813 um  ->  0.498 um
    total RMS         sqrt(2.186^2 - 0.813^2 + 0.498^2) = 2.089 um   = a 4.4% reduction

§40.5's convexity table prices a 4.4% reduction at **well under +0.005** — below the readable floor
and far below the 30% gate.

⚠ **And the worst axis is the uncorrectable one.** §25 measured per-axis residual scatter z **1.707**
µm against y 0.943 / x 1.039. z carries the most error and sits at `downsample = 1`, so rounding
returns it to exactly the lattice it started on.

☑ This also retroactively explains §25: its 0.498 µm quantisation figure was measured on
*submission* coordinates, which is precisely the post-rounding value derived above. §25 and §40.9
now reconcile.

### 42.4 §41.4's pre-registration was CONFIRMED — and for a better reason than it gave

Predicted **5.9%** total RMS reduction and a FAIL. Actual structural ceiling: **4.4%** and a FAIL.
The prediction was right, slightly optimistic, and the true mechanism (an integer submission format
with z uncorrectable) is stronger than the one predicted (parabola only removes 59% of
quantisation).

⇒ ☠ **The cheap localisation programme is closed.** All three routes are gone: two were already
shipped (§41), and the third cannot express its gain in the output format. §40.2's **+0.0259 remains
real and remains the largest identified cause of the association loss** — the snap oracle reached it
using *integer* GT coordinates, so it is expressible — but the 2.186 µm gap between our integers and
GT's integers is **genuine detector error, not quantisation**, and only a better detector removes it.

### 42.5 Standing plan after this

| # | lever | status |
|---|---|---|
| 1 | **divisions: wide gates + fitted selector** | §26.3's untried combination; needs `divdump`; +0.014..+0.028 |
| 2 | **division-aware assignment** | free — the replay owns `assign_pass`; §24.6 says the 1-1 Hungarian structurally cannot emit a division |
| 3 | detector fine-tune for localisation | the only route to §40.2's +0.0259; exp 21 scale (~194 min), not 534 GPU-h |
| ☠ | all coordinate post-processing | §40.4 raw-image CoM, §42 sub-voxel readout |

**Method note, and it is the fourth of its kind.** §36 established "before any kernel run, ask
whether replaying files on disk answers it". The integer-writer fact was in cell 11 the whole time
and would have killed this lever for free. §41 got this right for levers 1 and 2 by reading the
config first, and I then failed to apply the same discipline one section later. **Extend the rule:
before building a lever, trace its output all the way to the submission writer and confirm the gain
can be expressed there.**

## 43. ★ Division-aware assignment: the mechanism is CONFIRMED, the precision is not (2026-08-06)

Free, in the replay, zero GPU. §24.6 established that `motion_relink_edges` is a strict 1-1 matching
and therefore *structurally* cannot emit a division; §26.3 measured that **8/25 (32%)** of division
FNs are "FN-consumed" — the daughter was detected but the matching gave her to a different parent —
and said no gate or selector bolted onto the existing assignment can reach them. §34.3 priced the
fix at **+0.0123** and filed it as "requires rewriting the assignment". §36's replay means we own
`assign_pass`, so it cost minutes.

**Formulation.** Each source is duplicated: row `i` is its ordinary link, row `i+S` a second child at
`cost + DIV_PENALTY`. The Hungarian then solves a degree-constrained matching where a source may take
two targets.

| div penalty | forks | edge_J | adj | div_J | score | Δ | edge FP | dTP | dFP | dFN |
|---|---|---|---|---|---|---|---|---|---|---|
| **inf (acceptance)** | **0** | **0.9046** | **0.8930** | 0.0000 | 0.8930 | — | 876 | 0 | 0 | 25 |
| 8.0 µm | 12,956 | 0.8934 | 0.8788 | 0.0249 | 0.8813 | **−0.0117** | 1,165 | **8** | 296 | 17 |
| 6.0 µm | 12,958 | 0.8934 | 0.8788 | 0.0249 | 0.8813 | −0.0117 | 1,165 | 8 | 296 | 17 |
| 4.0 µm | 12,972 | 0.8934 | 0.8788 | 0.0249 | 0.8813 | −0.0117 | 1,165 | 8 | 296 | 17 |
| 2.0 µm | 13,026 | 0.8933 | 0.8787 | 0.0248 | 0.8812 | −0.0119 | 1,166 | 8 | 297 | 17 |
| 1.0 µm | 13,091 | 0.8933 | 0.8785 | 0.0248 | 0.8810 | −0.0120 | 1,167 | 8 | 298 | 17 |

☑ **Acceptance test passed**: the infinite-penalty arm emits **0 forks** and reproduces the replay
baseline exactly. ⚠ The first version failed it — it filtered eligibility as
`s in un_s or n_child[s] == 1` in *both* arms, so the relaxed pass handed a second child to an
already-matched source at ordinary cost and the inf arm emitted 13,007 forks. Caught by the
acceptance test, which is why it is there.

### 43.1 ★★ §26.3's pre-registered 8/25 fired exactly

**Division FN goes 25 → 17, i.e. dTP 0 → 8 — precisely the 8/25 FN-consumed bucket §26.3
predicted.** A division-aware assignment recovers exactly the divisions the 1-1 constraint was
destroying, and nothing else. That is the cleanest mechanism confirmation in the division programme
and it validates §24.6's structural reading.

It also gets **4× the division recall of the shipped post-hoc scavenge** (dTP 8 vs §26.2's 2).

### 43.2 ☠ And it is destroyed by precision, exactly as §16 was

**12,956 forks to obtain 8 true ones** — a 0.06% hit rate — driving division FP to 296 and edge FP
876 → 1,165 (**+33%**). Net **−0.0117**.

The penalty is almost inert from 8 µm down to 1 µm, and the reason is structural: with the source
rows duplicated, `linear_sum_assignment` matches every target it can, so **every orphan inside the
gate gets adopted by somebody**. Genuine track births are adopted along with real daughters. The
penalty only decides *which* source adopts, not *whether*.

This is §16's `divgeom` failure in a new costume — there it was 13,495 coincidental candidates from
widened gates, here 12,956 from a relaxed degree constraint — and it is the fourth appearance of
§21.1's pattern: **admissibility was never the binding constraint; precision is.**

### 43.3 ★ But it prices the combined lever precisely, and confirms §34.3

The two halves are now measured separately and they compose arithmetically:

    division-aware assignment  ->  dTP 8, dFP 296   div_J 8/(8+296+17)  = 0.0249
    shipped post-hoc scavenge  ->  dTP 2, dFP  40   div_J 2/(2+40+23)   = 0.0308
    assignment RECALL + scavenge PRECISION -> 8/(8+40+17) = 0.123  => contributes 0.0123

**That +0.0123 is exactly §34.3's independent estimate for this lever**, derived by a completely
different route. Two independent derivations agreeing is the strongest support the division
programme has had.

⇒ **Division-aware assignment is not a standalone lever; it is the RECALL half of one.** It must be
paired with a precision selector that keeps ~8 of 12,956 candidates. That is §26.3's "wide gates +
fitted selector" — except the pool is now known to be the *assignment's* second-child candidates,
not a geometric gate's, and §24.1's fitted GBM (AP 0.0889 → 0.2827) is the natural ranker.

⚠ **The base rate is brutal and must be stated:** ~8 real in 12,956 candidates is **0.06%**. §24.1's
selector was fitted on a pool of 47 candidates/volume with a ~2% base rate. Extrapolating its AP to
a pool 30× sparser is exactly the error §24.2 flagged. **The selector must be refitted on this pool**,
which the replay can now generate for free — no `divdump` kernel needed, which retires that
prerequisite.

### 43.4 Standing plan

| # | lever | status |
|---|---|---|
| **1** | **fitted selector over the assignment's second-child candidates** | free to generate the pool (§43.3); ceiling +0.0123; base rate 0.06% is the risk |
| 2 | detector fine-tune for localisation | the only route to §40.2's +0.0259; exp 21 scale |
| ☠ | division-aware assignment alone | −0.0117 (§43) |
| ☠ | all coordinate post-processing | §40.4, §42 |
| ☠ | node budget, relink knobs, learned cost | §38.4, §33.3, §37 |

## 44. ★ Board census + competitive intel: the field moved, and the 0.915 line is readable (2026-08-07)

### 44.1 Where we actually stand (full leaderboard downloaded, 2,091 teams)

    leader                 0.949
    top 5                  0.938+
    top 10 (gold zone)     ~0.932+
    top 20                 0.922+
    ██ 104 teams at exactly 0.915  <-- the public-notebook shelf
    35 teams at 0.914      <-- us, rank ~166

Two readings, both load-bearing:

* **We are BELOW the public shelf.** 104 teams sit at exactly 0.915 by forking the current public
  line; our best genuine config is 0.914. Whatever the 0.915 line does that we don't is worth
  ≥+0.001 and is free to read.
* **The real prize bar is probably below 0.949.** The division metric exploit (§1) is known to be
  worth up to ~+0.08 on `div_J` and its own author excludes it from final-2. Some of the
  0.938–0.949 band is likely exploit-inflated on the public board and will not survive private
  final-2 picks. The honest bar to plan against is the ~0.932 gold shelf, not 0.949.

### 44.2 The 0.915 public line, decomposed (yusuketogashi "Biohub 159B/162", pulled 2026-08-07)

Same `pilkwang` support pack, same `unet_transformer` method, same dual-seed near-even detection
blend. The deltas against our `d4edge`/`edgethr30` line:

| component | ours | theirs |
|---|---|---|
| density-adaptive gap closing | **already ON** (`BIOHUB_GAP_DENSITY_ADAPTIVE=1`, verified in `evalset_d4edge`) | ON — **not a diff** |
| association evidence | transformer edge prob + motion relink | **public 22-feature local ranker at 85%** (`pilkwang/biohub-local-association-ranker-unet300-v1`) + "JS-TTA reliability" prob at 15% |
| multi-frame term | none (cost is strictly pairwise, §35.5) | **"three-frame forward acceleration lookahead"** (Biohub 162, targeting 0.916) |
| `MOTION_RELINK_LEARNED_BONUS` | 1.0 | 1.0 |
| gap close | max_gap 1, 6.0 µm | **max_gap 2, 5.8 µm** |

The ranker's 22 features are §35.5's list almost verbatim: `source/target_density_7um`,
`candidate_rank`, `candidate_count`, `velocity_um`, per-axis displacements, node degrees, `t_norm`.
⚠ Our §37 measured a same-family learned cost at −0.0002 out of sample — so the ranker's marginal
value is NOT established by their score; the whole-config delta over us is only +0.001. But every
one of their mechanisms is testable in the §36 replay in minutes, for free, before any kernel run.

### 44.3 Literature anchors for the two live internal levers (recorded so the plan has citations)

* **Detector fine-tune for localisation + movement vectors** — this dataset is the linajea setting
  (sparse point annotations on light-sheet embryo volumes). Malin-Mayor et al., *Nature Biotech*
  2023: train cell-indicator maxima AND per-cell **movement vectors** (offset to the same nucleus in
  the adjacent frame) from sparse annotations; link on predicted-parent position, not raw distance.
  That is precisely the cue §40 says we lack: it routes around localisation error instead of asking
  the linker to overcome it. Our owned pipeline already has an offset-head implementation
  (`detect/unet3d.py`). Sub-voxel training-side readout (CenterNet-style offset branch; subpixel
  heatmap regression, Bulat et al. 2021; soft-argmax/integral regression) is the standard fix for
  argmax quantisation — trained on the FINE grid it can move z by whole fine-voxels toward truth,
  which §42.3's integer-writer argument does NOT cap (the snap oracle used integer GT coords).
* **Division selector** — TGMM (Amat et al. 2014) and CTC-winning pipelines all use a fitted
  division classifier over geometric+appearance features, precision-first. Supports §43.4 lever 1.
* **Dense pseudo-labels** — ELEPHANT (Sugawara et al., eLife 2022) and noisy-student self-training:
  sparse annotation → confident predictions → retrain. §37.1's root cause (supervision exists for
  3.6% of cells) is exactly what this attacks, and §37.4 lists it as never attempted.

### 44.4 The plan, ranked by EV against the ~0.932 gold bar

| # | lever | ceiling | cost | first gate (all local, free) |
|---|---|---|---|---|
| 1 | **port + replay-test the 0.915 line's deltas** (22-feature ranker, acceleration lookahead, max_gap 2) | +0.001..+0.004 | free (replay) then 1 kernel run | replay Δadj > 0 on 18 thick volumes |
| 2 | **detector fine-tune: offset/movement-vector head + sub-voxel target** (linajea §44.3) | ≤+0.0259 (§40.2) + motion cue for relink | ~194 min/attempt (exp 21 scale) | **RMS −30% on matched nodes (§40.5)**; movement-vector arm: swap flips in replay |
| 3 | **division selector refit on the §43 second-child pool** | +0.0123 | free pool, minutes/fit | held-out div_J ↑ with edge FP flat |
| 4 | **dense pseudo-labels** for edge/ranker/selector retraining (ELEPHANT) | unlocks 2 and 3 | M1-scale first | pseudo-label purity ≥95% vs GT on held-out annotated cells |
| 5 | cross-family detector ensemble (tophat UNet à la xiaoleilian) | unknown; attacks §41.1's shared bias | high (training) | only if 2 lands and quota remains |

Honest arithmetic: 1+2+3 all landing ≈ 0.914 + 0.002 + 0.010 + 0.006 ≈ **0.932** — the gold shelf,
if the exploit-inflated band above it thins out at private rerun. Winning outright still requires
the detector bet (2, then 5) to land at the top of its range.

## 45. The §44.4 plan executed end-to-end in one day — and what its four kills mean (2026-08-07)

All five items of §44.4 were run to a verdict in a single session, at a cost of one kernel run,
one submission, and ~3 MPS-hours. Rows 34–37 in `experiments.md` hold the numbers.

| item | verdict | evidence |
|---|---|---|
| 1. port 0.915-line deltas | **mechanisms inert; port submitted** | replay: ranker +0.0000, lookahead +0.0001 (row 34); `pub162_port` local dev-4 **0.9261 vs d4edge 0.9216**, submitted, LB pending |
| 3. division selector | ☠ **dead** | oracle +0.032 but no realisable arm clears the floor; 0/8 true forks in any top-5 (row 35) |
| 2. detector fine-tune | ☠ **dead at the §40.5 screen** | residual ~90% unpredictable: −8.7% vs −30% required (row 37), zero GPU spent |
| 4. pseudo-labels | gate passes (99.90%), **no consumer** | row 36; every model that would train on them died in §37/row 35/row 37 |
| 5. cross-family ensemble | closed | conditional on item 2, which failed |

### 45.1 The annotation-noise ceiling — why the association oracle was never reachable

Row 37's screen supplies the missing explanation for the whole §27–§43 arc. The +0.0954
association oracle and the +0.0259 snap oracle both define "error" as distance from **human point
annotations**. A human clicking the centre of a 3–8 µm nucleus carries 1–2 µm of click noise, and
**no model can predict another annotator's noise** — yet the oracles count it as recoverable error.
The predictable fraction of our 2.0 µm residual measures at ~10%; the rest is jointly detector
ambiguity and GT noise, indistinguishable and equally unreachable. This is §38.5's "the information
does not exist" made quantitative, and it now covers association, localisation, node budget AND
divisions.

### 45.2 Standing: the 0.913–0.915 shelf is the information ceiling of this pipeline

Every lever family is now measured-dead except two, both expensive and speculative:
1. **Detection recall via a differently-trained detector** (§27 caps it at +0.0236 with the current
   linker; a new detector family also shifts the linker's operating point — unbounded but unpriced).
   ~50–100 GPU-h against a ~225 GPU-h remaining budget. The only route with a ceiling above +0.01.
2. **Exploit-adjacent metric play** — explicitly declined (§1).

**Final-2 guidance as of today:** best genuine configs are `thr096` (0.914) and, if the LB
confirms, `pub162_port` (expected ~0.915) with `pub162_thr096` (queued) as its one-change
improvement candidate. Reserve final-2 for the two highest CONFIRMED genuine scores.

**Update (same day, evening): `pub162_port` CONFIRMED at LB 0.915** (row 38) — new best,
exactly on the shelf. Current final-2 picks: **`pub162_port` (0.915)** + the better of
`pub162_thr096` (scoring) / `thr096` (0.914). The +0.002 over d4edge lives in the
detection/fusion/gap stack, not the association mechanisms row 34 proved inert.

**Update (night): `pub162_thr096` also 0.915 (row 41)** — the threshold lever is dead on the
new base, harmlessly. **Final-2 = the two confirmed 0.915s** until the detector programme
(row 40, §45.2 route 1) produces something that clears its gates.

## 47. ⚠ The exploit was PATCHED and the board RESCORED — §44.1/§45.2's band thesis is WRONG (2026-08-08)

Discussion-forum sweep (thread 727154 + the official royerlab/kaggle-cell-tracking-competition
repo, now cloned locally). Three load-bearing corrections and one asset:

### 47.1 ⚠ WRONG in §44.1/§45.2: "the 0.92–0.949 band is mostly exploit-inflated"

The organizers **patched the division-jaccard exploit and RESCORED every submission**
("submissions that did not actively exploit this were not affected"). The current
0.916–0.949 band is **genuine engineering**. Consequences:
* The honest bar is the leader's **0.949**, not a "gold shelf ~0.932 after review attrition."
  No attrition is coming — review-risk reasoning about the band above us is dead.
* The +0.034 gap is real headroom someone found — most plausibly their own trained
  detector/pipeline, which independently re-motivates route 1 at larger scale.
* Retroactive coherence: §11's "the hack is negative" LB reads and probe_pad's clean
  penalty-side fit were all measured on the POST-patch scorer without us knowing.

### 47.2 §39.1 SETTLED: the node-ratio multiplier is UNCAPPED above

Official code: `J_adj = max(0, J · (1 − α·total_node_ratio))` — clamped at 0 below, **no upper
cap**. §40.6's delta test had it right; §39.1's level test was the confounded one. The
node-budget lever stays dead regardless (§38.4: 1% capture), but the scorer property is
no longer unknown.

### 47.3 Row 35's division kill is STRONGER than we knew

Official division matching works on parent→divider→children→**grandchildren** windows (±1
timepoint), and **unmatched forks can still be FALSE POSITIVES via child/grandchild
cross-component evidence** — our reimplementation excluded unmatched forks entirely. The
12,639 "div-invisible" forks in row 35's pool were likely FP-eligible officially, making
every keep-forks arm MORE negative than we measured. The kill stands, harder.

### 47.4 ★ Asset: the OFFICIAL scorer is public — vendor it

`github.com/royerlab/kaggle-cell-tracking-competition` (cloned to scratchpad) carries the
exact metrics + division logic + tests. Every "community reimplementation, one link
unconfirmed" caveat (§17, §26, §39, docs/metric.md) can now be retired by scoring locally
with the official code. **Next instrument upgrade: wire the official scorer into
score_submission.py / the evalset scoring path** — do this before reading blend3's verdict
if time permits, else immediately after.

## 8. Operational notes

- GPU: Kaggle caps **2 concurrent GPU sessions**; kernels default to P100 (sm_60, torch-incompatible)
  → always set `machine_shape=NvidiaTeslaT4`. No CLI "cancel"; `kernels_delete` frees a slot (source
  is local, re-pushable).
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`. `ctrack` shipped to kernels via the
  `ctrack-src` dataset; detector weights via `biohub-detector-weights`.
- 5 submissions/day. Calibrate against the ~0.04 offset before spending one.
