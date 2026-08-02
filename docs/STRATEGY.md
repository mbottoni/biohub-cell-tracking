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

⇒ Priority: (1) **`divoff`** — divisions as a *removal* lever; (2) even-weight ensembling of the two
400ep seeds (`secw`, `swapw`); (3) architectural / different-family diversity — but note the
`cellmot-baseline-artifacts` dataset is *attached but never referenced* in both notebooks that ship
it, so that lead is unverified; (4) stop spending submissions on post-processing knobs.

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
Still unmeasured: s/iter on a T4 — the number that decides whether training is feasible here at all.

## 8. Operational notes

- GPU: Kaggle caps **2 concurrent GPU sessions**; kernels default to P100 (sm_60, torch-incompatible)
  → always set `machine_shape=NvidiaTeslaT4`. No CLI "cancel"; `kernels_delete` frees a slot (source
  is local, re-pushable).
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`. `ctrack` shipped to kernels via the
  `ctrack-src` dataset; detector weights via `biohub-detector-weights`.
- 5 submissions/day. Calibrate against the ~0.04 offset before spending one.
