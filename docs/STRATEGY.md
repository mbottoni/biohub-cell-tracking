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

## 8. Operational notes

- GPU: Kaggle caps **2 concurrent GPU sessions**; kernels default to P100 (sm_60, torch-incompatible)
  → always set `machine_shape=NvidiaTeslaT4`. No CLI "cancel"; `kernels_delete` frees a slot (source
  is local, re-pushable).
- Data staged at `/Volumes/extSSD/kaggle/biohub/{raw,gt}/train`. `ctrack` shipped to kernels via the
  `ctrack-src` dataset; detector weights via `biohub-detector-weights`.
- 5 submissions/day. Calibrate against the ~0.04 offset before spending one.
