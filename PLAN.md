# Plan to Win: Biohub — Cell Tracking During Development

**Competition:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
**Written:** 2026-07-23 · **Assumed deadline:** 2026-09-29 (UNVERIFIED — confirm in Phase 0) → ~10 weeks
**Type:** Research code competition, CC0 data, hosted by Chan Zuckerberg Biohub (Royer Group)

---

## 0. What we actually know vs. what we're assuming

Be honest about this, because the plan below is only as good as these assumptions and
several are load-bearing.

**Confirmed (from public sources):**
- Task: detect and track cells in zebrafish embryos through **3D space + time** (light-sheet
  microscopy, Zebrahub-derived data). Full lineages including divisions.
- Data is the largest publicly available cell-tracking dataset by annotation count, CC0.
- Metrics are from the **Cell Tracking Challenge** family — TRA / DET / SEG, with **CHOTA**
  also mentioned.
- The incumbent method is **Ultrack** (Royer group, *Nature Methods* 2025): multi-hypothesis
  segmentation + integer-linear-programming linking. Open source. This is the bar.

**Unverified — resolve on day 1 (Phase 0):**
- [ ] Exact scoring metric and its weights (is it plain TRA? a blend? CHOTA?)
- [ ] Submission format (CTC `res_track.txt` + label TIFFs? a CSV of points? RLE masks?)
- [ ] Whether detection/segmentation is scored, or only linking given provided detections
- [ ] Code-competition compute budget (GPU hours, runtime cap, internet off?)
- [ ] Dataset size on disk and number of held-out sequences
- [ ] Prize structure and team-merge deadline

> **The strategy below is written to be robust to these unknowns**, but revisit it the moment
> Phase 0 lands. If, say, detections are *provided* and only linking is scored, Phase 2
> collapses and everything moves into Phases 3–5.

---

## 1. The core insight: where the score actually lives

CTC's TRA is `1 - min(AOGM, AOGM_0) / AOGM_0`, where AOGM is a weighted edit distance between
the predicted and ground-truth lineage graphs. The standard weights are:

| Error | Weight |
|---|---|
| **FN vertex** (missed cell) | **10** |
| NS vertex (two cells merged into one) | 5 |
| ED (edge deleted) | 1 |
| EA (edge added) | 1.5 |
| EC (edge with wrong semantics) | 1 |
| **FP vertex** (spurious cell) | **1** |

**A missed cell costs 10× a spurious one.** That single asymmetry should drive most of our
design decisions:

1. **Bias detection hard toward recall.** Sweep the detection threshold against the *real*
   metric, not against F1. The optimum will sit well below the F1-optimal threshold.
2. **Never merge two cells into one** — a merge is an NS (5) *and* usually cascades into edge
   errors. Split-happy is cheaper than merge-happy.
3. Edge errors are cheap individually but there are ~N of them per cell per frame, so at
   tens of thousands of cells across hundreds of frames, linking still dominates in absolute
   terms. Both stages matter; detection just has the sharper gradient.

**Corollary:** we must be able to compute the official metric locally, exactly, before we
write a single model. That is Phase 0's real deliverable.

---

## 2. Architecture we're aiming for

A three-stage pipeline, each stage independently improvable and measurable:

```
  raw 3D+t volumes (zarr)
        │
        ▼
  [A] preprocessing        drift/rigid registration · intensity normalization ·
                           anisotropy handling · denoising
        │
        ▼
  [B] detection            3D nucleus centers + (optional) instance masks
                           ensemble → multi-hypothesis candidate set
        │
        ▼
  [C] linking              learned affinities → global optimization (ILP/flow)
                           → division detection → gap closing → track repair
        │
        ▼
  submission (CTC res_track.txt + labels, or comp-specific format)
```

Deliberately **not** end-to-end. End-to-end trackers are elegant but at this scale the
debuggability of a staged pipeline is worth more than the joint optimum, and each stage can
be scored in isolation against the real metric.

---

## 3. Phased plan

### Phase 0 — Recon & measurement harness (Week 1) ⟵ *nothing else starts until this is done*

The single highest-leverage week. Goal: a local scorer we trust and a submission we can make
end-to-end, however bad the score.

- [ ] Fix Kaggle API auth (`~/.kaggle/kaggle.json` currently 401s), pull data + rules
- [ ] Read Overview / Data / Evaluation / Rules tabs in full; fill in the unknowns in §0
- [ ] Characterize the data: voxel spacing (expect strong z anisotropy), frame count, cell
      count per frame, intensity drift over time, embryo motion magnitude
- [ ] **Build `ctrack.eval`: a local CTC/CHOTA scorer** (via `traccuracy`) and verify it
      reproduces the leaderboard on a known submission
- [ ] **Design the validation split**: hold out *entire timelapses / embryos*, never random
      frames. Random frame splits leak catastrophically here — adjacent frames are ~identical.
      If only one timelapse exists, hold out contiguous *time blocks* plus spatial crops.
- [ ] **Submit something trivial** (e.g. thresholded blobs + nearest-neighbour linking) to
      lock in the format and measure CV↔LB correlation early
- [ ] Log the compute budget; decide local-GPU vs Kaggle-GPU vs cloud

**Exit criterion:** local score on a held-out sequence lands within a known offset of the LB.

### Phase 1 — Ultrack baseline (Week 2)

- [ ] Run stock Ultrack on the competition data, tuned only lightly
- [ ] Score it locally; submit it
- [ ] Profile it: runtime, memory, where its errors are (FN? merges? divisions? gaps?)

**Why:** it is the published SOTA on exactly this data from exactly this lab. If we can't
beat it we have no result; if we can, we know precisely which error class we improved. It
also gives us a strong candidate-generation backend for free.

**Exit criterion:** a reproducible Ultrack score + a per-error-class breakdown table.

### Phase 2 — Detection (Weeks 3–5) — *biggest expected gain*

- [ ] Baseline 3D U-Net predicting a **Gaussian center heatmap + boundary/foreground map**;
      peak-detect with a local-max + NMS in physical (anisotropy-corrected) coordinates
- [ ] Compare against: **StarDist-3D**, **Cellpose-SAM**, and a center+offset regression head
      (EmbedTrack-style). Pick on held-out DET, not on loss.
- [ ] Handle anisotropy explicitly — resample to isotropic physical spacing, or use
      anisotropic kernels/strides. Do **not** treat z like x/y.
- [ ] Tile/patch inference with overlap-blending; deduplicate detections at tile seams
- [ ] **Threshold sweep against the real metric** (recall-biased, per §1)
- [ ] TTA: flips + 90° rotations in xy, flip in z
- [ ] Ensemble 2–3 architectures → feed the *union* as candidate hypotheses to Phase 3
      rather than voting them down to one answer

**Exit criterion:** held-out DET clearly above the Ultrack baseline's detection stage.

### Phase 3 — Linking (Weeks 4–7, overlaps Phase 2)

Escalating ladder — stop when the gain per week flattens:

1. **Nearest neighbour + motion model.** After global drift correction, cell displacement
   between frames is small; a Kalman/constant-velocity prior gets surprisingly far.
2. **Global bipartite matching per frame pair** (Hungarian) with a cost combining distance,
   volume/intensity similarity, and local-neighbourhood consistency.
3. **ILP / min-cost-flow over a temporal window** (Ultrack's core). Handles appearance,
   disappearance, division, and merge resolution jointly under global constraints.
4. **Learned edge affinities.** Replace hand-tuned costs with a model:
   - **Trackastra** (transformer over detection sets, pretrained on microscopy) — fine-tune it
   - or a GNN over the candidate graph (Cell-Tracker-GNN style)
   Feed learned affinities *into* the ILP rather than replacing it — learned scoring +
   combinatorial feasibility is the combination that wins.

**Exit criterion:** each rung measured against the same held-out set; keep the best.

### Phase 4 — Divisions (Week 7)

Divisions are rare, structurally important, and disproportionately expensive to get wrong
(a missed division corrupts an entire subtree's lineage).

- [ ] Dedicated division classifier on candidate parent→(child, child) triplets: local 3D+t
      crop → CNN, plus features (volume drop, roundness spike, intensity change)
- [ ] Enforce biological constraints in the ILP: ≤2 children, no merges, division cooldown
- [ ] Report a division-specific precision/recall separate from the aggregate metric

### Phase 5 — Track repair & gap closing (Week 8)

- [ ] Close gaps across ≥1 missed frame (detection FNs are the dominant error — recovering
      them at the track level is a direct 10-weight win each)
- [ ] Re-detect at *predicted* locations where a track breaks: run a low-threshold local
      search where the motion model says a cell should be. Cheap, high yield.
- [ ] Prune implausibly short orphan tracks — *but* verify against the metric first; given
      FP costs only 1, pruning is often net-negative.

### Phase 6 — Ensembling & final push (Week 9)

- [ ] Detection ensembles as multi-hypothesis input to the ILP (Ultrack is *designed* for this)
- [ ] Seed/fold ensembling of the linking model
- [ ] Final metric-directed hyperparameter sweep on the full pipeline
- [ ] Blend candidates: keep several full pipelines and pick by held-out score, not by feel

### Phase 7 — Hardening & write-up (Week 10)

- [ ] Runtime: chunked zarr I/O, mixed precision, streaming inference — must fit the budget
- [ ] Deterministic seeds, pinned deps, clean rerun from scratch
- [ ] Reproducible submission notebook
- [ ] Solution write-up (this is a *research* competition — clarity and reproducibility carry
      weight beyond the raw score)

---

## 4. Differentiators — where we expect to actually beat the field

Ranked by expected value:

1. **Metric-directed thresholding.** Most competitors will tune detection on F1/IoU. The
   FN:FP = 10:1 asymmetry means the metric-optimal operating point is materially different.
   Nearly free to exploit; measurable in week 1.
2. **Global drift correction before linking.** Light-sheet embryo timelapses drift and the
   embryo itself moves/grows. Rigidly registering consecutive volumes first shrinks the
   per-cell displacement distribution dramatically, which makes *every* downstream linker
   better. Easy to skip, big payoff.
3. **Multi-hypothesis segmentation + ILP** instead of committing to one segmentation. This is
   Ultrack's key idea and it generalizes: never force an irreversible decision in stage B that
   stage C has more information to make.
4. **Learned affinities inside a combinatorial solver**, not instead of it.
5. **Re-detection at track breaks.** Directly attacks the highest-weight error class using
   information (motion continuity) that the detector alone doesn't have.
6. **Rigorous sequence-level validation.** Many teams will leak via random frame splits, then
   overfit the LB. A trustworthy CV is a compounding advantage over 10 weeks.

---

## 5. Validation protocol (non-negotiable)

- Split by **whole sequence / embryo**; if impossible, by contiguous time blocks + spatial crops.
- Score with the **same code path** as the leaderboard metric.
- Track CV and LB in a single table; investigate any divergence immediately rather than
  trusting the LB.
- Keep a fixed "golden" held-out sequence never used for any tuning decision until Week 9.
- Every experiment gets a row: config hash, CV, LB, runtime, notes. No undocumented runs.

---

## 6. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Data is TB-scale, local disk/GPU insufficient | Blocks everything | Assess in Phase 0; chunked zarr, train on crops, rent GPU if needed |
| Metric is not what we assume (e.g. CHOTA-weighted) | Mis-optimized pipeline | Phase 0 exit criterion is exact local reproduction |
| Runtime cap in code competition | Late-stage rework | Profile from Phase 1; keep a fast path always working |
| Strong z-anisotropy breaks off-the-shelf 3D models | Poor detection | Resample to isotropic physical spacing early |
| Photobleaching / intensity drift across time | Detector degrades late in sequence | Per-frame percentile normalization; check DET vs. frame index |
| Overfitting the public LB | Shakeup on private | Sequence-level CV + golden holdout |
| Ultrack is already near the ceiling | Marginal gains only | Attack its specific failure modes (from Phase 1 profiling), not the average case |
| Time: 10 weeks, ambitious scope | Incomplete pipeline | Phases are ordered so that stopping after *any* phase still leaves a valid submission |

---

## 7. Weekly milestones

| Week | Milestone | Ship |
|---|---|---|
| 1 | Data + rules understood, local scorer trusted, CV split fixed | Trivial submission |
| 2 | Ultrack baseline reproduced and profiled | Baseline submission |
| 3–4 | 3D detector trained, beats baseline detection | Detector + simple linker |
| 5 | Detector ensemble + metric-directed thresholds | Improved submission |
| 6–7 | ILP linking with learned affinities | Major submission |
| 7 | Division handling | — |
| 8 | Gap closing + re-detection at breaks | Improved submission |
| 9 | Ensembling, final sweeps, golden holdout check | Best submission |
| 10 | Hardening, reproducibility, write-up | Final + solution post |

---

## 8. Immediate next actions

1. Regenerate the Kaggle API token (kaggle.com/settings → API → Create New Token) — the
   current key 401s and everything is blocked behind it.
2. `make data` → pull competition files, inspect structure.
3. Fill in every `[ ]` in §0 from the actual rules pages.
4. Build and validate `ctrack.eval` against a known submission.
5. Make the trivial submission. Get on the board.

---

## References

- Ultrack — *Nature Methods* 2025, Royer Group · https://github.com/royerlab/ultrack
- Zebrahub — https://zebrahub.org
- Cell Tracking Challenge metrics — http://celltrackingchallenge.net/evaluation-methodology/
- `traccuracy` (Python CTC/CHOTA metrics) — https://github.com/live-image-tracking-tools/traccuracy
- CHOTA metric — https://arxiv.org/abs/2408.11571
- Trackastra — https://github.com/weigertlab/trackastra
- StarDist-3D — https://github.com/stardist/stardist
