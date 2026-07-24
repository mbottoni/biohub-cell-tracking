# Plan to Win: Biohub — Cell Tracking During Development

**Competition:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
**Updated:** 2026-07-23 (v2 — rewritten after gaining API access; v1 was written blind and got
most of this wrong)

| | |
|---|---|
| **Prize** | $60,000 USD |
| **Deadline** | **2026-09-29** 23:59 UTC (~9.5 weeks left; comp opened 2026-06-29) |
| **Team merger / new entrant deadline** | **2026-09-22** |
| **Type** | Research · **kernels-only** (notebook submission) · 5 subs/day · max team 5 |
| **Metric** | **Custom** ("CZI Biohub Zebrafish 133605") — *not* standard CTC TRA |
| **Rules accepted** | ✅ (`userHasEntered: true`) |
| **Public LB state** | top public notebooks at **~0.897**; strong rule-based ~0.857 |

---

## 1. Ground truth about the problem (verified, not assumed)

### Data — full details in [docs/data.md](docs/data.md)
- **87.6 GB / 24,886 files / 204 stores**: 199 `train/`, 4 `test/`, `sample_submission.csv`.
  Downloaded to `/Volumes/extSSD/kaggle/biohub/raw` (won't fit on the internal disk).
- Zarr **v3**, **blosc2**-compressed, one chunk per timepoint (`0/c/{t}/0/0/0`).
- Every volume is **`(100, 64, 256, 256)` uint16** — 100 frames of a **104 µm cube**. These are
  small crops, not whole embryos, so local training is entirely feasible.
- **Voxel size `[1.625, 0.40625, 0.40625]` µm (Z, Y, X)** → exactly **4× anisotropy in Z**.
  Hence the community's `SUBSAMPLE = [1, 4, 4]`: downsampling Y/X by 4 gives isotropic 1.625 µm.
- Only **two source embryos** (`6bba`: 128 crops, `44b6`: 71). CV folds must group by embryo.
- `zarr.json` ships `image_statistics.quantiles` — use them for normalization.
- Ground truth is **`.geff`** (zarr-backed graph: node ids, `t/z/y/x` props, edge ids), present
  for `train/` only. Read with `geff`, manipulate with `tracksdata`.

### The visible test set is a development sample, not the scored set
All 4 test IDs also exist in `train/` — and are **byte-identical** (same `zarr.json` md5, same
chunk md5, same shape). The train copies carry `.geff` ground truth.

This is normal for a kernels-only competition: `test/` is swapped for hidden data at rerun. It
can't be the scored set — otherwise everyone would read the train geff and score 1.0, and the
public LB is ~0.897.

**Use it as a fixture, not a validation set.** We hold exact GT for the four volumes that flow
through the real submission code path, which makes them ideal for verifying the local metric
end-to-end (same loader, same inference, same CSV writer, known answer). Model selection must
use a held-out, embryo-grouped split of the 199 train stores.

### Submission format — a point graph, *not* segmentation masks
A single `submission.csv` with columns:

```
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

- `row_type == "node"` → a detected cell centroid at `(t, z, y, x)`, identified by `node_id`
- `row_type == "edge"` → a link, `source_id → target_id` (unused fields are `-1`)
- The graph must be a **directed forest**: unique `node_id`s, no duplicated `target_id`
  (each cell has ≤1 parent). Divisions = a node with 2 children.

**Consequence: segmentation is not scored at all.** Everything in v1 about SEG, IoU, StarDist
masks and CTC TIFF layout is dead. This is **point detection + graph linking**.

### The metric
Custom. From the reference implementation used by top notebooks
(`biohub_tracking.metrics`: `evaluate`, `node_recall`, `per_sample_metrics`, `summarise`) it is
built from:
- `node_recall(pred_graph, truth_graph)`
- edge TP / FP / FN
- division TP / FP / FN
- normalized against **`GeffMetadata.extra["estimated_number_of_nodes"]`**

That last term is the whole ballgame — see §2.

**Open item:** we have not yet read the exact formula. Extracting it from the support pack /
evaluation page is task #1. Everything below is provisional until we have it.

---

## 2. The elephant: the metric is exploitable, and the field knows it

Two of the most-upvoted public notebooks are literally titled *"metric hack"*. Here is the
exploit, reconstructed from `kirneo/metric-hack-last-call-update`:

```python
MAX_COMPONENTS = 3400
FORKS = 32
# add one hub node at t=-1000, (z,y,x) = (-10000,-10000,-10000)
# add an edge hub -> root, for the root of every connected component
# then chain 32 fake "fork" nodes, each with 2 children, at t = -999, -997, ...
```

The injected nodes sit at **negative timepoints and impossible coordinates**, so they can never
match a ground-truth node. What they do:
1. **Collapse ~3400 connected components into one**, and
2. **inject 32 free division events**.

That this *helps* tells you the metric rewards low fragmentation and/or division recall, while
**barely penalizing false-positive nodes**. And there's a good scientific reason for that: the
GT is **partially annotated** — hence `estimated_number_of_nodes`. You cannot penalize a
predicted cell that might be a real, un-annotated one. The exploit weaponizes that leniency.

`MAX_COMPONENTS` and `FORKS` are visibly LB-probed (the code comments track
`20 → 30 → 40`, `2600 → 3400 → "хочу 3800"`).

### Position on this — my recommendation

**Understand it, reproduce it for calibration, but do not build the solution on it.**

- It is public, so it confers **no edge** — everyone at the top has it. It's table stakes for
  LB position, not a differentiator.
- It is **fragile**: this is a Research competition whose stated purpose is to establish
  community benchmarking standards. Hosts patch metrics mid-competition, and Kaggle rules let
  them address degenerate submissions. A solution whose score depends on the hub trick can
  evaporate.
- A **$60k research prize with a required write-up** is decided partly by hosts reading the
  method. "We added a fake hub node" is not a winning write-up.

So: keep the legitimate pipeline as the product, keep the hack as a **detachable
post-processing flag** (`--augment-hub`) that we can toggle to measure its LB delta and switch
off instantly if the metric changes. Track both scores separately, always.

> **Decision needed from you:** confirm this framing. If you'd rather optimize LB position
> at any cost, the plan reorders. I'd argue against it.

---

## 3. What the field has converged on (our starting point, not our destination)

From `kirneo/metric-hack-last-call-update` and `yusuketogashi/lb897-baseline`:

```
zarr volume (T,Z,Y,X)
  └─ subsample [1,4,4]  → isotropic 1.625 µm
      └─ TemporalUNet3D on 64³ patches, TIME_LENGTH=2
          ├─ head A: point probability   → max-pool NMS peaks, POINT_THRESHOLD ≈ 0.955
          └─ head B: per-voxel edge features
              └─ SimpleNodeTransformer over (nodes_t, nodes_t+1) → edge probabilities
                  └─ candidate edges: strong ≥0.50, top-3 parents ≥0.25,
                     top-1 child ≥0.18, max displacement 10 µm
                      └─ ILP (tracksdata): edge -1.0, appear 0.0,
                         disappear +1.4, division +1.0
                          └─ post: 1-frame gap closing (≤10 µm, refine to local
                             intensity centroid), min track length 3,
                             trajectory smoothing (w=0.15, ≤1.5 µm shift)
                              └─ [hub + fake forks]
```

Notable: `POINT_THRESHOLD = 0.955` is *high*, and there is heavy TTA + multi-GPU. Detection is
tuned tight, then gaps are repaired downstream.

**We should reproduce this to ~0.89 first.** Reproducing the public state of the art is week 1,
not the goal. Everything after is where the prize is.

---

## 4. Where I think the remaining headroom is

Ranked by expected value. All of these attack the *legitimate* metric.

1. **Reverse-engineer the exact metric and optimize it directly.** The public notebooks tune
   `POINT_THRESHOLD` and ILP weights by LB probing at 5 subs/day. With the true formula and a
   trustworthy local harness we can sweep offline over thousands of configs. Given the metric
   weights recall and divisions asymmetrically, the jointly-optimal operating point is almost
   certainly not where hand-probing landed.
2. **Divisions.** The metric scores division TP/FP/FN explicitly, and the hack shows divisions
   are worth a lot. Everyone currently gets divisions as a *byproduct* of the ILP. A dedicated
   division classifier on candidate parent→(child,child) triplets — local 3D+t crop CNN plus
   volume/roundness/intensity features — is an obvious, unexploited lever.
3. **Longer temporal context.** The public model uses `TIME_LENGTH = 2` — pairwise frames only.
   A tracker attending over a window of 3–5 frames resolves ambiguities (crossings, near-misses,
   the frame *after* a division) that no pairwise model can. This is the biggest architectural
   gap in the public solution.
4. **Gap closing beyond one frame.** Public post-processing closes exactly 1-frame gaps. Multi-
   frame gap closing with motion-model re-detection (search for a peak at the *predicted*
   location with a lowered threshold) directly converts FN nodes into TPs.
5. **Global drift/rigid registration before linking.** Still unexploited as far as I can see in
   the public code. Embryos drift and grow; removing bulk motion shrinks the displacement
   distribution and makes every linker better.
6. **Ensembling detection heads** and feeding the *union* of candidates into the ILP rather
   than one thresholded set — the ILP is built to arbitrate among hypotheses.

---

## 5. Validation protocol (non-negotiable)

- **Split by whole dataset/embryo.** Never random frames — adjacent timepoints are ~identical
  and a frame-level split leaks catastrophically.
- Score with a **local reimplementation of the official metric**, verified to reproduce a known
  LB score to within a small constant.
- Track **two numbers per experiment**: score with the hub augmentation and without. If they
  ever move in opposite directions, trust the clean one.
- One **golden holdout** dataset, untouched by any tuning decision until the final two weeks.
- 5 subs/day is scarce. Spend them on *reducing uncertainty* (probing the metric, confirming
  CV↔LB offset), not on hill-climbing that could have been done offline.

---

## 6. Phases

### Phase 0 — Recon (this week) ← *in progress*
- [x] Kaggle API access (bearer token; CLI ≥2.x required, 1.5.13 can't do it)
- [x] Confirm prize / deadline / kernels-only / submission format / voxel geometry
- [ ] Finish download, unpack, characterize: how many datasets, T per dataset, volume shape,
      node counts, division rate, annotation density (how partial *is* the GT?)
- [ ] **Extract the exact metric formula** and build `ctrack.eval` around it
- [ ] Establish the CV split and the golden holdout
- [ ] Reproduce the public baseline end-to-end; confirm our local score ≈ its LB

### Phase 1 — Baseline parity (week 2)
- [ ] UNet3D detection + transformer edges + ILP running in our own repo, ~0.89 clean
- [ ] Full error decomposition: node FN, node FP, edge FP/FN, division FP/FN, per dataset
- [ ] This table decides all subsequent priorities. Do not skip it.

### Phase 2 — Metric-directed tuning (weeks 3–4)
- [ ] Offline sweep of `POINT_THRESHOLD`, edge thresholds, ILP weights against the real metric
- [ ] Expect the single largest cheap gain here

### Phase 3 — Model improvements (weeks 4–7)
- [ ] Temporal window > 2 frames (highest-value architectural change)
- [ ] Dedicated division classifier
- [ ] Drift/rigid registration preprocessing
- [ ] Multi-frame gap closing with motion-guided re-detection

### Phase 4 — Ensembling & hardening (weeks 8–9)
- [ ] Seed/fold ensembles; union-of-candidates into ILP
- [ ] Notebook must run inside the kernel time/GPU limit — profile from Phase 1, not week 9
- [ ] Golden holdout check

### Phase 5 — Write-up (week 10)
- [ ] Research competition: reproducibility and a clear method write-up carry real weight

---

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **Host patches the metric** (plausible given public hacks) | Score collapse for hack-dependent solutions | Keep the hack detachable; optimize the clean metric as primary |
| Hack-based solutions excluded from prizes | Wasted effort if we build on it | Same |
| Kernel runtime limit | Late rework | Profile from Phase 1; keep a fast path always green |
| We're 3.5 weeks late to a 13-week comp | Less time than the field | Start from the public baseline, don't rebuild from scratch |
| Partial GT annotation misleads local CV | Wrong decisions all the way down | Understand `estimated_number_of_nodes` before trusting any local number |
| 5 subs/day | Slow LB feedback | Offline sweeps against a trusted local metric |
| Public notebooks leapfrog us | Being 0.89 in week 9 is worthless | Monitor top notebooks weekly; our edge must be in §4, not in reproduction |

---

## 8. Immediate next actions

1. Finish download; unpack and characterize the data (`notebooks/01_eda.ipynb`)
2. **Extract the exact metric implementation** — highest priority, everything depends on it
3. Build `ctrack.eval` against it, verify on the public baseline
4. Fix the CV split + golden holdout
5. Reproduce the public pipeline in-repo, get the error decomposition table

---

## References

- Ultrack — *Nature Methods* 2025 · https://github.com/royerlab/ultrack
- `tracksdata` / `geff` — Royer-lab graph tooling (used by the public baselines)
- Zebrahub — https://zebrahub.org
- Public notebooks worth reading: `inversion/cell-tracking-getting-started-w-nearest-neighbor`
  (official starter, defines the format), `yusuketogashi/lb897-baseline`,
  `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` (clean CV setup),
  `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction`,
  `kirneo/metric-hack-last-call-update` (the exploit)
