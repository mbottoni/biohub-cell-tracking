# Learned detector — design notes

## The sparse-label problem (must solve before training)

The ground truth annotates only ~6% of cells (mean; 0.13–20% range), and — verified across
several volumes — the labeled cells are a **spatially and temporally representative sample**,
not a complete sub-region:

| volume | nodes | density | frames covered | spatial extent |
|---|---|---|---|---|
| 6bba_062c8d37 | 930 | 15.4% | 99/100 | full (z 1–62, y 2–239, x 4–252) |
| 6bba_07e24132 | 359 | 1.7% | 100/100 | full |
| 44b6_0113de3b | 52 | 0.2% | 52/100 | full |

So the labeled cells are scattered throughout the volume among many **unlabeled real nuclei**.

**Consequence:** a naive heatmap detector — Gaussian blobs at GT points as positives,
everything else as background — trains against contaminated negatives. ~94% of the "background"
is actually real nuclei. That teaches the model to suppress true detections, which is the
opposite of what we want (recall drives the score, per the sweep).

### Options

1. **Masked / positive-unlabeled loss.** Supervise the loss *only* in a small radius around GT
   points (clear positives) and in regions confidently far from any cell (clear negatives from
   low intensity), leaving the ambiguous majority unsupervised. Avoids penalizing correct
   detections of unlabeled cells.
2. **Peak-ranking / metric-learning.** Don't regress a dense map; learn to rank candidate peaks
   so annotated cells score high. Sidesteps the negative-labeling question.
3. **Detect-all then filter.** Train an intensity-driven "any nucleus" detector (the unlabeled
   cells become weak positives, since they *are* nuclei), accepting that it detects everything,
   then rely on the linker + node-budget tradeoff. Closest to the classical baseline.
4. **Self-training.** Bootstrap: detect all bright blobs, treat high-confidence ones as
   pseudo-labels, refine. Risky with contaminated labels.

Leaning toward **(1)**: a foreground/background U-Net with a loss masked to (near-GT ∪
confident-background), which respects the label semantics. Validate the masking against the
real metric — recall on held-out GT points is the proxy that matters, and edge Jaccard is the
real target.

## Compute

Volumes are 64³ isotropic (whole volume, no tiling). A small 3D U-Net fits comfortably; train
on Apple MPS locally, or a Kaggle GPU. Inference at submission is one forward pass per
timepoint × 100 × 4 volumes — trivial.

## Targets

Gaussian-splat GT node centres on the isotropic grid, σ ≈ one nucleus radius (~2.5 µm → ~1.5
isotropic voxels). Predict a heatmap; NMS peaks as detections, exactly as the blob baseline
does — so the detector is a drop-in replacement for `ctrack.detect.blob` and everything
downstream (linking, gap closing, submission) is reused unchanged.

## Validation

Embryo-stratified 5-fold (`ctrack.data.splits.kfold`, now known leakage-free). Report held-out
edge Jaccard via the verified metric. Remember `6bba` carries 85% of the signal — weight it.

## Open

- [ ] Confirm the masked-loss approach beats detect-all on held-out edge Jaccard
- [ ] σ and NMS radius sweep
- [ ] Does temporal context (2–3 frame input) help detection, or only linking?
