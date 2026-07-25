---
name: detector-status
description: Detector over-detection finding + node-ratio-aware fix; current training run state (2026-07-25)
metadata: 
  node_type: memory
  type: project
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T18:34:38.268Z
---

**Key finding (2026-07-25)**: the 3D U-Net detector over-detected WORSE as training "improved"
val peak-recall. On the golden-8 pipeline eval: epoch 0 -> score 0.402, ~42k nodes/vol; epoch 6
(recall 0.990) -> score 0.259, ~93k nodes/vol. Cause: peak-recall rewards firing everywhere, but
the metric penalizes total node count. True target is ~20-30k nodes/vol (the estimated total, see
[[metric-and-contamination]]). The heatmap saturates (flat threshold sweep, cliff only at 0.99).

**Fix (committed b70bb3a in src/ctrack/detect/train.py)**: select/early-stop on
`sel = recall * exp(-0.1*|node_ratio|)` where `node_ratio` compares predicted peak count to
`estimated_number_of_nodes/100`; `pos_weight` 20->4 to curb over-firing. `peak_recall()` now also
returns peak count. Kernel driver kernels/train/train_detector_gpu.py passes POS_WEIGHT and logs
sel/node_ratio. Fixed T4 run relaunched 2026-07-25 (~15 epochs, val-stride 8, patience 4);
best checkpoint uploads mid-run to Kaggle dataset `maruanbottoni/biohub-detector-weights`.

**Fix confirmed working (2026-07-25)**: new pos_weight=4 detector epoch-0 golden-8 sweep RESPONDS
to threshold (0.7->0.9 sheds nodes) instead of the old flat-until-0.99 saturation; score 0.429
(distance-ILP) vs old 0.402. Best T4 checkpoint stuck at epoch-0 (sel 0.847, node_ratio 1.06) for
a while — later epochs may over-detect more; investigate.

**Pipeline expansion (2026-07-25, commits after b70bb3a)** — implemented all of the literature
plan (see [[pipeline-architecture]]): fast greedy divisions-aware linker (link/greedy.py, no
SCIP), offset-to-parent head (UNet3D out_offset + build_offset_target + train_fold offset_weight),
learned edge model + motion features (link/edge_model.train_edge_model), multi-hypothesis
candidates (link/candidates.py + ilp node_scores), TTA/ensemble (detect/infer.py). Integrated in
scripts/local/eval_full.py. **Measured on golden-8, same epoch-0 detector: greedy + multi-hyp =
score 0.489 / edge_J 0.506 vs distance-ILP 0.429 (+0.06).** Next: train the OFFSET detector, fit
the edge model on DETECTOR candidates with motion features (the dominant-edge_jaccard win), add
TTA/ensemble. Deferred: temporal 2-frame input (redundant with offset head). Data at
/Volumes/extSSD/kaggle/biohub/{raw,gt}/train. See [[competition-objective]], [[metric-and-contamination]].
