---
name: pipeline-architecture
description: "Module map of the owned cell-tracking pipeline (detector, linkers, edge model, eval)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T18:34:52.936Z
---

Owned pipeline modules (src/ctrack/), built to incorporate the linajea/Ultrack/Trackastra ideas:

- **detect/unet3d.py** — `UNet3D(base, out_offset)`; optional 3-ch offset-to-parent head
  (`forward(x, return_offset=True)`). Losses: `masked_bce_loss`, `masked_offset_loss`.
- **detect/targets.py** — `build_frame_target` (heatmap+trusted mask), `build_offset_target`
  (per-voxel parent-offset + validity mask from geff parent links).
- **detect/dataset.py** — `DetectionFrames(with_offset=)`; 3-tuple or 5-tuple (adds offset,
  offset_mask); offset-aware flips negate the flipped component.
- **detect/train.py** — `train_fold(pos_weight=4, offset_weight=, ...)`; node-ratio-aware
  selection (see [[detector-status]]).
- **detect/infer.py** — `predict_heatmap(tta=)` flip-TTA, `predict_heatmap_ensemble` (work with
  existing weights).
- **link/greedy.py** — `link_greedy`: fast divisions-aware decoder (<=1 parent, <=2 children), no
  ILP. Use this to avoid the slow SCIP fallback (no Gurobi license locally).
- **link/candidates.py** — `build_nodes` (multi-hypothesis candidates + per-node confidence),
  `prune_singletons`.
- **link/ilp.py** — `link_ilp(node_scores=, edge_affinity=)`; tracksdata global ILP.
- **link/edge_model.py** — `EdgeMLP`, `pair_features(tgt_offset=)` (motion-agreement features),
  `train_edge_model` (fits on GT graphs in seconds), `affinity_map`.
- **eval/biohub.py** — verified metric (`evaluate`, `per_sample_metrics`, `summarise`).

**scripts/local/eval_full.py** — integrated evaluator: detector(+TTA/ensemble) -> multi-hypothesis
candidates -> greedy/ILP(+edge model) -> metric, `--sweep` thresholds. This is the fast local
golden-8 scorer to run before spending a Kaggle submission. Golden-8 held out via
`ctrack.data.splits.hold_out_golden(n=8, seed=0)`.
