---
name: metric-and-contamination
description: "The Biohub scoring metric internals, the visible-test contamination, and local-vs-LB offset"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T17:50:53.774Z
---

**Metric** (verified field-for-field against reference `biohub_tracking.metrics`):
`total_node_ratio=(N_pred-N_total)/N_total`; `edge_jaccard=|matched|/(|GT edges|+|valid pred edges|-|matched|)`;
`adj_edge_jaccard=max(0, edge_jaccard*(1-0.1*total_node_ratio))`;
`score=adj_edge_jaccard + 0.1*division_jaccard`.
Only edges touching ANNOTATED GT nodes count. Node match = centroid distance <=7µm (anisotropy-scaled).
Crucially, `N_total` in total_node_ratio is the volume's ESTIMATED total cell count
(`estimated_number_of_nodes` in the geff, ~20-30k/vol), NOT the sparse annotated count -> the
detector must predict ~all cells, and over-detection is punished hard through the adj multiplier.

**Contamination**: the 4 visible test volumes are BYTE-IDENTICAL to 4 train volumes. Pretrained
pipelines score inflated locally (E005: 0.969 local vs 0.888 LB). Avoid by scoring on a proper
held-out CV split (golden holdout of 8, seed 0, via `ctrack.data.splits.hold_out_golden`).

**Local-vs-LB offset**: local metric runs ~0.04 HIGHER than LB (blob baseline: local 0.644 ->
LB 0.601). Budget for this gap before submitting.

See [[competition-objective]], [[detector-status]].
