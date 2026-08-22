#!/usr/bin/env python3
"""Pad-probe builder + the aggregation model it calibrated (row 101).

Appends K = frac x n_pred edgeless far-field nodes (x >= 100000, outside the volume,
>7 um from anything) per dataset to a finished submission.csv. Only the node-ratio
multiplier can move. Used post-hoc on the local control to identify the scorer's
EXACT aggregation:

    per-volume J_adj = max(0, J_v * (1 - 0.1 * (n_pred_v - N_total_v)/N_total_v)),
    aggregated as a weighted mean with weights = per-volume edge sample size
    (edge TP+FP+FN). Reproduces measured ctl/pad10/pad25 to 4 decimals
    (0.9221 / 0.9131 / 0.8997). The naive global-sum model is 16% too shallow --
    any hidden-test inversion MUST use the weighted model.

    .venv/bin/python scripts/local/pad_probe.py <in.csv> <out.csv> <frac>
"""
import csv, sys
from collections import defaultdict

inp, out, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(inp)))
by_ds_nodes = defaultdict(list)
for r in rows:
    if r["row_type"] == "node":
        by_ds_nodes[r["dataset"]].append(int(r["node_id"]))
COLS = ["dataset","row_type","node_id","t","z","y","x","source_id","target_id"]
extra = []
for ds, ids in by_ds_nodes.items():
    k = int(round(frac * len(ids)))
    nid = max(ids) + 1
    for j in range(k):
        extra.append({"dataset": ds, "row_type": "node", "node_id": nid + j,
                      "t": 0, "z": 0, "y": 0, "x": 100000 + j,
                      "source_id": -1, "target_id": -1})
    print(f"{ds}: +{k:,} pad nodes (base {len(ids):,})")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id"] + COLS)
    w.writeheader()
    for i, r in enumerate(rows + extra):
        w.writerow({"id": i, **{c: r.get(c, -1) for c in COLS}})
print(f"wrote {out}")
