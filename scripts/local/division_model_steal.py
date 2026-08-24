#!/usr/bin/env python
"""Model-guided daughter steal — the untested cell of rows 115 x 116.

Row 115: stealing weak links WITHOUT a model loses (wrong selection).
Row 116: the model WITHOUT stealing cannot reach correct daughters (no pool — they are
claimed by continuing tracks; TP pinned at 4 for every K).
This script tests the combination: the DIVISION MODEL picks which parent forks (prob
floor + tight K), and its nearest non-child neighbour at t+1 may be CLAIMED — in which
case the existing incoming edge is removed (the steal). Protections: never steal from a
parent that already has 2 children; one action per parent and per target; dt=1 only.

Usage: division_model_steal.py <base_submission.csv> <out_dir>
Env: DIVDET_OUT (model dir). Scores every (K, prob_floor) variant with the official
scorer. Reject-only: this either clearly beats 0.9412 locally or the postproc division
book closes for good.
"""
from __future__ import annotations

import csv
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from division_detector import CAND_RADIUS, extract  # noqa: E402

BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
N = os.environ["DIVDET_OUT"]

with open(f"{N}/div_model.pkl", "rb") as f:
    model = pickle.load(f)["model"]

rows = list(csv.DictReader(open(BASE)))
fields = list(rows[0].keys())
nodes = defaultdict(dict)
out_edges = defaultdict(lambda: defaultdict(list))   # ds -> source -> [target]
in_edge_row = {}                                     # (ds, target) -> row index in `rows`
outdeg = defaultdict(int)
for idx, r in enumerate(rows):
    if r["row_type"] == "node":
        nodes[r["dataset"]][int(r["node_id"])] = (
            int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
    else:
        ds, s, t = r["dataset"], int(r["source_id"]), int(r["target_id"])
        out_edges[ds][s].append(t)
        outdeg[(ds, s)] += 1
        in_edge_row[(ds, t)] = idx

scored = []  # (prob, ds, parent, candidate, candidate_is_claimed)
for v in sorted(nodes):
    nd = nodes[v]
    by_t = defaultdict(list)
    for nid, (t, *_) in nd.items():
        by_t[t].append(nid)
    trees = {t: (cKDTree(np.array([nd[n][1:] for n in ns])), ns)
             for t, ns in by_t.items() if ns}
    want = defaultdict(list)
    pair = {}
    for p, (t, z, y, x) in nd.items():
        if outdeg[(v, p)] != 1:
            continue
        tr = trees.get(t + 1)
        if tr is None:
            continue
        own_child = out_edges[v][p][0]
        best = None
        for j in tr[0].query_ball_point(np.array([z, y, x]), CAND_RADIUS):
            c = tr[1][j]
            if c == own_child:
                continue
            # claimed candidates allowed IF their current parent keeps another child
            claimed = (v, c) in in_edge_row
            if claimed:
                cur_parent_row = rows[in_edge_row[(v, c)]]
                cur_parent = int(cur_parent_row["source_id"])
                if outdeg[(v, cur_parent)] != 1:
                    continue  # stealing would break a division or a repaired branch
                if cur_parent == p:
                    continue
            d = float(np.linalg.norm(np.array(nd[c][1:]) - np.array([z, y, x])))
            if best is None or d < best[0]:
                best = (d, c, claimed)
        if best is None:
            continue
        want[t].append(((v, p), z, y, x))
        pair[(v, p)] = (best[1], best[2])
    got = extract(v, want)
    if got:
        keys = list(got)
        P = model.predict_proba(np.nan_to_num(
            np.array([got[k] for k in keys], dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0))[:, 1]
        scored += [(float(pr), k[0], k[1], pair[k][0], pair[k][1])
                   for pr, k in zip(P, keys)]
    print(f"  {v}: scored {len(got):,} (running {len(scored):,})", flush=True)

scored.sort(key=lambda r: -r[0])

for K, floor in [(50, 0.90), (100, 0.90), (250, 0.80), (100, 0.0)]:
    take = [s for s in scored[:K] if s[0] >= floor]
    drop_rows = set()
    add = []
    used_targets = set()
    used_parents = set()
    n_steal = 0
    for pr, ds, p, c, claimed in take:
        if (ds, c) in used_targets or (ds, p) in used_parents:
            continue
        if claimed:
            drop_rows.add(in_edge_row[(ds, c)])
            n_steal += 1
        add.append((ds, p, c))
        used_targets.add((ds, c))
        used_parents.add((ds, p))
    name = f"msteal_k{K}_f{int(floor*100)}"
    d = OUT / name
    d.mkdir(exist_ok=True)
    with open(d / "submission.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        i = 0
        for idx, r in enumerate(rows):
            if idx in drop_rows:
                continue
            r = dict(r); r["id"] = i; i += 1
            w.writerow(r)
        for ds, p, c in add:
            w.writerow({"id": i, "dataset": ds, "row_type": "edge", "node_id": -1,
                        "t": -1, "z": -1, "y": -1, "x": -1,
                        "source_id": p, "target_id": c}); i += 1
    print(f"\n=== {name}: {len(add)} forks ({n_steal} stolen)")
    subprocess.run([str(REPO / ".venv/bin/python"),
                    str(REPO / "scripts/local/score_official.py"),
                    str(d / "submission.csv")], check=False)
