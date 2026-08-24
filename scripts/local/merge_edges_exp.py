#!/usr/bin/env python
"""Cross-family edge-ensemble experiments: 917 (donor) edges onto the div917 (base) graph.

PLAN 3.1f.2, unlocked by row 115 (local 917 replay exists). Matches donor nodes to base
nodes per (dataset, frame) with a KD-tree in micrometres, maps donor edges through the
match, then scores merge strategies:

  transplant_divs  donor DIVISIONS only: where a mapped donor parent has 2 mapped
                   daughters and the base parent has exactly 1 child, add the second
                   edge (a division-pool source rows 114/115 could not create).
  union_add        add every mapped donor edge whose target has no base parent and
                   whose source has < 2 children (recall play, no node changes).
  drop_leaf       drop base edges NOT confirmed by the donor whose target is a leaf
                   (conservative precision play).
  intersect        drop ALL base edges not confirmed by the donor (aggressive control
                   arm — expected to lose; bounds the confirmation signal's value).

Invariants preserved: single parent per target, <= 2 children per source, dt == 1 only,
node rows untouched. Every variant is scored with the official scorer. Reject-only.

Usage: merge_edges_exp.py <base_submission.csv> <donor_submission.csv> <out_dir>
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 3.0

BASE, DONOR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]


def load(path):
    nodes = defaultdict(dict)   # dataset -> node_id -> (t, z, y, x)
    edges = defaultdict(list)   # dataset -> [(source, target)]
    with open(path) as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds][int(r["node_id"])] = (int(r["t"]), int(r["z"]), int(r["y"]), int(r["x"]))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))
    return nodes, edges


def match_nodes(bnodes, dnodes):
    """donor_id -> base_id per dataset, nearest-neighbour in um within MATCH_UM, per frame."""
    mapping = {}
    by_t_base = defaultdict(list)
    for nid, (t, z, y, x) in bnodes.items():
        by_t_base[t].append((nid, z, y, x))
    trees = {}
    ids_by_t = {}
    for t, rows in by_t_base.items():
        pts = np.array([[z, y, x] for _, z, y, x in rows], float) * SCALE
        trees[t] = cKDTree(pts)
        ids_by_t[t] = [nid for nid, *_ in rows]
    for did, (t, z, y, x) in dnodes.items():
        tree = trees.get(t)
        if tree is None:
            continue
        dist, idx = tree.query(np.array([z, y, x], float) * SCALE)
        if dist <= MATCH_UM:
            mapping[did] = ids_by_t[t][int(idx)]
    return mapping


def build_variant(strategy, bnodes, bedges, dedges, mapping):
    """Return the base edge list transformed by the strategy."""
    children = defaultdict(list)
    parent = {}
    for s, t in bedges:
        children[s].append(t)
        parent[t] = s
    confirmed = set()
    donor_pairs = []
    for ds_, dt_ in dedges:
        bs, bt = mapping.get(ds_), mapping.get(dt_)
        if bs is None or bt is None or bs == bt:
            continue
        if bnodes[bt][0] - bnodes[bs][0] != 1:
            continue
        donor_pairs.append((bs, bt))
        confirmed.add((bs, bt))

    if strategy == "transplant_divs":
        dchildren = defaultdict(set)
        for bs, bt in donor_pairs:
            dchildren[bs].add(bt)
        out = list(bedges)
        added = 0
        for bs, kids in dchildren.items():
            if len(kids) < 2 or len(children.get(bs, [])) != 1:
                continue
            existing = children[bs][0]
            for k in kids:
                if k != existing and k not in parent and len(children[bs]) < 2:
                    out.append((bs, k))
                    children[bs].append(k)
                    parent[k] = bs
                    added += 1
                    break
        return out, {"added": added}
    if strategy == "union_add":
        out = list(bedges)
        added = 0
        for bs, bt in donor_pairs:
            if bt in parent or len(children.get(bs, [])) >= 2 or (bs, bt) in set(out):
                continue
            out.append((bs, bt))
            children[bs].append(bt)
            parent[bt] = bs
            added += 1
        return out, {"added": added}
    if strategy == "drop_leaf":
        sources = set(children.keys())
        out = [(s, t) for s, t in bedges if (s, t) in confirmed or t in sources]
        return out, {"dropped": len(bedges) - len(out)}
    if strategy == "intersect":
        out = [(s, t) for s, t in bedges if (s, t) in confirmed]
        return out, {"dropped": len(bedges) - len(out)}
    raise ValueError(strategy)


def write_submission(path, all_nodes, all_edges):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        rid = 0
        for ds in sorted(all_nodes):
            for nid, (t, z, y, x) in sorted(all_nodes[ds].items()):
                w.writerow([rid, ds, "node", nid, t, z, y, x, -1, -1])
                rid += 1
            for s, t in all_edges[ds]:
                w.writerow([rid, ds, "edge", -1, -1, -1, -1, -1, s, t])
                rid += 1


bnodes, bedges = load(BASE)
dnodes, dedges = load(DONOR)
mappings = {ds: match_nodes(bnodes[ds], dnodes.get(ds, {})) for ds in bnodes}
for ds in sorted(bnodes):
    print(f"[match] {ds}: {len(mappings[ds])}/{len(dnodes.get(ds, {}))} donor nodes matched "
          f"to {len(bnodes[ds])} base nodes")

for strategy in ["transplant_divs", "union_add", "drop_leaf", "intersect"]:
    merged = {}
    stats_tot = defaultdict(int)
    for ds in bnodes:
        out, st = build_variant(strategy, bnodes[ds], bedges[ds], dedges.get(ds, []), mappings[ds])
        merged[ds] = out
        for k, v in st.items():
            stats_tot[k] += v
    dst = OUT / f"merge_{strategy}.csv"
    write_submission(dst, bnodes, merged)
    print(f"\n=== {strategy} {dict(stats_tot)}")
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "scripts/local/score_official.py"), str(dst)],
                   check=False)
