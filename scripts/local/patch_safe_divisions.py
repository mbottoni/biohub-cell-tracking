#!/usr/bin/env python3
"""POST-LINK SAFE-DIVISION RECOVERY — ported from the public LB-0.917 line (row 96).

WHY THIS IS NOT A REPEAT OF ROWS 24/76/85/86.
Our division attempts all failed, but each failed on a mechanism this rule avoids:

  row 24/76  loosened the SAFE_DIV geometric gates (12/14.4/12) -> TP 3->3, FP 18->46.
             Conclusion: "widening the gates admits the missed events and still cannot
             pick them -- the bottleneck is SELECTION, not candidate generation."
  row 86     ranked candidates by APPEARANCE (AUC 0.863) and inserted top-K. Every K
             lost, because a useful fork re-wires an already-parented node and trips
             the cross-component / merged-branch FP rules.

This rule keeps the loose gates AND adds the selector row 76 said was missing -- and
the selector is TEMPORAL, not appearance and not static geometry (§35 killed those):

  * q must be an ORPHAN at t+1 (no predecessor). We only ever ATTACH A FREE NODE and
    never steal a parented one, so row 86's cross-component FP mode cannot fire.
  * p must have exactly one predecessor -> no forks at track starts.
  * MUTUAL nearest neighbour between c1 and q, both directions.
  * DIVERGENCE: dist(c1_next, q_next) - dist(c1, q) >= DIV_DIVERGE_UM. The two
    children must MOVE APART in the next frame. That is a look-ahead confirmation
    that they are two real cells rather than one cell detected twice -- which is
    precisely the false-positive population that sank row 86.
  * per-frame and global caps identical to our own shipped SAFE_DIV caps.

Operates on a finished submission.csv, so it costs no inference: apply, rescore,
compare. Writes a NEW file; never mutates the input (row 93).

    .venv/bin/python scripts/local/patch_safe_divisions.py <in.csv> <out.csv>
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])          # (z, y, x) um per voxel
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375
COLS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _d(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * SCALE))


def add_safe_divisions(nodes, edges, stats):
    """nodes: {nid: {'t':int,'xyz':np.array}}  edges: list[(src,dst)] -> n added."""
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b); pred[b].append(a)
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)

    orphan_tree, orphan_ids = {}, {}
    for t, ids in by_t.items():
        orph = [g for g in ids if not pred.get(g)]
        orphan_ids[t] = orph
        if orph:
            orphan_tree[t] = cKDTree(np.asarray([nodes[g]["xyz"] for g in orph], float) * SCALE)

    proposals = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        c1 = ch[0]; t = n["t"]
        if nodes[c1]["t"] != t + 1:
            continue
        if len(pred.get(p, [])) != 1:                 # mid-track parents only
            continue
        if _d(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = orphan_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = orphan_ids[t + 1][idx]
            if q == c1:
                continue
            d_pq = _d(n["xyz"], nodes[q]["xyz"])
            d_s = _d(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            d_q, q_idx = tree.query(nodes[c1]["xyz"] * SCALE)   # mutual-NN check
            if orphan_ids[t + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            # DIVERGENCE: the sisters must move APART next frame
            if _d(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q))

    stats["proposals"] = len(proposals)
    proposals.sort()
    used_p, used_q, per_frame = set(), set(), defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    stats["cap_global"] = max_global
    added = 0
    for _score, t, p, q in proposals:
        if added >= max_global:
            stats["hit_global_cap"] = 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            per_frame[t] += 0
            stats["blocked_frame_cap"] += 1
            continue
        edges.append((p, q)); succ[p].append(q); pred[q].append(p)
        used_p.add(p); used_q.add(q); per_frame[t] += 1; added += 1
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    args = ap.parse_args()
    assert args.inp != args.out, "refusing to overwrite the input"

    rows = list(csv.DictReader(open(args.inp)))
    by_ds = defaultdict(lambda: ([], []))
    for r in rows:
        by_ds[r["dataset"]][0 if r["row_type"] == "node" else 1].append(r)

    out_rows, total = [], 0
    print(f"{'dataset':<16}{'nodes':>9}{'edges':>9}{'proposals':>11}{'ADDED':>7}{'cap':>7}")
    print("-" * 59)
    for ds in sorted(by_ds):
        nrows, erows = by_ds[ds]
        nodes = {int(r["node_id"]): {"t": int(float(r["t"])),
                                     "xyz": np.array([float(r["z"]), float(r["y"]), float(r["x"])])}
                 for r in nrows}
        edges = [(int(r["source_id"]), int(r["target_id"])) for r in erows]
        n0 = len(edges)
        stats = defaultdict(int)
        added = add_safe_divisions(nodes, edges, stats)
        total += added
        print(f"{ds:<16}{len(nodes):>9,}{n0:>9,}{stats['proposals']:>11,}{added:>7}{stats['cap_global']:>7}")
        for r in nrows:
            out_rows.append(r)
        for a, b in edges:
            out_rows.append({"dataset": ds, "row_type": "edge", "node_id": -1, "t": -1,
                             "z": -1, "y": -1, "x": -1, "source_id": a, "target_id": b})
    print("-" * 59)
    print(f"{'TOTAL':<16}{'':>9}{'':>9}{'':>11}{total:>7}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + COLS)
        w.writeheader()
        for i, r in enumerate(out_rows):
            w.writerow({"id": i, **{c: r.get(c, -1) for c in COLS}})
    print(f"\nwrote {args.out}  ({len(out_rows):,} rows)")


if __name__ == "__main__":
    main()
