#!/usr/bin/env python3
"""Part C: per-GT-track residual trajectories around association-error frames.
Distinguishes 'cell detected but off-centre' (residual stays ~2um, neighbour flip) from
'cell missed, GT matched to a neighbour' (residual jumps at error frames; pred count near the GT drops)."""
from __future__ import annotations
import csv, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
REPO = Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking"); sys.path.insert(0, str(REPO / "src"))
from ctrack.data.geff import load_geff
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
SUB = Path(sys.argv[1] if len(sys.argv) > 1 else "~/biohub_replay947_runs/ds_ctl/submission.csv").expanduser()
SHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 10

def load_submission(path):
    nodes, edges = defaultdict(list), defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            if r["row_type"] == "node":
                nodes[ds].append((int(r["node_id"]), int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"])))
            else:
                edges[ds].append((int(r["source_id"]), int(r["target_id"])))
    return {ds: Graph(nodes=np.asarray(v, float), edges=np.asarray(edges[ds], np.int64).reshape(-1, 2)) for ds, v in nodes.items()}

def main():
    preds = load_submission(SUB)
    res_err, res_ok, n9_err, n9_ok, res_pm1 = [], [], [], [], []
    per_track_err = Counter()
    shown = 0
    for ds in sorted(preds):
        pred = preds[ds]; gt = load_geff(GT_ROOT / f"{ds}.geff"); g = gt.graph
        matched = match_nodes(g, pred, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        g2p = {gid: pid for pid, gid in matched.items()}
        ppos = {int(r[0]): r[2:5] * SCALE for r in pred.nodes}; pt = {int(r[0]): int(r[1]) for r in pred.nodes}
        gpos = {int(r[0]): r[2:5] * SCALE for r in g.nodes}; gtt = {int(r[0]): int(r[1]) for r in g.nodes}
        frame_ids = defaultdict(list)
        for pid, t in pt.items(): frame_ids[t].append(pid)
        trees = {t: (cKDTree(np.asarray([ppos[i] for i in ids])), np.asarray(ids)) for t, ids in frame_ids.items()}
        pred_edges = {(int(s), int(t)) for s, t in pred.edges}
        # pred track id = component root
        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
            return x
        for s, t in pred_edges:
            ra, rb = find(s), find(t)
            if ra != rb: parent[ra] = rb
        # short labels for pred tracks
        lab = {}
        def L(n):
            r = find(n)
            if r not in lab: lab[r] = f"T{len(lab)}"
            return lab[r]
        gt_edges = {(int(a), int(b)) for a, b in g.edges}
        gt_parent = {b: a for a, b in gt_edges}; gt_out = defaultdict(set)
        for a, b in gt_edges: gt_out[a].add(b)
        def root(n):
            while n in gt_parent: n = gt_parent[n]
            return n
        err_edges = []
        for a, b in gt_edges:
            pa, pb = g2p.get(a), g2p.get(b)
            if pa is not None and pb is not None and (pa, pb) not in pred_edges:
                err_edges.append((a, b))
        err_nodes = set()
        for a, b in err_edges: err_nodes.add(a); err_nodes.add(b)
        # pooled stats: residual + n pred within 9um, at error-endpoint nodes vs other matched nodes, plus +-1 frames
        for gid, pid in g2p.items():
            r = float(np.linalg.norm(ppos[pid] - gpos[gid]))
            t = gtt[gid]; n9 = len(trees[t][0].query_ball_point(gpos[gid], 9.0)) if t in trees else 0
            if gid in err_nodes: res_err.append(r); n9_err.append(n9)
            else: res_ok.append(r); n9_ok.append(n9)
        for a, b in err_edges: per_track_err[(ds, root(a))] += 1
        # show trajectories
        for a, b in sorted(err_edges, key=lambda e: gtt[e[0]]):
            if shown >= SHOW: break
            shown += 1
            t0 = gtt[a]
            # walk GT track from t0-4 to t0+5
            chain = {}; cur = a; k = 0
            while cur is not None and k < 4:
                cur = gt_parent.get(cur); k += 1
                if cur is not None: chain[gtt[cur]] = cur
            chain[t0] = a; cur = a
            for k in range(5):
                nxt = gt_out.get(cur)
                if not nxt: break
                cur = min(nxt); chain[gtt[cur]] = cur
            print(f"\n--- {ds} GT edge {a}->{b} at t={t0} (GT track root {root(a)}) ---")
            for t in sorted(chain):
                n = chain[t]; pid = g2p.get(n)
                tree, ids = trees[t]
                idx = tree.query_ball_point(gpos[n], 9.0)
                near = sorted(((float(np.linalg.norm(ppos[int(ids[i])] - gpos[n])), int(ids[i])) for i in idx))
                nl = " ".join(f"{L(i)}:{d:.1f}{'*' if i == pid else ''}" for d, i in near)
                mark = "ERR" if (n in err_nodes) else "   "
                gp = gpos[n]
                print(f"  t={t:>2} {mark} gt=({gp[0]:.0f},{gp[1]:.0f},{gp[2]:.0f}) matched={L(pid) if pid is not None else '--'} res={np.linalg.norm(ppos[pid]-gpos[n]) if pid is not None else float('nan'):.1f} | within9um: {nl}")
    print("\n== POOLED: matched GT nodes that are endpoints of an assoc-FN edge vs all other matched GT nodes ==")
    for name, r, n9 in (("assocFN endpoints", res_err, n9_err), ("other matched", res_ok, n9_ok)):
        r = np.asarray(r); n9 = np.asarray(n9)
        print(f"{name:>18}: n={r.size} residual med={np.median(r):.2f} mean={r.mean():.2f} p75={np.percentile(r,75):.2f} | pred nodes within 9um med={np.median(n9):.0f} mean={n9.mean():.2f} frac>=2: {(n9>=2).mean():.2f}")
    print(f"\nassoc-FN edges per GT track (ds, root): {per_track_err.most_common(12)}   n_tracks_with_err={len(per_track_err)}")

if __name__ == "__main__":
    main()
