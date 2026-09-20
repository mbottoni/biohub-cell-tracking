#!/usr/bin/env python3
"""Part D: how much of the 'association' loss is clean (both endpoints well-localised) vs detection-contaminated;
recall vs local crowding; recall vs endpoint residual; realistic association headroom."""
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
    rows = []; fp_paired_by_edge = Counter(); fp_total = 0
    per_vol = {}
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
        pred_out, pred_in = defaultdict(set), defaultdict(set)
        for s, t in pred_edges: pred_out[s].add(t); pred_in[t].add(s)
        gt_edges = {(int(a), int(b)) for a, b in g.edges}
        gt_out, gt_in = defaultdict(set), defaultdict(set)
        for a, b in gt_edges: gt_out[a].add(b); gt_in[b].add(a)
        vol_rows = []
        for a, b in gt_edges:
            pa, pb = g2p.get(a), g2p.get(b)
            if pa is None or pb is None: continue
            ra = float(np.linalg.norm(ppos[pa] - gpos[a])); rb = float(np.linalg.norm(ppos[pb] - gpos[b]))
            d_gt = float(np.linalg.norm(gpos[b] - gpos[a]))
            n9a = len(trees[gtt[a]][0].query_ball_point(gpos[a], 9.0)); n9b = len(trees[gtt[b]][0].query_ball_point(gpos[b], 9.0))
            good = (pa, pb) in pred_edges
            r = dict(ds=ds, good=good, ra=ra, rb=rb, rmax=max(ra, rb), d_gt=d_gt, n9=max(n9a, n9b), pa=pa, pb=pb)
            rows.append(r); vol_rows.append(r)
        # FPs and pairing to FN edges
        fps = []
        for s, t in pred_edges:
            gs, gt_ = matched.get(s), matched.get(t)
            ov = gs is not None and len(gt_out.get(gs, ())) > 0; iv = gt_ is not None and len(gt_in.get(gt_, ())) > 0
            if not (ov or iv): continue
            if gs is not None and gt_ is not None and (gs, gt_) in gt_edges: continue
            fps.append((s, t))
        fp_total += len(fps)
        for r in vol_rows:
            if not r["good"]:
                r["fp_paired"] = sum(1 for s, t in fps if s == r["pa"] or t == r["pb"])
        per_vol[ds] = dict(fp=len(fps), n_gt=len(gt_edges), tp=sum(r["good"] for r in vol_rows), both=len(vol_rows))
    fn = [r for r in rows if not r["good"]]
    print(f"association-testable edges {len(rows)}, assoc-FN {len(fn)}, FP total {fp_total}")
    print("\n== P(correct | max endpoint residual) ==")
    for lo, hi in ((0, 1.7), (1.7, 2.3), (2.3, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.1)):
        m = [r for r in rows if lo <= r["rmax"] < hi]
        if m: print(f"  rmax {lo:.1f}-{hi:.1f}: n={len(m):>5} P(correct)={np.mean([r['good'] for r in m]):.3f} lost={sum(not r['good'] for r in m)}")
    print("\n== P(correct | max pred nodes within 9um of either GT endpoint) ==")
    for k in (1, 2, 3):
        m = [r for r in rows if (r["n9"] == k if k < 3 else r["n9"] >= 3)]
        if m: print(f"  n9={'>=3' if k==3 else k}: n={len(m):>5} P(correct)={np.mean([r['good'] for r in m]):.3f} lost={sum(not r['good'] for r in m)}")
    print("\n== P(correct | GT displacement) x (clean vs dirty endpoints, clean = both residuals <= 2.3um) ==")
    for lo, hi in ((0, 4), (4, 6.5), (6.5, 99)):
        for nm, cond in (("clean", lambda r: r["rmax"] <= 2.3), ("dirty", lambda r: r["rmax"] > 2.3)):
            m = [r for r in rows if lo <= r["d_gt"] < hi and cond(r)]
            if m: print(f"  d_gt {lo}-{hi} {nm}: n={len(m):>5} P(correct)={np.mean([r['good'] for r in m]):.3f} lost={sum(not r['good'] for r in m)}")
    print("\n== ASSOC-FN split ==")
    for thr in (2.3, 3.0):
        clean = [r for r in fn if r["rmax"] <= thr]
        fast_clean = [r for r in clean if r["d_gt"] >= 6.5]
        print(f"  both endpoints residual <= {thr}: {len(clean)}/{len(fn)} (of which fast d_gt>=6.5: {len(fast_clean)}; slow: {len(clean)-len(fast_clean)}); paired FPs on clean: {sum(r['fp_paired'] for r in clean)}")
    dirty = [r for r in fn if r["rmax"] > 3.0]
    print(f"  at least one endpoint residual > 3.0 (dubious detection): {len(dirty)}/{len(fn)}; paired FPs {sum(r['fp_paired'] for r in dirty)}; of these with another pred node within 9um: {sum(r['n9']>=2 for r in dirty)}")
    fast = [r for r in fn if r["d_gt"] >= 6.5]
    print(f"  fast (d_gt >= 6.5um): {len(fast)}/{len(fn)}; paired FPs {sum(r['fp_paired'] for r in fast)}; per volume {dict(Counter(r['ds'] for r in fast))}")
    print(f"  slow & dirty: {sum(1 for r in fn if r['d_gt'] < 6.5 and r['rmax'] > 3.0)}; slow & clean(<=3): {sum(1 for r in fn if r['d_gt'] < 6.5 and r['rmax'] <= 3.0)}")
    # realistic headroom: if only clean(<=3.0) assoc-FN and their paired FPs were fixed
    TP = sum(v["tp"] for v in per_vol.values()); FP = fp_total; FN = sum(v["n_gt"] - v["tp"] for v in per_vol.values())
    for thr in (2.3, 3.0):
        clean = [r for r in fn if r["rmax"] <= thr]; dfp = sum(r["fp_paired"] for r in clean)
        j0 = TP / (TP + FP + FN); j1 = (TP + len(clean)) / (TP + len(clean) + FP - dfp + FN - len(clean))
        print(f"  micro edge_J now {j0:.4f} -> fixing clean(<= {thr}) assoc-FN + paired FPs: {j1:.4f} (+{j1-j0:.4f})")
    fast_fix = (TP + len(fast)) / (TP + len(fast) + FP - sum(r['fp_paired'] for r in fast) + FN - len(fast))
    print(f"  fixing all FAST assoc-FN (+paired FPs): {fast_fix:.4f} (+{fast_fix - TP/(TP+FP+FN):.4f})")
    print("\n== per-volume assoc-FN by class (fast / slow-dirty / slow-clean) ==")
    for ds in per_vol:
        f = [r for r in fn if r["ds"] == ds]
        print(f"  {ds}: fast={sum(r['d_gt']>=6.5 for r in f)} slow_dirty={sum(r['d_gt']<6.5 and r['rmax']>3.0 for r in f)} slow_clean={sum(r['d_gt']<6.5 and r['rmax']<=3.0 for r in f)}  (FP {per_vol[ds]['fp']})")

if __name__ == "__main__":
    main()
