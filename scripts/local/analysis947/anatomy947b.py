#!/usr/bin/env python3
"""Part B: TP baselines, duplicate/ambiguity analysis, z-vs-xy residual decomposition, track structure
of the competing nodes. Scratch wrapper (no repo edits)."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking")
sys.path.insert(0, str(REPO / "src"))
from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes  # noqa: E402

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


def summ(a):
    a = np.asarray(a, float)
    if not a.size:
        return "n=0"
    return f"n={a.size} mean={a.mean():.2f} med={np.median(a):.2f} p90={np.percentile(a,90):.2f}"


def main():
    preds = load_submission(SUB)
    TP, FN, DET = [], [], []
    dup_stats = []
    for ds in sorted(preds):
        pred = preds[ds]
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        g = gt.graph
        matched = match_nodes(g, pred, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        g2p = {gid: pid for pid, gid in matched.items()}
        ppos = {int(r[0]): r[2:5] * SCALE for r in pred.nodes}
        pt = {int(r[0]): int(r[1]) for r in pred.nodes}
        gpos = {int(r[0]): r[2:5] * SCALE for r in g.nodes}
        gtt = {int(r[0]): int(r[1]) for r in g.nodes}
        frame_ids = defaultdict(list)
        for pid, t in pt.items():
            frame_ids[t].append(pid)
        trees = {t: (cKDTree(np.asarray([ppos[i] for i in ids])), np.asarray(ids)) for t, ids in frame_ids.items()}

        def within(t, x, r):
            if t not in trees:
                return []
            tree, ids = trees[t]
            idx = tree.query_ball_point(x, r)
            out = [(int(ids[i]), float(np.linalg.norm(ppos[int(ids[i])] - x))) for i in idx]
            return sorted(out, key=lambda z: z[1])

        pred_edges = {(int(s), int(t)) for s, t in pred.edges}
        pred_out, pred_in = defaultdict(set), defaultdict(set)
        for s, t in pred_edges:
            pred_out[s].add(t); pred_in[t].add(s)
        gt_edges = {(int(a), int(b)) for a, b in g.edges}
        gt_out = defaultdict(set)
        for a, b in gt_edges:
            gt_out[a].add(b)

        # track length (component size via undirected union) for pred nodes
        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        for s, t in pred_edges:
            union(s, t)
        comp_size = Counter(find(n) for n in ppos)
        def tlen(n):
            return comp_size[find(n)]

        # --- duplicate rate: predicted node pairs within r um in the same frame
        n_pairs = {3.0: 0, 4.0: 0, 5.0: 0}
        for t, (tree, ids) in trees.items():
            for r in n_pairs:
                n_pairs[r] += len(tree.query_pairs(r))
        # fraction of matched pred nodes that have a same-frame pred neighbour within r
        mp = list(matched.keys())
        near = {r: 0 for r in n_pairs}
        for p in mp:
            for r in n_pairs:
                if len(within(pt[p], ppos[p], r)) > 1:
                    near[r] += 1
        dup_stats.append((ds, len(ppos), n_pairs, len(mp), near))

        for a, b in sorted(gt_edges):
            pa, pb = g2p.get(a), g2p.get(b)
            ta = gtt[a]
            if pa is None or pb is None:
                for n, p_ in ((a, pa), (b, pb)):
                    if p_ is None:
                        cands = within(gtt[n], gpos[n], 15.0)
                        if cands:
                            nid, d = cands[0]
                            res = ppos[nid] - gpos[n]
                            DET.append(dict(ds=ds, n=n, t=gtt[n], d=d, dz=abs(res[0]), dxy=float(np.hypot(res[1], res[2])),
                                            nid_matched=nid in matched, nid_tlen=tlen(nid), n_in7=len(within(gtt[n], gpos[n], 7.0)),
                                            n_in10=len(cands and within(gtt[n], gpos[n], 10.0))))
                continue
            d_gt = float(np.linalg.norm(gpos[b] - gpos[a]))
            good = (pa, pb) in pred_edges
            ra, rb = ppos[pa] - gpos[a], ppos[pb] - gpos[b]
            row = dict(ds=ds, t=ta, d_gt=d_gt, d_pred=float(np.linalg.norm(ppos[pb] - ppos[pa])),
                       loc_a=float(np.linalg.norm(ra)), loc_b=float(np.linalg.norm(rb)),
                       loc_a_z=abs(ra[0]), loc_a_xy=float(np.hypot(ra[1], ra[2])),
                       loc_b_z=abs(rb[0]), loc_b_xy=float(np.hypot(rb[1], rb[2])),
                       zfrac=abs(gpos[b][0] - gpos[a][0]) / d_gt if d_gt > 0 else 0.0,
                       cand_a=len(within(ta, gpos[a], 7.0)), cand_b=len(within(ta + 1, gpos[b], 7.0)),
                       dens_a10=len(within(ta, ppos[pa], 10.0)), tlen_a=tlen(pa), tlen_b=tlen(pb),
                       indeg_a=len(pred_in[pa]), outdeg_b=len(pred_out[pb]))
            if good:
                TP.append(row)
            else:
                taken = pred_out.get(pa, set())
                stolen = pred_in.get(pb, set())
                row["kind"] = ("swap_both" if taken and stolen else "swap_src" if taken else "term_tgt_stolen" if stolen else "gap")
                # candidates within 7um of GT a at t and GT b at t+1 (metric-eligible alternates)
                ca = within(ta, gpos[a], 7.0); cb = within(ta + 1, gpos[b], 7.0)
                row["alt_a"] = [(i, round(d, 1), i == pa, tlen(i)) for i, d in ca]
                row["alt_b"] = [(i, round(d, 1), i == pb, tlen(i)) for i, d in cb]
                if taken:
                    tk = min(taken, key=lambda q: np.linalg.norm(ppos[q] - ppos[pa]))
                    row["tk"] = tk
                    row["d_tk_gtb"] = float(np.linalg.norm(ppos[tk] - gpos[b]))
                    row["d_tk_pb"] = float(np.linalg.norm(ppos[tk] - ppos[pb]))
                    row["tk_tlen"] = tlen(tk); row["tk_indeg"] = len(pred_in[tk])
                    row["tk_in7_of_gtb"] = row["d_tk_gtb"] <= 7.0
                    # does the taken node's track continue and is it "the" cell? follow one step: does tk's successor exist
                    row["tk_has_succ"] = len(pred_out[tk]) > 0
                if stolen:
                    st = next(iter(stolen))
                    row["st"] = st
                    row["d_st_gta"] = float(np.linalg.norm(ppos[st] - gpos[a]))
                    row["d_st_pa"] = float(np.linalg.norm(ppos[st] - ppos[pa]))
                    row["st_tlen"] = tlen(st); row["st_in7_of_gta"] = row["d_st_gta"] <= 7.0
                FN.append(row)

    print("== TP vs ASSOC-FN BASELINES ==")
    for k in ("d_gt", "d_pred", "loc_a", "loc_b", "loc_a_z", "loc_a_xy", "loc_b_z", "loc_b_xy", "zfrac", "cand_a", "cand_b", "dens_a10", "tlen_a", "tlen_b"):
        print(f"{k:>9}: TP {summ([r[k] for r in TP])}   |  assocFN {summ([r[k] for r in FN])}")
    print(f"TP: d_pred>7um {sum(r['d_pred']>7 for r in TP)}/{len(TP)}; indeg_a==0 {sum(r['indeg_a']==0 for r in TP)}; outdeg_b==0 {sum(r['outdeg_b']==0 for r in TP)}")
    print(f"TP: cand_a>=2 (another pred node within 7um of GT src) {sum(r['cand_a']>=2 for r in TP)}/{len(TP)}; cand_b>=2 {sum(r['cand_b']>=2 for r in TP)}")
    print(f"FN: cand_a>=2 {sum(r['cand_a']>=2 for r in FN)}/{len(FN)}; cand_b>=2 {sum(r['cand_b']>=2 for r in FN)}; both>=2 {sum(r['cand_a']>=2 and r['cand_b']>=2 for r in FN)}; either>=2 {sum(r['cand_a']>=2 or r['cand_b']>=2 for r in FN)}")
    print(f"FN: loc_a>3.5 or loc_b>3.5: {sum(r['loc_a']>3.5 or r['loc_b']>3.5 for r in FN)}/{len(FN)}   TP: {sum(r['loc_a']>3.5 or r['loc_b']>3.5 for r in TP)}/{len(TP)}")
    print(f"FN: z-dominated residual (loc_z > loc_xy) src {sum(r['loc_a_z']>r['loc_a_xy'] for r in FN)}/{len(FN)} tgt {sum(r['loc_b_z']>r['loc_b_xy'] for r in FN)}   TP src {sum(r['loc_a_z']>r['loc_a_xy'] for r in TP)}/{len(TP)}")
    sw = [r for r in FN if "tk" in r]
    print(f"\n== SWAP: the TAKEN node ==  n={len(sw)}")
    print(f"taken within 7um of GT target (metric-eligible alternate match): {sum(r['tk_in7_of_gtb'] for r in sw)}/{len(sw)}")
    print(f"d(taken, GT b) {summ([r['d_tk_gtb'] for r in sw])}   d(taken, pb) {summ([r['d_tk_pb'] for r in sw])}")
    print(f"taken node track length {summ([r['tk_tlen'] for r in sw])}  (pb track length {summ([r['tlen_b'] for r in sw])}; pa track length {summ([r['tlen_a'] for r in sw])})")
    print(f"taken has predecessor already (in-deg>0, i.e. would be a merge): {sum(r['tk_indeg']>0 for r in sw)}; taken has successor {sum(r['tk_has_succ'] for r in sw)}")
    print(f"taken track len <= 6: {sum(r['tk_tlen']<=6 for r in sw)}; pb track len <= 6: {sum(r['tlen_b']<=6 for r in sw)}; pb track len < 20: {sum(r['tlen_b']<20 for r in sw)}")
    st = [r for r in FN if "st" in r]
    print(f"\n== TARGET STOLEN: the STEALING source ==  n={len(st)}")
    print(f"stealer within 7um of GT a (alternate match for source): {sum(r['st_in7_of_gta'] for r in st)}/{len(st)}   d(st, GT a) {summ([r['d_st_gta'] for r in st])}  d(st, pa) {summ([r['d_st_pa'] for r in st])}  st track len {summ([r['st_tlen'] for r in st])}")
    print("\n== DUPLICATE RATE (same-frame predicted pairs within r um) ==")
    for ds, n, pairs, nm, near in dup_stats:
        print(f"{ds}: n_pred={n} pairs<3um={pairs[3.0]} (<4: {pairs[4.0]}, <5: {pairs[5.0]}) per-1k-nodes<3um={1000*pairs[3.0]/n:.1f} | matched nodes {nm}: with neighbour <3um {near[3.0]}, <4um {near[4.0]}, <5um {near[5.0]}")
    print("\n== ASSOC-FN ALTERNATES (pred nodes within 7um of GT src @t / GT tgt @t+1: (id, dist, is_matched_one, tracklen)) ==")
    for r in sorted(FN, key=lambda r: (r["ds"], r["t"])):
        extra = ""
        if "tk" in r:
            extra += f" | taken {r['tk']} d(tk,GTb)={r['d_tk_gtb']:.1f} d(tk,pb)={r['d_tk_pb']:.1f} tklen={r['tk_tlen']} tkin={r['tk_indeg']}"
        if "st" in r:
            extra += f" | stealer {r['st']} d(st,GTa)={r['d_st_gta']:.1f} d(st,pa)={r['d_st_pa']:.1f} stlen={r['st_tlen']}"
        print(f"{r['ds']} t={r['t']:>2} {r['kind']:>15} d_gt={r['d_gt']:.1f} zf={r['zfrac']:.2f} locA={r['loc_a']:.1f}(z{r['loc_a_z']:.1f}) locB={r['loc_b']:.1f}(z{r['loc_b_z']:.1f}) A:{r['alt_a']} B:{r['alt_b']}{extra}")
    print("\n== DETECTION-FN residual decomposition (nearest pred within 15um of unmatched GT node) ==")
    print(f"n={len(DET)} d {summ([r['d'] for r in DET])} dz {summ([r['dz'] for r in DET])} dxy {summ([r['dxy'] for r in DET])}  z-dominated {sum(r['dz']>r['dxy'] for r in DET)}/{len(DET)}")
    print(f"nearest pred node is matched to another GT: {sum(r['nid_matched'] for r in DET)}; its track len {summ([r['nid_tlen'] for r in DET])}; pred nodes within 7um {summ([r['n_in7'] for r in DET])} within 10um {summ([r['n_in10'] for r in DET])}")
    for r in sorted(DET, key=lambda r: (r["ds"], r["t"])):
        print(f"  {r['ds']} t={r['t']:>2} gt={r['n']} d={r['d']:.1f} dz={r['dz']:.1f} dxy={r['dxy']:.1f} nearest_tlen={r['nid_tlen']} nearest_matched={r['nid_matched']}")


if __name__ == "__main__":
    main()
