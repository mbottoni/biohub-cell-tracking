"""One pass at the widest gate, then filter -- recall vs pool for every setting."""
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/Users/maruanottoni/home/kaggle/biohub-cell-tracking/src")
from ctrack.data.geff import load_geff
S = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
     "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
GT = "/Volumes/extSSD/kaggle/biohub/gt/train"
MATCH, RMAX, SMAX = 10.0, 34.0, 40.0
nodes, edges = defaultdict(dict), defaultdict(list)
for r in csv.DictReader(open(f"{S}/runs/reverify_stack/submission.csv")):
    ds = r["dataset"]
    if r["row_type"] == "node":
        nodes[ds][int(r["node_id"])] = (int(r["t"]), float(r["z"]), float(r["y"]), float(r["x"]))
    else: edges[ds].append((int(r["source_id"]), int(r["target_id"])))
recs = []   # (d_par, d_sis, rank_at_RMAX, is_truth)
for v in sorted(nodes):
    nd = nodes[v]
    by_t = defaultdict(list)
    for nid, (t, *_) in nd.items(): by_t[t].append(nid)
    trees = {t: (cKDTree(np.array([nd[n][1:] for n in ns])), ns) for t, ns in by_t.items() if ns}
    oe = defaultdict(list)
    for s_, t_ in edges[v]: oe[s_].append(t_)
    G = load_geff(f"{GT}/{v}.geff").graph
    kids = defaultdict(list)
    for s_, t_ in G.edges: kids[s_].append(t_)
    idx = {nid: i for i, nid in enumerate(G.ids)}
    def m(gid):
        i = idx[gid]; tr = trees.get(int(G.t[i]))
        if tr is None: return None
        d, j = tr[0].query(np.array(G.zyx[i]))
        return tr[1][j] if d <= MATCH else None
    truth = set()
    for pid, ch in kids.items():
        if len(ch) < 2: continue
        mp, m1, m2 = m(pid), m(ch[0]), m(ch[1])
        if None in (mp, m1, m2) or len({mp, m1, m2}) < 3: continue
        linked = [c for c in oe.get(mp, []) if c in (m1, m2)]
        if not linked: continue
        truth.add((mp, frozenset((linked[0], m2 if linked[0] == m1 else m1))))
    for (p, c1) in edges[v]:
        if p not in nd or c1 not in nd: continue
        tr = trees.get(nd[p][0] + 1)
        if tr is None: continue
        pz = np.array(nd[p][1:]); c1z = np.array(nd[c1][1:])
        cands = []
        for j in tr[0].query_ball_point(pz, RMAX):
            c2 = tr[1][j]
            if c2 == c1: continue
            ds_ = float(np.linalg.norm(c1z - np.array(nd[c2][1:])))
            if ds_ > SMAX: continue
            cands.append((float(np.linalg.norm(pz - np.array(nd[c2][1:]))), ds_, c2))
        cands.sort()
        for rk, (dp_, ds_, c2) in enumerate(cands, 1):
            recs.append((dp_, ds_, rk, 1 if (p, frozenset((c1, c2))) in truth else 0))
a = np.array(recs)
print(f"single pass: {len(a):,} candidate pairs, {int(a[:,3].sum())} truth\n")
print(f"{'RADIUS':>7} {'SIS':>5} {'MC':>3} {'recall':>9} {'pool':>11} {'pos rate':>9}")
for RAD, SIS in ((11.0,18.0),(20.0,25.0),(26.0,30.0),(30.0,34.0),(34.0,40.0)):
    for MC in (3, 8, 15, 999):
        sel = (a[:,0] <= RAD) & (a[:,1] <= SIS) & (a[:,2] <= MC)
        k, n = int(a[sel,3].sum()), int(sel.sum())
        print(f"{RAD:7.1f} {SIS:5.1f} {MC:3d} {k:6d}/16 {n:11,d} {100*k/max(n,1):8.4f}%")
