"""F10 lever: GT lineages almost never BEGIN in the crop interior (1.6%).
So a PREDICTED track that starts in the interior is a fragmentation error.
Question: how many of our 116 FN edges are recoverable by joining such fragments?

The metric makes this an ASYMMETRIC bet: edges between unannotated cells are free
(metrics.md: a FP needs a matched endpoint), so a wrong join on an unannotated fragment
costs nothing, while a right join on an annotated lineage buys a TP.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO=Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking"); sys.path.insert(0,str(REPO/"src"))
from ctrack.data.geff import load_geff
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes
GT=Path("/Volumes/extSSD/kaggle/biohub/gt/train")
SUB=Path.home()/"biohub_replay947_runs/ds_ctl/submission.csv"
MARGIN_UM=8.0   # "at a face" tolerance
nodes=defaultdict(list); edges=defaultdict(list)
with SUB.open() as f:
    for r in csv.DictReader(f):
        if r["row_type"]=="node": nodes[r["dataset"]].append((int(r["node_id"]),int(r["t"]),float(r["z"]),float(r["y"]),float(r["x"])))
        else: edges[r["dataset"]].append((int(r["source_id"]),int(r["target_id"])))
T={}; tot=defaultdict(int)
print(f"{'volume':17s} {'pred starts':>11} {'interior':>9} {'%':>6} {'FN edges':>9} {'FN w/ interior-start tgt':>24}")
for ds in sorted(nodes):
    arr=np.asarray(nodes[ds],float)
    E=set(map(tuple,edges[ds]))
    pos={int(r[0]):np.asarray(r[2:5])*SCALE for r in arr}; tt={int(r[0]):int(r[1]) for r in arr}
    has_pred={b for a,b in E}
    tmax=max(tt.values()); 
    lo=np.array([min(p[0] for p in pos.values()),min(p[1] for p in pos.values()),min(p[2] for p in pos.values())])
    hi=np.array([max(p[0] for p in pos.values()),max(p[1] for p in pos.values()),max(p[2] for p in pos.values())])
    tmin=min(tt.values())
    starts=[i for i in pos if i not in has_pred]
    def interior(i):
        if tt[i]<=tmin: return False
        p=pos[i]
        return bool(((p-lo)>MARGIN_UM).all() and ((hi-p)>MARGIN_UM).all())
    inter=[i for i in starts if interior(i)]
    g=load_geff(GT/f"{ds}.geff").graph
    pred=Graph(nodes=arr,edges=np.asarray(sorted(E),np.int64).reshape(-1,2))
    m=match_nodes(g,pred,max_distance=MAX_MATCH_DISTANCE_UM,scale=SCALE)
    g2p={gt:p for p,gt in m.items()}
    GE=set(map(tuple,g.edges.tolist()))
    fn=[(a,b) for a,b in GE if (g2p.get(a),g2p.get(b)) not in E]
    iset=set(inter)
    rec=[(a,b) for a,b in fn if g2p.get(b) in iset]
    tot['starts']+=len(starts); tot['inter']+=len(inter); tot['fn']+=len(fn); tot['rec']+=len(rec)
    print(f"{ds:17s} {len(starts):>11} {len(inter):>9} {100*len(inter)/max(len(starts),1):>5.1f}% {len(fn):>9} {len(rec):>24}")
print(f"\nPOOLED: predicted track starts {tot['starts']}, of which INTERIOR {tot['inter']} ({100*tot['inter']/tot['starts']:.1f}%)")
print(f"FN edges {tot['fn']}; FN whose TARGET is an interior predicted track start: {tot['rec']} ({100*tot['rec']/max(tot['fn'],1):.0f}%)")
print(f"\nIf ALL {tot['rec']} were recovered with zero new FP: edge_J 3266/3506 -> {(3266+tot['rec'])/3506:.4f} (base 0.9315)")
