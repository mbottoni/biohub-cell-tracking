#!/usr/bin/env python3
"""Absolute ceiling of any z-localisation mechanism: snap every MATCHED predicted
node's z to its GT node's z. Upper bound on z-flip TTA (and on §25 z refinement)."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking"); sys.path.insert(0,str(REPO/"src"))
from ctrack.data.geff import load_geff
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
SRC = Path("~/biohub_replay947_runs/ds_ctl/submission.csv").expanduser()
OUT = Path(sys.argv[1])
nodes, rows = defaultdict(list), []
with SRC.open() as f:
    rd=csv.DictReader(f); fn=rd.fieldnames
    for r in rd:
        rows.append(r)
        if r["row_type"]=="node":
            nodes[r["dataset"]].append((int(r["node_id"]),int(r["t"]),float(r["z"]),float(r["y"]),float(r["x"])))
newz={}
for ds,v in nodes.items():
    pred=Graph(nodes=np.asarray(v,float),edges=np.zeros((0,2),np.int64))
    gt=load_geff(GT_ROOT/f"{ds}.geff"); g=gt.graph
    m=match_nodes(g,pred,max_distance=MAX_MATCH_DISTANCE_UM,scale=SCALE)
    gn={int(r[0]):r for r in g.nodes}
    for pid,gid in m.items(): newz[(ds,int(pid))]=gn[int(gid)][2]
    print(ds,"matched",len(m),"of",len(v))
n=0
for r in rows:
    if r["row_type"]=="node":
        k=(r["dataset"],int(r["node_id"]))
        if k in newz: r["z"]=str(int(round(newz[k]))); n+=1
with OUT.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(rows)
print("snapped",n,"->",OUT)
