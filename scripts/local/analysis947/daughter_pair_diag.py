"""Are the detection-FN nodes post-division DAUGHTER PAIRS merged into one detection?

Row 151 recorded "0 at/near divisions", but GT lineages are SPARSE: a daughter's track
may start with no annotated parent, so an unannotated division would look like a plain
track start. That is precisely the case row 151 could not see. Test it directly:

for each unmatched GT node, ask
  (a) is it a GT track START (no GT predecessor)?
  (b) is there ANOTHER GT track start at the same t within 15um (a sibling)?
  (c) is there exactly ONE predicted node near the PAIR (a merged detection)?
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
nodes=defaultdict(list)
with SUB.open() as f:
    for r in csv.DictReader(f):
        if r["row_type"]=="node":
            nodes[r["dataset"]].append((int(r["node_id"]),int(r["t"]),float(r["z"]),float(r["y"]),float(r["x"])))
tot=defaultdict(int); rows=[]
for ds in sorted(nodes):
    arr=np.asarray(nodes[ds],float)
    g=load_geff(GT/f"{ds}.geff").graph
    pred=Graph(nodes=arr,edges=np.zeros((0,2),np.int64))
    m=match_nodes(g,pred,max_distance=MAX_MATCH_DISTANCE_UM,scale=SCALE)
    g2p={gt:p for p,gt in m.items()}
    gn=g.nodes  # (id,t,z,y,x)
    gpos={int(r[0]):np.asarray(r[2:5])*SCALE for r in gn}
    gt_t={int(r[0]):int(r[1]) for r in gn}
    has_pred=set(int(b) for a,b in g.edges.tolist())
    starts=[i for i in gpos if i not in has_pred]
    ppos={int(r[0]):np.asarray(r[2:5])*SCALE for r in arr}
    pt=defaultdict(list)
    for i,p in ppos.items(): pt[int(arr[list(ppos).index(i)][1]) if False else 0].append(i)
    pby=defaultdict(list)
    for r in arr: pby[int(r[1])].append((int(r[0]),np.asarray(r[2:5])*SCALE))
    unmatched=[i for i in gpos if i not in g2p]
    tot['gt_nodes']+=len(gpos); tot['unmatched']+=len(unmatched); tot['starts']+=len(starts)
    startset=set(starts)
    for u in unmatched:
        t=gt_t[u]; isstart=u in startset
        sib=None; sibd=None
        for o in gpos:
            if o==u or gt_t[o]!=t: continue
            d=float(np.linalg.norm(gpos[o]-gpos[u]))
            if d<=15.0 and (sibd is None or d<sibd):
                sib, sibd = o, d
        sib_is_start = sib in startset if sib is not None else False
        near=[(pid,float(np.linalg.norm(pp-gpos[u]))) for pid,pp in pby[t]]
        near.sort(key=lambda z:z[1])
        tot['is_start']+=isstart
        tot['has_sib15']+= sib is not None
        tot['sib_is_start']+= bool(sib_is_start)
        tot['start_and_sibstart']+= bool(isstart and sib_is_start)
        rows.append((ds,u,t,isstart,sib,round(sibd,2) if sibd else None,sib_is_start,
                     round(near[0][1],2) if near else None))
print(f"GT nodes {tot['gt_nodes']} | unmatched (det-FN) {tot['unmatched']} | GT track starts {tot['starts']}")
print(f"\nof the {tot['unmatched']} unmatched GT nodes:")
print(f"  is a GT track START                : {tot['is_start']}")
print(f"  has another GT node <=15um same t  : {tot['has_sib15']}")
print(f"  that neighbour is ALSO a track start: {tot['sib_is_start']}")
print(f"  BOTH are track starts (daughter-pair signature): {tot['start_and_sibstart']}")
print("\nper-node detail (ds, gt_id, t, is_start, sibling, sib_dist_um, sib_is_start, nearest_pred_um):")
for r in rows: print("  ",r)
