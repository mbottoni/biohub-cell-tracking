"""Is there a SYSTEMATIC coordinate offset between our detections and the GT?

This is NOT refuted by the row-153 z-oracle. That oracle moved only ALREADY-MATCHED
nodes, which cannot change a cardinality-dominated assignment. A systematic global
shift moves EVERY node, including unmatched ones -- and can pull a node that currently
has no detection within 7um INSIDE the radius. That is the one lever the anatomy says
can still pay.

Measure: the mean residual VECTOR (pred - gt) over all matched pairs, per axis, per
volume. A zero-mean scatter means no offset. A consistent non-zero mean is a bias.
Then: how many currently-unmatched GT nodes would come inside 7um if we removed it?
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
allres=[]; print(f"{'volume':17s} {'n':>5} {'mean dz':>8} {'mean dy':>8} {'mean dx':>8} {'med|d|':>7}")
per_vol={}
for ds in sorted(nodes):
    arr=np.asarray(nodes[ds],float)
    g=load_geff(GT/f"{ds}.geff").graph
    pred=Graph(nodes=arr,edges=np.zeros((0,2),np.int64))
    m=match_nodes(g,pred,max_distance=MAX_MATCH_DISTANCE_UM,scale=SCALE)
    idx={int(r[0]):i for i,r in enumerate(arr)}
    gpos={int(r[0]):np.asarray(r[2:5])*SCALE for r in g.nodes}
    res=[]
    for p,gt in m.items():
        res.append(arr[idx[int(p)]][2:5]*SCALE - gpos[int(gt)])
    R=np.asarray(res); per_vol[ds]=R; allres.append(R)
    print(f"{ds:17s} {len(R):>5} {R[:,0].mean():>+8.3f} {R[:,1].mean():>+8.3f} {R[:,2].mean():>+8.3f} {np.median(np.linalg.norm(R,axis=1)):>7.3f}")
A=np.vstack(allres)
print(f"\nPOOLED n={len(A)}  mean residual vector (um): dz={A[:,0].mean():+.4f} dy={A[:,1].mean():+.4f} dx={A[:,2].mean():+.4f}")
print(f"  std: dz={A[:,0].std():.3f} dy={A[:,1].std():.3f} dx={A[:,2].std():.3f}")
sem=A.std(axis=0)/np.sqrt(len(A))
print(f"  SEM: dz={sem[0]:.4f} dy={sem[1]:.4f} dx={sem[2]:.4f}")
print(f"  t-stat: dz={A[:,0].mean()/sem[0]:+.1f} dy={A[:,1].mean()/sem[1]:+.1f} dx={A[:,2].mean()/sem[2]:+.1f}")
print(f"  => |mean offset| = {np.linalg.norm(A.mean(axis=0)):.4f} um vs median residual {np.median(np.linalg.norm(A,axis=1)):.3f} um")
