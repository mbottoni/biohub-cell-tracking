"""Phase 0: what does a SPURIOUS division actually cost?

Row 81 assumed every wrong prediction is an FP. The metric says otherwise:
a fork is only FP if it matches an annotated GT node within 7um. ~96% of cells
are unannotated. Inject K random forks into the measured stack submission and
read the real dFP/dK off the official scorer.
"""
import csv, os, sys, random
from collections import defaultdict

S=("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
   "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad")
N=("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
   "2ad68fab-98b7-4dc4-b103-ce7227df708f/scratchpad")
SRC=f"{S}/runs/reverify_stack/submission.csv"
K=int(sys.argv[1]); random.seed(int(sys.argv[2]) if len(sys.argv)>2 else 0)

rows=list(csv.DictReader(open(SRC))); fields=list(rows[0].keys())
nodes=defaultdict(dict); outdeg=defaultdict(int); indeg=defaultdict(int)
for r in rows:
    if r["row_type"]=="node":
        nodes[r["dataset"]][int(r["node_id"])]=(int(r["t"]),float(r["z"]),float(r["y"]),float(r["x"]))
    else:
        outdeg[(r["dataset"],int(r["source_id"]))]+=1
        indeg[(r["dataset"],int(r["target_id"]))]+=1

# candidate spurious forks: a node with outdeg==1 gains a 2nd child = nearest
# free node at t+1 (indeg 0), so we never create indegree>1.
import numpy as np
from scipy.spatial import cKDTree
add=[]
for ds,nd in nodes.items():
    by_t=defaultdict(list)
    for nid,(t,*_) in nd.items(): by_t[t].append(nid)
    trees={t:(cKDTree(np.array([nd[n][1:] for n in ns])),ns) for t,ns in by_t.items() if ns}
    srcs=[n for n in nd if outdeg[(ds,n)]==1]
    random.shuffle(srcs)
    for p in srcs:
        tp=nd[p][0]; tr=trees.get(tp+1)
        if tr is None: continue
        d,j=tr[0].query(np.array(nd[p][1:]),k=min(6,len(tr[1])))
        for dd,jj in zip(np.atleast_1d(d),np.atleast_1d(j)):
            c=tr[1][jj]
            if indeg[(ds,c)]==0 and outdeg[(ds,p)]<2:
                add.append((ds,p,c)); outdeg[(ds,p)]+=1; indeg[(ds,c)]+=1; break
        if len(add)>=K: break
    if len(add)>=K: break

out=f"{N}/inject_k{K}"; os.makedirs(out,exist_ok=True)
with open(f"{out}/submission.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    i=0
    for r in rows:
        r=dict(r); r["id"]=i; i+=1; w.writerow(r)
    for ds,p,c in add:
        w.writerow({"id":i,"dataset":ds,"row_type":"edge","node_id":-1,"t":-1,
                    "z":-1,"y":-1,"x":-1,"source_id":p,"target_id":c}); i+=1
print(f"injected {len(add)} spurious forks -> {out}/submission.csv")
