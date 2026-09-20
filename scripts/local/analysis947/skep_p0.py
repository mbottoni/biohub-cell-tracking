"""Contrarian arm: HARD PRE-COMMIT of mutual-top candidate pairs with prob>=p0 (both endpoints free),
   BEFORE tight/relaxed. Narrow, countable, interpretable. vs global prob-Hungarian pass-0."""
from __future__ import annotations
import csv,pickle,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
REPO=Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking"); sys.path.insert(0,str(REPO/"src"))
from ctrack.data.geff import load_geff
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM,SCALE,Graph,match_nodes
GT=Path("/Volumes/extSSD/kaggle/biohub/gt/train");CACHE=Path("/Volumes/extSSD/kaggle/biohub/replay947/cache_densad")
SUB=Path.home()/"biohub_replay947_runs/ds_ctl/submission.csv"
DENS={'low':(6.5,11.0,3.0),'middle':(6.5,9.0,8.0),'high':(5.5,10.0,1.0)}
nodes,edges=defaultdict(list),defaultdict(list)
with SUB.open() as f:
    for r in csv.DictReader(f):
        ds=r["dataset"]
        if r["row_type"]=="node": nodes[ds].append((int(r["node_id"]),int(r["t"]),float(r["z"]),float(r["y"]),float(r["x"])))
        else: edges[ds].append((int(r["source_id"]),int(r["target_id"])))
def score(arr,E,gtg):
    pred=Graph(nodes=arr,edges=np.asarray(sorted(E),np.int64).reshape(-1,2))
    m=match_nodes(gtg,pred,max_distance=MAX_MATCH_DISTANCE_UM,scale=SCALE);g2p={g:p for p,g in m.items()}
    GE=set(map(tuple,gtg.edges.tolist()));tp=sum(1 for a,b in GE if (g2p.get(a),g2p.get(b)) in E)
    od=defaultdict(int);idg=defaultdict(int)
    for a,b in GE: od[a]+=1;idg[b]+=1
    p2g={p:g for g,p in g2p.items()};v=0
    for s,t in E:
        gs=p2g.get(s);gt_=p2g.get(t)
        if (gs is not None and od[gs]>0) or (gt_ is not None and idg[gt_]>0): v+=1
    return tp,v-tp,len(GE)-tp
ARMS=["noP0","PC95","PC90","PC80","PC50","P0_global14"]
res={a:[0,0,0] for a in ARMS};per=defaultdict(dict);W={};nadd=defaultdict(dict)
for ds in sorted(nodes):
    arr=np.asarray(nodes[ds],float);ids=set(arr[:,0].astype(int))
    pos={int(r[0]):r[2:5]*SCALE for r in arr};tt={int(r[0]):int(r[1]) for r in arr}
    E0=set(map(tuple,edges[ds])); gtg=load_geff(GT/f"{ds}.geff").graph
    pk=pickle.loads((CACHE/f"{ds}.pkl").read_bytes());coords,alle=pk["ce"] if isinstance(pk,dict) else pk
    cand=defaultdict(float)
    for a,b,pr,_ in alle:
        a,b=int(a),int(b)
        if a in ids and b in ids: cand[(a,b)]=max(cand[(a,b)],float(pr))
    bs=defaultdict(list);bt=defaultdict(list)
    for (a,b),pr in cand.items(): bs[a].append((pr,b));bt[b].append((pr,a))
    topS={a:max(v)[1] for a,v in bs.items()};topT={b:max(v)[1] for b,v in bt.items()}
    mutual={(a,b) for (a,b) in cand if topS.get(a)==b and topT.get(b)==a}
    byt=defaultdict(list)
    for i in ids: byt[tt[i]].append(i)
    candT=defaultdict(list)   # index candidates by source frame
    for (a,b),pr in cand.items(): candT[tt[a]].append((a,b,pr))
    avg=len(ids)/max(len(byt),1);grp='low' if avg<120 else ('middle' if avg<400 else 'high')
    tight,relax,bonus=DENS[grp]
    def build(mode,p0=1.1,G=14.0):
        E=set();added=0
        for t in sorted(byt):
            S=set(byt.get(t,[]));T=set(byt.get(t+1,[]))
            if not S or not T: continue
            cl=candT.get(t,[])
            if mode=="pc":
                for a,b,pr in sorted(cl,key=lambda z:-z[2]):
                    if pr<p0 or (a,b) not in mutual: continue
                    if a in S and b in T and float(np.linalg.norm(pos[b]-pos[a]))<=G:
                        E.add((a,b));S.discard(a);T.discard(b);added+=1
            elif mode=="pre":
                Sl=sorted(S);Tl=sorted(T);si={s:i for i,s in enumerate(Sl)};ti={x:i for i,x in enumerate(Tl)}
                C=np.ones((len(Sl),len(Tl)));any_=False
                for a,b,pr in cl:
                    if a in si and b in ti and float(np.linalg.norm(pos[b]-pos[a]))<=G: C[si[a],ti[b]]=1.0-pr;any_=True
                if any_:
                    r,c=linear_sum_assignment(C)
                    for i,j in zip(r,c):
                        if C[i,j]<1.0: E.add((Sl[i],Tl[j]));S.discard(Sl[i]);T.discard(Tl[j]);added+=1
            for gate in (tight,relax):
                S2=sorted(S);T2=sorted(T)
                if not S2 or not T2: break
                big=gate*1000+1;C2=np.full((len(S2),len(T2)),big)
                for i,s in enumerate(S2):
                    ps=pos[s]
                    for j,tg in enumerate(T2):
                        d=float(np.linalg.norm(pos[tg]-ps))
                        if d<=gate: C2[i,j]=d*1.05-bonus*cand.get((s,tg),0.0)
                r,c=linear_sum_assignment(C2)
                for i,j in zip(r,c):
                    if C2[i,j]<big: E.add((S2[i],T2[j]));S.discard(S2[i]);T.discard(T2[j])
        return E,added
    ES={"noP0":build("none"),"PC95":build("pc",0.95),"PC90":build("pc",0.90),
        "PC80":build("pc",0.80),"PC50":build("pc",0.50),"P0_global14":build("pre")}
    for k,(E,ad) in ES.items():
        tp,fp,fn=score(arr,E,gtg);res[k][0]+=tp;res[k][1]+=fp;res[k][2]+=fn;per[ds][k]=tp/(tp+fp+fn);nadd[ds][k]=ad
    W[ds]=sum(score(arr,E0,gtg))
    print(f"{ds:16s} "+" ".join(f"{k}={per[ds][k]:.4f}({nadd[ds][k]})" for k in ARMS),flush=True)
tot=sum(W.values());sb=[d for d in per if d.startswith('6bba')];tb=sum(W[d] for d in sb)
b=sum(W[d]*per[d]['noP0'] for d in per)/tot; b6=sum(W[d]*per[d]['noP0'] for d in sb)/tb
print()
for k in ARMS:
    tp,fp,fn=res[k];w=sum(W[d]*per[d][k] for d in per)/tot;w6=sum(W[d]*per[d][k] for d in sb)/tb
    print(f"{k:12s} micro={tp/(tp+fp+fn):.4f} TP{tp} FP{fp} FN{fn}  wAll={w:.4f} dw={w-b:+.4f}  w6bba={w6:.4f} dw6={w6-b6:+.4f}  precommitted={sum(nadd[d][k] for d in per)}")
