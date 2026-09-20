"""I06 refutation instrument: ORACLE upper bound on any sub-voxel z (or xyz) scheme.

Stage 1 can only change each node's z by at most 0.5 raw voxel (it replaces
int(round(z_float)) by z_float).  So an oracle that moves every predicted node's
z toward its nearest GT node in the same frame, clamped to +/- DELTA voxels, is a
STRICT UPPER BOUND on stage 1 (DELTA=0.5) and on stage1+stage2 (DELTA=1.0).
If the oracle is flat, the real mechanism is dead.
"""
import sys, csv
from pathlib import Path
import numpy as np
REPO = Path('/Users/maruanottoni/home/kaggle/biohub-cell-tracking')
sys.path.insert(0, str(REPO/'src'))
from ctrack.data.geff import load_geff
from ctrack.eval.biohub import SCALE

GT = Path('/Volumes/extSSD/kaggle/biohub/gt/train')
SUB = Path('/Users/maruanottoni/biohub_replay947_runs/ds_ctl/submission.csv')

MODE = sys.argv[1]          # 'z' | 'xyz' | 'jitz'
DELTA = float(sys.argv[2])  # voxels
OUT = Path(sys.argv[3])

rows = list(csv.DictReader(SUB.open()))
by_ds = {}
for i, r in enumerate(rows):
    if r['row_type'] == 'node':
        by_ds.setdefault(r['dataset'], []).append(i)

rng = np.random.default_rng(0)
moved = {'z': [], 'y': [], 'x': []}
for ds, idxs in by_ds.items():
    g = load_geff(GT/f'{ds}.geff').graph
    gt = g.nodes  # id,t,z,y,x
    gt_by_t = {}
    for t in np.unique(gt[:,1]):
        gt_by_t[int(t)] = gt[gt[:,1]==t][:,2:5]
    P = np.array([[float(rows[i]['t']), float(rows[i]['z']), float(rows[i]['y']), float(rows[i]['x'])] for i in idxs])
    newzyx = P[:,1:4].copy()
    for t in np.unique(P[:,0]):
        m = P[:,0]==t
        gtp = gt_by_t.get(int(t))
        if gtp is None or len(gtp)==0: continue
        pp = P[m,1:4]
        d = np.linalg.norm((pp[:,None,:]-gtp[None,:,:])*SCALE[None,None,:], axis=2)
        j = d.argmin(1); dmin = d.min(1)
        tgt = gtp[j]
        delta = tgt - pp
        near = dmin <= 7.0   # only nodes that could plausibly match
        if MODE == 'z':
            step = np.clip(delta[:,0], -DELTA, DELTA)
            step = np.where(near, step, 0.0)
            newzyx[m,0] = pp[:,0] + step
        elif MODE == 'xyz':
            step = np.clip(delta, -DELTA, DELTA)
            step = np.where(near[:,None], step, 0.0)
            newzyx[m] = pp + step
        elif MODE == 'jitz':
            newzyx[m,0] = pp[:,0] + rng.uniform(-DELTA, DELTA, size=m.sum())
    moved['z'].append(np.abs(newzyx[:,0]-P[:,1]))
    for i, k in zip(idxs, range(len(idxs))):
        rows[i]['z'] = f'{newzyx[k,0]:.4f}'
        rows[i]['y'] = f'{newzyx[k,1]:.4f}'
        rows[i]['x'] = f'{newzyx[k,2]:.4f}'

mz = np.concatenate(moved['z'])
print(f'MODE={MODE} DELTA={DELTA} |dz| mean={mz.mean():.4f} vox = {mz.mean()*1.625:.4f} um ; frac moved={np.mean(mz>1e-9):.4f}')

with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in rows:
        if r['row_type'] != 'node':
            r['z']='-1.0'; r['y']='-1.0'; r['x']='-1.0'
        w.writerow(r)
print('wrote', OUT)
