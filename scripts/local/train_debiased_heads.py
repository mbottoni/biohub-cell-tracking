"""Row 185: V1284 heads that keep PER-NODE correction but control the MEAN shift.

Rows 160/179: the population-mean coordinate shift is what costs ~-0.005; our 40-movie head
(mean shift +0.585/+0.255/+0.120 um, ~ the full bias) lost -0.005 despite better per-node
accuracy. The author's head (mean +0.194/+0.038/-0.108) pays. So re-centre the TARGETS:
  'amean' : y - mean(y) + author_mean   -> per-node accuracy, the author's (paying) mean shift
  'zmean' : y - mean(y)                 -> per-node accuracy, zero mean shift
Same arch / save format / bounded output as x138's refine(); group split by movie as before.
Usage: train_debiased_heads.py <pairs.npz> <out_dir>
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch

spec = importlib.util.spec_from_file_location('v', 'scripts/v1284_train_cell.py'); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
pairs, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
d = np.load(pairs); X, Y, G = d['X'], d['Y'], d['G']
AUTHOR_MEAN = np.array([0.194, 0.038, -0.108], np.float32)   # measured on these pairs, row 175
rng = np.random.default_rng(0); movies = np.unique(G)
vm = np.isin(G, rng.choice(movies, size=max(2, len(movies) // 5), replace=False))
ymean = Y[~vm].mean(0)
report = {'pairs': int(len(X)), 'movies': int(len(movies)), 'target_mean': ymean.tolist()}
for tag, shift in (('amean', AUTHOR_MEAN), ('zmean', np.zeros(3, np.float32))):
    Yt = (Y - ymean + shift).astype(np.float32)
    nrm = np.linalg.norm(Yt, axis=1, keepdims=True); Yt = np.where(nrm > 1.99, Yt * 1.99 / nrm, Yt).astype(np.float32)
    head, mean, scale, curve = v.fit(X[~vm], Yt[~vm], 150, Xv=X[vm], Yv=Yt[vm])
    ep = int(np.argmin(curve)) + 1
    final, mean, scale, _ = v.fit(X, Yt, ep)
    torch.save({'state_dict': final.state_dict(), 'mean': torch.from_numpy(mean.astype(np.float32)),
                'scale': torch.from_numpy(scale.astype(np.float32))}, out / f'v1284_head_{tag}.pt')
    # held-out, split-model: per-node CENTRED residual (what we want to improve) + realised mean shift
    head.eval()
    with torch.no_grad():
        p = v.bounded(head, torch.from_numpy((X[vm] - mean) / scale)).numpy()
    yc = Y[vm] - Y[vm].mean(0)
    base_c = np.linalg.norm(yc, axis=1).mean()
    res_c = np.linalg.norm(yc - (p - p.mean(0)), axis=1).mean()
    report[tag] = {'best_epoch': ep, 'heldout_centred_base_um': float(base_c), 'heldout_centred_after_um': float(res_c),
                   'heldout_centred_rel': float((res_c - base_c) / base_c), 'heldout_mean_shift_um': p.mean(0).round(3).tolist()}
    print(tag, json.dumps(report[tag]))
(out / 'report.json').write_text(json.dumps(report, indent=1))
