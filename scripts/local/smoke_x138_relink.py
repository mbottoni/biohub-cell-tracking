"""Smoke: run x138's REAL motion_relink_edges (exec'd from a built notebook) on a synthetic
drifting volume, once with densad's middle-group settings and once with the notebook's own
density table, and report how many edges differ. Usage: smoke_x138_relink.py <kernel_dir>"""
import ast, collections, json, math, os, re, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

k = sys.argv[1]
cells = [''.join(c['source']) for c in json.load(open(f'kernels/{k}/{k}.ipynb'))['cells']]
for key, val in re.findall(r'os\.environ\[["\'](BIOHUB_\w+)["\']\]\s*=\s*["\']([^"\']*)["\']', cells[0]):
    os.environ[key] = val
ns = {'os': os, 'np': np, 'math': math, 'linear_sum_assignment': linear_sum_assignment, 'cKDTree': cKDTree}
exec('from __future__ import annotations\nfrom typing import *', ns)
for i in (2, 5):
    for node in ast.parse(cells[i]).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            try: exec(compile(ast.Module([node], []), f'c{i}', 'exec'), ns)
            except Exception: pass
rng = np.random.default_rng(0); nodes, probs, prev, nid = {}, {}, {}, 0
base = rng.uniform(0, 300, (200, 3))
for t in range(6):
    pos = base + t * np.array([3.5, 5.0, -4.0]) + rng.normal(0, 2.6, base.shape); cur = {}
    for i, p in enumerate(pos):
        nodes[nid] = {'id': nid, 't': t, 'z': p[0], 'y': p[1], 'x': p[2]}; cur[i] = nid
        if t: probs[(prev[i], nid)] = float(rng.uniform(0.3, 1.0))
        nid += 1
    prev = cur
def run(cfg):
    for a, b in (('tight_um', 'TIGHT_UM'), ('relaxed_um', 'RELAXED_UM'), ('velocity_weight', 'VELOCITY_WEIGHT'), ('learned_bonus', 'LEARNED_BONUS')):
        ns['MOTION_RELINK_' + b] = cfg[a]
    st = collections.defaultdict(int)
    return {(e['source_id'], e['target_id']) for e in ns['motion_relink_edges'](dict(nodes), st, probs)}
g = ns['determine_density_group'](nodes)
e_var = run(ns['DENSITY_GROUP_OVERRIDES'][g])
e_den = run({'tight_um': 6.5, 'relaxed_um': 9.0, 'velocity_weight': 0.0, 'learned_bonus': 6.0})
print(f'{k}: group {g} | variant {ns["DENSITY_GROUP_OVERRIDES"][g]} | edges {len(e_var)} vs densad {len(e_den)} | differing {len(e_var ^ e_den)}')
