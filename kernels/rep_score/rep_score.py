"""Score the already-computed REP-VAL submissions on CPU, and sweep post-hoc node pruning.

Why this kernel exists: `rep_full`, `rep_lean`, `rep_ens0`, `rep_ens5` all ran their pipelines
to completion (~35 min GPU each) but their in-notebook scoring cell died on a stale `ctrack`
import path, so the numbers were lost. The predictions survived in each kernel's output, so we
re-score them here on CPU in minutes -- no GPU slot, no submission quota, no pipeline re-run.

It also sweeps two things that are pure transformations of the submission graph, so they need no
re-run either:

  * global min-track-length pruning -- drop connected components below L nodes
  * ORACLE per-volume pruning to a target node ratio (uses the GT `estimated_number_of_nodes`,
    which is NOT readable at test time -- this measures the CEILING of the per-volume
    calibration lever, it is not a deployable config)

Measured beforehand on these submissions (see docs/STRATEGY.md): the pipeline OVER-detects on
8 of 12 REP-VAL volumes, so the mean node-budget multiplier is 0.9904 (FULL) / 0.9947 (LEAN) --
below 1.0, i.e. we are paying a penalty where the metric offers a bonus. The sweep asks how much
of that is recoverable and what it costs in edge Jaccard.
"""

import glob
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- deps
# The plain CPU image has no zarr (the GPU pipeline image gets it via the support pack), and
# load_geff needs it to read the GT stores. Needs enable_internet in kernel-metadata.json.
try:
    import zarr  # noqa: F401
except ModuleNotFoundError:
    import subprocess

    print("zarr missing -- installing", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=True)
    import zarr  # noqa: F401
print("zarr", zarr.__version__, flush=True)

# --------------------------------------------------------------------------- ctrack import
# The ctrack-src dataset ships the package CONTENTS at the root (no ctrack/ folder), so the
# path has to be discovered and, if needed, re-rooted under a real `ctrack` package dir.
# This is the form proven by the biohub-ctrack-test kernel.


def _add_ctrack():
    root = None
    for sig in ("eval/biohub.py", "detect/blob.py"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{sig}", recursive=True)
                if "tracking_repo" not in h]
        if hits:
            root = Path(hits[0]).parents[1]
            break
    if root is None:
        raise FileNotFoundError("ctrack-src not found under /kaggle/input")
    if root.name == "ctrack":
        sys.path.insert(0, str(root.parent))
    else:
        dest = Path("/kaggle/working/_ctrackpkg/ctrack")
        if not dest.exists():
            shutil.copytree(root, dest)
        sys.path.insert(0, str(dest.parent))
    print("ctrack root:", root, flush=True)


_add_ctrack()
from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import Graph, evaluate, per_sample_metrics, summarise  # noqa: E402

print("ctrack import OK", flush=True)

# --------------------------------------------------------------------------- inputs
TRAIN = next(p for p in ["/kaggle/input/competitions/biohub-cell-tracking-during-development/train",
                         "/kaggle/input/biohub-cell-tracking-during-development/train"]
             if os.path.isdir(p))
print("TRAIN =", TRAIN, flush=True)

# The four submissions ride in as the `biohub-repval-submissions` dataset (gzipped CSV).
# They cannot come in via kernel_sources: Kaggle rejects outputs of kernels that errored, and
# all four rep_* kernels errored in their scoring cell.
WANT = ["rep-full", "rep-lean", "rep-ens0", "rep-ens5"]
subs = {}
for path in glob.glob("/kaggle/input/**/*submission.csv*", recursive=True):
    for w in WANT:
        if w in os.path.basename(path):
            subs[w] = path
print("submissions found:", {k: v for k, v in sorted(subs.items())}, flush=True)
missing = [w for w in WANT if w not in subs]
if missing:
    print("!! MISSING:", missing, flush=True)
if not subs:
    raise SystemExit("no submissions mounted -- check dataset_sources in kernel-metadata.json")

# --------------------------------------------------------------------------- GT cache
GT = {}


def gt_for(ds):
    if ds not in GT:
        GT[ds] = load_geff(Path(TRAIN) / f"{ds}.geff")
    return GT[ds]


# --------------------------------------------------------------------------- graph helpers
def parse(path):
    """submission.csv -> {dataset: (nodes (N,5) float, edges (E,2) int64)}."""
    df = pd.read_csv(path)
    out = {}
    for ds, g in df.groupby("dataset"):
        nodes = g[g.row_type == "node"][["node_id", "t", "z", "y", "x"]].values.astype(float)
        edges = g[g.row_type == "edge"][["source_id", "target_id"]].values.astype(np.int64)
        out[ds] = (nodes, edges)
    return out


def components(nodes, edges):
    """Connected components over the undirected graph. Returns {node_id: comp_size}."""
    adj = defaultdict(list)
    for s, t in edges:
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    size = {}
    seen = set()
    for nid in nodes[:, 0].astype(np.int64):
        nid = int(nid)
        if nid in seen:
            continue
        stack, comp = [nid], []
        seen.add(nid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        for u in comp:
            size[u] = len(comp)
    return size


def prune_by_min_len(nodes, edges, min_len, size=None):
    """Drop every component with fewer than `min_len` nodes, and its incident edges."""
    size = size if size is not None else components(nodes, edges)
    keep = {n for n, s in size.items() if s >= min_len}
    nmask = np.array([int(i) in keep for i in nodes[:, 0].astype(np.int64)])
    emask = np.array([int(s) in keep and int(t) in keep for s, t in edges]) if len(edges) \
        else np.zeros(0, dtype=bool)
    return nodes[nmask], (edges[emask] if len(edges) else edges)


def prune_to_ratio(nodes, edges, est_n, target_ratio, size=None):
    """ORACLE: drop shortest components until N_pred/est_n <= 1 + target_ratio.

    Uses est_n, which is geff metadata absent from test volumes. Ceiling measurement only.
    """
    size = size if size is not None else components(nodes, edges)
    budget = (1.0 + target_ratio) * est_n
    if len(nodes) <= budget:
        return nodes, edges, None
    # Remove whole components, shortest first, until under budget.
    by_comp = defaultdict(list)
    for n, s in size.items():
        by_comp[s].append(n)
    dropped, n_left = set(), len(nodes)
    for s in sorted(by_comp):
        for n in by_comp[s]:
            if n_left <= budget:
                break
            dropped.add(n)
            n_left -= 1
        if n_left <= budget:
            break
    keep = {int(i) for i in nodes[:, 0].astype(np.int64)} - dropped
    nmask = np.array([int(i) in keep for i in nodes[:, 0].astype(np.int64)])
    emask = np.array([int(s) in keep and int(t) in keep for s, t in edges]) if len(edges) \
        else np.zeros(0, dtype=bool)
    # effective min length actually applied
    kept_sizes = [size[n] for n in keep] or [0]
    return nodes[nmask], (edges[emask] if len(edges) else edges), min(kept_sizes)


def score(graphs, label, transform=None):
    """Score a whole submission (dict of per-volume graphs), optionally transformed."""
    rows, npred, mults = [], 0, []
    for ds, (nodes, edges) in sorted(graphs.items()):
        gt = gt_for(ds)
        if transform is not None:
            nodes, edges = transform(ds, nodes, edges, gt)
        r = per_sample_metrics(evaluate(gt.graph, Graph(nodes=nodes, edges=edges)),
                               gt.estimated_number_of_nodes)
        rows.append(r)
        npred += len(nodes)
        mults.append(1 - 0.1 * r["total_node_ratio"])
    s = summarise(rows)
    print(f"  {label:<34} score={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"adj={s['adj_edge_jaccard']:.4f}  div_J={s['division_jaccard']:.4f}  "
          f"N={npred:,}  mean_mult={np.mean(mults):.4f}", flush=True)
    return s


# --------------------------------------------------------------------------- 1. as-is scores
print("\n" + "=" * 100)
print("1. AS-IS SCORES  (the locked LEAN-vs-FULL question, plus the #2-lite blend sweep)")
print("=" * 100, flush=True)

parsed = {}
results = {}
for w in WANT:
    if w not in subs:
        continue
    parsed[w] = parse(subs[w])
    results[w] = score(parsed[w], w)

# --------------------------------------------------------------------------- 2. pruning sweep
print("\n" + "=" * 100)
print("2. POST-HOC MIN-TRACK-LENGTH PRUNING  (recover the node-budget bonus we are not collecting)")
print("=" * 100, flush=True)

# Cache components once per (submission, volume) -- they do not change under pruning by length.
comp_cache = {w: {ds: components(n, e) for ds, (n, e) in g.items()} for w, g in parsed.items()}

for w in [x for x in ("rep-lean", "rep-full") if x in parsed]:
    print(f"\n-- {w} --", flush=True)
    score(parsed[w], f"{w} baseline (L=asis)")
    for L in (8, 10, 12, 14, 16, 20):
        score(parsed[w], f"{w} min_track_len>={L}",
              transform=lambda ds, n, e, gt, L=L, w=w:
                  prune_by_min_len(n, e, L, size=comp_cache[w][ds]))

# --------------------------------------------------------------------------- 3. oracle ceiling
print("\n" + "=" * 100)
print("3. ORACLE per-volume pruning to a target node ratio")
print("   !! uses GT estimated_number_of_nodes -- NOT available at test time.")
print("   This measures the CEILING of per-volume calibration, to decide whether predicting")
print("   est_n_nodes from image statistics is worth building.")
print("=" * 100, flush=True)

for w in [x for x in ("rep-lean", "rep-full") if x in parsed]:
    print(f"\n-- {w} --", flush=True)
    for target in (-0.05, -0.10, -0.20, -0.30, -0.50):
        def tf(ds, n, e, gt, target=target, w=w):
            n2, e2, _ = prune_to_ratio(n, e, gt.estimated_number_of_nodes, target,
                                       size=comp_cache[w][ds])
            return n2, e2
        score(parsed[w], f"{w} ORACLE ratio<={target:+.2f}", transform=tf)

# --------------------------------------------------------------------------- summary
print("\n" + "=" * 100)
print("SUMMARY (as-is)")
for w, s in sorted(results.items()):
    print(f"  {w:<12} score={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f} "
          f"adj={s['adj_edge_jaccard']:.4f} div_J={s['division_jaccard']:.4f}")
print("=" * 100, flush=True)
