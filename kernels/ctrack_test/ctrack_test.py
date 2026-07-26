"""Fast test: robustly locate + import ctrack from the ctrack-src dataset (content-at-root)."""
import sys, glob, shutil
from pathlib import Path

def _add_ctrack():
    # ctrack-src ships the package CONTENTS at the dataset root (no ctrack/ folder, no zip).
    # Find the dir that holds eval/biohub.py (our signature) and expose it as an importable 'ctrack'.
    root = None
    for sig in ("eval/biohub.py", "detect/blob.py"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{sig}", recursive=True) if "tracking_repo" not in h]
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
    print("ctrack root found:", root, flush=True)

_add_ctrack()
import numpy as np
from ctrack.eval.biohub import Graph, evaluate, per_sample_metrics, summarise
from ctrack.data.geff import load_geff
# tiny self-test of the metric
nodes = np.array([[1,0,0,0,0],[2,1,0,0,1],[3,2,0,0,2]], float)
edges = np.array([[1,2],[2,3]])
g = Graph(nodes=nodes, edges=edges)
print("ctrack import + Graph OK:", len(g.nodes), "nodes,", len(g.edges), "edges", flush=True)
print("SUCCESS: ctrack usable in kernel", flush=True)
