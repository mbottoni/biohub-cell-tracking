#!/usr/bin/env python3
"""STRATEGY 34 -- is there APPEARANCE information a better association model could use?

§33 closed association to every knob: bonus, edge threshold, velocity weight,
relink radius, neighbourhood flow. The remaining +0.0954 of headroom is real
(§27's oracle is a rigorous bound) but unreachable by anything in the current
cost function, which sees only geometry.

That leaves "train a better association model" -- and §12 prices it at 534 GPU-h
against a ~240 h quota. Before treating that as *the* plan, there is a prior
question nobody has asked, and it is free to answer:

    Does the image contain information that distinguishes the correct successor
    from the one we wrongly take?

If a cell's appearance is conserved frame to frame, the correct target should
resemble the source more than the distractor does. If it does NOT, then the
+0.0954 is not merely compute-blocked, it is INFORMATION-blocked: no association
model of any size can recover it from these images, and the honest conclusion is
that the remaining headroom is unreachable rather than expensive.

Method: for each of the 379 known swaps take the source S at t, the correct
target C at t+1, and the wrongly-taken target W at t+1. Extract a 3D patch around
each and ask whether S is more similar to C than to W, under three independent
measures:

  * normalised cross-correlation of the patches (texture/shape)
  * absolute difference of mean intensity (brightness, roughly conserved)
  * absolute difference of peak intensity

50% = coin flip = no signal. Meaningfully above 50% = a learned model has
something to work with.

⚠ This tests only whether the signal EXISTS. It does not show any model could
exploit it, and it uses GT-matched identities, so it is an upper bound on what
appearance can offer.

    python scripts/local/appearance_signal.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.data.volume import open_volume  # noqa: E402
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
RAW_ROOT = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
PZ, PY, PX = 2, 7, 7          # half-widths in voxels -> 5 x 15 x 15 patch


def patch(frame: np.ndarray, z: int, y: int, x: int) -> np.ndarray | None:
    Z, Y, X = frame.shape
    if not (PZ <= z < Z - PZ and PY <= y < Y - PY and PX <= x < X - PX):
        return None
    return frame[z - PZ:z + PZ + 1, y - PY:y + PY + 1, x - PX:x + PX + 1].astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    # Collect swaps first, so each frame is decompressed at most once.
    swaps: dict[str, list[tuple[int, tuple, tuple, tuple]]] = defaultdict(list)
    for ds, gn in nodes.groupby("dataset"):
        ge = edges[edges.dataset == ds]
        s = ge.source_id.astype(int).to_numpy()
        t = ge.target_id.astype(int).to_numpy()
        pe = set(zip(s.tolist(), t.tolist()))
        child = defaultdict(list)
        for a, b in zip(s, t):
            child[a].append(b)
        vox = {int(r.node_id): (int(r.t), int(round(r.z)), int(round(r.y)), int(round(r.x)))
               for r in gn.itertuples()}
        gt = load_geff(args.gt_root / f"{ds}.geff")
        pred = Graph(
            nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                             dtype=np.float64),
            edges=np.asarray(list(zip(s, t)), dtype=np.int64),
        )
        m = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
        g2p = {g: p for p, g in m.items()}
        for ga, gb in gt.graph.edges:
            pa, pb = g2p.get(int(ga)), g2p.get(int(gb))
            if pa is None or pb is None or (pa, pb) in pe:
                continue
            k = child.get(pa)
            if not k or k[0] == pb:
                continue
            swaps[ds].append((vox[pa][0], vox[pa][1:], vox[pb][1:], vox[k[0]][1:]))

    res = {"ncc": [], "mean": [], "peak": []}
    n_used = 0
    for ds, items in sorted(swaps.items()):
        raw = args.raw_root / f"{ds}.zarr"
        if not raw.exists():
            continue
        vol = open_volume(raw)
        cache: dict[int, np.ndarray] = {}

        def frame(tt: int) -> np.ndarray:
            if tt not in cache:
                if len(cache) > 6:
                    cache.clear()
                cache[tt] = np.asarray(vol.frame(tt))
            return cache[tt]

        for t, S, C, W in items:
            try:
                f0, f1 = frame(t), frame(t + 1)
            except Exception:
                continue
            ps, pc, pw = patch(f0, *S), patch(f1, *C), patch(f1, *W)
            if ps is None or pc is None or pw is None:
                continue
            n_used += 1
            res["ncc"].append(ncc(ps, pc) > ncc(ps, pw))
            res["mean"].append(abs(ps.mean() - pc.mean()) < abs(ps.mean() - pw.mean()))
            res["peak"].append(abs(ps.max() - pc.max()) < abs(ps.max() - pw.max()))

    print(f"swaps with all three patches in-bounds: {n_used}\n")
    if not n_used:
        print("no usable swaps")
        return
    print(f"{'appearance measure':<40} {'correct beats distractor':>26}")
    for k, label in (("ncc", "normalised cross-correlation"),
                     ("mean", "mean-intensity similarity"),
                     ("peak", "peak-intensity similarity")):
        v = np.array(res[k], bool)
        se = np.sqrt(0.25 / len(v))
        z = (v.mean() - 0.5) / se
        print(f"{label:<40} {v.mean():>17.1%}   z={z:>+5.1f}")
    print("\n  50% = coin flip = the image carries NO usable identity signal on these cases.")
    best = max(np.mean(res[k]) for k in res)
    if best < 0.55:
        print("  => APPEARANCE DOES NOT DISCRIMINATE. The remaining association headroom is")
        print("     INFORMATION-blocked, not merely compute-blocked: no association model")
        print("     can recover from these images what is not in them.")
    else:
        print("  => appearance carries real signal a learned model could exploit;")
        print("     the +0.0954 is compute-blocked rather than information-blocked.")


if __name__ == "__main__":
    main()
