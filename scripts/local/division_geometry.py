#!/usr/bin/env python3
"""Is our safe-division heuristic better than random at hitting REAL divisions?

STRATEGY 14 argues the division term is a precision play worth up to +0.1. That
argument has a hole: `div_J = TP/(TP+FP+FN)`, so if TP is 0 the term is 0 no
matter how few FPs we emit. Removing FPs only pays if we ever hit a real one.

So measure the GT side directly. For every annotated dividing node, compute:

  * parent -> daughter displacement (vs BIOHUB_SAFE_DIV_MAX_UM = 4.66)
  * sister <-> sister separation   (vs BIOHUB_SAFE_DIV_SISTER_MAX_UM = 8.5)

and compare those against the same quantities for NON-dividing nodes, which is
what sets the false-positive rate. If real divisions look like ordinary motion,
the gates cannot separate them and the whole thesis is dead.

    python scripts/local/division_geometry.py --root /Volumes/extSSD/kaggle/biohub/gt/train
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE  # noqa: E402

SAFE_DIV_MAX_UM = 4.66
SAFE_DIV_SISTER_MAX_UM = 8.5


def collect(path: Path) -> dict:
    gt = load_geff(path)
    g = gt.graph
    pos = {int(r[0]): r[2:5] for r in g.nodes}
    tof = {int(r[0]): float(r[1]) for r in g.nodes}

    children = defaultdict(list)
    for a, b in g.edges:
        children[int(a)].append(int(b))

    div_parent_child, div_sister = [], []
    lin_parent_child = []  # non-dividing single-child steps: the FP population

    for p, kids in children.items():
        d = [float(np.linalg.norm((pos[k] - pos[p]) * SCALE)) for k in kids]
        if len(kids) >= 2:
            div_parent_child.extend(d)
            for i in range(len(kids)):
                for j in range(i + 1, len(kids)):
                    div_sister.append(
                        float(np.linalg.norm((pos[kids[i]] - pos[kids[j]]) * SCALE))
                    )
        elif len(kids) == 1:
            lin_parent_child.extend(d)

    # How many *unrelated* same-frame cell pairs fall inside the sister gate?
    # That is the population a distance-only sister test would wrongly admit.
    by_t = defaultdict(list)
    for n, t in tof.items():
        by_t[t].append(pos[n])
    near_pairs = tot_pairs = 0
    for t, ps in by_t.items():
        if len(ps) < 2:
            continue
        a = np.asarray(ps) * SCALE
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        iu = np.triu_indices(len(a), k=1)
        dd = d[iu]
        tot_pairs += dd.size
        near_pairs += int((dd <= SAFE_DIV_SISTER_MAX_UM).sum())

    return {
        "dataset": gt.dataset,
        "n_div": len(div_sister),
        "div_parent_child": div_parent_child,
        "div_sister": div_sister,
        "lin_parent_child": lin_parent_child,
        "near_pairs": near_pairs,
        "tot_pairs": tot_pairs,
        "est_n": gt.estimated_number_of_nodes,
        "nodes": gt.n_nodes,
    }


def q(a, name):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        print(f"  {name:<26} (empty)")
        return
    print(
        f"  {name:<26} n={a.size:<7} median={np.median(a):6.2f}  "
        f"p10={np.percentile(a,10):6.2f}  p90={np.percentile(a,90):6.2f}  max={a.max():7.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    stores = find_geff_stores(args.root)
    print(f"{len(stores)} geff stores", flush=True)

    acc = defaultdict(list)
    near = tot = 0
    n_vol_with_div = 0
    for i, p in enumerate(stores, 1):
        try:
            r = collect(p)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        for k in ("div_parent_child", "div_sister", "lin_parent_child"):
            acc[k].extend(r[k])
        near += r["near_pairs"]
        tot += r["tot_pairs"]
        n_vol_with_div += r["n_div"] > 0
        if i % 50 == 0:
            print(f"  ...{i}/{len(stores)}", flush=True)

    print(f"\n== division geometry, all {len(stores)} volumes ==")
    print(f"volumes with >=1 annotated division: {n_vol_with_div}")
    print(f"total annotated divisions: {len(acc['div_sister'])}")
    q(acc["div_parent_child"], "division parent->daughter")
    q(acc["div_sister"], "division sister<->sister")
    q(acc["lin_parent_child"], "NON-div parent->child")

    dpc = np.asarray(acc["div_parent_child"])
    ds = np.asarray(acc["div_sister"])
    lpc = np.asarray(acc["lin_parent_child"])

    print(f"\n== gate pass rates (SAFE_DIV_MAX_UM={SAFE_DIV_MAX_UM}, "
          f"SISTER_MAX_UM={SAFE_DIV_SISTER_MAX_UM}) ==")
    if dpc.size:
        print(f"  real divisions passing parent->child gate : "
              f"{(dpc <= SAFE_DIV_MAX_UM).mean():.3f}")
    if ds.size:
        print(f"  real divisions passing sister gate        : "
              f"{(ds <= SAFE_DIV_SISTER_MAX_UM).mean():.3f}")
    if lpc.size:
        print(f"  ordinary motion passing parent->child gate: "
              f"{(lpc <= SAFE_DIV_MAX_UM).mean():.3f}   <-- FP driver")
    if tot:
        print(f"  same-frame cell pairs within sister gate  : "
              f"{near/tot:.5f}  ({near}/{tot})   <-- FP driver")


if __name__ == "__main__":
    main()
