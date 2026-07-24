#!/usr/bin/env python3
"""Characterise the ground truth across all train volumes.

Answers the questions Phase 0 is blocked on (PLAN.md, docs/data.md):

  * how sparse is the annotation, and does it vary?
  * how many volumes contain divisions at all -- the division term is ~11% of the score, and
    the one volume inspected by hand had zero
  * what is the inter-frame displacement distribution (sets the linking radius)
  * do the two embryo prefixes (6bba, 44b6) differ enough to matter for CV grouping

Usage:
    python scripts/analyze_gt.py --root /Volumes/extSSD/kaggle/biohub/raw/train
    python scripts/analyze_gt.py --root ... --csv docs/gt_stats.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE  # noqa: E402


def summarise_one(path: Path) -> dict:
    gt = load_geff(path)
    g = gt.graph

    pos = {int(i): p for i, p in zip(g.ids, g.zyx)}
    disp = np.array(
        [np.linalg.norm((pos[int(b)] - pos[int(a)]) * SCALE) for a, b in g.edges]
    ) if gt.n_edges else np.array([])

    t = g.t
    # Do GT edges ever skip a frame? This decides whether gap-closing may emit skip edges or
    # must insert an intermediate node. An unmatched skip edge touching an annotated node is a
    # pure false positive, so getting this wrong is costly.
    tmap = {int(i): int(tt) for i, tt in zip(g.ids, g.t)}
    dt = np.array([tmap[int(b)] - tmap[int(a)] for a, b in g.edges]) if gt.n_edges else np.array([])

    return {
        "dataset": gt.dataset,
        "embryo": gt.dataset.split("_")[0],
        "nodes": gt.n_nodes,
        "edges": gt.n_edges,
        "tracks": gt.n_tracks(),
        "divisions": gt.n_divisions(),
        "est_n_nodes": gt.estimated_number_of_nodes,
        "density": gt.density,
        "t_min": int(t.min()) if gt.n_nodes else -1,
        "t_max": int(t.max()) if gt.n_nodes else -1,
        "edge_dt_max": int(dt.max()) if dt.size else -1,
        "edge_dt_skip_frac": float((dt > 1).mean()) if dt.size else float("nan"),
        "disp_median_um": float(np.median(disp)) if disp.size else float("nan"),
        "disp_p90_um": float(np.percentile(disp, 90)) if disp.size else float("nan"),
        "disp_max_um": float(disp.max()) if disp.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory containing *.geff stores")
    ap.add_argument("--csv", help="write the per-volume table here")
    args = ap.parse_args()

    stores = find_geff_stores(args.root)
    if not stores:
        sys.exit(f"no .geff stores under {args.root}")
    print(f"{len(stores)} geff stores\n")

    rows = []
    for i, p in enumerate(stores, 1):
        try:
            rows.append(summarise_one(p))
        except Exception as exc:  # noqa: BLE001 -- report and continue; one bad store shouldn't abort
            print(f"  FAILED {p.name}: {exc}", file=sys.stderr)
        if i % 25 == 0:
            print(f"  ...{i}/{len(stores)}", file=sys.stderr)

    df = pd.DataFrame(rows)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}\n")

    pd.set_option("display.width", 140)
    print("== per-volume distribution ==")
    print(df[["nodes", "edges", "tracks", "divisions", "est_n_nodes", "density",
              "edge_dt_max", "disp_median_um", "disp_p90_um", "disp_max_um"]].describe().T)

    print("\n== totals ==")
    print(f"  volumes                 {len(df)}")
    print(f"  annotated nodes (total) {df.nodes.sum():,}")
    print(f"  annotated edges (total) {df.edges.sum():,}")
    print(f"  divisions (total)       {df.divisions.sum():,}")
    print(f"  volumes with >=1 div    {(df.divisions > 0).sum()} / {len(df)} "
          f"({(df.divisions > 0).mean():.1%})")
    print(f"  mean annotation density {df.density.mean():.4%}")

    print("\n== by embryo (CV grouping must respect this) ==")
    print(df.groupby("embryo").agg(
        volumes=("dataset", "count"), nodes=("nodes", "sum"), edges=("edges", "sum"),
        divisions=("divisions", "sum"), density=("density", "mean"),
        disp_p90=("disp_p90_um", "mean"),
    ))

    skip = df.edge_dt_skip_frac.fillna(0)
    if (skip == 0).all():
        print("\n  All GT edges connect CONSECUTIVE frames (dt=1). Gap closing must therefore")
        print("  insert an intermediate node plus two dt=1 edges -- emitting a skip edge from")
        print("  an annotated node would be a pure false positive.")
    else:
        print(f"\n  {(skip > 0).sum()} volumes contain skip edges (max dt {df.edge_dt_max.max()}) --")
        print("  linking must be able to emit them.")

    if df.divisions.sum() == 0:
        print("\n  !! NO divisions anywhere: the 0.1 division term never engages and the")
        print("     public 'fake forks' augmentation cannot be buying score through it.")
    else:
        share = df.divisions.sum() / df.edges.sum()
        print(f"\n  divisions per annotated edge: {share:.4f} -- rare-event Jaccard, expect")
        print("  high variance in the division term.")


if __name__ == "__main__":
    main()
