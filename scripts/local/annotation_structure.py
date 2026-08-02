#!/usr/bin/env python3
"""Is "this cell was annotated" predictable from track structure?

Context: the pad probe (LB 0.868 vs 0.912 base) pins us at ~97% of the node
budget, so the multiplier is ~1.003 and pruning is the only way to grow it.
`probe_len20` showed the trade is real but mispriced: dropping 22% of nodes
cost 4.8% of edge_J, where break-even needs <2.1%.  A prune only pays if it is
selective, so the question is whether annotated cells are structurally marked.

This reads the 199 train GEFF graphs and characterises the annotated tracks.
It cannot see unannotated cells directly -- that is what "unannotated" means --
but if annotated lineages are, say, near-full-movie in span, then any predicted
component that is short is almost certainly not carrying scored edges, which is
exactly the signal a ranker would exploit.

Run: .venv/bin/python scripts/local/annotation_structure.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import find_geff_stores, load_geff  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def track_spans(gt) -> tuple[list[int], list[int], int, int]:
    """Per-connected-component node count and temporal span."""
    ids = gt.graph.nodes[:, 0].astype(np.int64)
    t = gt.graph.nodes[:, 1]
    index = {int(i): k for k, i in enumerate(ids)}

    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in gt.graph.edges:
        a, b = index.get(int(s)), index.get(int(d))
        if a is None or b is None:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps: dict[int, list[int]] = defaultdict(list)
    for k in range(len(ids)):
        comps[find(k)].append(k)

    sizes, spans = [], []
    for members in comps.values():
        tt = t[members]
        sizes.append(len(members))
        spans.append(int(tt.max() - tt.min()) + 1)
    return sizes, spans, int(t.min()), int(t.max())


def main() -> None:
    stores = find_geff_stores(GT_ROOT)
    if not stores:
        raise SystemExit(f"no .geff stores under {GT_ROOT}")

    all_sizes: list[int] = []
    all_spans: list[float] = []   # span as a fraction of the movie
    per_vol = []

    for path in stores:
        gt = load_geff(path)
        if gt.n_nodes == 0:
            continue
        sizes, spans, t0, t1 = track_spans(gt)
        movie = max(t1 - t0 + 1, 1)
        all_sizes += sizes
        all_spans += [s / movie for s in spans]
        per_vol.append((gt.dataset, gt.n_nodes, gt.estimated_number_of_nodes, len(sizes), movie))

    sizes = np.array(all_sizes)
    spans = np.array(all_spans)
    print(f"volumes: {len(per_vol)}   annotated tracks: {len(sizes)}   annotated nodes: {sizes.sum():,}")
    print(f"\nannotated component SIZE (nodes)")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"   p{q:<3} {np.percentile(sizes, q):8.1f}")
    print(f"   mean {sizes.mean():8.1f}")
    print(f"\nannotated component SPAN (fraction of movie length)")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"   p{q:<3} {np.percentile(spans, q):8.3f}")

    print(f"\n--- what a min-length rule would cost on ANNOTATED tracks ---")
    print(f"{'min_len':>8}{'ann. tracks kept':>19}{'ann. nodes kept':>17}")
    for thr in (2, 4, 6, 10, 15, 20, 30, 40):
        keep = sizes >= thr
        print(f"{thr:>8}{keep.mean()*100:>18.1f}%{sizes[keep].sum()/sizes.sum()*100:>16.1f}%")

    dens = np.array([n / e for _, n, e, _, _ in per_vol if e and np.isfinite(e)])
    print(f"\nannotated / est_n_nodes: median {np.median(dens):.4f}  mean {dens.mean():.4f}")


if __name__ == "__main__":
    main()
