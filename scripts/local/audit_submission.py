#!/usr/bin/env python3
"""STRATEGY 23.5 -- structural audit of a submission.csv against facts we know.

Cheap checks that need no GPU and no leaderboard read. Each one is a fact
established from the corpus that the pipeline could be violating silently:

1. **GT is 100% `dt = 1`.** Across all 199 volumes `edge_dt_max = 1` and
   `edge_dt_skip_frac = 0`, no exceptions. So a predicted edge whose endpoints
   are more than one frame apart cannot match any GT edge -- it is a guaranteed
   non-TP that can only ever be an FP (when an endpoint matches an annotated
   node) or invisible. Gap-closing inserts synthetic midpoint nodes so it
   *should* be compliant; this has never been checked.

2. **Out-degree >= 3 is biologically impossible** for a division, and out-degree
   0 in mid-movie is a track end. Both are cheap sanity reads on the graph.

3. **Edge length** against `OUTPUT_EDGE_MAX_UM = 14.0`, and against the observed
   GT displacement distribution (corpus median 1.82 um, p90 3.63, max 60.8).

4. **Division siting** -- where the predicted divisions land relative to the
   annotated subgraph, which is the only place they can score (23.3: only ~1.4%
   of inserted divisions land on annotated nodes at all).

    python scripts/local/audit_submission.py submissions/d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import SCALE, match_nodes, Graph  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]
    print(f"{args.submission}: {len(nodes):,} nodes, {len(edges):,} edges, "
          f"{df.dataset.nunique()} volumes\n")

    tot_bad_dt = 0
    tot_div = 0
    tot_div_annotated = 0
    for ds, gn in nodes.groupby("dataset"):
        ge = edges[edges.dataset == ds]
        t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], dtype=float) * SCALE
               for r in gn.itertuples()}

        src = ge.source_id.astype(int).to_numpy()
        tgt = ge.target_id.astype(int).to_numpy()
        dt = np.array([t_of[b] - t_of[a] for a, b in zip(src, tgt)])
        bad = int((dt != 1).sum())
        tot_bad_dt += bad

        length = np.array([float(np.linalg.norm(pos[b] - pos[a])) for a, b in zip(src, tgt)])
        outdeg = Counter(src)
        deg_hist = Counter(outdeg.values())
        divisions = [n for n, d in outdeg.items() if d >= 2]
        tot_div += len(divisions)

        # Which predicted divisions sit on an annotated GT node -- the only ones
        # the metric can see at all (23.3).
        gt = load_geff(args.gt_root / f"{ds}.geff")
        pred = Graph(
            nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                             dtype=np.float64),
            edges=np.asarray(list(zip(src, tgt)), dtype=np.int64),
        )
        matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
        div_on_annotated = sum(1 for d in divisions if d in matched)
        tot_div_annotated += div_on_annotated

        print(f"  {ds}")
        print(f"    edges dt!=1            {bad:>6,}  ({bad/len(ge):.3%})"
              f"{'   <-- CANNOT EVER BE A TP' if bad else ''}")
        print(f"    dt histogram           {dict(sorted(Counter(dt).items()))}")
        print(f"    edge length um         med {np.median(length):.2f}  p90 "
              f"{np.percentile(length,90):.2f}  max {length.max():.2f}  "
              f"(>14.0: {int((length>14.0).sum())})")
        print(f"    out-degree histogram   {dict(sorted(deg_hist.items()))}")
        print(f"    divisions (outdeg>=2)  {len(divisions):>6,}  of which on an "
              f"annotated node: {div_on_annotated} ({div_on_annotated/max(len(divisions),1):.2%})")
        gt_div = sum(1 for _, c in Counter(int(a) for a, _ in gt.graph.edges).items() if c >= 2)
        print(f"    GT divisions           {gt_div:>6}")

    print(f"\nTOTALS")
    print(f"  edges with dt != 1: {tot_bad_dt:,}" +
          ("  -- all edges are single-frame, as GT requires" if tot_bad_dt == 0
           else "  <-- STRUCTURAL FPs, none can score"))
    print(f"  predicted divisions: {tot_div:,}, on annotated nodes: {tot_div_annotated} "
          f"({tot_div_annotated/max(tot_div,1):.2%})")
    print("  ^ only the annotated ones can enter division_jaccard at all (23.3),")
    print("    so the FP budget is set by this fraction, not by the raw insert count.")


if __name__ == "__main__":
    main()
