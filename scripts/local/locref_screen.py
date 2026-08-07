#!/usr/bin/env python3
"""Screen a localisation change against the STRATEGY 40.5 gate.

40.2 measured the payoff curve for localisation: perfect coordinates are worth
+0.0259 transferable adj, but the response is strongly CONVEX (lambda 0.5 buys
+0.0080, lambda 1.0 buys +0.0305; loss ~ RMS^1.7), so a small improvement buys
nothing measurable. 40.5 therefore pre-registered a hard gate:

    no localisation intervention proceeds to a submission unless it first
    reduces matched-node RMS by >= 30%

This applies that gate. It reports, for two submissions over the same volumes:

  * matched-node residual RMS (localisation quality of the cells the metric can
    actually see -- computed from `match_nodes`, which uses position only and is
    therefore independent of the linker)
  * the convexity-implied score this RMS change is worth
  * the ACTUAL end-to-end score, which is the only thing that counts (38.5)

    python scripts/local/locref_screen.py submissions/d4edge/submission.csv \\
                                           submissions/locref/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import (  # noqa: E402
    MAX_MATCH_DISTANCE_UM, SCALE, Graph, match_nodes,
)
from score_submission import load_submission, score  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")

# 40.2's measured anchors: fractional RMS reduction -> transferable adj gain.
SNAP_FULL = 0.0259
EXPONENT = 1.7


def residuals(path: Path) -> dict[str, tuple[float, int]]:
    out = {}
    for ds, pred in sorted(load_submission(path).items()):
        gt = load_geff(GT_ROOT / f"{ds}.geff")
        matched = match_nodes(gt.graph, pred, max_distance=MAX_MATCH_DISTANCE_UM,
                             scale=SCALE)
        row_of = {int(n[0]): k for k, n in enumerate(pred.nodes)}
        gid_row = {int(g[0]): k for k, g in enumerate(gt.graph.nodes)}
        p_um = pred.nodes[:, 2:5] * SCALE
        g_um = gt.graph.nodes[:, 2:5] * SCALE
        d = [float(np.linalg.norm(p_um[row_of[int(p)]] - g_um[gid_row[int(g)]]))
             for p, g in matched.items() if int(p) in row_of and int(g) in gid_row]
        out[ds] = (float(np.sqrt(np.mean(np.square(d)))) if d else float("nan"), len(d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    rb, rc = residuals(Path(args.baseline)), residuals(Path(args.candidate))
    common = sorted(set(rb) & set(rc))
    if not common:
        raise SystemExit("the two submissions share no volumes")

    print(f"{'volume':>16} {'matched':>8} {'RMS base':>9} {'RMS cand':>9} {'reduction':>10}")
    for ds in common:
        b, n = rb[ds]
        c, _ = rc[ds]
        print(f"{ds:>16} {n:>8} {b:>8.3f}u {c:>8.3f}u {1 - c / b:>9.1%}")

    B = float(np.sqrt(np.mean([rb[d][0] ** 2 for d in common])))
    C = float(np.sqrt(np.mean([rc[d][0] ** 2 for d in common])))
    red = 1 - C / B
    print(f"\n  matched-node RMS   {B:.3f} um -> {C:.3f} um   ({red:+.1%})")

    implied = SNAP_FULL * (1 - (1 - max(red, 0.0)) ** EXPONENT)
    print(f"  40.5 convexity implies this RMS change is worth ~{implied:+.4f} adj")
    print(f"  40.5 GATE (>=30% reduction): "
          f"{'PASS' if red >= 0.30 else 'FAIL'}  ({red:+.1%})")

    (ab, _), (ac, _) = score(Path(args.baseline)), score(Path(args.candidate))
    print(f"\n  {'metric':<20} {'baseline':>9} {'candidate':>10} {'delta':>9}")
    for k in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"):
        print(f"  {k:<20} {ab[k]:>9.4f} {ac[k]:>10.4f} {ac[k] - ab[k]:>+9.4f}")

    print("\n  End-to-end score is the verdict (38.5: never rank by a per-item metric).")
    print("  RMS is the SCREEN -- it says whether the mechanism moved at all.")


if __name__ == "__main__":
    main()
