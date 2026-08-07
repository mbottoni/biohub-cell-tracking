#!/usr/bin/env python3
"""Per-volume decomposition of the GT-snap localisation oracle.

19.2 / 39.3 made this mandatory: a single edge on a 49-GT-edge volume moves the
macro mean by +0.0096, so ANY headline macro delta must be shown to survive the
removal of thin volumes before it is believed. 39.3 was written because a
headline was published without it. This is that check for the snap oracle.

    python scripts/local/snap_decompose.py submissions/evalset_d4edge/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from localisation_doseresponse import load_submission, run_volume  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--level", type=float, default=1.0)
    args = ap.parse_args()

    subs = load_submission(Path(args.submission))
    order = sorted(subs)
    jb = [(ds, subs[ds].nodes, "noise", 0.0, 0) for ds in order]
    js = [(ds, subs[ds].nodes, "snap", args.level, 0) for ds in order]
    with ProcessPoolExecutor() as ex:
        base = list(ex.map(run_volume, jb))
        snap = list(ex.map(run_volume, js))

    rows = []
    for b, s in zip(base, snap):
        rows.append((b["dataset"], b["edge_tp"] + b["edge_fn"],
                     b["adj_edge_jaccard"], s["adj_edge_jaccard"],
                     s["adj_edge_jaccard"] - b["adj_edge_jaccard"],
                     b["edge_fp"], s["edge_fp"]))

    print(f"snap lambda={args.level}\n")
    print(f"{'volume':>16} {'GTedges':>8} {'adj base':>9} {'adj snap':>9} {'delta':>8} "
          f"{'eFP base':>9} {'eFP snap':>9}")
    for ds, gtE, ab, asn, d, fb, fs in sorted(rows, key=lambda r: -abs(r[4])):
        flag = "  <-- THIN" if gtE < 200 else ""
        print(f"{ds:>16} {gtE:>8} {ab:>9.4f} {asn:>9.4f} {d:>+8.4f} "
              f"{fb:>9} {fs:>9}{flag}")

    thick = [r for r in rows if r[1] >= 200]
    print(f"\n  headline macro adj delta (all {len(rows)}):           "
          f"{np.mean([r[4] for r in rows]):+.4f}")
    print(f"  TRANSFERABLE (mean over {len(thick)} thick volumes): "
          f"{np.mean([r[4] for r in thick]):+.4f}")
    print(f"  improved {sum(1 for r in rows if r[4] > 0.0005)}/{len(rows)}   "
          f"worsened {sum(1 for r in rows if r[4] < -0.0005)}/{len(rows)}")
    print(f"  edge FP total {sum(r[5] for r in rows)} -> {sum(r[6] for r in rows)}")


if __name__ == "__main__":
    main()
