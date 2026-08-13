#!/usr/bin/env python3
"""Score a submission.csv with the OFFICIAL competition scorer (STRATEGY 47.4).

Thin orchestrator over royerlab/kaggle-cell-tracking-competition's own scripts,
used verbatim (csv_to_geffs.py -> evaluate.py) so there is zero reimplementation
surface: division windows with grandchild evidence, sample-size weighting, and
the uncapped-above node multiplier are exactly the shipped semantics.

CALIBRATION (measured on evalset_d4edge, 24 volumes, 2026-08-08):
    official score 0.9117   vs   ctrack instrument 0.8947   (+0.0170)
    edge_J agrees to 0.0001; division TP/FP/FN agree EXACTLY (2/40/23).
    The delta is aggregation: official weights volumes by sample size, ours
    macro-averages. Relative comparisons on either instrument remain valid;
    absolute levels are only trustworthy HERE.

    .venv/bin/python scripts/local/score_official.py <submission.csv>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Durable location: the original vendored copy lived in a session scratchpad and
# was destroyed by the /private/tmp sweep (row 67). Re-clone with:
#   git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git official_metric
# ⚠ Re-cloning takes upstream HEAD, which is NOT necessarily the revision row 46
# calibrated against — verify against a known artifact before trusting a number.
OFFICIAL = Path(os.environ.get(
    "BIOHUB_OFFICIAL_METRIC",
    "/Volumes/extSSD/kaggle/biohub/replay_assets/official_metric"))
GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
PY = REPO / ".venv/bin/python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-dir", default=str(GT_ROOT))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="official_geffs_") as tmp:
        r = subprocess.run(
            [str(PY), "scripts/csv_to_geffs.py", "--csv",
             str(Path(args.submission).resolve()), "--out-dir", tmp],
            cwd=OFFICIAL, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(1)
        print(r.stdout.strip().splitlines()[-1])

        r = subprocess.run(
            [str(PY), "scripts/evaluate.py", "--pred-dir", tmp,
             "--gt-dir", args.gt_dir],
            cwd=OFFICIAL, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if ("WARNING" in line or "_base_graph" in line
                    or not line.strip()):
                continue
            print(line)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit(1)


if __name__ == "__main__":
    main()
