#!/usr/bin/env python3
"""STRATEGY 24.3's pre-registered test -- WHY is each GT division missed?

§24.3 found, on the 2 division FNs the 4-volume dev set contains, that neither
was reachable by any selector: one daughter was never detected, the other was
detected but already consumed by the 1-1 assignment (§24.6). That was **n = 2**,
which is exactly the 3-division instrument §20.3 warned about and that §23.3
criticised §20 for trusting. It was recorded as a mechanism hypothesis, not a
conclusion.

This is the test at scale. Every GT division in the scored volumes is placed in
exactly one bucket:

  TP           source matched, predicted out-degree >= 2            -> already scored
  FN-parent    source not matched to any predicted node             -> detection, parent
  FN-daughter  source matched, second daughter never detected       -> detection, daughter
  FN-consumed  daughter detected but in-degree 1 (another parent)   -> the 1-1 assignment
  FN-REACHABLE daughter detected AND orphan, inside the wide gates  -> a SELECTOR could fix it

Only the last bucket is addressable by lever #2 (a fitted division selector).
The ceiling on that lever is `FN-REACHABLE / all GT divisions`; everything else
needs better detection (§18) or a division-aware assignment (§24.7).

⚠ Read `FN-consumed` carefully: the daughter being taken by another parent may
mean the assignment is WRONG there (a swap, §18.2) rather than the division
being unrecoverable. Those overlap with the association problem.

    python scripts/local/division_fn_diagnosis.py submissions/evalset_d4edge/submission.csv
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
from ctrack.eval.biohub import SCALE, Graph, match_nodes  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")
WIDE_PARENT_UM = 12.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt-root", type=Path, default=GT_ROOT)
    ap.add_argument("--gate", type=float, default=WIDE_PARENT_UM)
    args = ap.parse_args()

    df = pd.read_csv(args.submission)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    buckets: Counter[str] = Counter()
    detail: list[tuple[str, str, str]] = []

    for ds, gn in nodes.groupby("dataset"):
        ge = edges[edges.dataset == ds]
        src = ge.source_id.astype(int).to_numpy()
        tgt = ge.target_id.astype(int).to_numpy()
        outdeg: Counter[int] = Counter(src)
        indeg: Counter[int] = Counter(tgt)
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], float) * SCALE
               for r in gn.itertuples()}
        t_of = dict(zip(gn.node_id.astype(int), gn.t.astype(int)))
        by_t: dict[int, list[int]] = defaultdict(list)
        for n, t in t_of.items():
            by_t[t].append(n)

        gt = load_geff(args.gt_root / f"{ds}.geff")
        gt_ch: dict[int, list[int]] = defaultdict(list)
        for a, b in gt.graph.edges:
            gt_ch[int(a)].append(int(b))
        gt_divs = [n for n, ks in gt_ch.items() if len(ks) >= 2]
        if not gt_divs:
            continue

        pred = Graph(
            nodes=np.asarray([[r.node_id, r.t, r.z, r.y, r.x] for r in gn.itertuples()],
                             dtype=np.float64),
            edges=np.asarray(list(zip(src, tgt)), dtype=np.int64),
        )
        matched = match_nodes(gt.graph, pred, max_distance=7.0, scale=SCALE)
        gt2pred = {g: p for p, g in matched.items()}

        for gdiv in gt_divs:
            p = gt2pred.get(gdiv)
            if p is None:
                buckets["FN-parent"] += 1
                detail.append((ds, str(gdiv), "parent not detected"))
                continue
            if outdeg.get(p, 0) >= 2:
                buckets["TP"] += 1
                detail.append((ds, str(gdiv), "already a division"))
                continue

            # Source is live (out-degree <= 1). Where are its daughters?
            linked = {int(b) for a, b in zip(src, tgt) if a == p}
            states = []
            for d in gt_ch[gdiv]:
                dp = gt2pred.get(d)
                if dp is None:
                    states.append("undetected")
                elif dp in linked:
                    states.append("is-the-linked-child")
                elif indeg.get(dp, 0) >= 1:
                    states.append("consumed")
                elif float(np.linalg.norm(pos[dp] - pos[p])) <= args.gate:
                    states.append("orphan-in-gate")
                else:
                    states.append("orphan-out-of-gate")

            others = [s for s in states if s != "is-the-linked-child"]
            if "orphan-in-gate" in others:
                buckets["FN-REACHABLE"] += 1
                detail.append((ds, str(gdiv), "orphan daughter inside gate"))
            elif "consumed" in others:
                buckets["FN-consumed"] += 1
                detail.append((ds, str(gdiv), "daughter taken by another parent"))
            elif "undetected" in others:
                buckets["FN-daughter"] += 1
                detail.append((ds, str(gdiv), "daughter not detected"))
            else:
                buckets["FN-other"] += 1
                detail.append((ds, str(gdiv), f"states={others}"))

    total = sum(buckets.values())
    print(f"{args.submission}: {total} GT divisions across "
          f"{nodes.dataset.nunique()} volumes\n")
    order = ["TP", "FN-REACHABLE", "FN-consumed", "FN-daughter", "FN-parent", "FN-other"]
    for k in order:
        if buckets.get(k):
            note = {
                "TP": "already scored",
                "FN-REACHABLE": "<-- a fitted SELECTOR could fix these (lever #2)",
                "FN-consumed": "<-- the 1-1 assignment took the daughter (§24.6)",
                "FN-daughter": "<-- detection failure on the daughter",
                "FN-parent": "<-- detection failure on the parent",
            }.get(k, "")
            print(f"  {k:<14} {buckets[k]:>4}  ({buckets[k]/total:>5.1%})  {note}")

    reach = buckets.get("FN-REACHABLE", 0)
    print(f"\nCEILING ON LEVER #2 (division selector): {reach}/{total} = "
          f"{reach/max(total,1):.1%} of GT divisions")
    if reach == 0:
        print("  => §24.3's hypothesis CONFIRMED at this scale: no division FN is")
        print("     reachable by better selection. The lever is FP-trimming only.")
    else:
        print("  => §24.3's hypothesis is REFUTED in part; a selector has real recall")
        print("     headroom. Re-rank lever #2 upward accordingly.")

    if len(detail) <= 40:
        print("\nper-division detail:")
        for ds, gd, why in detail:
            print(f"    {ds:>16}  {gd:>10}  {why}")


if __name__ == "__main__":
    main()
