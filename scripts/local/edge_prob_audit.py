#!/usr/bin/env python3
"""STRATEGY 18.4 -- is the CORRECT target even a candidate the relinker can see?

`motion_relink_edges.learned_prob` returns **0.0** for any pair absent from
`learned_edge_probs`, and `cost = motion + 0.05*raw - BONUS*prob`. So for a swap
whose correct edge is not a candidate, the bonus term is identically zero and NO
value of MOTION_RELINK_LEARNED_BONUS can ever recover it.

This joins the dumped probabilities against the known-wrong cases from
link_ceiling.py and splits them into the two worlds of 18.4:

  A) correct edge IS a candidate  -> a bigger bonus can work, if prob is high enough
  B) correct edge is NOT a candidate -> the lever is dead, full stop

    python scripts/local/edge_prob_audit.py <edge_probs.csv> <submission.csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/local"))

from ctrack.data.geff import load_geff  # noqa: E402
from ctrack.eval.biohub import MAX_MATCH_DISTANCE_UM, SCALE, match_nodes  # noqa: E402
from link_ceiling import load_submission  # noqa: E402

GT_ROOT = Path("/Volumes/extSSD/kaggle/biohub/gt/train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("edge_probs")
    ap.add_argument("submission")
    args = ap.parse_args()

    probs: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    with open(args.edge_probs) as f:
        for r in csv.DictReader(f):
            probs[r["dataset"]][(int(r["source_id"]), int(r["target_id"]))] = float(r["prob"])
    print(f"candidate edges dumped: {sum(len(v) for v in probs.values())} "
          f"over {len(probs)} volumes")

    n = miss = 0
    p_correct, p_taken, both = [], [], 0
    for ds, pred in sorted(load_submission(Path(args.submission)).items()):
        pv = probs.get(ds, {})
        g = load_geff(GT_ROOT / f"{ds}.geff").graph
        m = match_nodes(g, pred, max_distance=MAX_MATCH_DISTANCE_UM, scale=SCALE)
        g2p = {gid: pid for pid, gid in m.items()}
        pe = {(int(s), int(t)) for s, t in pred.edges}
        po = defaultdict(set)
        for s, t in pe:
            po[int(s)].add(int(t))
        for gs, gt_ in g.edges:
            ps, pt = g2p.get(int(gs)), g2p.get(int(gt_))
            if ps is None or pt is None or (ps, pt) in pe or not po.get(ps):
                continue
            n += 1
            pc = pv.get((ps, pt))
            if pc is None:
                miss += 1
                continue
            taken = max((pv.get((ps, q), 0.0) for q in po[ps]), default=0.0)
            p_correct.append(pc)
            p_taken.append(taken)
            both += pc > taken

    print(f"\nswaps analysed: {n}")
    print(f"  correct edge NOT a candidate (prob is identically 0): {miss}/{n} = {miss/max(1,n):.0%}")
    print(f"  correct edge IS a candidate                         : {n-miss}/{n}")
    if p_correct:
        pc, pt_ = np.array(p_correct), np.array(p_taken)
        print(f"    prob(correct) median {np.median(pc):.4f}   prob(taken) median {np.median(pt_):.4f}")
        print(f"    transformer ranks CORRECT above TAKEN: {both}/{len(pc)} = {both/len(pc):.0%}")
        print(f"    max achievable bonus swing = prob(correct)-prob(taken), "
              f"median {np.median(pc-pt_):+.4f}")
    print("\n  => a swap is recoverable by ANY bonus only if the correct edge is a candidate")
    print(f"     AND the transformer prefers it. Upper bound on recoverable swaps: "
          f"{both}/{n} = {both/max(1,n):.0%}")


if __name__ == "__main__":
    main()
