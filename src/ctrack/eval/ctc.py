"""Local CTC / CHOTA metric harness.

Phase 0's key deliverable: score predictions locally with the *same* metric as the
leaderboard, so every downstream decision is metric-directed rather than proxy-directed.

Expects CTC layout:
    <gt_dir>/mask*.tif   + man_track.txt
    <pred_dir>/mask*.tif + res_track.txt

Verified against traccuracy 0.4.3. See tests/test_ctc_eval.py for the round-trip check.

NOTE: the competition's exact scoring is not yet confirmed (PLAN.md section 0). Until a
submission with a known leaderboard score reproduces here, treat these as indicative only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# AOGM weights from the CTC evaluation methodology, cross-checked against traccuracy's
# reported AOGM. The 10:1 FN:FP asymmetry on nodes is the single most important fact in
# this competition -- it means the metric-optimal detection threshold sits well below the
# F1-optimal one. See PLAN.md section 1.
AOGM_WEIGHTS = {
    "fn_nodes": 10.0,  # missed cell
    "ns_nodes": 5.0,   # two cells merged into one
    "fp_nodes": 1.0,   # spurious cell
    "fn_edges": 1.5,   # edge that should exist but doesn't (EA)
    "fp_edges": 1.0,   # spurious edge (ED)
    "ws_edges": 1.0,   # edge with wrong semantics (EC)
}


def score_ctc(gt_dir: str | Path, pred_dir: str | Path, chota: bool = True) -> dict:
    """Score one sequence. Returns a flat dict: AOGM, TRA, DET, LNK, error counts, CHOTA."""
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CHOTAMetric, CTCMetrics

    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt = load_ctc_data(str(gt_dir), str(gt_dir / "man_track.txt"))
    pred = load_ctc_data(str(pred_dir), str(pred_dir / "res_track.txt"))

    metrics = [CTCMetrics()] + ([CHOTAMetric()] if chota else [])
    results, _matched = run_metrics(
        gt_data=gt, pred_data=pred, matcher=CTCMatcher(), metrics=metrics
    )

    out: dict = {}
    for r in results:
        out.update(r["results"])
    return out


def error_breakdown(results: dict) -> dict:
    """Weighted AOGM cost per error class -- shows where the score is actually being lost."""
    return {
        k: {"count": results.get(k, 0), "weight": w, "cost": (results.get(k, 0) or 0) * w}
        for k, w in AOGM_WEIGHTS.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a CTC prediction dir against ground truth.")
    ap.add_argument("--gt", required=True, help="ground-truth CTC dir (with man_track.txt)")
    ap.add_argument("--pred", required=True, help="prediction CTC dir (with res_track.txt)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    res = score_ctc(args.gt, args.pred)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    for key in ("TRA", "DET", "LNK", "CHOTA", "AOGM"):
        if key in res:
            print(f"{key:12s} {res[key]:.6f}" if isinstance(res[key], float) else f"{key:12s} {res[key]}")

    print("\n-- where the cost is --")
    breakdown = error_breakdown(res)
    total = sum(d["cost"] for d in breakdown.values())
    for k, d in sorted(breakdown.items(), key=lambda kv: -kv[1]["cost"]):
        share = d["cost"] / total * 100 if total else 0.0
        print(f"{k:12s} n={d['count']:<8} x{d['weight']:<5} = {d['cost']:<10.1f} ({share:4.1f}%)")


if __name__ == "__main__":
    main()
