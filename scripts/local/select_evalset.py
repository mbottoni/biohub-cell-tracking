#!/usr/bin/env python3
"""STRATEGY 23.2 -- choose the widened local eval set.

The four `test/` volumes are 1.65% of the ground truth we own (2,127 of 128,883
GT edges; 3 of 151 divisions), and they are not even a representative 1.65%:

    median est_n_nodes        dev 32,795   vs   corpus 17,890   (1.8x denser)
    median annotation cover   dev  1.76%   vs   corpus  3.56%   (2x thinner)

Denser in cells and thinner in annotation is the worst combination a measurement
fixture can have, and every thin-volume pathology of the last three sessions
(19.2's "80% one edge", 20.3's 3-division statistic, 22.1's 2x robustness bug)
traces back to it.

The shipped kernel already reads `BIOHUB_TEST_DIR` from the environment and
globs `*.zarr` out of it (`list_test_stems`), so pointing a commit run at a
subset of `train/` costs one run and zero submissions.

This script picks that subset. Two requirements pull against each other:

* **Representative** -- the estimate has to transfer to a hidden set drawn like
  the corpus, so we stratify on `est_n_nodes` (which drives the node-budget
  multiplier) crossed with embryo (the only two sources that exist).
* **Cross-checkable** -- the 4 dev volumes are force-included so the widened run
  reproduces the numbers we already hold for them (d4edge local 0.9216). If they
  do not reproduce exactly, the instrument is wrong and nothing else here counts.

Deliberately NOT enriched for divisions. A representative draw already yields
~0.76 divisions/volume, and enrichment would bias `division_jaccard` -- which is
MICRO-pooled (23.3), so it is a single ratio over the whole set rather than a
per-volume mean. Division *model fitting* uses all 199 GT graphs offline and does
not need inference, so it has no claim on this budget.

Run: python scripts/local/select_evalset.py [--n 24]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GT_STATS = REPO / "docs/gt_stats.csv"
OUT = REPO / "configs/evalset.json"

# Force-included: these are the four `test/` volumes, byte-identical to train
# (docs/data.md). Keeping them lets the widened run reproduce known numbers.
DEV = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

SEED = 20260804
N_STRATA = 4


def load() -> list[dict]:
    rows = list(csv.DictReader(GT_STATS.open()))
    for r in rows:
        r["est_n_nodes"] = float(r["est_n_nodes"])
        r["edges"] = int(r["edges"])
        r["divisions"] = int(r["divisions"])
        r["density"] = float(r["density"])
    return rows


def quantiles(vals: list[float], k: int) -> list[float]:
    s = sorted(vals)
    return [s[int(len(s) * i / k)] for i in range(1, k)]


def stratum(value: float, cuts: list[float]) -> int:
    return sum(value >= c for c in cuts)


def describe(name: str, rows: list[dict]) -> None:
    n = len(rows)
    e = sum(r["edges"] for r in rows)
    d = sum(r["divisions"] for r in rows)
    est = sorted(r["est_n_nodes"] for r in rows)
    den = sorted(r["density"] for r in rows)
    emb = defaultdict(int)
    for r in rows:
        emb[r["embryo"]] += 1
    print(f"  {name:<10} n={n:<4} GT edges={e:<7,} divisions={d:<4} "
          f"med est_n={est[n // 2]:>8,.0f}  med cover={den[n // 2]:.4f}  "
          f"embryo={dict(emb)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="target volume count")
    args = ap.parse_args()

    rows = load()
    by_id = {r["dataset"]: r for r in rows}
    cuts = quantiles([r["est_n_nodes"] for r in rows], N_STRATA)

    # Bucket the corpus on (embryo, est_n_nodes stratum) so the draw preserves
    # both the 128/71 embryo split and the node-count distribution.
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["embryo"], stratum(r["est_n_nodes"], cuts))].append(r)

    chosen = {d for d in DEV}
    # Allocate ALL n slots proportionally to bucket size, largest remainder
    # first so rounding never silently drops a whole stratum. The forced dev
    # volumes count against their own bucket's quota (subtracted at draw time)
    # rather than being added on top -- otherwise they are counted twice and
    # the set comes up short by len(DEV).
    quota = {k: len(v) / len(rows) * args.n for k, v in buckets.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: -(quota[k] - take[k]))[: args.n - sum(take.values())]:
        take[k] += 1

    rng = random.Random(SEED)
    for k in sorted(buckets):
        pool = sorted(b["dataset"] for b in buckets[k] if b["dataset"] not in chosen)
        rng.shuffle(pool)
        # Dev volumes already sitting in this bucket count against its quota.
        already = sum(1 for d in DEV if (by_id[d]["embryo"], stratum(by_id[d]["est_n_nodes"], cuts)) == k)
        for s in pool[: max(0, take[k] - already)]:
            chosen.add(s)

    sel = sorted(chosen)
    sel_rows = [by_id[s] for s in sel]

    print(f"est_n_nodes stratum cuts: {[f'{c:,.0f}' for c in cuts]}")
    print("\ndistribution check (the point is that EVALSET tracks CORPUS, not DEV):")
    describe("CORPUS", rows)
    describe("DEV", [by_id[d] for d in DEV])
    describe("EVALSET", sel_rows)

    corpus_e = sum(r["edges"] for r in rows)
    dev_e = sum(by_id[d]["edges"] for d in DEV)
    sel_e = sum(r["edges"] for r in sel_rows)
    print(f"\n  GT edges      dev {dev_e:>6,} ({dev_e/corpus_e:.2%})  ->  "
          f"evalset {sel_e:>6,} ({sel_e/corpus_e:.2%})   = {sel_e/dev_e:.1f}x")
    dev_d = sum(by_id[d]["divisions"] for d in DEV)
    sel_d = sum(r["divisions"] for r in sel_rows)
    print(f"  divisions     dev {dev_d:>6}          ->  evalset {sel_d:>6}"
          f"          = {sel_d/max(dev_d,1):.1f}x")
    thin = [r for r in sel_rows if r["edges"] < 200]
    print(f"  thin volumes (<200 GT edges): {len(thin)}/{len(sel_rows)} "
          f"(dev had 2/4 -- half the instrument was unusable)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seed": SEED,
        "n_strata": N_STRATA,
        "forced": DEV,
        "stems": sel,
        "gt_edges": sel_e,
        "gt_divisions": sel_d,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(sel)} volumes)")


if __name__ == "__main__":
    main()
