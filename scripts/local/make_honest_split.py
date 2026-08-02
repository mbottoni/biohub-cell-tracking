#!/usr/bin/env python3
"""Build an honest train/validation split for training our own model.

The public checkpoints were trained with `dataset_splits_alltrain.json`:
`split_manifest.json` reports train=199, validation=40, **overlap=40**. Every
validation volume was also a training volume, so their `validation_score` is a
training-set score. It cannot detect overfitting, and "best epoch 381" was
selected on it. Two of those 40 volumes (44b6_0113de3b, 44b6_0b24845f) are the
visible test volumes as well.

Their selection metric compounds the problem: `score = acc * recall`, where acc
sits at 0.9998, so selection is effectively "maximise node recall" -- on a
pipeline whose detection recall is already 97.5%, whose bottleneck is edge_J,
and whose metric penalises over-detection through the node-ratio term.

So 400 epochs is not an evidence-backed target, it is an artefact of a leaky
metric. This script builds a split that can actually answer when to stop.

Design:
  * stratified by embryo (44b6 / 6bba) x annotation-density tercile, so the
    holdout matches the corpus rather than repeating the 9.3 golden-8 mistake
    where the holdout was 2x denser than train;
  * the 4 dev-test volumes are forced into the holdout, so our validation score
    is measured on exactly the volumes the visible LB scores;
  * never-trained-on, so early stopping means something.

Run: .venv/bin/python scripts/local/make_honest_split.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GT_STATS = REPO / "docs/gt_stats.csv"
OUT = REPO / "configs/honest_split.json"

# Byte-identical to the four visible test volumes (docs/data.md).
DEV_TEST = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
HOLDOUT_TARGET = 30
SEED = 0


def main() -> None:
    rows = list(csv.DictReader(GT_STATS.open()))
    vols = {
        r["dataset"]: {
            "embryo": r["embryo"],
            "density": float(r["density"]),
            "nodes": int(r["nodes"]),
        }
        for r in rows
    }
    print(f"corpus: {len(vols)} volumes")

    # Density terciles computed over the whole corpus.
    dens = sorted(v["density"] for v in vols.values())
    t1, t2 = dens[len(dens) // 3], dens[2 * len(dens) // 3]

    def bucket(name: str) -> tuple[str, int]:
        v = vols[name]
        tier = 0 if v["density"] <= t1 else (1 if v["density"] <= t2 else 2)
        return (v["embryo"], tier)

    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for name in sorted(vols):
        strata[bucket(name)].append(name)

    holdout: list[str] = [d for d in DEV_TEST if d in vols]
    forced = set(holdout)
    print(f"forced into holdout (dev-test): {holdout}")

    # Proportional allocation across strata, deterministic pick within each.
    rng_order = sorted(strata.items())
    remaining = HOLDOUT_TARGET - len(holdout)
    total = len(vols)
    for key, members in rng_order:
        pool = [m for m in members if m not in forced]
        want = round(len(members) / total * HOLDOUT_TARGET) - sum(
            1 for f in forced if bucket(f) == key
        )
        want = max(0, min(want, len(pool), remaining))
        step = max(1, len(pool) // want) if want else 1
        picked = pool[::step][:want]
        holdout += picked
        remaining -= len(picked)

    holdout = sorted(set(holdout))
    train = sorted(set(vols) - set(holdout))
    assert not (set(train) & set(holdout)), "split leaks"

    def summarise(names: list[str], label: str) -> None:
        emb = defaultdict(int)
        for n in names:
            emb[vols[n]["embryo"]] += 1
        d = sorted(vols[n]["density"] for n in names)
        med = d[len(d) // 2]
        print(f"  {label:<10} n={len(names):<4} embryos={dict(emb)}  median density={med:.4f}")

    print("\nsplit:")
    summarise(train, "train")
    summarise(holdout, "holdout")
    summarise(sorted(vols), "corpus")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([{"split": 0, "train": train, "test": holdout}], indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(train)} train / {len(holdout)} holdout, overlap 0)")
    print("contrast: the public split was 199 train / 40 validation with overlap 40.")


if __name__ == "__main__":
    main()
