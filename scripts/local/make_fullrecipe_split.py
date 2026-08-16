#!/usr/bin/env python3
"""PHASE 1 of the full-recipe retrain — build the split that makes local honest.

WHY THIS EXISTS (row 90)
------------------------
`stack_adjonly` measured +0.0036 on evalset-24 and scored **-0.003** on the LB:
sign inverted, total error ~0.0066. The cause is structural, not a bug --
evalset-24 is drawn from TRAIN volumes and the pilkwang detector trained on
EVERY train volume, so no uncontaminated local set for the public pipeline
exists anywhere. We can reject configs locally; we can never confirm one.

Training our own detector fixes that, but ONLY if the split is built to. The
whole point of this file is the quarantine:

    HOLDOUT   = the evalset-24 stems, EXCLUDED from train and selection alike.
                Never trained on, never early-stopped on, never staged to Colab.
    SELECTION = a stratified sample used ONLY for early stopping (--patience).
    TRAIN     = everything else, capped by whatever Colab disk allows (Phase 0).

Holding out evalset-24 SPECIFICALLY is the trick. It is the set every historical
local number in this project was measured on, so the new honest score is directly
comparable to the 0.9135 line -- except this time our model has never seen it.

⚠ The old probe split is NOT reusable: `w3_probe_splits48.json` has
`44b6_5f15d135`, `44b6_9be80b04` and others in `train`, and those ARE evalset-24
stems. Reusing it would silently reproduce exactly the contamination row 90 is
about.

⚠ HONEST, NOT PERFECT. Even with our own detector the public *ranker* checkpoint
and the hand-tuned post-processing knobs were fitted on train volumes. This
removes the detector's contamination, which is the dominant term; it does not
remove all of it. Say so when quoting any number that comes out of it.

    .venv/bin/python scripts/local/make_fullrecipe_split.py --train-cap 160
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = Path("/Volumes/extSSD/kaggle/biohub/raw/train")
GT_STATS = REPO / "docs/gt_stats.csv"
EVALSET = REPO / "configs/evalset.json"
OUT = REPO / "configs/fullrecipe_splits.json"
VOLLIST = REPO / "configs/fullrecipe_volumes.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=160,
                    help="max TRAIN volumes; set from the Phase-0 disk measurement")
    ap.add_argument("--selection", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    print("=" * 78)
    print("PHASE 1 — full-recipe split (row 90: build the honest instrument)")
    print("=" * 78)

    all_stems = sorted(p.stem for p in RAW.glob("*.zarr"))
    holdout = sorted(json.loads(EVALSET.read_text())["stems"])
    print(f"\n[1/5] inventory")
    print(f"      volumes on disk           : {len(all_stems)}")
    print(f"      HOLDOUT (evalset-24)      : {len(holdout)}")

    missing = [s for s in holdout if s not in all_stems]
    assert not missing, f"evalset stems missing from raw/: {missing}"

    pool = [s for s in all_stems if s not in set(holdout)]
    print(f"      pool after quarantine     : {len(pool)}")

    # ---- stratify the selection set -------------------------------------
    stats: dict[str, dict] = {}
    with open(GT_STATS) as f:
        for r in csv.DictReader(f):
            stats[r["dataset"]] = r
    dens = {s: float(stats[s]["density"]) for s in pool if s in stats}
    print(f"\n[2/5] stratification")
    print(f"      volumes with gt_stats     : {len(dens)} / {len(pool)}")

    # Two axes the project has repeatedly found to matter: embryo (the corpus is
    # 35/65 by embryo, and golden-8's 50/50 skew is why REP-VAL was built) and
    # annotation density (row 58: density is the strongest gap predictor at
    # -0.577). Balance the selection set on both so early stopping is not driven
    # by an unrepresentative slice.
    ranked = sorted(dens, key=lambda s: dens[s])
    n_str = 4
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for i, s in enumerate(ranked):
        strata[(s.split("_")[0], min(i * n_str // len(ranked), n_str - 1))].append(s)
    for k in sorted(strata):
        print(f"        {k[0]} density-q{k[1]}      : {len(strata[k]):3d}")

    import random
    rng = random.Random(args.seed)
    # ⚠ Sample PROPORTIONALLY to the pool's embryo mix, not round-robin. A plain
    # round-robin over (embryo, density) strata returns a 50/50 selection set on
    # a 36/64 corpus -- which is precisely the golden-8 skew that forced this
    # project to build REP-VAL (SESSION_LOG 07-25: "golden-8 is SKEWED ...
    # embryo-balanced (50/50 vs train 35/65). CV not representative").
    # Early stopping should fire on a slice that looks like the corpus.
    emb_pool: dict[str, list[str]] = defaultdict(list)
    for s in dens:
        emb_pool[s.split("_")[0]].append(s)
    sel: list[str] = []
    for emb in sorted(emb_pool):
        want = round(args.selection * len(emb_pool[emb]) / len(dens))
        # stratify by density WITHIN the embryo so the quota is spread, not clumped
        ranked_e = sorted(emb_pool[emb], key=lambda s: dens[s])
        if want:
            step = len(ranked_e) / want
            for i in range(want):
                lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
                sel.append(rng.choice(ranked_e[lo:hi]))
    sel = sorted(set(sel))
    while len(sel) < args.selection:            # rounding shortfall
        c = rng.choice([s for s in dens if s not in sel])
        sel.append(c)
    sel = sorted(sel[:args.selection])

    train = sorted(s for s in pool if s not in set(sel))
    capped = len(train) > args.train_cap
    if capped:
        # ⚠ Cap PROPORTIONALLY. Dropping from the largest embryo first (the
        # obvious implementation) drives TRAIN toward 50/50 on a 36/64 corpus --
        # the golden-8 skew again, and the third time this bug has appeared in
        # this file's history. TRAIN must look like HOLDOUT or the model is
        # trained on a different distribution than it is judged on.
        rng2 = random.Random(args.seed + 1)
        by_emb: dict[str, list[str]] = defaultdict(list)
        for s in train:
            by_emb[s.split("_")[0]].append(s)
        n_before = sum(len(v) for v in by_emb.values())
        keep: list[str] = []
        for emb in sorted(by_emb):
            quota = round(args.train_cap * len(by_emb[emb]) / n_before)
            # density-stratified thinning inside the embryo, so the cap does not
            # preferentially discard sparse or dense volumes
            ranked_e = sorted(by_emb[emb], key=lambda s: dens.get(s, 0.0))
            if quota >= len(ranked_e):
                keep.extend(ranked_e)
                continue
            step = len(ranked_e) / quota if quota else 0
            for i in range(quota):
                lo = int(i * step)
                hi = max(lo + 1, int((i + 1) * step))
                keep.append(rng2.choice(ranked_e[lo:hi]))
        train = sorted(set(keep))
        while len(train) > args.train_cap:
            train.remove(rng2.choice(train))

    print(f"\n[3/5] split")
    print(f"      TRAIN                     : {len(train)}"
          f"{f'  (capped from {len(pool) - len(sel)})' if capped else ''}")
    print(f"      SELECTION (early stopping): {len(sel)}")
    print(f"      HOLDOUT (honest, unseen)  : {len(holdout)}")
    print(f"      embryo mix TRAIN          : {dict(Counter(s.split('_')[0] for s in train))}")
    print(f"      embryo mix SELECTION      : {dict(Counter(s.split('_')[0] for s in sel))}")
    print(f"      embryo mix HOLDOUT        : {dict(Counter(s.split('_')[0] for s in holdout))}")

    # ---- the guard that makes this file worth having ---------------------
    print(f"\n[4/5] CONTAMINATION GUARD")
    hs, ts, ss = set(holdout), set(train), set(sel)
    checks = [
        ("HOLDOUT ∩ TRAIN", hs & ts),
        ("HOLDOUT ∩ SELECTION", hs & ss),
        ("TRAIN ∩ SELECTION", ts & ss),
    ]
    ok = True
    for name, bad in checks:
        print(f"      {name:24s} : {len(bad)}" + (f"  !! {sorted(bad)[:4]}" if bad else "  OK"))
        ok &= not bad
    assert ok, "SPLIT IS CONTAMINATED — refusing to write it"

    old = json.loads((REPO / "configs/w3_probe_splits48.json").read_text())[0]
    leak = set(old["train"]) & {f"{s}.zarr" for s in holdout}
    print(f"      (for contrast) the OLD w3 probe split put {len(leak)} evalset-24")
    print(f"       volumes in TRAIN — reusing it would have reproduced row 90's bug)")

    payload = [{
        "train": [f"{s}.zarr" for s in train],
        "test": [f"{s}.zarr" for s in sel],
        "_note": (f"full-recipe retrain (row 90). HOLDOUT = evalset-24 ({len(holdout)} "
                  f"volumes) is EXCLUDED from both lists and never staged to Colab, so "
                  f"scoring on it is honest. seed={args.seed}, train_cap={args.train_cap}."),
        "_holdout": [f"{s}.zarr" for s in holdout],
    }]
    OUT.write_text(json.dumps(payload, indent=1))
    VOLLIST.write_text("".join(f"{s}\n" for s in train + sel))
    print(f"\n[5/5] wrote")
    print(f"      {OUT.relative_to(REPO)}   (trainer split: train/test)")
    print(f"      {VOLLIST.relative_to(REPO)}  ({len(train) + len(sel)} volumes to stage — "
          f"HOLDOUT deliberately absent)")
    print("\nNEXT: Phase 0 measures Colab disk, which sets --train-cap. Re-run this "
          "with the measured cap before staging.")


if __name__ == "__main__":
    main()
