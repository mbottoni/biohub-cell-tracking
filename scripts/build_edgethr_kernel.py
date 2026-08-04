#!/usr/bin/env python3
"""STRATEGY 21.4 -- open the candidate gate, since the association model is already right.

`probdump` (21.1/21.2) measured, over the 65 known association failures:

    correct edge is NOT a candidate (prob identically 0)   47/65 = 72%
    correct edge IS a candidate                            18/65
    ...transformer ranks it ABOVE the taken edge           18/18 = 100%
    median prob(correct) vs prob(taken)                    0.838 vs 0.000

So the shipped association model is not the weak component -- it is right every
time it is allowed to speak. The gate in front of it is what loses the score, and
21.3 identified that gate by elimination as the probability threshold: the
missing edges are NOT beyond any distance cap (0% exceed the longest observed
candidate) and NOT outside a neighbourhood (the correct target is the median
2nd-nearest node at t+1), they simply scored under 0.48.

    BIOHUB_DUAL_SEED_EDGE_THRESHOLD   0.48 -> 0.15 / 0.30
    BIOHUB_MOTION_RELINK_LEARNED_BONUS 1.0 -> 6.0

The two knobs are COUPLED and ship together: admitting a candidate does nothing
unless the bonus can lift it over the motion margin (18.3: median 6.93 um), and
the bonus does nothing for a candidate that is absent (21.1). Two thresholds
because the effect is a trade -- a lower gate also admits wrong edges, and where
that turns over is exactly what we cannot predict.

RISK, stated plainly: this is the same shape as divgeom (16) -- flood a candidate
pool, trust a ranking. What is different this time is that the diagnosis is
measured rather than inferred (72%), the ranking is validated at 18/18 rather
than assumed, and score_submission.py grades it locally before anything is spent.
If the ROBUST delta does not clear ~0.005 it does not go to the board.

Ceiling (18): macro association headroom +0.077, of which the two thick volumes
contribute ~+0.044 robust. Realistic target +0.005 .. +0.015 robust.

Run: python scripts/build_edgethr_kernel.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"

# (cell index, old, new) -- the threshold lives in cell 7, the bonus in cell 4.
BONUS_OLD = "os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '1.0'"
BONUS_NEW = ("# STRATEGY 21.4: a newly-admitted candidate is useless unless the bonus\n"
             "# can lift it over the motion margin (18.3: median 6.93 um). Coupled knob.\n"
             "os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '6.0'")
THR_OLD = 'os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"'
DIAG_OLD = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
DIAG_NEW = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"'


def slug(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def sub(cell: str, old: str, new: str, label: str) -> str:
    if cell.count(old) != 1:
        raise RuntimeError(f"{label}: anchor count = {cell.count(old)}, expected 1")
    return cell.replace(old, new, 1)


def build(name: str, title: str, thr: str) -> None:
    nb = json.loads(BASE_NB.read_text())

    c4 = "".join(nb["cells"][4]["source"])
    c4 = sub(c4, BONUS_OLD, BONUS_NEW, "bonus")
    c4 = sub(c4, DIAG_OLD, DIAG_NEW, "diagnostics")
    nb["cells"][4]["source"] = c4.splitlines(keepends=True)

    c7 = "".join(nb["cells"][7]["source"])
    c7 = sub(c7, THR_OLD,
             "# STRATEGY 21.3: 72% of correct successors scored under 0.48 and so were\n"
             "# never candidates -- learned_prob returns 0.0 for them and no bonus can\n"
             "# reach them. Not a distance cap and not kNN; this is the gate.\n"
             f'os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "{thr}"', "threshold")
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"maruanbottoni/{slug(title)}", "title": title,
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
    }, indent=2))
    print(f"wrote kernels/{name}  ->  maruanbottoni/{slug(title)}   (thr={thr}, bonus=6.0)")


if __name__ == "__main__":
    build("edgethr15", "Biohub edge candidate threshold 0 15", "0.15")
    build("edgethr30", "Biohub edge candidate threshold 0 30", "0.30")
