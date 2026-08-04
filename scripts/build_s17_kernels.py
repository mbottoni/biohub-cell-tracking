#!/usr/bin/env python3
"""STRATEGY 17.6 items 1 and 2. Both are pure env-var changes off d4edge (LB 0.913/0.914).

These are COMMIT RUNS, not submissions. `scripts/local/score_submission.py`
scores the resulting `submission.csv` against real GT before anything is spent
on the board (17.1).

  relinkw3 / relinkw6 -- MOTION_RELINK_LEARNED_BONUS 1.0 -> 3.0 / 6.0
      17.4: on 6bba_05db0fb1, node recall is 0.991 but edge_J is 0.836 -- 133 FP
      and 83 FN edges built on CORRECT nodes. That volume dominates micro edge_J
      and carries 25% of the macro score, so the loss there is association, not
      detection. 17.5: motion_relink discards 100% of the transformer's edges and
      re-derives them from a Hungarian with
          cost = motion_dist + 0.05*raw_dist - BONUS * prob
      with distances in um and competitors inside RELAXED_UM=10 differing by
      several um, while prob is in [0,1]. At BONUS=1.0 the learned association is
      worth <=1 um of tiebreak -- which is also why edgeft (exp 21) was a no-op.
      Raising it lets the trained model actually reach the objective.

      Two points, not one: the effect need not be monotone. A bonus large enough
      to override motion entirely would re-introduce the transformer's own FPs.

  divveto -- DEEPCENTER_SAFE_DIV_VETO 0 -> 1, on the ORIGINAL gates
      17.2 measured the division term for the first time: d4edge scores
      TP=1, FP=5, FN=2 -> div_J = 0.125, worth +0.0125. Hold TP and drive FP to 0
      and div_J = 1/3 -> +0.033.
      16.4: after divgeom, the lever is precision on the EXISTING candidate pool
      via an INDEPENDENT signal -- the DeepCenter model -- not more of the same
      geometry. This leaves gates, pool size and node retention untouched, so it
      stays a clean single lever. The veto is already wired and defaults ON in
      the pipeline; our config explicitly disables it.

Run: python scripts/build_s17_kernels.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"

DIAG = ('os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"',
        'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"')


def slug(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def patch(cfg: str, pairs) -> str:
    for old, new in pairs:
        if cfg.count(old) != 1:
            raise RuntimeError(f"anchor count {cfg.count(old)} != 1 for {old!r}")
        cfg = cfg.replace(old, new, 1)
    return cfg


def build(name: str, title: str, pairs) -> None:
    nb = json.loads(BASE_NB.read_text())
    nb["cells"][4]["source"] = patch(
        "".join(nb["cells"][4]["source"]), [*pairs, DIAG]).splitlines(keepends=True)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))  # must still parse
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
    print(f"wrote kernels/{name}  ->  maruanbottoni/{slug(title)}")


BONUS = "os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '1.0'"
VETO = "os.environ[\"BIOHUB_DEEPCENTER_SAFE_DIV_VETO\"] = '0'"

if __name__ == "__main__":
    for v in ("3.0", "6.0"):
        build(f"relinkw{v[0]}", f"Biohub relink learned bonus {v}",
              [(BONUS, f"# STRATEGY 17.5: the learned edge prob is worth <=1 um of tiebreak\n"
                       f"# against a um-scale motion cost. 17.4: the dominant volume is\n"
                       f"# linking-limited (recall 0.991, edge_J 0.836). Let it matter.\n"
                       f"os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '{v}'")])
    build("divveto", "Biohub division deepcenter veto",
          [(VETO, "# STRATEGY 17.2: d4edge scores division TP=1, FP=5, FN=2 (div_J 0.125).\n"
                  "# Hold TP, drive FP to 0 -> div_J 0.33 -> +0.033. Precision on the\n"
                  "# EXISTING pool via an INDEPENDENT signal, not more geometry (16.4).\n"
                  "os.environ[\"BIOHUB_DEEPCENTER_SAFE_DIV_VETO\"] = '1'")])
