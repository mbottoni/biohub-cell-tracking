#!/usr/bin/env python3
"""Build the stacked pub162 winner: disappearance 2.0 + UNCLAMPED gap-close 3.

Measured on the row-60 local replay against the official scorer, 24-volume
evalset, identical machinery throughout (rows 62-65):

    pub162 base                     0.9099
    + ILP disappearance 1.5 -> 2.0  0.9119   (+0.0020, saturating curve, 2.0 = plateau top)
    + gap clamp lifted, max_gap 3   0.9135   (+0.0036 cumulative, unimodal peak at 3)

The gap edit is a CODE patch, not a config change. Cell 15 ships

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)

which pins gap closing to one-frame holes and makes BIOHUB_GAP_CLOSE_MAX_GAP inert
for every value >= 1 -- the public line's `max_gap = 2`, credited by row 38 as a
carrier of its +0.002, has never executed. Lifting the clamp buys detection-FN
recovery (node recall 0.9829 -> 0.9878) and cuts division FPs (37 -> 18) exactly as
13/22-23 predict. Beyond 3 the node inflation (+1.7%, +2.3%) and decaying adjacent
edge Jaccard overtake the recall gain.

Cell 10's freeze guard would reject the disappearance change as drift, so its
expected value is updated to 2.0 -- the guard still catches every OTHER change.
GAP_CLOSE_MAX_GAP is not in the guard's tables and needs no rewrite.

PRE-REGISTERED READ: pub162 scored 0.915 on the LB at local 0.9099. If the local
delta translates at the ~half strength of row 38 (+0.0045 local -> +0.002 LB),
expect ~0.917; at full strength ~0.9186. Flat-or-worse kills the stack on hidden
test and the two 0.915s remain the final-2.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/pub162_port/pub162_port.ipynb"
BASE_META = REPO / "kernels/pub162_port/kernel-metadata.json"
NAME = "pub162_stack"

EDITS = [
    # (cell-matching substring, old, new, label)
    ('os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"]',
     'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"',
     'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2.0"',
     "env: ILP disappearance -> 2.0"),
    ('os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"]',
     'os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"',
     'os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "3"',
     "env: gap max_gap -> 3"),
    ('"ILP_DISAPPEARANCE_WEIGHT":',
     '"ILP_DISAPPEARANCE_WEIGHT": 1.5,',
     '"ILP_DISAPPEARANCE_WEIGHT": 2.0,',
     "freeze guard: expected disappearance -> 2.0"),
    ('effective_gap_max = min(',
     '    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)',
     '    effective_gap_max = max(1, int(GAP_CLOSE_MAX_GAP))',
     "code: lift the 1-frame gap clamp"),
]


def main() -> None:
    nb = json.loads(BASE.read_text())
    for _probe, old, new, label in EDITS:
        hits = 0
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if old in src:
                hits += src.count(old)
                cell["source"] = src.replace(old, new).splitlines(keepends=True)
        assert hits == 1, f"{label}: expected exactly 1 match, found {hits}"
        print(f"  [ok] {label}")

    out_dir = REPO / "kernels" / NAME
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1))

    meta = json.loads(BASE_META.read_text())
    meta["id"] = "maruanbottoni/biohub-pub162-stack"
    meta["title"] = "biohub-pub162-stack"
    meta["code_file"] = f"{NAME}.ipynb"
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Post-conditions: the built notebook must carry all four changes and none
    # of the originals. A patch that silently fails looks exactly like success.
    reloaded = json.loads((out_dir / f"{NAME}.ipynb").read_text())
    built = "\n".join("".join(c["source"]) for c in reloaded["cells"])
    for _probe, old, new, label in EDITS:
        assert new.strip() in built, f"POST-CHECK FAILED, new text absent: {label}"
        assert old.strip() not in built, f"POST-CHECK FAILED, old text remains: {label}"
    print(f"all 4 edits verified in {out_dir}")


if __name__ == "__main__":
    main()
