#!/usr/bin/env python3
"""STRATEGY 27.2 -- run the BONUS value §22.3's own arithmetic prescribed.

§22.3 derived, correctly, that a swap flips only if `BONUS x prob > margin`. With
the candidate gate opened to 0.30, newly-admitted edges carry `prob <= 0.48`, so
at BONUS 6 the learned term contributes at most `6 x 0.48 = 2.9 um` against
§18.2's median swap margin of **6.93 um**. The arithmetic concludes:

    "Flipping them would need BONUS ~= 17."

§22.4 then declared association-by-reweighting exhausted -- on evidence from
`relinkw3` (BONUS 3), `relinkw6` (BONUS 6), `edgethr30` and `edgethr15`. **The
prescribed value was never run.** Concluding a lever is dead from settings well
below your own model's threshold is an untested hypothesis, not a null result.

Base is `edgethr30`, NOT `d4edge`, because §22.3's arithmetic is specifically
about the opened-gate configuration: the low-probability candidates that need the
large bonus only exist once the gate is at 0.30. So this is ONE change against
its own baseline:

    BIOHUB_MOTION_RELINK_LEARNED_BONUS   6.0 -> 17.0

and it is scored against `evalset_edgethr30` on the same 24 volumes.

⚠ THE RISK, STATED FIRST. §18.3's asymmetry is now ~40:1 correct edges to swaps
at this scale (379 swaps against ~15,000 correct links). A bonus large enough to
override geometry on the swaps can equally override it on links that are
currently right, and 2% corruption cancels a perfect fix of every swap. This is
a genuine two-sided bet, which is exactly why it needs measuring rather than
asserting -- in either direction.

PRE-REGISTERED READ, so it cannot be rationalised afterwards:

  * swaps DOWN materially (379 -> ~300 or below) and macro adj UP  => §22.4 is
    wrong, reweighting had headroom, and the lever scales further.
  * swaps down but FP up and adj flat/down => the 40:1 asymmetry dominates;
    §22.4's conclusion is right for the WRONG reason, and reweighting is dead
    because of corruption, not because of reach.
  * swaps ~flat => the margin model itself is wrong; revisit §22.3.

Run: python scripts/build_bonus17_kernel.py && \
     python scripts/build_evalset_kernel.py bonus17 edgethr30
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/edgethr30/edgethr30.ipynb"

OLD = "os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '6.0'"
NEW = ("# STRATEGY 27.2: 22.3's own arithmetic says a swap needs BONUS x prob >\n"
       "# margin, and with prob <= 0.48 at this gate against a 6.93 um median\n"
       "# margin that means BONUS ~= 17. 22.4 declared the lever dead having only\n"
       "# tested 3 and 6. This runs the prescribed value.\n"
       "os.environ[\"BIOHUB_MOTION_RELINK_LEARNED_BONUS\"] = '17.0'")


def main() -> None:
    nb = json.loads(BASE.read_text())
    cell = "".join(nb["cells"][4]["source"])
    if cell.count(OLD) != 1:
        raise RuntimeError(f"anchor count = {cell.count(OLD)}, expected 1")
    nb["cells"][4]["source"] = cell.replace(OLD, NEW, 1).splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    out = REPO / "kernels/bonus17"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bonus17.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((REPO / "kernels/edgethr30/kernel-metadata.json").read_text())
    meta.update({"id": "maruanbottoni/biohub-bonus17", "title": "Biohub bonus17",
                 "code_file": "bonus17.ipynb"})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote kernels/bonus17  (BONUS 6.0 -> 17.0, one change vs edgethr30)")


if __name__ == "__main__":
    main()
