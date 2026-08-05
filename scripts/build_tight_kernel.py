#!/usr/bin/env python3
"""STRATEGY 31.6 -- raise MOTION_RELINK_TIGHT_UM so the correct successor can compete.

Reading `motion_relink_edges` settles which knob matters, and it is NOT the one
§31.4 proposed. The relinker is a TWO-STAGE GREEDY assignment:

    for pass_name, gate_um in (("tight", TIGHT_UM), ("relaxed", RELAXED_UM)):
        pass_sources = [s for s in source_ids if s in unmatched_sources]
        pass_targets = [t for t in target_ids if t in unmatched_targets]
        matches = assign_pass(pass_sources, pass_targets, gate_um)

`assign_pass` builds a cost matrix where any pair with `raw > gate_um` is left at
`big` and therefore excluded, then runs `linear_sum_assignment`.

So in the TIGHT pass, a source whose correct successor is 9.37 um away (§31.2's
median) sees only the wrong node 1.86 um away, is matched to it, and is removed
from `unmatched_sources` before the relaxed pass ever runs. **The tight pass
pre-commits.** Widening RELAXED cannot help a source that never reaches it --
which is why §31.4's proposed 10 -> 14 sweep would have measured nothing, and why
it is abandoned here.

The lever is TIGHT_UM. At 6.0 it excludes **90.5%** of correct successors from the
only assignment that matters. Raising it puts the far-but-correct target and the
near-but-wrong target into the SAME global assignment, where the Hungarian can
weigh them jointly -- and, importantly, where the wrong target's own true parent
(usually an unannotated cell, so invisible to our diagnostics) can claim it and
force our source onto its real successor. That mechanism does not exist today,
because the competition never happens.

Two arms, each ONE change against `d4edge`, to get a curve rather than a point:

    tight9   MOTION_RELINK_TIGHT_UM 6.0 -> 9.0    (covers the p50 of 9.37)
    tight12  MOTION_RELINK_TIGHT_UM 6.0 -> 12.0   (covers the p75 of 11.47)

PRE-REGISTERED READ, fixed before the runs:

  * swaps DOWN and adj UP  => the pre-commitment was the mechanism; pursue a
    single-stage assignment over the full radius as the real fix.
  * swaps down but FP up / adj flat => widening trades one error for another;
    the assignment needs a better cost, not a bigger gate.
  * nothing moves => the cost is so distance-dominated that even in a joint
    assignment the near-wrong target wins, and association is closed to
    everything except a new model (§22.4/§28's conclusion, extended).

⚠ RISK: at 12 um in a volume with ~11.7 um spacing every cell has several
admissible targets, so the tight Hungarian gets much denser. §29 found the
oracle gap correlates with density and wider gates cost most exactly there;
under a macro mean, hurting the dense majority to help the fast minority can net
negative. This is a genuine two-sided bet.

Run: python scripts/build_tight_kernel.py && \
     python scripts/build_evalset_kernel.py tight9 tight12
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/d4edge/d4edge.ipynb"
ANCHOR = 'os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"'


def build(name: str, title: str, tight: str) -> None:
    nb = json.loads(BASE.read_text())
    cell = "".join(nb["cells"][4]["source"])
    if cell.count(ANCHOR) != 1:
        raise RuntimeError(f"anchor count = {cell.count(ANCHOR)}, expected 1")
    if "BIOHUB_MOTION_RELINK_TIGHT_UM" in cell:
        raise RuntimeError("cell 4 already sets BIOHUB_MOTION_RELINK_TIGHT_UM")
    patch = (
        "# STRATEGY 31.6: the relinker is a two-stage greedy assignment and the\n"
        "# TIGHT pass pre-commits -- a source whose correct successor is 9.37 um\n"
        "# away (31.2 median) is married to a wrong node 1.86 um away and never\n"
        "# reaches the relaxed pass. At 6.0 the gate excludes 90.5% of correct\n"
        "# successors from the only assignment that matters. Widen it so the far\n"
        "# correct target and the near wrong one compete in the SAME Hungarian.\n"
        f'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "{tight}"\n'
    )
    nb["cells"][4]["source"] = cell.replace(ANCHOR, patch + ANCHOR, 1).splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((REPO / "kernels/d4edge/kernel-metadata.json").read_text())
    meta.update({"id": f"maruanbottoni/biohub-{name}", "title": title,
                 "code_file": f"{name}.ipynb"})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote kernels/{name}  (MOTION_RELINK_TIGHT_UM 6.0 -> {tight}, one change vs d4edge)")


if __name__ == "__main__":
    build("tight9", "Biohub tight9", "9.0")
    build("tight12", "Biohub tight12", "12.0")
