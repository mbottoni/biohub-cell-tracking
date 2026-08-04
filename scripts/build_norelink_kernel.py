#!/usr/bin/env python3
"""STRATEGY 24.6 -- let the ILP's division-aware solution stand.

§24.6 measured, over all 116,134 `d4edge` edges, that **no node has in-degree
>= 2**: the linked graph is a strict bipartite 1-1 assignment. And the
out-degree-2 counts (39/66/10/208 = 323) equal `safe_divisions_added` exactly,
so *every* division we emit comes from the post-hoc orphan scavenge and none
from the ILP.

That is not because the ILP cannot express divisions. Cell 3's objective is

    min  -sum p_ij x_ij  +  0 sum a_i  +  1.5 sum b_i  +  1.0 sum v_i

with explicit division variables `v_i` priced at 1.0. The solution is simply
thrown away: `motion_relink_replaced_raw_edges` equals `raw_edges` on every
volume (§17.5), and the relinker re-derives everything from a Hungarian, which
by construction cannot give a node two children.

So the pipeline solves a division-aware global problem and then discards it for
a greedy per-frame 1-1 matching, after which a heuristic scavenges ~323
divisions back out of whatever the matching left orphaned. §24.3 showed the
cost: a real daughter that the assignment consumed as some other cell's child is
permanently inadmissible as a division candidate.

This is the cheapest possible test of that thesis -- ONE env var:

    BIOHUB_OUTPUT_MOTION_RELINK   1 -> 0

Everything else is byte-identical to `d4edge`, and it is scored on the 24-volume
eval set (§23.2) so `div_J` rests on 25 GT divisions rather than 3.

WHAT TO EXPECT, stated before the run so it cannot be rationalised after:

* The public author ships relink ON, and §17.5 shows it rewrites 100% of edges,
  so it is plausible that `edge_J` DROPS substantially -- the ILP output has
  never been scored on its own here.
* The read that matters is the DECOMPOSITION, not the total: if `div_J` rises
  materially while `edge_J` falls, the thesis holds and the fix is a joint
  formulation (division-aware assignment), not this switch. If `div_J` stays
  flat, the ILP is not finding divisions either and §24.6's lever is smaller
  than argued.

Run: python scripts/build_norelink_kernel.py && \
     python scripts/build_evalset_kernel.py norelink
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/d4edge/d4edge.ipynb"

ANCHOR = 'os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"'
PATCH = '''# STRATEGY 24.6: the relinker replaces 100% of the ILP's edges with a strict
# 1-1 Hungarian assignment (17.5), which structurally cannot emit a division --
# measured: no node in the output has in-degree >= 2, and all 323 divisions come
# from the post-hoc orphan scavenge, none from the ILP whose objective prices
# division variables v_i at 1.0. Turn the relinker off and let that solution
# stand. Read the DECOMPOSITION (div_J vs edge_J), not the total.
os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "0"
'''


def main() -> None:
    nb = json.loads(BASE.read_text())
    cell = "".join(nb["cells"][4]["source"])
    if cell.count(ANCHOR) != 1:
        raise RuntimeError(f"anchor count = {cell.count(ANCHOR)}, expected 1")
    if "BIOHUB_OUTPUT_MOTION_RELINK" in cell:
        raise RuntimeError("cell 4 already sets BIOHUB_OUTPUT_MOTION_RELINK")
    cell = cell.replace(ANCHOR, PATCH + ANCHOR, 1)
    nb["cells"][4]["source"] = cell.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e

    out = REPO / "kernels/norelink"
    out.mkdir(parents=True, exist_ok=True)
    (out / "norelink.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((REPO / "kernels/d4edge/kernel-metadata.json").read_text())
    meta.update({"id": "maruanbottoni/biohub-norelink", "title": "Biohub norelink",
                 "code_file": "norelink.ipynb"})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote kernels/norelink  (BIOHUB_OUTPUT_MOTION_RELINK=0, one change vs d4edge)")


if __name__ == "__main__":
    main()
