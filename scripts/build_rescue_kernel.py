#!/usr/bin/env python3
"""pub162_stack_adjonly + the pipeline's OWN short-track rescue, switched on.

WHY THIS EXISTS (row 88)
------------------------
Row 87 closed the detection-side routes: raw detection recall on annotated cells
is already 0.9958 and the whole DET_THRESHOLD ladder hides only 9 more cells. But
it left a residue -- final-submission recall is LOWER than raw detection recall,
so post-processing deletes cells detection already found correctly.

Measured on the matched 21-volume set against the sha-87d51dda stack:

    node recall  raw 0.9958 -> final 0.9907     112 annotated cells DELETED
    GT edges 15,705 | both endpoints detected 15,600 | currently matched 14,934
    RECOVERABLE GT edges (endpoints detected, edge absent): 666
        by RETENTION (an endpoint was deleted): 139  (20.9%)
        by LINKING   (both kept, never linked): 527  (79.1%)

and from the pipeline's own instrumentation over 24 volumes:

    short_track_components_removed    1,516
    short_track_nodes_removed         6,410
    short_track_edges_removed         4,894      (1.02% of final edges)
    short_track_rescue_triggered          0      <-- NEVER FIRES

THE LEVER
---------
The public pipeline already implements a SELECTIVE rescue for exactly this: it
re-admits removed components of length in [RESCUE_MIN_LEN, OUTPUT_MIN_TRACK_LEN)
that pass quality gates (mean edge prob >= 0.82, mean edge distance <= 3.25 um),
ranked by `mean_prob - 0.02*mean_dist + 0.004*len` and filled to a hard node
budget of min(180, 1.8% of nodes). It is GT-blind and budget-capped by
construction. It ships DISABLED (`BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE = "0"`).

⚠ Enabling it alone is a NO-OP. The rescue only runs when the removed fraction
reaches `SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC` (default 0.10). We remove
6,410/495,653 = **1.29%**, far below that gate, so the block is never entered.
Both knobs must move together. This is precisely the row-82 failure mode -- a
knob whose output is byte-identical to base because a guard upstream stopped it
from ever taking effect -- so the local run asserts `short_track_rescue_triggered
== 1` before any number from it is believed.

Why this is not row 75. Row 75 swept OUTPUT_MIN_TRACK_LEN (5 and 8, both lose,
6 optimal) and correctly called it a genuine null: that is the BLANKET knob, and
it trades every short track at once. The rescue is the SELECTIVE version of the
same decision -- it keeps the short tracks that look real and still deletes the
rest -- which is a different lever on the same quantity.

    .venv/bin/python scripts/build_rescue_kernel.py [--tag <name>] [--loose]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels/pub162_stack_adjonly/pub162_stack_adjonly.ipynb"
CELL = 4

# (old, new) substitutions applied to cell 4. Each must match exactly once.
def subs(loose: bool) -> list[tuple[str, str]]:
    out = [(
        'os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "0"',
        'os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"\n'
        '# row 88: the trigger defaults to 0.10 but we only remove ~1.29% of nodes,\n'
        '# so the rescue block is never entered. Lower it or enabling above is inert.\n'
        'os.environ["BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC"] = "0.001"',
    )]
    if loose:
        # ⚠ These four values ARE the measured `r11_aggr` config (0.9138). All
        # four must ship together: the shipped gates rescue 1,691 nodes (0.9136)
        # and only the widened set reaches 4,935 (0.9138). Emitting a subset
        # would build an artifact that is NOT the one measured -- the row-66 trap.
        out[0] = (out[0][0], out[0][1] +
                  '\nos.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "2"'
                  '\nos.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.70"'
                  '\nos.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "4.50"'
                  '\nos.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "600"')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pub162_rescue")
    ap.add_argument("--loose", action="store_true",
                    help="widen the rescue quality gates (prob 0.82->0.70, dist 3.25->4.50)")
    args = ap.parse_args()

    out_dir = REPO / "kernels" / args.tag
    nb = json.loads(SRC.read_text())
    src = "".join(nb["cells"][CELL]["source"])
    assert "ADAPTIVE_SHORT_TRACK_RESCUE" in src, f"cell {CELL} is not the config cell"

    for old, new in subs(args.loose):
        n = src.count(old)
        assert n == 1, f"expected 1 match for {old!r}, found {n}"
        src = src.replace(old, new)
        print(f"[build] patched: {old.split('[')[1].split(']')[0]} (1 match)")

    nb["cells"][CELL]["source"] = src.splitlines(keepends=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.ipynb").write_text(json.dumps(nb, indent=1))

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    slug = args.tag.replace("_", "-")
    meta["id"] = f"maruanbottoni/biohub-{slug}"
    meta["title"] = f"biohub {args.tag.replace('_', ' ')}"
    meta["code_file"] = f"{args.tag}.ipynb"
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[build] wrote {out_dir}/  id={meta['id']}")


if __name__ == "__main__":
    main()
