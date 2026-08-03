#!/usr/bin/env python3
"""Re-center the safe-division heuristic on the MEASURED geometry of real divisions.

STRATEGY 15, from `scripts/local/division_geometry.py` over all 199 GT graphs
(151 annotated divisions, 128,581 ordinary steps):

    division sister <-> sister   median 10.57 um   (p90 14.36)
    division parent -> daughter  median  5.75 um   (near 4.08 / far 7.13)
    ordinary motion              median  1.82 um

Two independent defects in the shipped config:

1. THE GATES SIT BELOW THE TRUTH. `SAFE_DIV_SISTER_MAX_UM = 8.5` is under the
   median real sister separation of 10.57. Exact three-gate admission of a real
   division is **0.265**; at (12 / 12 / 14.4) it is **0.894**. We are
   structurally blind to ~3/4 of all divisions.

2. THE RANKING IS INVERTED. Proposals sort ascending on
   `parent_dist + 0.15*sister_dist`, so the smallest distances win -- but real
   divisions are the pairs that moved APART (median 5.75 um vs 1.82 um for
   ordinary motion). The sort systematically prefers stationary, non-dividing
   cells, and real divisions are the first dropped when `frame_cap` binds.

The fix is a single coherent intervention -- widen so real divisions are
*admissible*, then re-rank so they win the truncation, then cut the count so we
insert FEWER but better:

    SAFE_DIV_MAX_UM                4.66    -> 12.0     admission 0.265 -> 0.894
    SAFE_DIV_SISTER_MAX_UM          8.5    -> 14.4
    SAFE_DIV_EXISTING_CHILD_MAX_UM  7.65   -> 12.0
    SAFE_DIV_GLOBAL_FRAC_CAP        0.00375-> 0.0010   ~350 inserts -> ~120
    SAFE_DIV_FRAME_FRAC_CAP         0.0076 -> 0.0030
    score  parent + 0.15*sister  ->  0.30*|parent-5.75| + |sister-10.57|

Why this pays (STRATEGY 14.1): division FP is counted ONLY at matched nodes and
annotation is 1.6% dense, so ~120 inserts cost ~2 visible FPs while every real
division caught counts fully. `div_J = TP/(TP+FP+FN)`.

Widening the gates WITHOUT the re-rank and the tighter cap would flood the graph
with bad divisions -- the three moves are one hypothesis, not three, so they ship
together. The DeepCenter veto is a genuinely separate precision mechanism and
gets its own kernel (`divgeom_veto`), per the one-change-per-experiment rule.

Also instruments the `frame_cap` break: STRATEGY 11.6 read
`safe_division_skipped_cap = 0` and concluded the caps never bind, but that stat
only tracks the GLOBAL cap -- the frame_cap break increments nothing. Unmeasured
until now.

Run: python scripts/build_divgeom_kernel.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"  # our best, LB 0.913

# Measured centres (scripts/local/division_geometry.py, 199 GT graphs).
PARENT_UM = 5.75
SISTER_UM = 10.57

OLD_SCORE = "score = parent_dist + 0.15 * sister_dist"
NEW_SCORE = (
    "# STRATEGY 15.3: the shipped score sorts ASCENDING on raw distance, so it\n"
    "                # preferred stationary cells. Real divisions sit at parent~5.75 um,\n"
    "                # sister~10.57 um (199 GT graphs). Score deviation from that\n"
    "                # geometry instead, so ascending order now means MOST division-like.\n"
    f"                score = 0.30 * abs(parent_dist - {PARENT_UM}) + abs(sister_dist - {SISTER_UM})"
)

OLD_CAP = """            if added_this_frame >= frame_cap:
                break"""
NEW_CAP = """            if added_this_frame >= frame_cap:
                # STRATEGY 15.3: 11.6 read safe_division_skipped_cap = 0 and
                # concluded the caps never bind, but that stat only covers the
                # GLOBAL cap -- this break was silent. Instrument it.
                stats["safe_division_frame_cap_hit"] += 1
                break"""

# `stats` is a PLAIN pre-seeded dict in filter_output_graph, not a defaultdict,
# so a new counter must be declared there or the first `+= 1` raises KeyError.
# v1 of this kernel died exactly that way -- the local fixture used a
# defaultdict and masked it.
OLD_SEED = '        "safe_division_skipped_cap": 0,'
NEW_SEED = ('        "safe_division_skipped_cap": 0,\n'
            '        "safe_division_frame_cap_hit": 0,')

# (env var, old literal, new literal) -- literals include their original quoting.
CONFIG = [
    ("BIOHUB_SAFE_DIV_MAX_UM", '"4.66"', '"12.0"'),
    ("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "'8.5'", "'14.4'"),
    ("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", '"7.65"', '"12.0"'),
    ("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", '"0.0076"', '"0.0030"'),
    ("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", '"0.00375"', '"0.0010"'),
    ("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", '"0"', '"1"'),
]
VETO = ("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "'0'", "'1'")


def patch_config(cfg: str, entries) -> str:
    for var, old, new in entries:
        anchor = f'os.environ["{var}"] = {old}'
        if cfg.count(anchor) != 1:
            raise RuntimeError(f"{var}: anchor count = {cfg.count(anchor)}, expected 1")
        cfg = cfg.replace(anchor, f'os.environ["{var}"] = {new}', 1)
    return cfg


def patch_pipeline(cell: str) -> str:
    """Patch the safe-division scorer + cap instrumentation in cell 11."""
    for old, new, label in ((OLD_SCORE, NEW_SCORE, "score"),
                            (OLD_CAP, NEW_CAP, "frame_cap"),
                            (OLD_SEED, NEW_SEED, "stats_seed")):
        if cell.count(old) != 1:
            raise RuntimeError(f"{label}: anchor count = {cell.count(old)}, expected 1")
        cell = cell.replace(old, new, 1)
    compile(cell, "<cell11>", "exec")  # must still parse
    return cell


def slug(title: str) -> str:
    """Kaggle derives the kernel slug from the TITLE, not from the id we send.

    A mismatch silently creates the kernel at the title-derived slug, which then
    breaks `kernels status` and the `-k` submission flag. Derive the id from the
    title so the two cannot drift.
    """
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def write(nb: dict, name: str, title: str) -> None:
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
    print(f"wrote kernels/{name}/{name}.ipynb")


def build(name: str, title: str, with_veto: bool) -> None:
    nb = json.loads(BASE_NB.read_text())
    entries = [*CONFIG, VETO] if with_veto else CONFIG
    nb["cells"][4]["source"] = patch_config(
        "".join(nb["cells"][4]["source"]), entries).splitlines(keepends=True)
    nb["cells"][11]["source"] = patch_pipeline(
        "".join(nb["cells"][11]["source"])).splitlines(keepends=True)
    # Every cell must still be valid Python.
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise RuntimeError(f"cell {i} no longer parses: {e}") from e
    write(nb, name, title)


if __name__ == "__main__":
    build("divgeom", "Biohub division geometry re-center", with_veto=False)
    build("divgeom_veto", "Biohub division geometry + DeepCenter veto", with_veto=True)
