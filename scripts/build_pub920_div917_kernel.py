#!/usr/bin/env python
"""Build pub920_div917 — the 0.917-family division radii grafted onto the 0.920 base.

WHY (rows 106-110): our hidden decomposition is adj 0.906 + div 0.014 (div_J 0.13); the
0.917 family measures div 0.024 (div_J 0.24) via the SAME safe-div postlink rule our port
already carries — the ONLY delta is the radii. kunal's inline changelog (row 110) supplies
the recipe AND the negative control: parent 12.0 / sister 15.0 / existing-child 10.0 µm,
with mutual-NN + forward-divergence (2.25 µm) + DeepCenter safe-div veto (thr 0.12) all
REQUIRED — radii widened WITHOUT the filters measured 0.915→0.914 on LB (also our row 99).
Our port defaults all three filters ON; this build pins them explicitly so default drift
can never silently disarm the mechanism (the ret00 lesson, row 104: silent knobs kill).

ONE mechanism changed vs pub920_port (LB 0.920, sub 55678092): the safe-div geometry.
The cell01 config guard pins det/ILP/gap/min-len/bidir/retention — none touched here, so
it must still PASS; a guard failure on commit means this build touched something it
should not have.

Pre-registered read (submit as: additive graft of the 917 div term onto our 0.906 adj):
  ~0.930  the div term transfers additively -> new best, hybrid confirmed
  ~0.920  flat: wide radii find nothing on our graph (orphan starvation, row 97 redux)
  <=0.918 the graft LOSES on hidden -> geometric route dead on our graph, go prob-gated
          (backtracking-v6 in-linker SD2, row 110) as the A/B follow-up.
Local (cache-hit replay, biased instrument: local UNDER-prices divisions ~x1.5, row 106):
reject-only gate — a clear local LOSS kills the submission; local gain confirms nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_port" / "pub920_port.ipynb"
DST_DIR = REPO / "kernels" / "pub920_div917"
DST = DST_DIR / "pub920_div917.ipynb"

# (old exact line prefix, new full line) — matched on the env assignment, not the literal,
# because cell00/cell06 carry an unrelated knob that shares the literal "8.5" (gap span).
REWRITES = [
    (
        'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"',
        'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.0"  # DIV917 GRAFT: 0.917-family radius (row 110)',
    ),
    (
        'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"',
        'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "15.0"  # DIV917 GRAFT: 0.917-family radius (row 110)',
    ),
    (
        'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"',
        'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"  # DIV917 GRAFT: 0.917-family radius (row 110)',
    ),
]

# Filters the recipe REQUIRES (all default-ON in the port; pinned so drift is loud).
PIN_BLOCK = (
    "\n# DIV917 GRAFT: the filters ARE the mechanism (field control: radii without them = LB loss)\n"
    'os.environ["BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN"] = "1"\n'
    'os.environ["BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE"] = "1"\n'
    'os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"\n'
    'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"\n'
    'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.12"\n'
)

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

cell0 = "".join(cells[0]["source"])
counts = {}
for old, new in REWRITES:
    counts[old] = cell0.count(old)
    cell0 = cell0.replace(old, new)
for old, n in counts.items():
    assert n == 1, f"expected exactly 1 occurrence, got {n}: {old}"
# Drop the stale author comments now trailing the rewritten lines ("reverted ... back to
# the proven value" would contradict the new values).
cell0 = re.sub(r"(# DIV917 GRAFT: 0\.917-family radius \(row 110\))  # REVIEW:[^\n]*", r"\1", cell0)

# The veto env line already exists in cell00 (set to '1'); drop it from the pin block to
# avoid a duplicate assignment, then append the block after the last radius line.
existing_veto = 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = \'1\''
assert cell0.count(existing_veto) == 1, "veto line moved — re-audit cell00"
pin_block = PIN_BLOCK.replace('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"\n', "")
anchor = 'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"  # DIV917 GRAFT: 0.917-family radius (row 110)'
cell0 = cell0.replace(anchor, anchor + pin_block, 1)
cells[0]["source"] = cell0.splitlines(keepends=True)

# Sanity: the filter code paths must exist downstream with the expected defaults.
allsrc = "\n".join("".join(c["source"]) for c in cells)
for needle in [
    'SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "1")',
    'SAFE_DIV_DIVERGE_UM = float(os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25"))',
    'DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12"))',
]:
    assert needle in allsrc, f"filter code path missing: {needle}"
# The cell01 guard must not reference any knob we changed.
guard_src = "".join(cells[1]["source"])
assert "SAFE_DIV" not in guard_src, "cell01 guard now pins SAFE_DIV — rewrite it too"

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_ret00" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-div917"
meta["title"] = "biohub pub920 div917"
meta["code_file"] = "pub920_div917.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} ({DST.stat().st_size} bytes); rewrites: {list(counts.values())}, pins appended")
