#!/usr/bin/env python
"""Build pub920_combo — div917 graft + recalibrated detection threshold (Step 4, row 110).

Usage: build_pub920_combo_kernel.py <det_threshold>   e.g. 0.9375

Combines the two independently-gated changes:
  1. DIV917 graft (see build_pub920_div917_kernel.py — radii 12/15/10 + filters pinned).
  2. Fused-detection threshold recalibration (hiranorm mechanism, row 110): the 0.475
     blend averages away one-sided detections at 0.96875; the retention receipts show
     60% of frames losing 18-42% of primary's candidates and falling back. The sweep
     (det_sweep_results.txt) picks the threshold; local gate is reject-only.

SUBMIT ONLY AFTER both single-change LB reads are in (one-change discipline): this kernel
exists so the winning combination is ready the moment the reads justify it.

Det threshold has THREE sites (cell00 env, cell01 guard _EXPECTED_NUMERIC, cell06
provenance manifest) — all rewritten; a missed guard site fails the commit run loudly
(that is the guard's job; do not weaken it).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

THR = sys.argv[1] if len(sys.argv) > 1 else None
assert THR is not None, "pass the calibrated det threshold, e.g. 0.9375"
assert 0.85 <= float(THR) <= 0.96875, f"implausible threshold {THR}"

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_div917" / "pub920_div917.ipynb"  # start from the graft
DST_DIR = REPO / "kernels" / "pub920_combo"
DST = DST_DIR / "pub920_combo.ipynb"
assert SRC.exists(), "build the div917 kernel first"

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

REWRITES = [
    (0, 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"',
        f'os.environ["BIOHUB_DET_THRESHOLD"] = "{THR}"  # COMBO: fused-blend recalibration (row 110)'),
    (1, '"BIOHUB_DET_THRESHOLD": 0.96875,',
        f'"BIOHUB_DET_THRESHOLD": {THR},'),
    (6, '"detector_threshold": 0.96875,',
        f'"detector_threshold": {THR},'),
]
for idx, old, new in REWRITES:
    s = "".join(cells[idx]["source"])
    n = s.count(old)
    assert n == 1, f"cell{idx}: expected exactly 1 occurrence of {old!r}, got {n}"
    cells[idx]["source"] = s.replace(old, new).splitlines(keepends=True)

# Both mechanisms must be present in the result.
allsrc = "\n".join("".join(c["source"]) for c in cells)
assert 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.0"' in allsrc, "div917 graft missing"
assert f'os.environ["BIOHUB_DET_THRESHOLD"] = "{THR}"' in allsrc

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-combo"
meta["title"] = "biohub pub920 combo"
meta["code_file"] = "pub920_combo.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} (det={THR} + div917 graft)")
