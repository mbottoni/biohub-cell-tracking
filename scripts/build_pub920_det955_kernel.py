#!/usr/bin/env python
"""Build pub920_det955 — fused-detection threshold recalibration (Step 3, rows 110-112).

WHY: the 0.475 blend averages away detections only one seed sees; retention receipts show
the blend losing 18-42% of primary's candidates on ~60% of frames at 0.96875 (row 111).
hiranorm's real-LB series banked this mechanism (+0.0185 holdout, det 0.96875->0.90 in
their plain-average family). Our 1-volume sweep (row 112): edge_J jumps +0.016 at 0.955
and SATURATES (identical edge_J at 0.9375/0.92 while node-ratio penalty grows) -> 0.955
is the measured local optimum. ⚠ Local caveats: memorized volume (node_recall 1.0), and
ret00=0.920-flat says blend-vs-primary at 0.96875 was score-neutral on hidden — this
tests whether the THRESHOLD lever itself transfers.

ONE change vs pub920_port (LB 0.920): BIOHUB_DET_THRESHOLD 0.96875 -> 0.955.
Three sites rewritten: cell00 env, cell01 guard _EXPECTED_NUMERIC, cell06 provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

THR = "0.955"
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_port" / "pub920_port.ipynb"
DST_DIR = REPO / "kernels" / "pub920_det955"
DST = DST_DIR / "pub920_det955.ipynb"

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
REWRITES = [
    (0, 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"',
        f'os.environ["BIOHUB_DET_THRESHOLD"] = "{THR}"  # DET955: fused-blend recalibration (rows 110-112)'),
    (1, '"BIOHUB_DET_THRESHOLD": 0.96875,', f'"BIOHUB_DET_THRESHOLD": {THR},'),
    (6, '"detector_threshold": 0.96875,', f'"detector_threshold": {THR},'),
]
for idx, old, new in REWRITES:
    s = "".join(cells[idx]["source"])
    n = s.count(old)
    assert n == 1, f"cell{idx}: expected 1 occurrence of {old!r}, got {n}"
    cells[idx]["source"] = s.replace(old, new).splitlines(keepends=True)

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-det955"
meta["title"] = "biohub pub920 det955"
meta["code_file"] = "pub920_det955.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST}")
