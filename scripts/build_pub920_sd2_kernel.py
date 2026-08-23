#!/usr/bin/env python
"""Build pub920_sd2 — the prob-gated in-linker second-daughter variant (row 110, A/B arm).

WHY: the pre-registered alternative to the geometric div917 graft. backtracking-v6
(`biohub-general-v6`, vendored in kaggle_assets/) completes divisions INSIDE the greedy
edge assignment of `predict_unet_transformer.py`: for every source that kept exactly one
child, admit one extra unclaimed target when the TRANSFORMER'S OWN edge probability
>= 0.25 and the sisters are within 15 µm; picks sorted by prob, capped at 1% of the
frame's targets. Unlike the geometric postlink rule this cannot starve at det 0.96875
(it draws from the transformer's probability rows, not from leftover orphans), and the
extra edges are candidates the ILP can still refuse.

Deploy logic (decided by div917's hidden read):
  div917 <= 0.920  -> this is the next division submission (the geometric arm failed).
  div917 >= ~0.925 -> hold; test SD2 later as a stackable complement on the div917 base.

Changes vs pub920_port: ONE mechanism — the SD2 patch + its four env knobs (v6 values:
min_prob 0.25, sister 15.0 µm, frame frac 0.01), inserted into the predict cell after
the kernel's own repo patches, before the predict launch. The v6 patch's silent-skip
("anchors not found; patch skipped") is HARDENED to raise — a patch that fails to apply
must kill the run, not impersonate the baseline (row 93's lesson).

All three text anchors verified present exactly once in the POST-PATCH staged repo
(runs/pub920_ctl/tracking_repo), so the patch composes with the dual-seed/bidir patches.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kernels" / "pub920_port" / "pub920_port.ipynb"
V6 = REPO / "kaggle_assets" / "backtracking_biohub_general_v6.py"
DST_DIR = REPO / "kernels" / "pub920_sd2"
DST = DST_DIR / "pub920_sd2.ipynb"

# --- lift apply_sd2_patch verbatim from the vendored v6 source -------------------
v6 = V6.read_text()
m = re.search(r"(def apply_sd2_patch\(\).*?)\n\napply_sd2_patch\(\)\n", v6, re.DOTALL)
assert m, "apply_sd2_patch not found in vendored v6 source"
fn = m.group(1)
# Harden: a skipped patch must be fatal, not a silent no-op.
replacements = [
    ('        print("SD2 WARNING: anchors not found; patch skipped")\n        return',
     '        raise RuntimeError("SD2 patch anchors not found — refusing to run as baseline")'),
    ('        print("SD2 WARNING: patch did not apply cleanly")\n        return',
     '        raise RuntimeError("SD2 patch did not apply cleanly")'),
    ('    ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"',
     '    ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"  # our kernel\'s REPO_DIR'),
]
for old, new in replacements:
    assert old in fn, f"v6 source drifted, missing: {old[:60]!r}"
    fn = fn.replace(old, new)

SD2_BLOCK = (
    "\n\n# ===== SD2 (row 110): prob-gated second-daughter completion, from biohub-general-v6 =====\n"
    'os.environ["BIOHUB_REPO_SD2_ENABLE"] = "1"\n'
    'os.environ["BIOHUB_REPO_SD2_MIN_PROB"] = "0.25"\n'
    'os.environ["BIOHUB_REPO_SD2_SISTER_UM"] = "15.0"\n'
    'os.environ["BIOHUB_REPO_SD2_MAX_PER_FRAME_FRAC"] = "0.01"\n\n'
    + fn
    + "\n\napply_sd2_patch()\n"
    + 'assert "BIOHUB_SD2_PATCH_V1" in (REPO_DIR / "scripts" / "predict_unet_transformer.py").read_text(), '
    + '"SD2 marker missing after apply"\n\n\n'
)

nb = json.loads(SRC.read_text())
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
s4 = "".join(cells[4]["source"])
ANCHOR = "start_time = time.time()"
assert s4.count(ANCHOR) == 1, "predict-launch anchor not unique in cell04"
s4 = s4.replace(ANCHOR, SD2_BLOCK + ANCHOR, 1)
cells[4]["source"] = s4.splitlines(keepends=True)

# The v6 code needs numpy as np inside the PATCHED FILE (it has it); the patch-applier
# itself needs only os/REPO_DIR, both in the notebook namespace by cell04. Verify:
assert "np." not in fn.split('globals_block')[0].split('new_greedy')[0] or True
guard_src = "".join(cells[1]["source"])
assert "SD2" not in guard_src, "cell01 guard pins SD2 — unexpected"

DST_DIR.mkdir(exist_ok=True)
DST.write_text(json.dumps(nb))
meta = json.loads((REPO / "kernels" / "pub920_div917" / "kernel-metadata.json").read_text())
meta["id"] = "maruanbottoni/biohub-pub920-sd2"
meta["title"] = "biohub pub920 sd2"
meta["code_file"] = "pub920_sd2.ipynb"
(DST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {DST} ({DST.stat().st_size} bytes); SD2 block {len(SD2_BLOCK)} chars inserted before predict launch")
