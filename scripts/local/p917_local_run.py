#!/usr/bin/env python
"""Local replay of the 0.917 family (pub917_port — single-cell M001 architecture).

Row 115's side discovery: the 917 pipeline is one self-contained cell (xiaolei UNet3D
x2, Hungarian 1-1, orphan-pool safe divisions) — so unlike the pilkwang family it needs
no staged repo, no cell transforms, no caches. This runner materialises the cell's own
`data/` + `models/` fallbacks as symlinks and exec()s the cell with an MPS patch (the
cell only knows cuda/cpu; conv3d forward on MPS is ~5x CPU here).

Purpose (PLAN 3.1f.2): produce the 917 family's LOCAL submission on holdout6 so the
cross-family edge ensemble can be gated locally — the instrument row 108 lacked.

Usage: p917_local_run.py <run_dir> [test_dir]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
TEST = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path("/Volumes/extSSD/kaggle/biohub/holdout6")
REPO = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.environ.get(
    "BIOHUB_M001_WEIGHTS_DIR",
    "/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/914d4e78-261f-4ae8-9e3a-e159bcbcf840/scratchpad/m001_weights"))

RUN.mkdir(parents=True, exist_ok=True)
(RUN / "data").mkdir(exist_ok=True)
for link, target in [(RUN / "data" / "test", TEST), (RUN / "models", WEIGHTS_DIR)]:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
os.chdir(RUN)

nb = json.loads((REPO / "kernels" / "pub917_port" / "pub917_port.ipynb").read_text())
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

# MPS patch: extend the device fallback chain. Anchor = the cell's own CPU default line.
OLD = 'DEVICE = "cpu"\nif torch.cuda.is_available():'
NEW = (
    'DEVICE = "cpu"\n'
    'if not torch.cuda.is_available() and torch.backends.mps.is_available() '
    'and os.environ.get("BIOHUB_MPS", "1") != "0":\n'
    '    DEVICE = "mps"\n'
    'if torch.cuda.is_available():'
)
assert src.count(OLD) == 1, "device anchor drifted"
src = src.replace(OLD, NEW, 1)

print(f"[p917] run_dir={RUN} test={TEST}", flush=True)
exec(compile(src, "pub917_cell", "exec"), {"__name__": "__main__"})
print(f"[p917] DONE -> {RUN / 'submission.csv'}", flush=True)
