#!/usr/bin/env python3
"""One-knob variants of the pub162 port (STRATEGY 48.3 residual sweep, 2026-08-10).

Rationale: rows 34/39/41 killed the association, consensus, and DET_THRESHOLD levers on
the pub162 base, but the components row 38 credits with its +0.002 over our old line were
never *perturbed* on this base: the ILP disappearance weight, the gap-close radius, and
the short-track output filter. These four variants bracket those knobs, one change each.
EV is low (we are perturbing the public author's tuned optimum) but nonzero, and the
submission budget is otherwise unused since 08-07.

Gate (pre-registered): a variant is submitted ONLY if its dev-4 commit output beats the
re-scored pub162_port baseline on the official scorer (scripts/local/score_official.py,
same instrument for all readings). Flat-or-worse variants are recorded and dropped.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "kernels/pub162_port/pub162_port.ipynb"
BASE_META = REPO / "kernels/pub162_port/kernel-metadata.json"

VARIANTS = {
    "pub162_disapp10": (
        'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"',
        'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.0"',
    ),
    "pub162_disapp20": (
        'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"',
        'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2.0"',
    ),
    "pub162_gap65": (
        'os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.8"',
        'os.environ["BIOHUB_GAP_CLOSE_UM"] = "6.5"',
    ),
    "pub162_mtl5": (
        'os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"',
        'os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "5"',
    ),
}


def main() -> None:
    meta = json.loads(BASE_META.read_text())
    for name, (old, new) in VARIANTS.items():
        nb = json.loads(BASE.read_text())
        n = 0
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if old in src:
                n += src.count(old)
                cell["source"] = src.replace(old, new).splitlines(keepends=True)
        assert n == 1, f"{name}: expected exactly 1 occurrence of {old!r}, found {n}"
        out_dir = REPO / "kernels" / name
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        m = dict(meta)
        slug = name.replace("_", "-")
        m["id"] = f"maruanbottoni/biohub-{slug}"
        m["title"] = f"biohub-{slug}"
        m["code_file"] = f"{name}.ipynb"
        (out_dir / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
        print(f"{name}: 1 replacement applied -> {out_dir}")


if __name__ == "__main__":
    main()
