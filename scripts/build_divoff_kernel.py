#!/usr/bin/env python3
"""Turn safe-division insertion OFF, on top of the 0.913 d4edge best.

`kernels/divprobe` raised the safe-division rate caps 5x and moved almost
nothing: candidates 355 -> 355, added 323 -> 350, `safe_division_skipped_cap`
= 0 in both runs. The caps were never the binding constraint, so that probe
measures noise and is not worth a submission.

The ground truth says the lever points the other way. The 4 visible test
volumes are byte-identical to train volumes (docs/data.md), so their GT is in
docs/gt_stats.csv:

    44b6_0113de3b  0 divisions      6bba_05b6850b  0 divisions
    44b6_0b24845f  0 divisions      6bba_05db0fb1  3 divisions

Three annotated divisions in total, against the 350 we insert. Across all 199
train volumes the median is 0 divisions per volume (mean 0.8, max 5, and 112
volumes have none at all).

So we are adding ~90 divisions per volume where ~1 exists. Every one of those
also introduces a second-child edge, which is a plain FP against edge_J -- the
term that carries ~99% of the score. This probe removes them all and asks what
they were costing us. A rise means the division machinery is net negative; no
movement means the term is inert and can be retired from the roadmap.

Run: python scripts/build_divoff_kernel.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"   # our best, LB 0.913
OUT_DIR = REPO / "kernels/divoff"


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    cfg = "".join(nb["cells"][4]["source"])

    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    if cfg.count(marker) != 1:
        raise RuntimeError(f"diagnostics anchor count = {cfg.count(marker)}")
    cfg = cfg.replace(
        marker,
        "# DIVISION PROBE 2: safe divisions OFF.\n"
        "# GT has 3 divisions across the 4 test volumes; we insert 350. Each one\n"
        "# also adds a second-child edge that is an FP against edge_J. Diagnostics\n"
        "# ON so run_stats confirms safe_divisions_added drops to 0.\n"
        'os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"\n'
        + marker.replace('"0"', '"1"'),
        1,
    )
    nb["cells"][4]["source"] = cfg.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "divoff.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "maruanbottoni/biohub-divisions-off",
                "title": "Biohub divisions off",
                "code_file": "divoff.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": False,
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "dataset_sources": [
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_DIR}/divoff.ipynb  (BIOHUB_OUTPUT_SAFE_DIVISIONS=0)")


if __name__ == "__main__":
    main()
