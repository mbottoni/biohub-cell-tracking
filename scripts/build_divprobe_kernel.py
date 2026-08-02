#!/usr/bin/env python3
"""Build the division-term probe off our 0.913 d4edge best.

`0.1 * division_jaccard` is worth up to 0.1 of the score and we have never
measured it on real test data.  STRATEGY 9.2 called divisions dead, but that
verdict came from the leaky golden-8 harness *and* from our own division
metric, which 10.6 established does not match the official one (ours uses
out-degree; the reference verifies daughter lineages via grandchildren and a
bipartite matching).  Both instruments are discredited, so the call is unproven.

Single lever: raise only the *rate caps* on safe-division insertion by 5x.
Geometry gates (parent/sister/existing-child radii) stay fixed, so this asks
"are we leaving divisions on the table?" and not "are these the right ones?".

Run: python scripts/build_divprobe_kernel.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"   # our best, LB 0.913
OUT_DIR = REPO / "kernels/divprobe"

# 5x the caps that throttle how many safe divisions may be added.
CAP_CHANGES = {
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": ("0.0076", "0.038"),
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": ("0.00375", "0.01875"),
}


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    cfg = "".join(nb["cells"][4]["source"])

    for var, (old, new) in CAP_CHANGES.items():
        line_old = f'os.environ["{var}"] = "{old}"'
        line_new = f'os.environ["{var}"] = "{new}"'
        if cfg.count(line_old) != 1:
            raise RuntimeError(f"{var}: expected one match of {line_old!r}, found {cfg.count(line_old)}")
        cfg = cfg.replace(line_old, line_new, 1)

    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    assert cfg.count(marker) == 1
    cfg = cfg.replace(
        marker,
        "# DIVISION PROBE: rate caps 5x (geometry gates unchanged) to measure\n"
        "# whether the 0.1*division_jaccard term has anything in it on real test.\n"
        '# Diagnostics ON so run_stats reports safe_division_candidates/added.\n'
        + marker.replace('"0"', '"1"'),
        1,
    )
    nb["cells"][4]["source"] = cfg.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "divprobe.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "maruanbottoni/biohub-division-probe",
                "title": "Biohub division probe",
                "code_file": "divprobe.ipynb",
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
    print(f"wrote {OUT_DIR}/divprobe.ipynb")
    for var, (old, new) in CAP_CHANGES.items():
        print(f"  {var}: {old} -> {new}")


if __name__ == "__main__":
    main()
