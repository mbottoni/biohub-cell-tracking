#!/usr/bin/env python3
"""Two kernels that test whether we are running the weaker checkpoint as primary.

The weights diagnostic (kernels/wdiag) showed the two attached checkpoints have
identical architecture but very different provenance:

  seed314159-v1   ships history.csv + training_config.json: 400 epochs on all
                  199 volumes, best epoch 381, validation_score 0.978 and still
                  climbing (0.946 at epoch 50 -> 0.978 at 381, i.e. +0.032).
  support-pack    ships config.json only -- no history, no training config.
  -50ep-v1        Its name is the sole evidence of its epoch count.

We run the support pack as PRIMARY and blend seed314159 in at only 0.15 edge
weight.  Since edge_J is ~99% of the score (STRATEGY 11.2) and detection recall
is already 97.5%, the edge role is the one that matters.  If the name is honest
we have the weaker model in the dominant seat.

  swapw  swap the roles outright: 400ep model primary, 50ep pack secondary
  secw   keep the roles, raise SECONDARY_EDGE_WEIGHT 0.15 -> 0.50

Together they bracket the question, and either outcome is informative: no
movement means the primary is trained comparably and the ceiling is elsewhere.

Run: python scripts/build_weights_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"   # our best, LB 0.913

SECONDARY_EDGE_ASSIGN = 'os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"'

SWAP_CELL = '''# SWAP: promote the documented 400-epoch checkpoint to PRIMARY.
#
# Cell 7 materialised both models and verified the seed314159 checksum; we only
# re-point which one plays which role, after that verification has run.
#   primary   <- seed314159 (400 epochs, val 0.978, best epoch 381)
#   secondary <- support-pack "50ep" (undocumented)
# Architectures match (unet_out_channels 32, layers [32,64,128], downsample
# [1,4,4], window 2), so the secondary-loading guard on window/downsample holds.
PRIMARY_WEIGHTS_ARG = str(SECONDARY_WEIGHTS_PATH)
os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(REPO_DIR / WEIGHTS_RELATIVE)

for _p in (Path(PRIMARY_WEIGHTS_ARG), Path(os.environ["BIOHUB_SECONDARY_WEIGHTS"])):
    if not _p.is_file():
        raise FileNotFoundError(f"swap target missing: {_p}")
    if not (_p.parent / "config.json").is_file():
        raise FileNotFoundError(f"swap target has no config.json: {_p.parent}")

print("SWAPPED primary  :", PRIMARY_WEIGHTS_ARG)
print("SWAPPED secondary:", os.environ["BIOHUB_SECONDARY_WEIGHTS"])
'''


def write_kernel(nb: dict, name: str, kid: str, title: str) -> None:
    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kid,
                "title": title,
                "code_file": f"{name}.ipynb",
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
    print(f"wrote kernels/{name}/{name}.ipynb")


def build_swap() -> None:
    nb = json.loads(BASE_NB.read_text())

    # predict_cmd must take the swapped primary rather than WEIGHTS_RELATIVE.
    cell9 = "".join(nb["cells"][9]["source"])
    old = '"--weights",\n    WEIGHTS_RELATIVE,'
    if cell9.count(old) != 1:
        alt = '"--weights", WEIGHTS_RELATIVE,'
        if cell9.count(alt) != 1:
            raise RuntimeError(
                f"--weights anchor not found uniquely (got {cell9.count(old)} / {cell9.count(alt)})"
            )
        old = alt
    cell9 = cell9.replace(old, old.replace("WEIGHTS_RELATIVE", "PRIMARY_WEIGHTS_ARG"), 1)
    nb["cells"][9]["source"] = cell9.splitlines(keepends=True)

    # Insert the swap right after the artifact/weights setup cell.
    swap_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": SWAP_CELL.splitlines(keepends=True)}
    nb["cells"].insert(8, swap_cell)

    write_kernel(nb, "swapw", "maruanbottoni/biohub-swap-primary-weights",
                 "Biohub swap primary weights")


def build_secondary_weight() -> None:
    nb = json.loads(BASE_NB.read_text())
    cell7 = "".join(nb["cells"][7]["source"])
    if cell7.count(SECONDARY_EDGE_ASSIGN) != 1:
        raise RuntimeError(f"secondary-edge-weight anchor count = {cell7.count(SECONDARY_EDGE_ASSIGN)}")
    cell7 = cell7.replace(
        SECONDARY_EDGE_ASSIGN,
        "# Raise the 400-epoch model's say over edge logits (was 0.15).\n"
        'os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.50"',
        1,
    )
    nb["cells"][7]["source"] = cell7.splitlines(keepends=True)
    write_kernel(nb, "secw", "maruanbottoni/biohub-secondary-edge-weight-050",
                 "Biohub secondary edge weight 050")


if __name__ == "__main__":
    build_swap()
    build_secondary_weight()
