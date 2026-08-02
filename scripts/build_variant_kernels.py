#!/usr/bin/env python3
"""Build the harmonic-fusion (idea 2) and bidirectional-edge (idea 4) kernels.

Both are single-lever edits to the dual-seed runtime patch in cell 9 of the
0.912 two-seeds base.  The patch lives inside a Python string literal, so we
rewrite the ``_ensemble_replacements`` assignment through ``ast`` rather than
doing text surgery on escaped source.

Run: python scripts/build_variant_kernels.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/two_seeds_repro/biohub-cell-tracking-two-seeds-logit-blend.ipynb"

# --- idea 2: harmonic-probability detection consensus -------------------------
# yusuketogashi's 0.915 line.  Harmonic mean behaves like AND: a peak only one
# seed believes gets suppressed, cutting N_pred at little recall cost, which is
# the node-ratio multiplier term from STRATEGY 10.5.
ARITHMETIC_BLEND = """                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )"""

HARMONIC_BLEND = """                    primary_det_prob = torch.sigmoid(primary_det.float()).clamp(1e-6, 1.0 - 1e-6)
                    secondary_det_prob = torch.sigmoid(
                        secondary_det_aligned.float()
                    ).clamp(1e-6, 1.0 - 1e-6)
                    consensus_det_prob = 1.0 / (
                        (1.0 - secondary_detection_weight) / primary_det_prob
                        + secondary_detection_weight / secondary_det_prob
                    )
                    det_logits[f] = torch.logit(
                        consensus_det_prob.clamp(1e-6, 1.0 - 1e-6)
                    ).to(primary_det.dtype)"""

# --- idea 4: bidirectional edge logits ---------------------------------------
BI_ANCHOR = """            )  # (1, n_src, n_tgt)

            if secondary_model is not None:"""


def load_bidirectional_block() -> str:
    """Return yusuketogashi's reverse-edge block.

    Their patch text spans the same anchor we do, so strip the leading
    ``predict_edges`` call.  The trailing ``if secondary_model is not None:``
    line stays -- it closes the anchor and must not be duplicated.
    """
    src = Path("/tmp/yusuke_bi_new.py").read_text()
    head = """            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)
"""
    assert src.startswith(head)
    block = src[len(head):]
    assert block.rstrip("\n").endswith("if secondary_model is not None:")
    return block


def cell_source(nb: dict, index: int) -> str:
    return "".join(nb["cells"][index]["source"])


def rewrite_replacements(cell: str, mutate) -> str:
    """Re-emit the _ensemble_replacements literal after applying ``mutate``."""
    tree = ast.parse(cell)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "_ensemble_replacements"
        ):
            old_segment = ast.get_source_segment(cell, node)
            pairs = ast.literal_eval(node.value)
            pairs = mutate(pairs)
            new_segment = "_ensemble_replacements = " + repr(pairs)
            assert cell.count(old_segment) == 1
            return cell.replace(old_segment, new_segment, 1)
    raise RuntimeError("_ensemble_replacements not found")


def add_config(cell: str, block: str) -> str:
    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    assert cell.count(marker) == 1
    return cell.replace(marker, block + marker, 1)


def write_kernel(nb: dict, out_dir: Path, stem: str, kid: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.ipynb").write_text(json.dumps(nb, indent=1))
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kid,
                "title": title,
                "code_file": f"{stem}.ipynb",
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
    print(f"wrote {out_dir}/{stem}.ipynb")


def build_harmonic() -> None:
    nb = json.loads(BASE_NB.read_text())

    def mutate(pairs):
        old, new = pairs[1]
        assert new.count(ARITHMETIC_BLEND) == 1, "arithmetic blend anchor missing"
        pairs[1] = (old, new.replace(ARITHMETIC_BLEND, HARMONIC_BLEND, 1))
        return pairs

    nb["cells"][9]["source"] = rewrite_replacements(
        cell_source(nb, 9), mutate
    ).splitlines(keepends=True)
    nb["cells"][4]["source"] = add_config(
        cell_source(nb, 4),
        "# IDEA 2: harmonic-probability two-seed detection consensus\n"
        "# (yusuketogashi 0.915).  Suppresses single-seed peaks -> lower N_pred\n"
        "# -> larger negative node-ratio bonus.  Marker only; the blend itself\n"
        "# is patched into predict_video below.\n"
        'os.environ["BIOHUB_DETECTION_FUSION_MODE"] = "harmonic_probability"\n',
    ).splitlines(keepends=True)
    write_kernel(
        nb,
        REPO / "kernels/harmonic",
        "harmonic",
        "maruanbottoni/biohub-harmonic-detection-fusion",
        "Biohub harmonic detection fusion",
    )


def build_bidirectional() -> None:
    nb = json.loads(BASE_NB.read_text())
    block = load_bidirectional_block()

    def mutate(pairs):
        old, new = pairs[2]
        assert new.count(BI_ANCHOR) == 1, "bidirectional anchor missing"
        pairs[2] = (
            old,
            new.replace(
                BI_ANCHOR,
                ")  # (1, n_src, n_tgt)\n" + block.rstrip("\n"),
                1,
            ),
        )
        return pairs

    nb["cells"][9]["source"] = rewrite_replacements(
        cell_source(nb, 9), mutate
    ).splitlines(keepends=True)
    nb["cells"][4]["source"] = add_config(
        cell_source(nb, 4),
        "# IDEA 4: bidirectional edge logits (yusuketogashi).  Score each frame\n"
        "# pair in reverse too, calibrate onto the forward distribution, blend.\n"
        'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"\n',
    ).splitlines(keepends=True)
    write_kernel(
        nb,
        REPO / "kernels/bidir",
        "bidir",
        "maruanbottoni/biohub-bidirectional-edges",
        "Biohub bidirectional edges",
    )


if __name__ == "__main__":
    build_harmonic()
    build_bidirectional()
