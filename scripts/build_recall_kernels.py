#!/usr/bin/env python3
"""Three experiments on the recall lever identified by the STRATEGY 13 EDA.

The EDA showed association is geometrically trivial (displacement/spacing =
0.103) and `edge_J ~ recall^2`, so our edge_J = 0.909 implies effective node
recall ~0.953 and detection recall is the bottleneck.

The model predicts a result it was not fitted to. Harmonic fusion is an AND: it
cut nodes 2.9%, which at edge_J ~ r^2 costs ~2.8% of edge_J => ~-0.025. The
measured score was 0.889, i.e. -0.023. So the mirror should pay.

Trade arithmetic (13.4): adding nodes costs 0.1*dk on the multiplier while
recall multiplies edge_J by (r'/r)^2, so net positive when dr/r > 0.05*dk.
Adding 10% more nodes pays if recall rises just 0.5% relative.

  thr096   DET_THRESHOLD 0.96875 -> 0.96   (small step)
  thr095   DET_THRESHOLD 0.96875 -> 0.95   (brackets the curve; with the
                                            current config these give three
                                            points to locate the optimum)
  orfuse   two-seed detection fusion by OR instead of the default blend --
           the exact dual of the harmonic (AND) form that scored 0.889

Run: python scripts/build_recall_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"   # our best, LB 0.913

# The arithmetic blend currently in the dual-seed patch.
ARITHMETIC = """                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )"""

# Dual of the harmonic mean: a weighted harmonic mean on the COMPLEMENT.
# Harmonic on p suppresses a peak unless BOTH seeds believe it (AND).
# Harmonic on (1-p) admits a peak if EITHER seed believes it (OR) -> more
# nodes, higher recall, which is what edge_J ~ r^2 says we want.
OR_FUSION = """                    primary_det_prob = torch.sigmoid(primary_det.float()).clamp(1e-6, 1.0 - 1e-6)
                    secondary_det_prob = torch.sigmoid(
                        secondary_det_aligned.float()
                    ).clamp(1e-6, 1.0 - 1e-6)
                    consensus_not = 1.0 / (
                        (1.0 - secondary_detection_weight) / (1.0 - primary_det_prob)
                        + secondary_detection_weight / (1.0 - secondary_det_prob)
                    )
                    consensus_det_prob = (1.0 - consensus_not).clamp(1e-6, 1.0 - 1e-6)
                    det_logits[f] = torch.logit(
                        consensus_det_prob
                    ).to(primary_det.dtype)"""

import ast


def rewrite_merged(cell: str, old: str, new: str) -> str:
    """Swap a block inside d4edge's `_merged` predict_video literal.

    d4edge replaced predict_video wholesale (build_d4edge_kernel.py), so cell 9
    holds a `_merged` string rather than the `_ensemble_replacements` list the
    pre-d4edge notebooks used. The detection blend lives inside that string.
    """
    tree = ast.parse(cell)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_merged":
            seg = ast.get_source_segment(cell, node)
            merged = ast.literal_eval(node.value)
            if merged.count(old) != 1:
                raise RuntimeError(f"blend anchor count in _merged = {merged.count(old)}")
            merged = merged.replace(old, new, 1)
            compile(merged, "<merged>", "exec")      # patched function must still parse
            assert cell.count(seg) == 1
            return cell.replace(seg, "_merged = " + repr(merged), 1)
    raise RuntimeError("_merged not found (is the base d4edge?)")


def write(nb: dict, name: str, kid: str, title: str) -> None:
    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": title, "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
    }, indent=2))
    print(f"wrote kernels/{name}/{name}.ipynb")


def build_threshold(value: str, name: str) -> None:
    nb = json.loads(BASE_NB.read_text())
    cfg = "".join(nb["cells"][4]["source"])
    old = 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"'
    assert cfg.count(old) == 1, cfg.count(old)
    cfg = cfg.replace(
        old,
        "# STRATEGY 13.4: edge_J ~ recall^2, and adding nodes is net positive\n"
        "# when dr/r > 0.05*dk. Diagnostics ON so run_stats gives dk directly.\n"
        f'os.environ["BIOHUB_DET_THRESHOLD"] = "{value}"',
        1,
    )
    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    assert cfg.count(marker) == 1
    cfg = cfg.replace(marker, marker.replace('"0"', '"1"'), 1)
    nb["cells"][4]["source"] = cfg.splitlines(keepends=True)
    write(nb, name, f"maruanbottoni/biohub-{name}", f"Biohub det threshold {value}")


def build_orfuse() -> None:
    nb = json.loads(BASE_NB.read_text())

    nb["cells"][9]["source"] = rewrite_merged(
        "".join(nb["cells"][9]["source"]), ARITHMETIC, OR_FUSION).splitlines(keepends=True)
    cfg = "".join(nb["cells"][4]["source"])
    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    cfg = cfg.replace(
        marker,
        "# STRATEGY 13: OR-fusion, the dual of the harmonic AND that scored\n"
        "# 0.889. Admits a peak either seed believes => more nodes, more recall.\n"
        'os.environ["BIOHUB_DETECTION_FUSION_MODE"] = "or_complement_harmonic"\n'
        + marker.replace('"0"', '"1"'), 1)
    nb["cells"][4]["source"] = cfg.splitlines(keepends=True)
    write(nb, "orfuse", "maruanbottoni/biohub-or-fusion", "Biohub OR fusion")


if __name__ == "__main__":
    build_threshold("0.96", "thr096")
    build_threshold("0.95", "thr095")
    build_orfuse()
