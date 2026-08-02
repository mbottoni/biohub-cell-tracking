#!/usr/bin/env python3
"""Build the D4 edge-TTA kernel: pilkwang's shared-point edge TTA merged with our
dual-seed secondary model, on top of the 0.912 two-seeds base notebook.

The support-pack repo is patched at kernel runtime by string surgery on
``scripts/predict_unet_transformer.py``.  The 0.912 base applies two patches:
an 8-view detection-TTA rewrite and a 6-part dual-seed rewrite.  pilkwang's
edge-TTA patch replaces ``predict_video`` wholesale, which collides with both.
So we build one merged function here instead of stacking the patches.

Run: python scripts/build_d4edge_kernel.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/two_seeds_repro/biohub-cell-tracking-two-seeds-logit-blend.ipynb"
PILKWANG_CELL = Path("/tmp/pilkwang_patch_cell.py")
OUR_REPLS = Path("/tmp/our_dualseed_repls.json")
OUT_DIR = REPO / "kernels/d4edge"


def load_pilkwang() -> tuple[str, str]:
    """Return (helpers, predict_video) source from pilkwang's patch cell."""
    tree = ast.parse(PILKWANG_CELL.read_text())
    out = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.targets[0].id] = node.value.value
    return out["_helpers"], out["_predict_video"]


# --- the four merge edits, applied to pilkwang's predict_video -----------------

SIGNATURE_OLD = (
    "    downsample: tuple[int, ...] = (1, 4, 4),\n"
    ") -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:"
)

DET_NORM_OLD = (
    "        if len(det_tta_names) > 1:\n"
    "            for f in range(W):\n"
    "                det_logits[f] = det_logits[f] / len(det_tta_names)\n"
    "\n"
    "        missing_feature_maps"
)

RAW_OLD = (
    "            if edge_logits_pair is None:\n"
    '                raise RuntimeError("Edge TTA produced no transformer logits")\n'
    "\n"
    "            raw = edge_logits_pair[0]"
)

CLEANUP_OLD = "        del feature_maps_cpu\n"
CLEANUP_NEW = (
    "        del feature_maps_cpu\n"
    "        if secondary_unet_out is not None:\n"
    "            del secondary_unet_out\n"
)


def build_merged_predict_video() -> str:
    helpers, predict_video = load_pilkwang()
    repls = json.loads(OUR_REPLS.read_text())
    sig_new = repls[0][1]
    secondary_encode = repls[1][1]
    secondary_edge = repls[2][1]

    src = predict_video

    # 1. secondary_* keyword parameters on the signature
    assert src.count(SIGNATURE_OLD) == 1
    src = src.replace(SIGNATURE_OLD, sig_new.rstrip("\n"), 1)

    # 2. secondary encode + detection blend, after the primary detection TTA
    #    average and before the feature-map completeness check.  Our dual-seed
    #    block opens with the primary `det_logits / _nv` normalisation and closes
    #    with `del imgs`; pilkwang normalises by len(det_tta_names) and deletes
    #    `imgs` further down, so splice out those two ends and keep the middle.
    body = secondary_encode
    head = "            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n"
    assert body.startswith(head), body[:120]
    body = body[len(head):]
    assert body.endswith("\n        del imgs")
    body = body[: -len("\n        del imgs")]

    assert src.count(DET_NORM_OLD) == 1
    det_norm_new = (
        "        if len(det_tta_names) > 1:\n"
        "            for f in range(W):\n"
        "                det_logits[f] = det_logits[f] / len(det_tta_names)\n"
        "\n"
        + body
        + "\n\n        missing_feature_maps"
    )
    src = src.replace(DET_NORM_OLD, det_norm_new, 1)

    # 3. secondary edge blend, applied once to the TTA-averaged primary logits.
    #    Our block opens by computing edge_logits_pair from a single
    #    predict_edges call -- pilkwang's TTA loop already produced it, so drop
    #    that call and keep the `if secondary_model is not None:` blend onward.
    marker = "            if secondary_model is not None:"
    assert marker in secondary_edge
    blend = secondary_edge[secondary_edge.index(marker):]
    assert blend.rstrip().endswith("raw = edge_logits_pair[0]")

    assert src.count(RAW_OLD) == 1
    raw_new = (
        "            if edge_logits_pair is None:\n"
        '                raise RuntimeError("Edge TTA produced no transformer logits")\n'
        "\n"
        + blend.rstrip("\n")
    )
    src = src.replace(RAW_OLD, raw_new, 1)

    # 4. release the secondary feature map at the end of each window
    assert src.count(CLEANUP_OLD) == 1
    src = src.replace(CLEANUP_OLD, CLEANUP_NEW, 1)

    merged = helpers + "\n\n" + src
    compile(merged, "<merged_predict_video>", "exec")  # syntax gate
    return merged


PATCH_CELL_TEMPLATE = '''# Fail fast instead of silently running volumetric inference on CPU.
import torch as _torch

if not _torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
    )
print("CUDA device:", _torch.cuda.get_device_name(0))

# Replace predict_video with the merged shared-point edge-TTA + dual-seed
# implementation.  Detection TTA is handled inside the new function via
# BIOHUB_DET_TTA_MODE, so the separate 8-view detection patch is gone.
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()

_merged = {merged!r}

_helper_start_marker = "# SHARED_POINT_TTA_HELPERS_START"
_helper_end_marker = "# SHARED_POINT_TTA_HELPERS_END"
if _helper_start_marker in _s:
    _hs = _s.index(_helper_start_marker)
    _he = _s.index(_helper_end_marker, _hs) + len(_helper_end_marker)
    _s = _s[:_hs] + _s[_he:]

_function_start = _s.index("@torch.no_grad()\\ndef predict_video(")
_function_end_marker = (
    "\\n\\n# ============================================================================="
    "\\n# Prediction loop"
)
_function_end = _s.index(_function_end_marker, _function_start)
_s = _s[:_function_start] + _merged + _s[_function_end:]

# Driver-side dual-seed wiring (load the secondary checkpoint, pass it through).
# These two anchors live outside predict_video and are untouched by the merge.
_driver_replacements = {driver!r}
for _i, (_old, _new) in enumerate(_driver_replacements, start=1):
    _n = _s.count(_old)
    if _n != 1:
        raise RuntimeError(f"Driver patch {{_i}} expected one match, found {{_n}}")
    _s = _s.replace(_old, _new, 1)

compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Merged edge-TTA + dual-seed patch applied |",
    "det=" + os.environ.get("BIOHUB_DET_TTA_MODE", "flip4"),
    "edge=" + os.environ.get("BIOHUB_EDGE_TTA_MODE", "original"),
    "identity_weight=" + os.environ.get("BIOHUB_EDGE_TTA_ORIGINAL_WEIGHT", "auto"),
)
'''


def main() -> None:
    merged = build_merged_predict_video()
    repls = json.loads(OUR_REPLS.read_text())
    driver = [repls[4], repls[5]]  # load_model + call-site kwargs

    nb = json.loads(BASE_NB.read_text())

    # --- cell 4: config.  Turn on D4 edge TTA. ---
    cfg = "".join(nb["cells"][4]["source"])
    assert "BIOHUB_DET_THRESHOLD" in cfg
    marker = 'os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"'
    assert cfg.count(marker) == 1
    cfg = cfg.replace(
        marker,
        "# IDEA 1: pilkwang's shared-point D4 edge TTA (their 0.916 line).\n"
        "# Eight inverse-aligned encoder views feed the node transformer; raw\n"
        "# association logits are averaged before sigmoid -> ILP.\n"
        'os.environ["BIOHUB_EDGE_TTA_MODE"] = "d4"\n'
        'os.environ["BIOHUB_DET_TTA_MODE"] = "d4"\n'
        'os.environ["BIOHUB_EDGE_TTA_ORIGINAL_WEIGHT"] = "0.125"\n'
        + marker,
        1,
    )
    nb["cells"][4]["source"] = cfg.splitlines(keepends=True)

    # --- cell 9: the runtime patch ---
    patch = PATCH_CELL_TEMPLATE.format(merged=merged, driver=driver)
    old = "".join(nb["cells"][9]["source"])
    tail_marker = "\ndef list_test_stems() -> list[str]:"
    assert old.count(tail_marker) == 1
    nb["cells"][9]["source"] = (patch + old[old.index(tail_marker):]).splitlines(
        keepends=True
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "d4edge.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "maruanbottoni/biohub-d4-edge-tta",
                "title": "Biohub D4 edge TTA",
                "code_file": "d4edge.ipynb",
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
    print(f"wrote {OUT_DIR}/d4edge.ipynb  ({len(merged)} B merged function)")


if __name__ == "__main__":
    main()
