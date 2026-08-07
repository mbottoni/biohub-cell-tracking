#!/usr/bin/env python3
"""Local smoke for the STRATEGY 41.2 sub-voxel patch, before spending a kernel run.

CLAUDE.md: "Any new patch to the trainer should be exercised locally before it is
pushed", and "Guard against silent no-ops -- assert and print match counts". Two
bugs in this project's history (an `os` NameError inside a patch, a read-only
symlink) each cost a ~6 min kernel round trip and would have been caught here.

Two independent checks:

  1. STRING SURGERY. Replay exactly what the kernel does -- strip the shared TTA
     helpers, swap in the merged predict_video, apply the driver replacements,
     then apply the four locref edits -- against the real
     `predict_unet_transformer.py` captured from a completed run. Assert every
     edit matches exactly once and the result compiles.

  2. THE MATH. Build a synthetic heatmap with peaks at KNOWN sub-voxel positions,
     run the patched `_detect_cells_pooled`, and check the recovered offsets. A
     patch that applies cleanly and computes the wrong offset looks like success.

    python scripts/local/test_locref.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "kernels/locref/locref.ipynb"
REPO_FILE = REPO / "submissions/d4edge/tracking_repo/scripts/predict_unet_transformer.py"


def literals(cell: str) -> dict:
    out = {}
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def apply_surgery() -> str:
    """Reconstruct the exact file state locref edits at kernel runtime.

    The captured `tracking_repo` was saved AFTER the kernel patched it, so it
    already carries the merged predict_video and the driver replacements. That
    makes it the right target: locref runs last, on precisely this state. We swap
    in the notebook's NEW _merged (the one carrying the int16-cast edit, applied at
    build time) and then apply the three runtime edits.
    """
    cell = "".join(json.loads(NB.read_text())["cells"][9]["source"])
    lit = literals(cell)
    src = REPO_FILE.read_text()

    assert "secondary_model = None" in src and src.count("det_tta_names") > 1, \
        "captured repo file is not in the post-merge state this test assumes"

    hs_m, he_m = "# SHARED_POINT_TTA_HELPERS_START", "# SHARED_POINT_TTA_HELPERS_END"
    hs = src.index(hs_m)
    he = src.index(he_m, hs) + len(he_m)
    src = src[:hs] + src[he:]

    start = src.index("@torch.no_grad()\ndef predict_video(")
    end_marker = ("\n\n# ============================================================================="
                  "\n# Prediction loop")
    end = src.index(end_marker, start)
    src = src[:start] + lit["_merged"] + src[end:]
    print("  merged swap       OK")

    for i, (old, new) in enumerate(lit["_locref_edits"], start=1):
        n = src.count(old)
        assert n == 1, f"locref edit {i}: {n} matches, expected 1\n---\n{old[:200]}"
        src = src.replace(old, new, 1)
    print(f"  locref edits      {len(lit['_locref_edits'])}/{len(lit['_locref_edits'])} matched")

    compile(src, "<patched>", "exec")
    print("  compiles          OK")

    assert "coords = coords.astype(np.int16)" not in src, \
        "the second int16 cast survived -- sub-voxel refinement would be a SILENT NO-OP"
    assert ".astype(np.int16)" not in src.split("def _detect_cells_pooled(")[1][:2000], \
        "_detect_cells_pooled still returns int16"
    print("  int16 casts gone  OK  (the silent-no-op guard)")
    return src


def test_math(src: str) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns: dict = {"torch": torch, "np": np, "F": F}
    fn_src = src[src.index("def _detect_cells_pooled("):]
    fn_src = fn_src[:fn_src.index("\n\n# SHARED_POINT") if "\n\n# SHARED_POINT" in fn_src
                    else fn_src.index("\n\n@torch.no_grad()")]
    exec(compile(fn_src, "<fn>", "exec"), ns)
    detect = ns["_detect_cells_pooled"]

    def arm(name: str, build, noise: float, seed: int):
        rng = np.random.default_rng(seed)
        Z, Y, X = 24, 48, 48
        grid = np.stack(np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X),
                                    indexing="ij"), -1).astype(float)
        truth = []
        hm = np.full((Z, Y, X), -12.0)
        for _ in range(14):
            c = np.array([rng.integers(4, Z - 4), rng.integers(4, Y - 4),
                          rng.integers(4, X - 4)], dtype=float) + rng.uniform(-0.45, 0.45, 3)
            truth.append(c)
            hm = np.maximum(hm, build(((grid - c) ** 2).sum(-1)))
        if noise:
            hm = hm + rng.normal(0, noise, hm.shape)

        out = detect(torch.from_numpy(hm).float().unsqueeze(0), t=3,
                     det_threshold=0.5, pool_kernel=(3, 3, 3))
        assert out.dtype == np.float32, f"expected float32 coords, got {out.dtype}"
        got = out[:, 1:].astype(float)
        errs, base = [], []
        for tr in truth:
            if not len(got):
                continue
            j = int(np.argmin(np.linalg.norm(got - tr, axis=1)))
            if np.linalg.norm(got[j] - tr) > 2.0:
                continue                       # a different peak, not this one
            errs.append(np.linalg.norm(got[j] - tr))
            base.append(np.linalg.norm(np.round(tr) - tr))
        errs, base = np.array(errs), np.array(base)
        red = 1 - errs.mean() / base.mean()
        print(f"  {name:<26} matched {len(errs):>2}/{len(truth)}  "
              f"argmax {base.mean():.4f}  parabolic {errs.mean():.4f}  "
              f"reduction {red:>+6.1%}")
        return red

    print()
    # Exact paraboloid: parabolic interpolation is exact by construction. This is a
    # CORRECTNESS check on the implementation, not evidence about real heatmaps.
    arm("paraboloid (correctness)", lambda d2: 8.0 - 0.9 * d2, 0.0, 0)

    # Realistic: a Gaussian probability blob pushed through a logit, which is NOT
    # parabolic away from the peak, plus heatmap noise. This is the honest arm.
    def gauss_logit(d2, sigma=1.3):
        p = np.clip(np.exp(-d2 / (2 * sigma ** 2)), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    reds = [arm(f"gaussian+noise seed {s}", gauss_logit, 0.15, s) for s in (1, 2, 3)]

    print(f"\n  realistic mean reduction {np.mean(reds):+.1%} of the sub-voxel error")
    print("  ^ CEILING ONLY. Real detector error is 2.186 um against a ~1.625 um")
    print("    lattice, so quantisation is a MINORITY of it; this arm bounds only the")
    print("    quantisation part. The 40.5 gate (>=30% of TOTAL RMS) is decided by the")
    print("    kernel run, not here.")
    assert np.mean(reds) > 0, "parabolic refinement is not beating the integer argmax"


if __name__ == "__main__":
    print("STRING SURGERY")
    patched = apply_surgery()
    print("\nMATH")
    test_math(patched)
    print("\nPASS -- patch applies, compiles, survives the int16 guard, and the "
          "estimator beats the argmax on synthetic peaks.")
