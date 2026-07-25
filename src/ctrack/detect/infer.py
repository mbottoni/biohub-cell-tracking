"""Inference-time robustness: test-time augmentation and multi-model ensembling.

Both work with already-trained weights (no retraining) and reduce detector variance, which the
metric rewards twice — cleaner peaks improve edge matching AND stabilise the node count that
total_node_ratio taxes.

- TTA: average the predicted heatmap over axis-flip variants (the same label-preserving flips
  used in training augmentation). Offsets are taken from the identity pass only (flipping a
  vector field back is fiddly and the heatmap carries the detection gain).
- Ensemble: average heatmaps across checkpoints (e.g. different folds/seeds).
"""

from __future__ import annotations

import itertools

import numpy as np


def predict_heatmap(model, frame_iso, device, *, tta: bool = False, return_offset: bool = False):
    """Predict the sigmoid heatmap for one isotropic frame. Optionally TTA-averaged over flips.

    Returns `hm` (D,H,W), or `(hm, offset)` if `return_offset` (offset from the identity pass,
    shape (3,D,H,W) or None).
    """
    import torch

    def _fwd(arr):
        x = torch.from_numpy(np.ascontiguousarray(arr))[None, None].float().to(device)
        with torch.no_grad():
            if return_offset:
                hm_l, off = model(x, return_offset=True)
                off_np = off[0].cpu().numpy() if off is not None else None
                return torch.sigmoid(hm_l)[0, 0].cpu().numpy(), off_np
            return torch.sigmoid(model(x))[0, 0].cpu().numpy()

    if not tta:
        return _fwd(frame_iso)

    # average over the 8 axis-flip combinations, un-flipping each prediction back
    acc = np.zeros_like(frame_iso, dtype=np.float64)
    offset0 = None
    n = 0
    for flips in itertools.product([False, True], repeat=3):
        arr = frame_iso
        for ax, f in enumerate(flips):
            if f:
                arr = np.flip(arr, ax)
        out = _fwd(arr)
        hm = out[0] if return_offset else out
        for ax, f in enumerate(flips):
            if f:
                hm = np.flip(hm, ax)
        acc += hm
        n += 1
        if return_offset and not any(flips):
            offset0 = out[1]
    hm_avg = (acc / n).astype(np.float32)
    return (hm_avg, offset0) if return_offset else hm_avg


def predict_heatmap_ensemble(models, frame_iso, device, *, tta: bool = False):
    """Average the heatmap over several models (ensemble). Offsets not ensembled here."""
    acc = None
    for m in models:
        hm = predict_heatmap(m, frame_iso, device, tta=tta)
        acc = hm.astype(np.float64) if acc is None else acc + hm
    return (acc / len(models)).astype(np.float32)
