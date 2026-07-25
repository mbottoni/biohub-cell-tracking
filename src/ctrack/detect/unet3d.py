"""A small 3D U-Net point detector.

Predicts a single-channel heatmap on the isotropic 64^3 grid (the whole volume — no tiling).
Peaks of the predicted heatmap are the detections, so this is a drop-in replacement for
ctrack.detect.blob: everything downstream (NMS, linking, gap closing, submission) is reused.

Deliberately small — 64^3 input, ~1M params, three downsampling levels. Fits Apple MPS and a
Kaggle GPU with room to spare, and the dataset is only 199 volumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.InstanceNorm3d(cout, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.InstanceNorm3d(cout, affine=True),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """3-level 3D U-Net. Input (B,1,D,H,W), output (B,1,D,H,W) heatmap logits.

    With `out_offset=True` an extra 3-channel head predicts, at each voxel, the offset (in
    isotropic voxels) to the cell's PARENT in the previous frame (linajea). This gives the linker
    motion-aware edge costs. `forward(x)` still returns only the heatmap logits (backward
    compatible); `forward(x, return_offset=True)` returns `(heatmap_logits, offset)`.
    """

    def __init__(self, base: int = 16, in_ch: int = 1, out_offset: bool = False):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = _block(in_ch, c1)
        self.enc2 = _block(c1, c2)
        self.enc3 = _block(c2, c3)
        self.bott = _block(c3, c4)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = _block(c4, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = _block(c3, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = _block(c2, c1)

        self.head = nn.Conv3d(c1, 1, 1)
        self.offset_head = nn.Conv3d(c1, 3, 1) if out_offset else None

    def forward(self, x: torch.Tensor, return_offset: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        hm = self.head(d1)  # logits; caller applies sigmoid
        if return_offset:
            off = self.offset_head(d1) if self.offset_head is not None else None
            return hm, off
        return hm


def masked_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 20.0
) -> torch.Tensor:
    """BCE-with-logits supervised only where `mask` is 1 (the trusted region).

    `pos_weight` upweights the heatmap's positive lobe since cells are a tiny fraction of
    voxels. The mask excludes the ambiguous bright middle (unlabeled cells) so the detector is
    not punished for finding them -- see ctrack.detect.targets and docs/detector.md.
    """
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    per_voxel = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    denom = mask.sum().clamp_min(1.0)
    return (per_voxel * mask).sum() / denom


def masked_offset_loss(
    pred_offset: torch.Tensor, target_offset: torch.Tensor, offset_mask: torch.Tensor
) -> torch.Tensor:
    """Smooth-L1 on the parent-offset vectors, supervised only where `offset_mask` is 1 (voxels
    belonging to a cell whose parent is known). `*_offset` are (B,3,D,H,W); mask is (B,1,D,H,W).
    """
    per_voxel = nn.functional.smooth_l1_loss(pred_offset, target_offset, reduction="none")
    per_voxel = per_voxel.sum(dim=1, keepdim=True)          # sum over the 3 vector components
    denom = offset_mask.sum().clamp_min(1.0)
    return (per_voxel * offset_mask).sum() / denom
