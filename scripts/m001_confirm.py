"""M001 tophat-UNet confirmation detector (row 110 idea 4, built 08-24).

Ported from `xiaoleilian/biohub-m001-ens3-sm6-sim2` (public, vendored notes in row 110):
xy 4x mean-pool -> grey_opening tophat -> percentile norm -> small UNet3D (base 24) ->
local-max peaks -> physical NMS 4 um. ONLY the tophat-trained checkpoint is used here:
it is the maximally decorrelated branch from the pilkwang detectors (different net,
different training data view, different preprocessing).

Purpose: NOT a detection source — a CONFIRMATION signal. The det955 gate (row 113)
showed blanket threshold-lowering fails because the 0.90–0.96875 band is noise on
non-memorized volumes; this module supplies the independent filter that band needs.
`confirm_trees(zarr_path)` returns {t: cKDTree of detections in um} for gating.

skimage is deliberately not used (peak_local_max -> scipy maximum_filter) so the module
runs in both the Kaggle offline image and the local venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import grey_opening, maximum_filter
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
UNET_THRESH = 0.15
NMS_UM = 4.0


def _block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base); self.e2 = _block(base, base * 2); self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2); self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2); self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2); self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2); self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


_MODEL = None
_DEVICE = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and os.environ.get("BIOHUB_MPS", "1") != "0":
        return torch.device("mps")
    return torch.device("cpu")


def _model() -> tuple[nn.Module, torch.device]:
    global _MODEL, _DEVICE
    if _MODEL is None:
        weights = os.environ["BIOHUB_M001_WEIGHTS"]
        _DEVICE = _device()
        ck = torch.load(weights, map_location=_DEVICE)
        m = UNet3D(base=ck.get("base", 24)).to(_DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _MODEL = m
        print(f"[m001] loaded {Path(weights).name} on {_DEVICE} | val_recall {ck.get('val_recall')}", flush=True)
    return _MODEL, _DEVICE


_ZC: dict[str, object] = {}


def _read_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = _read_meta(zp)
        buf = blosc2.decompress(open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])


def _pool_norm_tophat(vol: np.ndarray) -> np.ndarray:
    Z, Y, X = vol.shape
    Y2, X2 = (Y // POOL) * POOL, (X // POOL) * POOL
    p = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    p = p.reshape(Z, Y2 // POOL, POOL, X2 // POOL, POOL).mean(axis=(2, 4))
    p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50)); hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def _physical_nms(coords, scores, radius_um):
    if len(coords) <= 1:
        return coords
    pts = coords * SCALE[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    return coords[np.array(keep)]


@torch.no_grad()
def detect_frame_um(vol: np.ndarray) -> np.ndarray:
    """Detections for one raw (Z, Y, X) frame, as (N, 3) zyx in micrometres."""
    model, device = _model()
    x = _pool_norm_tophat(vol)
    h = torch.sigmoid(model(torch.from_numpy(x)[None, None].to(device)))[0, 0].float().cpu().numpy()
    is_peak = (h == maximum_filter(h, size=3)) & (h >= UNET_THRESH)
    pk = np.argwhere(is_peak)
    if len(pk) == 0:
        return np.zeros((0, 3))
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    coords = _physical_nms(coords, sc, NMS_UM)
    return coords * SCALE[None, :]


def confirm_trees(zarr_path) -> dict[int, cKDTree]:
    """{t: cKDTree of tophat-model detections (um)} for every frame of the video."""
    zp = Path(str(zarr_path))
    if not (zp / "0" / "zarr.json").exists():
        with_ext = zp.parent / (zp.name + ".zarr")
        if (with_ext / "0" / "zarr.json").exists():
            zp = with_ext
        else:
            raise FileNotFoundError(f"no zarr at {zp} or {with_ext}")
    zarr_path = zp
    meta = _read_meta(zarr_path)
    n_frames = meta["shape"][0]
    trees: dict[int, cKDTree] = {}
    total = 0
    for t in range(n_frames):
        pts = detect_frame_um(load_volume(zarr_path, t, meta))
        total += len(pts)
        trees[t] = cKDTree(pts) if len(pts) else None
    print(f"[m001] {Path(str(zarr_path)).name}: {total} confirmations over {n_frames} frames", flush=True)
    return trees
