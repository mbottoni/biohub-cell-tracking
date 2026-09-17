"""DivNet (giorgosi/biohub-divnet-v2) — reconstructed architecture + preprocessing.

The public checkpoint carries its own config (task: division_event_detection):
  input: 5 channels = image at lags [-1,0,+1,+2] + gaussian_center marker
         (sigma z,y,x = 1.5,2.0,2.0 in pooled voxels), crop_z 16, crop_yx 32,
         pool_xy 4  (0.40625*4 = 1.625 um -> isotropic with z)
  normalization: (x - p50) / (p99.5 - p50), clip [-0.5, 6.0]
  arch: divnet_unet3d_gap, base 16 -> UNet3D enc/dec + GAP over dec1 + Linear(16,1)

Architecture reconstructed from the state dict (50 tensors, strict-load verified).
Preprocessing assumptions (pool type, norm scope, channel order) are validated
against GT divisions locally before deployment — see scripts/divnet_local_check.py.

haideptry's public integration of this checkpoint is a double no-op: wrong class
(strict=False matches zero keys -> random weights) and a 6-D tensor that makes
conv3d raise inside its try/except. Nothing it "verified" ever ran. (row 128)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _norm(kind, c):
    # state dict has per-layer weight+bias but NO running stats and convs have NO bias:
    # InstanceNorm3d(affine=True) or GroupNorm both fit; discriminated empirically on GT
    # divisions by scripts/divnet_local_check.py.
    if kind == 'instance':
        return nn.InstanceNorm3d(c, affine=True)
    if kind.startswith('group'):
        return nn.GroupNorm(int(kind[5:]), c)
    raise ValueError(kind)


class _Block(nn.Module):
    def __init__(self, cin, cout, norm='instance'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _norm(norm, cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _norm(norm, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DivNetUNet3DGAP(nn.Module):
    def __init__(self, in_channels=5, base=16, norm='instance'):
        super().__init__()
        self.enc1 = _Block(in_channels, base, norm)
        self.enc2 = _Block(base, base * 2, norm)
        self.enc3 = _Block(base * 2, base * 4, norm)
        self.bottleneck = _Block(base * 4, base * 8, norm)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _Block(base * 8, base * 4, norm)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Block(base * 4, base * 2, norm)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = _Block(base * 2, base, norm)
        self.head = nn.Linear(base, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        g = d1.mean(dim=(2, 3, 4))
        return self.head(g)


def load_divnet(ckpt_path, device='cpu', norm='instance'):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DivNetUNet3DGAP(norm=norm)
    model.load_state_dict(ck['model_state'], strict=True)
    model.to(device).eval()
    return model, ck['config']


def pool_xy(frame, k=4):
    """Mean-pool a (Z,Y,X) frame by k in y and x."""
    Z, Y, X = frame.shape
    Yk, Xk = (Y // k) * k, (X // k) * k
    f = frame[:, :Yk, :Xk].reshape(Z, Yk // k, k, Xk // k, k)
    return f.mean(axis=(2, 4))


def gaussian_marker(shape, center, sigma):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    sz, sy, sx = sigma
    return np.exp(-(((zz - cz) ** 2) / (2 * sz ** 2)
                    + ((yy - cy) ** 2) / (2 * sy ** 2)
                    + ((xx - cx) ** 2) / (2 * sx ** 2))).astype(np.float32)


def build_input(frames_by_lag, z, y, x, cfg, norm_scope='crop', marker_last=True):
    """frames_by_lag: dict lag -> full (Z,Y,X) frame, ALREADY xy-pooled by cfg pool_xy.
    (z, y, x): node position in pooled-voxel coordinates.
    Returns (5, 16, 32, 32) float32."""
    icfg = cfg['input']
    ncfg = cfg['normalization']
    cz, cyx = icfg['crop_z'], icfg['crop_yx']
    hz, hyx = cz // 2, cyx // 2
    chans = []
    for lag in icfg['image_lags']:
        fr = frames_by_lag[lag]
        Z, Y, X = fr.shape
        z0, y0, x0 = int(round(z)) - hz, int(round(y)) - hyx, int(round(x)) - hyx
        crop = np.zeros((cz, cyx, cyx), dtype=np.float32)
        zs, ys, xs = max(0, z0), max(0, y0), max(0, x0)
        ze, ye, xe = min(Z, z0 + cz), min(Y, y0 + cyx), min(X, x0 + cyx)
        if ze > zs and ye > ys and xe > xs:
            crop[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = fr[zs:ze, ys:ye, xs:xe]
        chans.append(crop)
    img = np.stack(chans, axis=0)
    if norm_scope == 'crop':
        lo = np.percentile(img, ncfg['lo_pct'])
        hi = np.percentile(img, ncfg['hi_pct'])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), ncfg['clip'][0], ncfg['clip'][1])
    marker = gaussian_marker((cz, cyx, cyx), (hz, hyx, hyx), icfg['marker_sigma'])
    if marker_last:
        out = np.concatenate([img, marker[None]], axis=0)
    else:
        out = np.concatenate([marker[None], img], axis=0)
    return out.astype(np.float32)


@torch.no_grad()
def score_nodes(model, inputs, device='cpu', batch=32):
    """inputs: list of (5,16,32,32) arrays -> sigmoid probs."""
    probs = []
    for i in range(0, len(inputs), batch):
        x = torch.from_numpy(np.stack(inputs[i:i + batch])).to(device)
        probs.extend(torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist())
    return probs
