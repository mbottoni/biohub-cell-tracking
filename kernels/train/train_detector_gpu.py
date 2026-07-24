"""Kaggle GPU training kernel for the 3D U-Net detector.

MPS on the dev Mac is ~18-58 min/epoch (docs/detector.md); a Kaggle T4/P100 is 10-20x faster.
This trains one CV fold and writes weights to /kaggle/working, which becomes the kernel's
output and can be attached as a dataset to the inference/submission kernel.

Attach: the competition data + the `ctrack-src` dataset. GPU on, internet on (to pip install
zarr, which the image lacks). Set FOLD below (run once per fold).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --- config ---
FOLD = 0
K = 5
EPOCHS = 25
BATCH = 8
BASE = 32
LR = 1e-3
TRAIN_STRIDE = 2
VAL_STRIDE = 4
SIGMA_UM = 2.5

# zarr is not in the Kaggle image; training kernels may use the internet.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=False)


def _add_ctrack_to_path():
    """Make `import ctrack` work regardless of how the ctrack-src dataset was packed.

    Handles three layouts: a real .../ctrack/__init__.py; a ctrack.zip; and (what Kaggle
    actually produced) the package *contents* extracted to the dataset root, recognisable by a
    detect/unet3d.py sibling of __init__.py. In the last case we materialise a proper ctrack/
    dir under /kaggle/working so the internal `from ctrack...` imports resolve.
    """
    import shutil
    import zipfile

    for init in Path("/kaggle/input").glob("**/ctrack/__init__.py"):
        sys.path.insert(0, str(init.parent.parent))
        return
    for marker in Path("/kaggle/input").glob("**/detect/unet3d.py"):
        pkg_root = marker.parent.parent  # the dir holding __init__.py + detect/ + data/ ...
        if (pkg_root / "__init__.py").exists():
            dest = Path("/kaggle/working/_ctrack_pkg")
            (dest / "ctrack").parent.mkdir(parents=True, exist_ok=True)
            if not (dest / "ctrack").exists():
                shutil.copytree(pkg_root, dest / "ctrack")
            sys.path.insert(0, str(dest))
            return
    for z in Path("/kaggle/input").glob("**/ctrack.zip"):
        dest = Path("/kaggle/working/_ctrack_src")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        sys.path.insert(0, str(dest))
        return
    raise FileNotFoundError("ctrack package not found under /kaggle/input")


_add_ctrack_to_path()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ctrack.data.geff import find_geff_stores  # noqa: E402
from ctrack.data.splits import hold_out_golden, kfold  # noqa: E402
from ctrack.data.volume import ISOTROPIC_SUBSAMPLE, isotropic_scale  # noqa: E402
from ctrack.detect.dataset import DetectionFrames  # noqa: E402
from ctrack.detect.unet3d import UNet3D, masked_bce_loss  # noqa: E402


def _find(name):
    for p in Path("/kaggle/input").glob(f"**/{name}"):
        if p.is_dir() and any(p.iterdir()):
            return p
    raise FileNotFoundError(name)


RAW = _find("train")                       # competition train/ (zarr + geff together)
GT = RAW                                    # geff lives beside the zarr in the competition mount
print("data root:", RAW)


def peak_recall(pred_hm, gt_iso, spacing, thr, nms, radius=7.0):
    from scipy import ndimage
    if len(gt_iso) == 0:
        return 0, 0
    pooled = ndimage.maximum_filter(pred_hm, size=nms, mode="nearest")
    peaks = np.argwhere((pred_hm == pooled) & (pred_hm > thr)).astype(np.float64)
    if len(peaks) == 0:
        return 0, len(gt_iso)
    d = np.linalg.norm((gt_iso[:, None, :] - peaks[None, :, :]) * spacing, axis=2)
    return int((d.min(axis=1) <= radius).sum()), len(gt_iso)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

all_ds = sorted(p.name.replace(".geff", "") for p in find_geff_stores(GT))
remaining, golden = hold_out_golden(all_ds, n=8, seed=0)
split = kfold(remaining, k=K, seed=0)[FOLD]
print(f"fold {FOLD}: {len(split.train)} train, {len(split.val)} val datasets")

train = DetectionFrames(split.train, RAW, GT, augment=True, frame_stride=TRAIN_STRIDE, sigma_um=SIGMA_UM)
val = DetectionFrames(split.val, RAW, GT, augment=False, frame_stride=VAL_STRIDE, sigma_um=SIGMA_UM)
print(f"frames: {len(train)} train, {len(val)} val")
train_dl = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)

model = UNet3D(base=BASE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
spacing = float(isotropic_scale(subsample=ISOTROPIC_SUBSAMPLE)[0])
nms_vox = max(1, int(round(3.0 / spacing)) | 1)

best = -1.0
for epoch in range(EPOCHS):
    model.train()
    t0, tot = time.time(), 0.0
    for x, y, m in train_dl:
        x, y, m = x.to(device), y.to(device), m.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss = masked_bce_loss(model(x), y, m)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += float(loss.detach())
    sched.step()

    model.eval()
    matched = total = 0
    with torch.no_grad():
        for j in range(len(val)):
            ds, t = val.index[j]
            frame = val.frame_only(j)
            x = torch.from_numpy(np.ascontiguousarray(frame))[None, None].float().to(device)
            hm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            centres = val._centres[(ds, t)] / np.array(ISOTROPIC_SUBSAMPLE)
            a, b = peak_recall(hm, centres, spacing, 0.3, nms_vox)
            matched += a
            total += b
    recall = matched / max(1, total)
    print(f"epoch {epoch:2d}  loss {tot/max(1,len(train_dl)):.4f}  "
          f"val peak-recall {recall:.3f}  {time.time()-t0:.0f}s", flush=True)

    if recall > best:
        best = recall
        torch.save({"model": model.state_dict(), "base": BASE, "fold": FOLD, "recall": recall},
                   f"/kaggle/working/unet_fold{FOLD}.pt")

print(f"best val peak-recall: {best:.3f}")
