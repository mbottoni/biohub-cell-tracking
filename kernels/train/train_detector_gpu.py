"""Kaggle T4 training kernel for the 3D U-Net detector (path B).

Trains a bigger detector than MPS can (base-32, many epochs, all fold-0 train data) on a T4,
holding out the fold-0 val + golden sets so we keep a CLEAN local-scoring split. Weights go to
/kaggle/working -> the kernel output, which we download and evaluate locally.

Attach: competition data + the `ctrack-src` dataset. GPU=T4 (metadata machine_shape), internet
on (to pip install zarr, absent from the image). The heavy lifting is ctrack.detect.train --
this kernel is a thin driver.
"""

import os
import subprocess
import sys
from pathlib import Path

# --- config ---
FOLD = int(os.environ.get("FOLD", 0))
K = 5
EPOCHS = int(os.environ.get("EPOCHS", 25))
BATCH = int(os.environ.get("BATCH", 8))
BASE = int(os.environ.get("BASE", 32))
TRAIN_STRIDE = int(os.environ.get("TRAIN_STRIDE", 2))
VAL_STRIDE = int(os.environ.get("VAL_STRIDE", 4))

# zarr is not in the Kaggle image; training kernels may use the internet.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr", "scipy"], check=False)


def _add_ctrack_to_path():
    """Put the ctrack package on sys.path regardless of how ctrack-src was packed."""
    import shutil
    import zipfile

    for init in Path("/kaggle/input").glob("**/ctrack/__init__.py"):
        sys.path.insert(0, str(init.parents[1]))
        return
    for marker in Path("/kaggle/input").glob("**/detect/unet3d.py"):
        pkg_root = marker.parents[1]  # dir holding __init__.py + detect/ + data/ ...
        if (pkg_root / "__init__.py").exists():
            dest = Path("/kaggle/working/_pkg")
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

from ctrack.detect.train import train_fold  # noqa: E402


def _find_train_dir() -> Path:
    for c in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
              Path("/kaggle/input/biohub-cell-tracking-during-development/train")]:
        if c.is_dir() and any(c.glob("*.zarr")):
            return c
    for p in Path("/kaggle/input").glob("**/train"):
        if p.is_dir() and any(p.glob("*.zarr")):
            return p
    raise FileNotFoundError("competition train/ not found")


TRAIN = _find_train_dir()
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
print("train dir:", TRAIN, flush=True)

best = train_fold(
    TRAIN, TRAIN, fold=FOLD, k=K, epochs=EPOCHS, batch=BATCH, base=BASE,
    train_stride=TRAIN_STRIDE, val_stride=VAL_STRIDE,
    out_dir="/kaggle/working", num_workers=2, log_every=50, max_minutes=None,
)
print(f"FINISHED fold {FOLD} best val peak-recall {best:.4f}", flush=True)
