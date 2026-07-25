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
EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH = int(os.environ.get("BATCH", 8))
BASE = int(os.environ.get("BASE", 32))
TRAIN_STRIDE = int(os.environ.get("TRAIN_STRIDE", 2))
VAL_STRIDE = int(os.environ.get("VAL_STRIDE", 8))
PATIENCE = int(os.environ.get("PATIENCE", 4))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", 4.0))  # was 20; lower curbs over-detection
OFFSET_WEIGHT = float(os.environ.get("OFFSET_WEIGHT", 1.0))  # >0 trains the movement-vector head
WEIGHTS_DS = "maruanbottoni/biohub-detector-weights"

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


def upload_best(ckpt_path, epoch, sel):
    """Publish the new best checkpoint to a Kaggle dataset so it's downloadable MID-RUN and
    survives a kill/timeout (Kaggle only commits kernel output at the very end). `sel` is the
    node-ratio-aware selection score (recall x over-detection penalty), not bare recall."""
    import json
    import shutil

    wdir = Path("/kaggle/working/_weights_out")
    wdir.mkdir(exist_ok=True)
    shutil.copy(ckpt_path, wdir / ckpt_path.name)
    json.dump({"title": "biohub-detector-weights", "id": WEIGHTS_DS,
               "licenses": [{"name": "CC0-1.0"}]}, open(wdir / "dataset-metadata.json", "w"))
    exists = subprocess.run(["kaggle", "datasets", "status", WEIGHTS_DS],
                            capture_output=True, text=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", str(wdir), "-m", f"epoch {epoch} sel {sel:.3f}",
            "--dir-mode", "zip"] if exists
           else ["kaggle", "datasets", "create", "-p", str(wdir), "--dir-mode", "zip"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  uploaded checkpoint to {WEIGHTS_DS} (epoch {epoch}, sel {sel:.3f}): "
          f"{'ok' if r.returncode == 0 else r.stderr[-150:]}", flush=True)


# pos_weight defaults to 4.0 in train_fold now (was 20 -> curbs over-firing); selection is
# node-ratio-aware inside train_fold, so "best" is the metric-aligned checkpoint, not max-recall.
best = train_fold(
    TRAIN, TRAIN, fold=FOLD, k=K, epochs=EPOCHS, batch=BATCH, base=BASE,
    train_stride=TRAIN_STRIDE, val_stride=VAL_STRIDE, pos_weight=POS_WEIGHT,
    offset_weight=OFFSET_WEIGHT,  # linajea movement-vector head -> motion-aware linking
    out_dir="/kaggle/working", num_workers=2, log_every=50, max_minutes=None,
    early_stop_patience=PATIENCE, on_best=upload_best,
)
print(f"FINISHED fold {FOLD} best val sel-score {best:.4f}", flush=True)
