#!/usr/bin/env python3
"""Build the OWN-DETECTOR training kernel (STRATEGY 45.2 route 1).

Trains ctrack's UNet3D on the competition train mount with:
  * dense pseudo-label supervision (teacher peaks at 0.3, GT at 1.0; row 36's 99.90% purity);
  * sel-metric early stopping (recall x exp(-0.1|node_ratio|), patience 8);
  * RESUMABLE checkpoints -- checkpoint_last_fold0.pt written atomically every epoch to
    /kaggle/working. To continue past the 12 h cap, re-push with --resume-from <slug> so the
    previous version's output is attached; the boot cell seeds the working dir from it.
  * --seed and --iso-mode so TWO CONCURRENT T4 SESSIONS give a same-day A/B:
    seed 0 / stride (public-baseline downsampling) vs seed 1 / mean (area-averaged, ~16x
    the xy photons per voxel -- the A/B volume.py has promised since July).

Environment: internet OFF; zarr-v3 stack installed offline from the support pack's wheels/.

Usage:
  python scripts/build_dettrain_kernel.py --variant smoke                 # 2 epochs, 20 vols
  python scripts/build_dettrain_kernel.py --variant s0                    # seed 0, stride
  python scripts/build_dettrain_kernel.py --variant s1mean                # seed 1, mean-pool
  python scripts/build_dettrain_kernel.py --variant s0 --resume-from maruanbottoni/biohub-dettrain-s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

VARIANTS = {
    # name -> (seed, iso_mode, epochs, limit, max_minutes, train_stride, val_stride)
    "smoke": (0, "stride", 2, 20, 55, 4, 10),
    "s0": (0, "stride", 60, None, 640, 2, 6),
    "s1mean": (1, "mean", 60, None, 640, 2, 6),
}

DEPS_CELL = r'''# ------------------------------------------------- offline deps (zarr v3 stack)
import glob, subprocess, sys
from pathlib import Path

need = ["zarr>=3.0.10,<4", "blosc2", "numcodecs>=0.13,<0.16", "donfig>=0.8",
        "google-crc32c>=1.5", "crc32c", "deprecated", "wrapt", "typing-extensions>=4.13"]
wheel_dirs = sorted({str(Path(p).parent) for p in
                     glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)})
assert wheel_dirs, "no wheels/ dir found in attached datasets (need the support pack)"
cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
for d in wheel_dirs:
    cmd += ["--find-links", d]
r = subprocess.run(cmd + need, capture_output=True, text=True)
if r.returncode != 0:
    # tolerate specs with no wheel present; install what exists, then verify imports
    for spec in need:
        subprocess.run(cmd + [spec], capture_output=True, text=True)
import zarr, numpy
print("zarr", zarr.__version__, "OK", flush=True)
'''

BOOT_CELL = r'''# ---------------------------------------------------------------- boot
import glob, shutil, sys
from pathlib import Path

WORK = Path("/kaggle/working")
COMP = next(p for p in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
                        Path("/kaggle/input/biohub-cell-tracking-during-development")] if p.exists())
TRAIN_DIR = COMP / "train"
assert TRAIN_DIR.exists(), f"no train mount at {TRAIN_DIR}"
n_zarr = len(list(TRAIN_DIR.glob("*.zarr"))); n_geff = len(list(TRAIN_DIR.glob("*.geff")))
print(f"train mount: {n_zarr} zarr / {n_geff} geff", flush=True)
assert n_zarr >= 190 and n_geff >= 190

# ctrack package: dataset ships package CONTENTS (find by signature file, ctrack_test-style).
hits = [h for h in glob.glob("/kaggle/input/**/detect/train.py", recursive=True)
        if "tracking_repo" not in h]
assert hits, "ctrack-src not found under /kaggle/input"
root = Path(hits[0]).parents[1]
pkg = WORK / "src" / "ctrack"
if pkg.exists():
    shutil.rmtree(pkg)
shutil.copytree(root, pkg)
(pkg / "__init__.py").touch()
sys.path.insert(0, str(WORK / "src"))
import inspect
import ctrack.detect.train as T
sig = inspect.signature(T.train_fold)
for k in ("resume", "pseudo_dir", "seed", "iso_mode"):
    assert k in sig.parameters, f"STALE ctrack-src: '{k}' missing -- version the dataset first"
print("ctrack loaded from", root, flush=True)

# Teacher peaks.
peak_files = glob.glob("/kaggle/input/**/biohub-teacher-peaks/**/*.npz", recursive=True) or \
             glob.glob("/kaggle/input/**/teacher*peaks*/**/*.npz", recursive=True) or \
             [p for p in glob.glob("/kaggle/input/**/*.npz", recursive=True)
              if ("44b6_" in Path(p).name or "6bba_" in Path(p).name)]
assert peak_files, "no teacher-peaks npz found"
PEAKS = Path(peak_files[0]).parent
print(f"teacher peaks: {PEAKS} ({len(list(PEAKS.glob('*.npz')))} volumes)", flush=True)

# Resume seed: prior version's checkpoint, if attached.
(WORK / "models").mkdir(parents=True, exist_ok=True)
prev = sorted(glob.glob("/kaggle/input/**/checkpoint_last_fold0.pt", recursive=True))
if prev:
    shutil.copy(prev[0], WORK / "models" / "checkpoint_last_fold0.pt")
    print("RESUME SEED: copied", prev[0], flush=True)
else:
    print("fresh run (no prior checkpoint attached)", flush=True)
'''

TRAIN_CELL_TMPL = r'''# ---------------------------------------------------------------- train ({name})
from ctrack.detect.train import train_fold

best = train_fold(
    str(TRAIN_DIR), str(TRAIN_DIR),
    fold=0, k=5,
    epochs={epochs},
    batch=8,
    base=32,
    lr=1e-3,
    train_stride={train_stride},
    val_stride={val_stride},
    pos_weight=4.0,
    out_dir=str(WORK / "models"),
    limit={limit},
    num_workers=2,
    log_every=50,
    max_minutes={max_minutes},
    early_stop_patience=8,
    resume="auto",
    pseudo_dir=str(PEAKS),
    pseudo_weight=0.3,
    seed={seed},
    iso_mode="{iso_mode}",
)
print("BEST SEL:", best, flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="smoke")
    ap.add_argument("--resume-from", help="kernel slug whose output holds checkpoint_last")
    args = ap.parse_args()

    seed, iso_mode, epochs, limit, max_minutes, tr_stride, va_stride = VARIANTS[args.variant]
    name = f"dettrain-{args.variant}"
    out_dir = REPO / "kernels" / name.replace("-", "_")
    train_cell = TRAIN_CELL_TMPL.format(
        name=name, epochs=epochs, limit=limit, max_minutes=max_minutes,
        train_stride=tr_stride, val_stride=va_stride, seed=seed, iso_mode=iso_mode)

    nb = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": c.splitlines(keepends=True)}
            for c in (DEPS_CELL, BOOT_CELL, train_cell)
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name.replace('-', '_')}.ipynb").write_text(json.dumps(nb))
    meta = {
        "id": f"maruanbottoni/biohub-{name}",
        "title": f"biohub-{name}",
        "code_file": f"{name.replace('-', '_')}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["maruanbottoni/ctrack-src",
                            "maruanbottoni/biohub-teacher-peaks",
                            "pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "kernel_sources": ([args.resume_from] if args.resume_from else []),
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {out_dir}\npush: .venv/bin/kaggle kernels push -p {out_dir}")


if __name__ == "__main__":
    main()
