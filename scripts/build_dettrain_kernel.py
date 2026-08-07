#!/usr/bin/env python3
"""Build the OWN-DETECTOR training kernel (STRATEGY 45.2 route 1).

Trains ctrack's UNet3D (base 32) on the competition train mount with:
  * dense pseudo-label supervision -- teacher peaks at weight 0.3, GT at 1.0
    (noisy-student; purity 99.90%, experiments row 36);
  * sel-metric early stopping (recall x exp(-0.1|node_ratio|), patience 8);
  * RESUMABLE checkpoints -- checkpoint_last_fold0.pt is written atomically
    every epoch into /kaggle/working (kernel output). To continue past the
    12 h session cap, push the next version with the PREVIOUS version's output
    attached as a kernel data source; the boot cell copies any
    checkpoint_last_fold0.pt it finds there into the working dir and
    train_fold(resume="auto") picks it up.

Prerequisites (do once, after the overnight teacher dump completes):
    .venv/bin/kaggle datasets version -p <ctrack-src staging dir> -m "resume + pseudo"
    .venv/bin/kaggle datasets create  -p data/teacher_peaks_upload   # biohub-teacher-peaks

Budget: plain UNet3D, ~199 volumes, batch 8 -- expect ~15-25 min/epoch on T4
(calibration cell prints the real number before committing to the loop).
max_minutes=660 keeps the session under the 12 h cap with margin for setup+val.

Run: python scripts/build_dettrain_kernel.py [--resume-from <kernel-slug>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "kernels/dettrain"

BOOT_CELL = r'''# ---------------------------------------------------------------- boot
import os, shutil, sys
from pathlib import Path

WORK = Path("/kaggle/working")
COMP = next(p for p in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
                        Path("/kaggle/input/biohub-cell-tracking-during-development")] if p.exists())
TRAIN_DIR = COMP / "train"
assert TRAIN_DIR.exists(), f"no train mount at {TRAIN_DIR}"

# ctrack package from the ctrack-src dataset (root = the package directory).
CTRACK_SRC = next(p for p in Path("/kaggle/input").glob("**/ctrack-src") if p.is_dir())
pkg = WORK / "src" / "ctrack"
if pkg.exists():
    shutil.rmtree(pkg)
shutil.copytree(CTRACK_SRC, pkg)
(pkg / "__init__.py").touch()
sys.path.insert(0, str(WORK / "src"))
import ctrack.detect.train as T  # noqa: E402
print("ctrack loaded from", pkg)
assert "resume" in T.train_fold.__kwdefaults__ or True  # presence check below
import inspect
sig = inspect.signature(T.train_fold)
assert "resume" in sig.parameters and "pseudo_dir" in sig.parameters, \
    "STALE ctrack-src: version the dataset with the resume+pseudo trainer first"

# Teacher peaks dataset.
PEAKS = next((p for p in Path("/kaggle/input").glob("**/biohub-teacher-peaks") if p.is_dir()), None)
if PEAKS is None:
    cands = list(Path("/kaggle/input").glob("**/teacher_peaks*"))
    PEAKS = cands[0] if cands else None
n_peak_files = len(list(PEAKS.glob("*.npz"))) if PEAKS else 0
print("teacher peaks:", PEAKS, f"({n_peak_files} volumes)")
assert n_peak_files > 0, "no teacher-peaks dataset attached"

# Resume: copy a prior version's checkpoint into the working dir if attached.
prev = sorted(Path("/kaggle/input").glob("**/checkpoint_last_fold0.pt"))
if prev:
    shutil.copy(prev[0], WORK / "models" / "checkpoint_last_fold0.pt") \
        if (WORK / "models").exists() or not (WORK / "models").mkdir(parents=True) else None
    print("RESUME SEED: copied", prev[0])
else:
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    print("fresh run (no prior checkpoint attached)")
'''

TRAIN_CELL = r'''# ---------------------------------------------------------------- train
from ctrack.detect.train import train_fold

best = train_fold(
    str(TRAIN_DIR), str(TRAIN_DIR),
    fold=0, k=5,
    epochs=60,
    batch=8,
    base=32,
    lr=1e-3,
    train_stride=2,
    val_stride=6,
    pos_weight=4.0,
    out_dir=str(WORK / "models"),
    num_workers=2,
    log_every=50,
    max_minutes=660,
    early_stop_patience=8,
    resume="auto",
    pseudo_dir=str(PEAKS),
    pseudo_weight=0.3,
)
print("BEST SEL:", best)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-from", help="kernel slug whose output holds checkpoint_last "
                                          "(attached as a kernel data source)")
    args = ap.parse_args()

    nb = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": BOOT_CELL.splitlines(keepends=True)},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": TRAIN_CELL.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dettrain.ipynb").write_text(json.dumps(nb))
    meta = {
        "id": "maruanbottoni/biohub-dettrain",
        "title": "biohub-dettrain",
        "code_file": "dettrain.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": ["maruanbottoni/ctrack-src",
                            "maruanbottoni/biohub-teacher-peaks"],
        "kernel_sources": ([args.resume_from] if args.resume_from else []),
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {OUT_DIR}. Push with: .venv/bin/kaggle kernels push -p {OUT_DIR}")


if __name__ == "__main__":
    main()
