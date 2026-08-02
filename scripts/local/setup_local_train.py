#!/usr/bin/env python3
"""Materialise a locally-runnable copy of the trainer for M1/MPS debugging.

Why bother when the T4 is ~14x faster (STRATEGY 12.9): iteration latency, not
throughput. A Kaggle cycle is push -> queue -> ~50 s setup -> read captured
logs, several minutes per attempt, and the logs are filtered. Locally the loop
is seconds, the traceback is immediate, and the trainer prints a per-phase
breakdown we never see on Kaggle:

    [timing] data: 11.6s (22%) | forward: 29.7s (56%) | backward: 12.0s (23%)

Two bugs this session -- the `os` NameError in a patch and the read-only
weights symlink -- would each have been caught here in seconds instead of a
~6 min kernel round trip.

Two patches are required, both recorded in STRATEGY 12.9:
  1. device selection is hardcoded `cuda`-or-`cpu`      -> add an MPS branch
  2. three unconditional `torch.cuda.synchronize()`     -> guard them

MPS caveat: conv3d BACKWARD on MPS is ~2.5x slower than CPU, while forward is
~1.5x faster. With the UNet frozen (the config we actually run) that profile
suits us; for full fine-tuning, pass --cpu.

Usage:
    python scripts/local/setup_local_train.py            # materialise + patch
    python scripts/local/setup_local_train.py --run      # + a 10-iter smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = Path(os.environ.get("BIOHUB_LOCAL_TRAIN", "/tmp/localtrain"))
DATA = Path("/Volumes/extSSD/kaggle/biohub/raw/train")   # holds .zarr AND .geff
VENV_PY = REPO / ".venv/bin/python"

DEVICE_OLD = '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
DEVICE_NEW = '''    # MPS: conv3d BACKWARD is ~2.5x slower than CPU but forward is ~1.5x
    # faster; with the UNet frozen we only need forward. BIOHUB_MPS=0 to opt out.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and os.environ.get("BIOHUB_MPS", "1") != "0":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  device: {device}", flush=True)'''


def find_source() -> Path:
    """Locate a tracking_repo copy; they arrive via `kaggle kernels output`."""
    for c in [DEST, Path("/tmp/out_len20/tracking_repo"), REPO / "kaggle_assets/tracking_repo"]:
        if (c / "scripts/train_unet_transformer.py").exists():
            return c
    raise SystemExit(
        "No tracking_repo found. Fetch one from any completed kernel:\n"
        "  .venv/bin/kaggle kernels output maruanbottoni/biohub-probe-min-track-len-20 -p /tmp/out_len20\n"
        "then re-run this script."
    )


def patch(dest: Path) -> None:
    p = dest / "scripts/train_unet_transformer.py"
    s = p.read_text()
    changed = []

    # Idempotent: detect the MPS branch by behaviour, not by exact comment text
    # (a hand-applied patch will have worded its comments differently).
    if "torch.backends.mps.is_available()" not in s:
        if s.count(DEVICE_OLD) != 1:
            raise SystemExit(f"device anchor count = {s.count(DEVICE_OLD)}")
        s = s.replace(DEVICE_OLD, DEVICE_NEW, 1)
        changed.append("device -> cuda|mps|cpu")

    bare = "torch.cuda.synchronize()"
    guarded = "(torch.cuda.synchronize() if torch.cuda.is_available() else None)"
    n = s.count(bare) - s.count(guarded)
    if n > 0:
        s = s.replace(bare, guarded)
        changed.append(f"guarded {n} cuda.synchronize()")

    if "import os" not in s.split("def ")[0]:
        s = "import os\n" + s
        changed.append("import os")

    compile(s, str(p), "exec")
    p.write_text(s)
    print("patched:", ", ".join(changed) if changed else "already patched")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run a 10-iteration smoke")
    ap.add_argument("--cpu", action="store_true", help="force CPU (MPS conv3d backward is slow)")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    src = find_source()
    if src != DEST:
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(src, DEST)
        print(f"copied {src} -> {DEST}")
    patch(DEST)

    if not args.run:
        print(f"\nready. Debug a single volume with:\n"
              f"  cd {DEST} && PYTHONPATH=src {VENV_PY} scripts/train_unet_transformer.py \\\n"
              f"    --debug-video <vol.zarr> --epochs 1 --max-iters 10 --batch-size 4 \\\n"
              f"    --num-workers 0 --window-size 2 --single-gpu")
        return

    vols = sorted(DATA.glob("*.zarr"))
    if not vols:
        raise SystemExit(f"no .zarr under {DATA}")
    env = {**os.environ, "PYTHONPATH": "src"}
    if args.cpu:
        env["BIOHUB_MPS"] = "0"
    cmd = [str(VENV_PY), "scripts/train_unet_transformer.py",
           "--debug-video", str(vols[0]), "--method", "local_dbg",
           "--epochs", "1", "--max-iters", str(args.iters), "--batch-size", "4",
           "--num-workers", "0", "--window-size", "2", "--single-gpu"]
    print("$ " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=DEST, env=env).returncode)


if __name__ == "__main__":
    main()
