#!/usr/bin/env python3
"""Smoke-test the training loop on Kaggle before committing GPU-days.

Training our own model is the last lever standing (STRATEGY 11.5/11.6), but it
is expensive: the shipped history.csv records ~1217 s/epoch and 400 epochs,
i.e. ~138 GPU-hours on the author's hardware, against a Kaggle quota of roughly
30 GPU-h/week and a 12 h session cap. Before spending any of that we need five
things confirmed, and one number measured.

Confirm:
  1. the competition `train` mount carries both .zarr and .geff per volume;
  2. the training loop runs at all on a T4;
  3. a checkpoint is written, with its config.json;
  4. that checkpoint reloads through the *inference* loader -- the thing that
     actually has to consume it;
  5. the diverse config (window-size 3) constructs and trains, not just the
     baseline. window 3 is the interesting one: it changes temporal context,
     which targets edge_J, and edge_J is ~99% of the score.

Measure:
  seconds/iteration on a T4, so the full run can be sized honestly rather than
  extrapolated from someone else's hardware.

Built off d4edge: cells 0-7 (config, offline wheels, repo materialisation,
weights) are reused verbatim so the environment matches our inference runs.
Everything downstream of that is replaced by the smoke cell.

Note on window-size 3: the dual-seed blender rejects a secondary whose
window/downsample differs from the primary ("incompatible inference grids"), so
a window-3 model cannot drop into the existing secondary slot. It would run as
its own pipeline. That is a deliberate trade -- more diversity, more plumbing.

Run: python scripts/build_trainsmoke_kernel.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/d4edge/d4edge.ipynb"
OUT_DIR = REPO / "kernels/trainsmoke"

SMOKE_CELL = r'''# ---------------------------------------------------------------- smoke test
import json, os, re, subprocess, sys, time
from pathlib import Path

ITERS = 8          # iterations per config -- enough to time, cheap enough to be safe
BATCH = 4
SPLIT_TRAIN, SPLIT_VAL = 6, 2

TRAIN_DIR = COMP_DIR / "train"
print("TRAIN_DIR:", TRAIN_DIR, "exists:", TRAIN_DIR.exists(), flush=True)
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"no train mount at {TRAIN_DIR}")

# Smoke run 1 died here: dataspec sets WEIGHTS_PATH = <repo>/weights, and the
# setup cell materialises that as a SYMLINK into the read-only /kaggle/input
# mount (link_or_copy_tree prefers os.symlink). Training then cannot write its
# checkpoint -- "[Errno 30] Read-only file system". Replace the symlink with a
# real copy so the directory is writable while still holding the pretrained
# weights that --unet-weights warm-starts from.
import shutil
_w = REPO_DIR / "weights"
if _w.is_symlink():
    _target = _w.resolve()
    print(f"weights/ is a symlink -> {_target}; copying to make it writable", flush=True)
    _w.unlink()
    shutil.copytree(_target, _w)
probe = _w / ".write_probe"
probe.write_text("ok")
probe.unlink()
print(f"weights dir writable: {_w}", flush=True)

zarrs = {p.name[:-5] for p in TRAIN_DIR.glob("*.zarr")}
geffs = {p.name[:-5] for p in TRAIN_DIR.glob("*.geff")}
stems = sorted(zarrs & geffs)
print(f"train volumes: {len(zarrs)} zarr, {len(geffs)} geff, {len(stems)} usable", flush=True)
if len(stems) < SPLIT_TRAIN + SPLIT_VAL:
    raise RuntimeError(f"only {len(stems)} usable train volumes")

splits_path = REPO_DIR / "smoke_splits.json"
splits_path.write_text(json.dumps(
    [{"split": 0,
      "train": stems[:SPLIT_TRAIN],
      "test": stems[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_VAL]}], indent=2))
print("splits:", splits_path, flush=True)


def smoke(method, extra):
    """Run one tiny training job; return (rc, wall_s, train_s, max_nodes)."""
    cmd = [sys.executable, "scripts/train_unet_transformer.py",
           "--data-dir", str(TRAIN_DIR), "--splits", splits_path.name, "--split", "0",
           "--method", method, "--epochs", "1", "--max-iters", str(ITERS),
           "--batch-size", str(BATCH), "--num-workers", "2"] + extra
    print("\n" + "=" * 70 + f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"},
                          capture_output=True, text=True)
    wall = time.time() - t0
    tail = proc.stdout.strip().splitlines()[-25:]
    print("\n".join(tail), flush=True)
    if proc.returncode != 0:
        print("--- STDERR ---\n" + proc.stderr[-3000:], flush=True)
    m = re.search(r"train=([\d.]+)s", proc.stdout)
    n = re.search(r"max_nodes=(\d+)", proc.stdout)
    return (proc.returncode,
            wall,
            float(m.group(1)) if m else float("nan"),
            int(n.group(1)) if n else -1)


results = {}
results["window2 (blendable)"] = smoke("smoke_w2", ["--window-size", "2"])
results["window3 (diverse)"]   = smoke("smoke_w3", ["--window-size", "3"])

# ------------------------------------------------------------- verification
print("\n" + "=" * 70 + "\nCHECKPOINT VERIFICATION", flush=True)
sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(REPO_DIR / "src"))
ok = {}
for method in ("smoke_w2", "smoke_w3"):
    d = REPO_DIR / "weights" / method / "split_0"
    ckpt, cfg = d / "edge_predictor_best.pth", d / "config.json"
    print(f"\n{method}: {d}")
    print(f"   checkpoint {ckpt.exists()}  ({ckpt.stat().st_size/1e6:.1f} MB)" if ckpt.exists()
          else "   checkpoint MISSING")
    if cfg.exists():
        print("   config:", json.loads(cfg.read_text()))
    # The real test: does the INFERENCE loader accept what training produced?
    try:
        from predict_unet_transformer import load_model
        import torch
        model, ws, ds = load_model(ckpt, torch.device("cuda"))
        print(f"   inference load_model OK -> window_size={ws} downsample={ds}")
        ok[method] = True
    except Exception as exc:
        print(f"   inference load_model FAILED: {type(exc).__name__}: {exc}")
        ok[method] = False

# ------------------------------------------------------------------ costing
print("\n" + "=" * 70 + "\nCOST PROJECTION (this T4 session)", flush=True)
# Volumes are (T=100, ...), so a stride-1 window pass yields ~T-(W-1) windows.
N_VOL, T = len(stems), 100
print(f"{'config':<22}{'rc':>4}{'wall_s':>9}{'train_s':>9}{'s/iter':>9}"
      f"{'proj h/epoch':>14}{'proj h @100ep':>15}")
for name, (rc, wall, train_s, _mn) in results.items():
    w = 3 if "3" in name else 2
    per_iter = train_s / ITERS if train_s == train_s else float("nan")
    iters_epoch = N_VOL * (T - (w - 1)) / BATCH
    h_epoch = per_iter * iters_epoch / 3600
    print(f"{name:<22}{rc:>4}{wall:>9.0f}{train_s:>9.1f}{per_iter:>9.2f}"
          f"{h_epoch:>14.2f}{h_epoch*100:>15.0f}")
print(f"\n(projection assumes the full {N_VOL}-volume train set, batch {BATCH}, "
      f"~{T} frames/volume, and that s/iter holds at scale)")
print("Kaggle budget: ~30 GPU-h/week, 12 h max per session.")
print("\nSMOKE RESULT:", "PASS" if all(r[0] == 0 for r in results.values()) and all(ok.values())
      else "FAIL -- do not launch the full run", flush=True)
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    # Keep cells 0-7: config, offline deps, repo + weights materialisation.
    nb["cells"] = nb["cells"][:8]
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": SMOKE_CELL.splitlines(keepends=True)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "trainsmoke.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "maruanbottoni/biohub-train-smoke",
                "title": "Biohub train smoke",
                "code_file": "trainsmoke.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": False,
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "dataset_sources": [
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_DIR}/trainsmoke.ipynb  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
