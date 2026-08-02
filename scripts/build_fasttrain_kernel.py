#!/usr/bin/env python3
"""Smoke the fast-iteration training ideas (STRATEGY 12.4 rows 1, 2, 5 + cudnn).

Measured baseline (12.3): 3.90 s/iter on a T4, 5.34 h/epoch over 199 volumes,
so 100 epochs = 534 GPU-h against ~240 h of remaining quota. This kernel tests
the cheap half of the speedup plan and reports the speedup against that
baseline in the same units.

Ideas under test:
  1  warm-start the FULL model from the 400-epoch checkpoint
  2  freeze the UNet, train only the edge transformer
  5  batch 8 instead of 4
  -  cudnn.benchmark (shapes are a fixed 64^3, so autotuning pays back at once)

AMP (12.4 row 3) is deliberately NOT in this kernel. It needs surgery inside
train_epoch, and keeping one smoke to one class of change is what let run 1
identify its failure in eight seconds.

Two traps this guards against:

  * `--unet-weights` is NOT a warm start for our purpose. It does
    `unet.load_state_dict(state, strict=False)`, but edge_predictor_best.pth is
    a whole-model dict whose keys are prefixed `unet.` / `node_transformer.`,
    so nearly nothing would match and training would silently begin from random
    init. The patch below loads the full model and PRINTS the matched-key count
    so a silent no-op is impossible to miss -- the same failure mode as the
    "50ep" slug in 11.5.
  * freezing must reach the optimizer. Setting requires_grad=False while
    AdamW still receives model.parameters() would train nothing and look fine.
    The patch filters the optimizer and prints trainable vs total parameters.

Run: python scripts/build_fasttrain_kernel.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/trainsmoke/trainsmoke.ipynb"
OUT_DIR = REPO / "kernels/fasttrain"

ANCHOR_MODEL = """    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)"""

PATCH_MODEL = ANCHOR_MODEL + '''

    # ---- STRATEGY 12.4 idea 1: warm-start the FULL model -------------------
    # edge_predictor_best.pth holds the whole UNetNodeTransformer state dict,
    # not just UNet weights, so --unet-weights cannot load it correctly.
    import os as _os
    _init = _os.environ.get("BIOHUB_INIT_CKPT", "").strip()
    if _init:
        _sd = torch.load(_init, map_location="cpu", weights_only=True)
        _sd = {k.replace("unet.module.", "unet.", 1): v for k, v in _sd.items()}
        _own = model.state_dict()
        _use = {k: v for k, v in _sd.items() if k in _own and _own[k].shape == v.shape}
        _missing = [k for k in _own if k not in _use]
        model.load_state_dict(_use, strict=False)
        print(f"  WARM START from {_init}", flush=True)
        print(f"  matched {len(_use)}/{len(_own)} tensors; {len(_missing)} left at init",
              flush=True)
        if len(_use) == 0:
            raise RuntimeError("warm start matched zero tensors -- refusing to pretend")
        if _missing:
            print(f"  first unmatched: {_missing[:5]}", flush=True)

    # ---- idea 2: freeze the UNet, train only the edge transformer ----------
    # Detection recall is already 97.5% and saturated; edge_J is the bottleneck
    # and ~99% of the score. Freezing removes the volumetric backward pass.
    if _os.environ.get("BIOHUB_FREEZE_UNET", "0") != "0":
        for _p in model.unet.parameters():
            _p.requires_grad = False
        model.unet.eval()
        print("  UNet FROZEN (training the edge transformer only)", flush=True)

    _tot = sum(p.numel() for p in model.parameters())
    _trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {_trn:,} trainable / {_tot:,} total ({_trn/_tot:.1%})", flush=True)

    torch.backends.cudnn.benchmark = True   # fixed 64^3 shapes'''

ANCHOR_OPT = "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)"
PATCH_OPT = (
    "    # Frozen parameters must not reach the optimizer, or freezing is a no-op.\n"
    "    _trainable = [p for p in model.parameters() if p.requires_grad]\n"
    "    if not _trainable:\n"
    '        raise RuntimeError("no trainable parameters")\n'
    "    optimizer = torch.optim.AdamW(_trainable, lr=lr)"
)

SMOKE_CELL = r'''# --------------------------------------------- fast-iteration training smoke
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ITERS, BATCH = 24, 8          # 24 iters so steady state dominates CUDA warmup
SPLIT_TRAIN, SPLIT_VAL = 6, 2
BASELINE_S_PER_ITER = 3.90    # measured, batch 4, window 2 (STRATEGY 12.3)

TRAIN_DIR = COMP_DIR / "train"
_w = REPO_DIR / "weights"
if _w.is_symlink():
    _t = _w.resolve(); _w.unlink(); shutil.copytree(_t, _w)
print("weights writable:", _w, flush=True)

PRETRAINED = _w / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
print("pretrained 400ep checkpoint:", PRETRAINED, PRETRAINED.exists(), flush=True)

# ---- patch the trainer -----------------------------------------------------
ts = REPO_DIR / "scripts" / "train_unet_transformer.py"
src = ts.read_text()
for name, old, new in PATCHES:
    if src.count(old) != 1:
        raise RuntimeError(f"patch {name}: expected 1 match, found {src.count(old)}")
    src = src.replace(old, new, 1)
compile(src, str(ts), "exec")
ts.write_text(src)
print("trainer patched: warm-start + freeze + cudnn.benchmark + optimizer filter", flush=True)

stems = sorted({p.name[:-5] for p in TRAIN_DIR.glob("*.zarr")} &
               {p.name[:-5] for p in TRAIN_DIR.glob("*.geff")})
splits_path = REPO_DIR / "fast_splits.json"
splits_path.write_text(json.dumps(
    [{"split": 0, "train": stems[:SPLIT_TRAIN], "test": stems[SPLIT_TRAIN:SPLIT_TRAIN+SPLIT_VAL]}]))


def run(method, env_extra, label):
    env = {**os.environ, "PYTHONPATH": "src", **env_extra}
    cmd = [sys.executable, "scripts/train_unet_transformer.py",
           "--data-dir", str(TRAIN_DIR), "--splits", splits_path.name, "--split", "0",
           "--method", method, "--epochs", "1", "--max-iters", str(ITERS),
           "--batch-size", str(BATCH), "--num-workers", "2", "--window-size", "2"]
    print("\n" + "=" * 72 + f"\n{label}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_DIR, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    for line in p.stdout.splitlines():
        if any(k in line for k in ("WARM START", "matched", "FROZEN", "parameters:",
                                   "Epoch", "unmatched", "max_nodes")):
            print("   " + line.strip(), flush=True)
    if p.returncode != 0:
        print("--- STDERR ---\n" + p.stderr[-2500:], flush=True)
    m = re.search(r"train=([\d.]+)s", p.stdout)
    acc = re.search(r"acc=([\d.]+)", p.stdout)
    rec = re.search(r"recall=([\d.]+)", p.stdout)
    return dict(rc=p.returncode, wall=wall,
                train_s=float(m.group(1)) if m else float("nan"),
                acc=float(acc.group(1)) if acc else float("nan"),
                recall=float(rec.group(1)) if rec else float("nan"))


runs = {}
runs["A cold, batch8"]        = run("fast_a", {}, "A: no warm start, UNet trainable (control)")
runs["B warm, batch8"]        = run("fast_b", {"BIOHUB_INIT_CKPT": str(PRETRAINED)},
                                    "B: + warm start (idea 1)")
runs["C warm+frozen, batch8"] = run("fast_c", {"BIOHUB_INIT_CKPT": str(PRETRAINED),
                                               "BIOHUB_FREEZE_UNET": "1"},
                                    "C: + frozen UNet (ideas 1+2)")

print("\n" + "=" * 72 + "\nRESULT vs the 3.90 s/iter batch-4 baseline (STRATEGY 12.3)\n")
N_VOL, T = len(stems), 100
print(f"{'run':<24}{'rc':>4}{'s/iter':>9}{'h/epoch':>10}{'speedup':>10}{'val_acc':>10}{'val_recall':>12}")
for name, r in runs.items():
    spi = r["train_s"] / ITERS
    h_ep = spi * (N_VOL * (T - 1) / BATCH) / 3600
    base_h = BASELINE_S_PER_ITER * (N_VOL * (T - 1) / 4) / 3600
    print(f"{name:<24}{r['rc']:>4}{spi:>9.2f}{h_ep:>10.2f}{base_h/h_ep:>9.2f}x"
          f"{r['acc']:>10.4f}{r['recall']:>12.4f}")

warm_ok = runs["B warm, batch8"]["recall"] > 0
print(f"\nearly-stop metric non-zero under warm start: {warm_ok}"
      "   <- 12.3 left this untested")
print("SMOKE RESULT:", "PASS" if all(r["rc"] == 0 for r in runs.values()) and warm_ok
      else "FAIL", flush=True)
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    nb["cells"] = nb["cells"][:8]                     # setup cells only
    patch_defs = (
        "PATCHES = [\n"
        f"    ('model', {ANCHOR_MODEL!r}, {PATCH_MODEL!r}),\n"
        f"    ('optimizer', {ANCHOR_OPT!r}, {PATCH_OPT!r}),\n"
        "]\n\n"
    )
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": (patch_defs + SMOKE_CELL).splitlines(keepends=True)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fasttrain.ipynb").write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "maruanbottoni/biohub-fast-train-smoke",
        "title": "Biohub fast train smoke",
        "code_file": "fasttrain.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
    }, indent=2))
    print(f"wrote {OUT_DIR}/fasttrain.ipynb")


if __name__ == "__main__":
    main()
