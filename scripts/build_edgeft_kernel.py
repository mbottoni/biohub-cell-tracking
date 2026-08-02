#!/usr/bin/env python3
"""Fine-tune ONLY the edge transformer -- the component that decides 99% of the score.

Rationale (STRATEGY 11.2, 12.1, 12.6):
  * edge_J ~= 0.909 is ~99% of our score, and it is frozen at that value across
    the entire public field because everyone inherits the SAME edge head.
  * That head was checkpoint-selected by `score = acc * recall` with acc=0.9998,
    i.e. effectively node recall, on a split with overlap 40/40. Nobody has ever
    optimised the association head against association quality on honestly
    held-out data.
  * Freezing the UNet keeps the encoder bit-identical to the primary, so
    window_size and downsample match by construction and the result drops into
    d4edge's existing secondary slot with no plumbing. This is exactly what
    window-3 could not do.

Config corrections carried in from the fasttrain smoke (12.6):
  batch 4 (batch 8 was a 3x regression), no cudnn.benchmark, warm start ON
  (verified 136/136 tensors), freeze ON (1.25x, and best val recall of 3 arms).

Selection metric change: `acc * recall` -> `-test_loss`. Held-out loss carries
the edge term, so it stops on association quality rather than a saturated
detection proxy. `best_score` must move 0.0 -> -inf with it, or a negative
score would never beat the initial value and NO checkpoint would ever be saved.

Steps, not epochs (12.7): --max-iters defines a short epoch so validation and
early stopping fire often. SMOKE=1 runs a tiny version to prove the loop.

Run: python scripts/build_edgeft_kernel.py [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kernels/trainsmoke/trainsmoke.ipynb"
SPLIT = REPO / "configs/honest_split.json"

# --- patch 1: warm start + freeze (no cudnn.benchmark this time) -------------
ANCHOR_MODEL = """    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)"""

PATCH_MODEL = ANCHOR_MODEL + '''

    import os as _os
    _init = _os.environ.get("BIOHUB_INIT_CKPT", "").strip()
    if _init:
        _sd = torch.load(_init, map_location="cpu", weights_only=True)
        _sd = {k.replace("unet.module.", "unet.", 1): v for k, v in _sd.items()}
        _own = model.state_dict()
        _use = {k: v for k, v in _sd.items() if k in _own and _own[k].shape == v.shape}
        model.load_state_dict(_use, strict=False)
        print(f"  WARM START matched {len(_use)}/{len(_own)} tensors", flush=True)
        if len(_use) == 0:
            raise RuntimeError("warm start matched zero tensors -- refusing to pretend")

    if _os.environ.get("BIOHUB_FREEZE_UNET", "0") != "0":
        for _p in model.unet.parameters():
            _p.requires_grad = False
        model.unet.eval()
        print("  UNet FROZEN (training the edge transformer only)", flush=True)
    _tot = sum(p.numel() for p in model.parameters())
    _trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {_trn:,} trainable / {_tot:,} total ({_trn/_tot:.1%})", flush=True)'''

ANCHOR_OPT = "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)"
PATCH_OPT = (
    "    _trainable = [p for p in model.parameters() if p.requires_grad]\n"
    "    if not _trainable:\n"
    '        raise RuntimeError("no trainable parameters")\n'
    "    optimizer = torch.optim.AdamW(_trainable, lr=lr)"
)

# --- patch 2: select on held-out loss, not the saturated recall proxy --------
ANCHOR_INIT = "    best_score = 0.0\n"
PATCH_INIT = (
    "    # -test_loss is negative, so 0.0 would never be beaten and nothing\n"
    "    # would ever be checkpointed. Must be -inf.\n"
    "    best_score = float('-inf')\n"
)

ANCHOR_SCORE = "        score = test_acc * test_recall\n"
PATCH_SCORE = (
    "        # acc sits at 0.9998, so acc*recall is effectively node recall --\n"
    "        # a saturated DETECTION proxy, while edge_J is the bottleneck.\n"
    "        # Held-out loss carries the edge term. Higher-is-better => negate.\n"
    "        if os.environ.get('BIOHUB_SELECT_ON_LOSS', '0') != '0':\n"
    "            score = -test_loss\n"
    "        else:\n"
    "            score = test_acc * test_recall\n"
)

CELL = r'''# ------------------------------- edge-transformer fine-tune (STRATEGY 12.7)
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

SMOKE = __SMOKE__
MAX_ITERS = 20 if SMOKE else 500      # steps per short epoch
EPOCHS    = 2  if SMOKE else 20       # => 40 steps (smoke) / 10,000 steps (full)
BATCH     = 4                         # batch 8 was a 3x regression (12.6)
LR        = 3e-4                      # head-only fine-tune, above the 1e-4 used for the full model

TRAIN_DIR = COMP_DIR / "train"
_w = REPO_DIR / "weights"
if _w.is_symlink():
    _t = _w.resolve(); _w.unlink(); shutil.copytree(_t, _w)
PRETRAINED = _w / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
assert PRETRAINED.exists(), PRETRAINED

ts = REPO_DIR / "scripts" / "train_unet_transformer.py"
src = ts.read_text()
for name, old, new in PATCHES:
    if src.count(old) != 1:
        raise RuntimeError(f"patch {name}: expected 1 match, found {src.count(old)}")
    src = src.replace(old, new, 1)
compile(src, str(ts), "exec")
ts.write_text(src)
print("trainer patched: warm-start + freeze + optimizer filter + loss-based selection", flush=True)

# Honest split: 169 train / 30 holdout, overlap 0, stratified by embryo x density.
# The 4 dev-test volumes are in the holdout, so validation tracks what the LB scores.
splits_path = REPO_DIR / "honest_split.json"
splits_path.write_text(json.dumps(HONEST_SPLIT))
avail = {p.name[:-5] for p in TRAIN_DIR.glob("*.zarr")} & {p.name[:-5] for p in TRAIN_DIR.glob("*.geff")}
tr = [s for s in HONEST_SPLIT[0]["train"] if s in avail]
te = [s for s in HONEST_SPLIT[0]["test"]  if s in avail]
print(f"split: {len(tr)} train / {len(te)} holdout (overlap {len(set(tr)&set(te))})", flush=True)
if SMOKE:                                   # keep the smoke cheap
    tr, te = tr[:6], te[:2]
    splits_path.write_text(json.dumps([{"split": 0, "train": tr, "test": te}]))
    print(f"SMOKE: reduced to {len(tr)}/{len(te)}", flush=True)

cmd = [sys.executable, "scripts/train_unet_transformer.py",
       "--data-dir", str(TRAIN_DIR), "--splits", splits_path.name, "--split", "0",
       "--method", "edge_ft", "--epochs", str(EPOCHS), "--max-iters", str(MAX_ITERS),
       "--batch-size", str(BATCH), "--num-workers", "2", "--window-size", "2",
       "--lr", str(LR)]
env = {**os.environ, "PYTHONPATH": "src",
       "BIOHUB_INIT_CKPT": str(PRETRAINED),
       "BIOHUB_FREEZE_UNET": "1",
       "BIOHUB_SELECT_ON_LOSS": "1"}
print("$ " + " ".join(cmd), flush=True)

t0 = time.time()
p = subprocess.run(cmd, cwd=REPO_DIR, env=env, capture_output=True, text=True)
wall = time.time() - t0
for line in p.stdout.splitlines():
    if any(k in line for k in ("WARM START", "FROZEN", "parameters:", "Epoch", "Best score",
                               "max_nodes", "Starting training")):
        print("   " + line.strip(), flush=True)
if p.returncode != 0:
    print("--- STDERR ---\n" + p.stderr[-3000:], flush=True)

print(f"\nexit={p.returncode}  wall={wall/60:.1f} min", flush=True)

out = REPO_DIR / "weights" / "edge_ft" / "split_0"
ck, cf = out / "edge_predictor_best.pth", out / "config.json"
print(f"checkpoint: {ck.exists()}  config: {cf.exists()}", flush=True)

# The checkpoint must (a) load in the inference path and (b) still declare the
# SAME grid as the primary, or it cannot go in d4edge's secondary slot.
sys.path.insert(0, str(REPO_DIR / "scripts")); sys.path.insert(0, str(REPO_DIR / "src"))
ok = False
if ck.exists():
    shutil.copy(ck, "/kaggle/working/edge_ft_best.pth")
    shutil.copy(cf, "/kaggle/working/edge_ft_config.json")
    from predict_unet_transformer import load_model
    import torch
    m, ws, ds = load_model(ck, torch.device("cuda"))
    pm, pws, pds = load_model(PRETRAINED, torch.device("cuda"))
    ok = (ws, ds) == (pws, pds)
    print(f"   inference load OK -> window={ws} downsample={ds}", flush=True)
    print(f"   primary grid       -> window={pws} downsample={pds}", flush=True)
    print(f"   BLENDABLE into d4edge secondary slot: {ok}", flush=True)

# Steps-per-epoch and cost, so the full run can be sized from this run.
m_tr = re.findall(r"train=([\d.]+)s", p.stdout)
if m_tr:
    spi = float(m_tr[-1]) / MAX_ITERS
    print(f"\n{spi:.2f} s/iter (batch {BATCH}); baseline 3.90, fasttrain batch8 22.23")
    print(f"full run {20*500:,} steps => {spi*20*500/3600:.1f} GPU-h")
print("RESULT:", "PASS" if p.returncode == 0 and ok else "FAIL", flush=True)
'''


def main() -> None:
    smoke = "--smoke" in sys.argv
    nb = json.loads(BASE_NB.read_text())
    nb["cells"] = nb["cells"][:8]

    # Cell 7 ends with the dual-seed *inference* setup, which calls
    # _find_secondary_artifact_root() at module level and raises if the
    # seed314159 dataset is absent. Training never touches that checkpoint, and
    # copying it cost ~3.7 min of every fasttrain run (~80% of fixed overhead).
    # Truncate the cell there so the dataset can be dropped entirely.
    cell7 = "".join(nb["cells"][7]["source"])
    marker = "SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()"
    if cell7.count(marker) != 1:
        raise RuntimeError(f"secondary-setup marker count = {cell7.count(marker)}")
    cell7 = cell7[: cell7.index(marker)].rstrip() + "\nprint('setup done (secondary seed skipped: training does not use it)')\n"
    compile(cell7, "<cell7>", "exec")
    nb["cells"][7]["source"] = cell7.splitlines(keepends=True)

    header = (
        "PATCHES = [\n"
        f"    ('model', {ANCHOR_MODEL!r}, {PATCH_MODEL!r}),\n"
        f"    ('optimizer', {ANCHOR_OPT!r}, {PATCH_OPT!r}),\n"
        f"    ('best_init', {ANCHOR_INIT!r}, {PATCH_INIT!r}),\n"
        f"    ('selection', {ANCHOR_SCORE!r}, {PATCH_SCORE!r}),\n"
        "]\n"
        f"HONEST_SPLIT = {json.loads(SPLIT.read_text())!r}\n\n"
    )
    body = CELL.replace("__SMOKE__", "True" if smoke else "False")
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": (header + body).splitlines(keepends=True)})

    name = "edgeft_smoke" if smoke else "edgeft"
    out = REPO / "kernels" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"maruanbottoni/biohub-{'edge-ft-smoke' if smoke else 'edge-ft'}",
        "title": f"Biohub edge ft{' smoke' if smoke else ''}",
        "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        # NOTE: the secondary seed dataset is deliberately absent -- training
        # never uses it, and copying it cost ~3.7 min of every fasttrain run.
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
    }, indent=2))
    print(f"wrote kernels/{name}/{name}.ipynb  (smoke={smoke})")


if __name__ == "__main__":
    main()
