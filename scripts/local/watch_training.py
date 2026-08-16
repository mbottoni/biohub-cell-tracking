#!/usr/bin/env python3
"""WATCH THE TRAINING — losses, selection score, early-stop status, ETA.

    .venv/bin/python scripts/local/watch_training.py            # one snapshot
    .venv/bin/python scripts/local/watch_training.py --watch    # refresh forever

WHY IT WORKS THIS WAY. The trainer prints its per-epoch losses to
`/content/train.log` inside the Colab VM, and that is NOT reachable: row 55
established that the ssh/exec channel dies at 60-90 minutes while the runtime
keeps training happily for hours. Sessions were declared dead (and once
stopped!) while they were healthy, purely because the tunnel had dropped.

So the loss curve is carried OUT inside the checkpoint itself. The patched
trainer appends a row to `history` every epoch and `torch.save`s it into
`checkpoint_last.pth`, which the runner pushes to a Kaggle dataset. The Kaggle
API is the one channel that has never failed us, and because the history is
restored on resume the curve stays continuous across any number of dead
sessions. This script just pulls that dataset and renders it.

⚠ `score` (= test_acc x test_recall) is the SELECTION metric on the 12 selection
volumes and is what early stopping watches. It is NOT the competition score and
NOT comparable to the 0.9135 line. The honest number comes later, from scoring
the HOLDOUT (evalset-24, which this model never sees) with the official scorer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KAGGLE = REPO / ".venv/bin/kaggle"
PY = REPO / ".venv/bin/python"

SPARK = "▁▂▃▄▅▆▇█"


def spark(vals: list[float], width: int = 48) -> str:
    if not vals:
        return ""
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return SPARK[0] * len(v)
    return "".join(SPARK[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v)


def fetch(ckpt_ds: str, dest: Path) -> dict | None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(KAGGLE), "datasets", "download", ckpt_ds,
                        "-p", str(dest), "--unzip"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  (no checkpoint dataset yet: {r.stderr.strip().splitlines()[-1:] or ''})")
        return None
    ck = dest / "checkpoint_last.pth"
    if not ck.exists():
        cands = list(dest.rglob("checkpoint_last.pth"))
        if not cands:
            print("  (dataset present but no checkpoint_last.pth yet)")
            return None
        ck = cands[0]
    out = subprocess.run(
        [str(PY), "-c",
         "import torch,json,sys;"
         "ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
         "print(json.dumps({'epoch':int(ck.get('epoch',-1)),"
         "'best_score':float(ck.get('best_score',0)),"
         "'since_best':int(ck.get('since_best',0)),"
         "'history':ck.get('history',[])}))", str(ck)],
        capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-500:])
        return None
    return json.loads(out.stdout)


def render(d: dict, patience: int, target: int) -> None:
    h = d.get("history", [])
    print("=" * 76)
    print(f" FULL-RECIPE TRAINING   epoch {d['epoch']}"
          f"{f' / target {target}' if target else ''}"
          f"   best_sel {d['best_score']:.4f}   since_best {d['since_best']}/{patience}")
    print("=" * 76)
    if not h:
        print(" no history rows yet — the first epoch has not been banked.")
        print(" (a checkpoint appears only after epoch 0 completes)")
        return

    print(f"\n {'ep':>4} {'edge':>8} {'det':>8} {'test':>8} {'acc':>7} {'recall':>7}"
          f" {'sel':>7} {'':1} {'train':>7} {'test':>6}")
    print(" " + "-" * 72)
    for r in h[-12:]:
        star = "*" if r["since_best"] == 0 else " "
        print(f" {r['epoch']:>4} {r['edge_loss']:>8.4f} {r['det_loss']:>8.4f}"
              f" {r['test_loss']:>8.4f} {r['test_acc']:>7.4f} {r['test_recall']:>7.4f}"
              f" {r['score']:>7.4f} {star} {r['train_s']:>6.0f}s {r['test_s']:>5.0f}s")

    print(f"\n edge_loss  {spark([r['edge_loss'] for r in h])}"
          f"  {h[0]['edge_loss']:.4f} -> {h[-1]['edge_loss']:.4f}")
    print(f" det_loss   {spark([r['det_loss'] for r in h])}"
          f"  {h[0]['det_loss']:.4f} -> {h[-1]['det_loss']:.4f}")
    print(f" test_loss  {spark([r['test_loss'] for r in h])}"
          f"  {h[0]['test_loss']:.4f} -> {h[-1]['test_loss']:.4f}")
    print(f" sel score  {spark([r['score'] for r in h])}"
          f"  {h[0]['score']:.4f} -> {h[-1]['score']:.4f}")

    ep_s = sum(r["train_s"] + r["test_s"] for r in h[-5:]) / len(h[-5:])
    print(f"\n epoch cost {ep_s / 60:.1f} min  ({ep_s:.0f}s)   "
          f"A100 units/epoch ~{ep_s / 3600 * 11.8:.2f}")
    if target and d["epoch"] < target:
        left = target - d["epoch"]
        print(f" to target  {left} epochs ~ {left * ep_s / 3600:.1f} h "
              f"~ {left * ep_s / 3600 * 11.8:.0f} units")
    # Early-stop read: is the selection score still moving, or are we paying
    # for noise? This is the decision the whole budget hinges on.
    if len(h) >= 6:
        recent = [r["score"] for r in h[-5:]]
        prev = [r["score"] for r in h[-10:-5]] or recent
        delta = sum(recent) / len(recent) - sum(prev) / len(prev)
        verdict = ("still climbing" if delta > 0.002 else
                   "flattening" if delta > 0.0005 else "PLATEAU — consider stopping")
        print(f" sel trend  last-5 mean vs previous-5: {delta:+.4f}  => {verdict}")
    print(f" stop rule  early stop fires at since_best={patience} "
          f"(now {d['since_best']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-ds", default="maruanbottoni/biohub-fullrecipe-ckpt")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--target", type=int, default=0, help="TARGET_EPOCH, for ETA")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=300, help="seconds between polls")
    args = ap.parse_args()

    dest = Path("/Volumes/extSSD/kaggle/biohub/fullrecipe_logs/_ckpt_peek")
    while True:
        print("\033[2J\033[H" if args.watch else "", end="")
        print(f"[watch {time.strftime('%H:%M:%S')}] {args.ckpt_ds}")
        d = fetch(args.ckpt_ds, dest)
        if d:
            render(d, args.patience, args.target)
        if not args.watch:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
