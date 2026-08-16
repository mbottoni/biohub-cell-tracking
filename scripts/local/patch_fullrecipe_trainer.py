#!/usr/bin/env python3
"""Patch the full-recipe trainer for unit-sliced Colab training (STRATEGY 48/49).

The support-pack `train_unet_transformer.py` saves only `edge_predictor_best.pth`
(flat model weights) — a session that dies loses every epoch since the last
best. For Colab Pro's unit-sliced reality (sessions of a few hours, spread over
days), the trainer needs exactly what we built for ctrack in row 40:

  * `checkpoint_last.pth` — model + optimizer + epoch + best_score, written
    ATOMICALLY every epoch;
  * `--resume auto` — continue from it if present;
  * `--max-minutes` — refuse to start an epoch that can't finish inside the
    session budget, so the session ends cleanly with the checkpoint uploadable.

Applied with exact-count asserts (a patch that silently fails to apply looks
exactly like success). Shared by the local M1 smoke and the Colab notebook so
both run the same code.

    python scripts/local/patch_fullrecipe_trainer.py <path-to-train_unet_transformer.py>
"""

from __future__ import annotations

import sys
from pathlib import Path

ANCHOR_ARGS = '''    parser.add_argument("--single-gpu", dest="data_parallel", action="store_false",'''
PATCH_ARGS = '''    parser.add_argument("--resume", type=str, default="auto",
                        help="'auto' resumes from checkpoint_last.pth in the output dir if "
                             "present; 'none' starts fresh")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop before starting an epoch that would exceed this budget")
    parser.add_argument("--patience", type=int, default=10,
                        help="early-stop after this many epochs without a new best (0 = off)")
    parser.add_argument("--single-gpu", dest="data_parallel", action="store_false",'''

ANCHOR_SIG = '''    max_iters: int | None = None,
    debug_video: Path | None = None,'''
PATCH_SIG = '''    max_iters: int | None = None,
    resume: str = "auto",
    max_minutes: float | None = None,
    patience: int = 10,
    debug_video: Path | None = None,'''

ANCHOR_CALL = '''            max_iters=args.max_iters,'''
PATCH_CALL = '''            max_iters=args.max_iters,
            resume=args.resume,
            max_minutes=args.max_minutes,
            patience=args.patience,'''

ANCHOR_LOOP = '''    best_score = 0.0
    save_path = output_dir / "edge_predictor_best.pth"
    pbar = tqdm(range(n_epochs), desc="Training", disable=False)
    print(f"Detection loss: weight={det_loss_weight}, neg_weight={det_neg_weight}", flush=True)

    for epoch in pbar:
        t0 = time.monotonic()'''
PATCH_LOOP = '''    import json as _json  # history mirror; do not assume upstream imports json
    best_score = 0.0
    save_path = output_dir / "edge_predictor_best.pth"
    last_path = output_dir / "checkpoint_last.pth"

    # ---- resume (unit-sliced Colab sessions; STRATEGY 48) ----
    start_epoch = 0
    if resume == "auto" and last_path.exists():
        _ck = torch.load(last_path, map_location=device, weights_only=False)
        _state = _ck["model_state_dict"]
        if isinstance(model.unet, nn.DataParallel):
            _state = {(k.replace("unet.", "unet.module.", 1)
                       if k.startswith("unet.") and not k.startswith("unet.module.") else k): v
                      for k, v in _state.items()}
        model.load_state_dict(_state)
        optimizer.load_state_dict(_ck["optimizer_state_dict"])
        start_epoch = int(_ck["epoch"]) + 1
        best_score = float(_ck.get("best_score", 0.0))
        since_best = int(_ck.get("since_best", 0))
        _history = list(_ck.get("history", []))
        print(f"RESUMED from {last_path.name}: epoch {start_epoch} "
              f"(best {best_score:.4f}, {len(_ck['model_state_dict'])} tensors, "
              f"{len(_history)} history rows)", flush=True)
    elif resume == "auto":
        print("fresh run (no checkpoint_last.pth)", flush=True)

    since_best = locals().get("since_best", 0)
    # ---- LOSS HISTORY, carried INSIDE the checkpoint --------------------
    # The per-epoch losses are only ever printed to /content/train.log, which
    # dies with the session (row 55: the ssh/exec channel drops at 60-90 min
    # while the runtime keeps training). The checkpoint dataset is the ONLY
    # channel that reliably reaches the Mac, so the curve has to ride along
    # inside it or it is unobservable. Restored on resume above, so the curve
    # is continuous across an arbitrary number of dead sessions.
    _history = locals().get("_history", [])
    _run_t0 = time.monotonic()
    _est_epoch_s = None
    pbar = tqdm(range(start_epoch, n_epochs), desc="Training", disable=False)
    print(f"Detection loss: weight={det_loss_weight}, neg_weight={det_neg_weight}", flush=True)

    for epoch in pbar:
        if max_minutes is not None and _est_epoch_s is not None:
            _elapsed_min = (time.monotonic() - _run_t0) / 60.0
            if _elapsed_min + _est_epoch_s / 60.0 > max_minutes:
                print(f"BUDGET STOP: {_elapsed_min:.1f} min elapsed, next epoch "
                      f"~{_est_epoch_s/60:.1f} min exceeds {max_minutes:.0f} min. "
                      f"Resumable at epoch {epoch}.", flush=True)
                break
        t0 = time.monotonic()'''

ANCHOR_SAVE = '''        marker = "*" if is_best else " "'''
PATCH_SAVE = '''        # ---- atomic resumable checkpoint, every epoch ----
        _est_epoch_s = train_time + test_time
        _tmp = last_path.with_suffix(".tmp")
        since_best = 0 if is_best else since_best + 1
        _history.append({
            "epoch": int(epoch),
            "edge_loss": float(edge_loss),
            "det_loss": float(det_loss),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_recall": float(test_recall),
            "score": float(score),
            "best_score": float(best_score),
            "since_best": int(since_best),
            "train_s": round(float(train_time), 1),
            "test_s": round(float(test_time), 1),
        })
        # Mirror to a plain JSONL too: if a session dies mid-upload the file is
        # still on the VM, and it costs nothing.
        try:
            with open(output_dir / "history.jsonl", "a") as _hf:
                _hf.write(_json.dumps(_history[-1]) + "\\n")
        except Exception:
            pass
        torch.save({
            "epoch": epoch,
            "best_score": best_score,
            "since_best": since_best,
            "history": _history,
            "model_state_dict": {k.replace("unet.module.", "unet.", 1): v
                                 for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        }, _tmp)
        _tmp.replace(last_path)

        if patience and since_best >= patience:
            print(f"EARLY STOP: no new best in {patience} epochs "
                  f"(best {best_score:.4f}). Stopping at epoch {epoch}.", flush=True)
            _print_final = True
            marker = "*" if is_best else " "
            print(f"  Epoch {epoch:3d}/{n_epochs} | (early-stop epoch)", flush=True)
            break

        marker = "*" if is_best else " "'''


def patch(path: Path) -> None:
    s = path.read_text()
    if "BUDGET STOP" in s:
        print("already patched")
        return
    for name, old, new in [("args", ANCHOR_ARGS, PATCH_ARGS),
                           ("sig", ANCHOR_SIG, PATCH_SIG),
                           ("call", ANCHOR_CALL, PATCH_CALL),
                           ("loop", ANCHOR_LOOP, PATCH_LOOP),
                           ("save", ANCHOR_SAVE, PATCH_SAVE)]:
        n = s.count(old)
        assert n == 1, f"anchor {name!r} matched {n} times, expected 1"
        s = s.replace(old, new, 1)
    compile(s, str(path), "exec")
    path.write_text(s)
    print(f"patched {path} (5/5 anchors, compile OK)")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
