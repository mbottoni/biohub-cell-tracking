#!/bin/bash
# FULL-RECIPE session runner v2 (PLAN 3.1e — the funded retrain).
# Launched on a Colab VM by launch_runner.py (via `colab exec`). Supersedes
# colab/fullrecipe_runner.sh (row 91 era), which stays untouched as the record.
#
# CHANGES vs v1, and the evidence for each:
#   * TOKEN from /content/kaggle_access_token (uploaded file). The new colab CLI
#     (~/.local/bin/colab) has no `exec --env`, so the env-forwarding trick the
#     old orchestrator used is gone. The file is uploaded per-session and never
#     appears in logs or command lines.
#   * SEED + PATIENCE forwarded to the trainer. PLAN 3.1e wants TWO decorrelated
#     seeds; upstream main() never passed its own seed param (and that param
#     only seeded the DataLoader — patch_fullrecipe_trainer.py now adds full
#     seeding). Row 91's patience=10 fired at e32 while det_loss was still
#     falling — the funded run needs it raised (default here: 40).
#   * TERMINAL BEACON. Row 91's orchestrator read EARLY STOP over ssh, which is
#     unreachable during training — it respawned sessions against a finished
#     run, ~100 min waste per useless epoch. The final beacon now carries
#     STATE=done + the train.log tail, and the checkpoint itself carries
#     since_best, so the orchestrator never needs a live channel to the VM.
#   * DATA_MODE=dataset: stage from a small pre-packed Kaggle dataset (smoke).
#     The full run keeps DATA_MODE=zip — the 81 GB zip path is proven (rows
#     54/58b/91). Per-file competition download is DEAD for this competition:
#     the download endpoint 404s on every file the listing returns (measured
#     08-24 from the Mac; commit 01be9b7 abandoned the same route in July).
#
# Beacons the orchestrator polls: beacon dataset (phase + STATE + log tail),
# checkpoint dataset (epoch, since_best, history). Never ssh.
set -euo pipefail
cd /content
exec 9>/content/runner.lock
flock -n 9 || { echo "another runner instance holds the lock -- exiting"; exit 0; }

# ---- token (uploaded file; env fallback for compatibility) --------------
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f /content/kaggle_access_token ]; then
  KAGGLE_API_TOKEN=$(cat /content/kaggle_access_token)
fi
: "${KAGGLE_API_TOKEN:?no token: upload ~/.kaggle/access_token to /content/kaggle_access_token}"
export KAGGLE_API_TOKEN

MAX_MINUTES="${MAX_MINUTES:-100000}"
BATCH="${BATCH:-8}"
WINDOW="${WINDOW:-2}"
METHOD="${METHOD:-fullrecipe}"
CKPT_DS="${CKPT_DS:-biohub-fr400-s1-ckpt}"
BEACON_DS="${BEACON_DS:-biohub-fr400-s1-beacon}"
EPOCHS="${EPOCHS:-400}"
SEED="${SEED:-1337}"
PATIENCE="${PATIENCE:-40}"
DATA_MODE="${DATA_MODE:-zip}"
DATA_DS="${DATA_DS:-biohub-frsmoke-data}"   # only read when DATA_MODE=dataset

phase() {
  echo "$1" > /content/PHASE; echo "== $1 == $(date -u +%H:%M:%S)"
  ( B=/content/beacon_staging; rm -rf "$B" && mkdir -p "$B"
    echo "$(hostname) $1 $(date -u +%H:%M:%S)" > "$B/beacon.txt"
    printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$BEACON_DS" "$BEACON_DS" > "$B/dataset-metadata.json"
    kaggle datasets version -p "$B" -m "$1 $(date -u +%H:%M)" >/dev/null 2>&1 || true ) &
}

# rich terminal beacon: STATE line + train.log tail, so the orchestrator can
# distinguish done / early-stop / fatal WITHOUT any live channel to the VM
beacon_state() {  # $1 = state token, rest = message
  local ST="$1"; shift
  B=/content/beacon_final; rm -rf "$B" && mkdir -p "$B"
  { echo "$(hostname) $ST $(date -u +%H:%M:%S)"
    echo "STATE=$ST"
    echo "MSG=$*"
    echo "--- train.log tail ---"
    tail -40 /content/train.log 2>/dev/null | tr -d "\r"; } > "$B/beacon.txt"
  printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$BEACON_DS" "$BEACON_DS" > "$B/dataset-metadata.json"
  kaggle datasets version -p "$B" -m "$ST $(date -u +%H:%M)" >/dev/null 2>&1 || true
}

# ---- STAGING HEARTBEAT (5 min) — retired once the training pusher starts ----
( while true; do
    sleep 300
    B=/content/hb_staging; rm -rf "$B" && mkdir -p "$B"
    { echo "$(hostname) heartbeat $(date -u +%H:%M:%S)"
      echo "phase: $(cat /content/PHASE 2>/dev/null)"
      df -h /content | tail -1
      du -sh /content/data 2>/dev/null | head -1; } > "$B/beacon.txt"
    printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$BEACON_DS" "$BEACON_DS" > "$B/dataset-metadata.json"
    kaggle datasets version -p "$B" -m "hb $(date -u +%H:%M)" >/dev/null 2>&1 || true
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

phase deps
pip -q install -U kaggle >/dev/null
pip -q install "zarr>=3.0.10,<4" blosc2 "numcodecs>=0.13,<0.16" "geff>=1.1.3.1.1" \
    "geff-spec<1.2" tracksdata "polars>=1.36" dask tqdm rustworkx "sqlalchemy>=2" >/dev/null

phase auth
mkdir -p /root/.kaggle
echo -n "$KAGGLE_API_TOKEN" > /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token
kaggle competitions list -s biohub | head -2

# ---- ENSURE THE STORES EXIST (silent-forever bug, see v1 comment) -------
for _DS in "$CKPT_DS" "$BEACON_DS"; do
  if ! kaggle datasets files "maruanbottoni/${_DS}" >/dev/null 2>&1; then
    echo "store maruanbottoni/${_DS} missing -- creating"
    _B=/content/mk_${_DS}; rm -rf "$_B"; mkdir -p "$_B"
    echo "bootstrap $(date -u +%FT%T)" > "$_B/beacon.txt"
    printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$_DS" "$_DS" > "$_B/dataset-metadata.json"
    kaggle datasets create -p "$_B" 2>&1 | tail -1
    sleep 20
  else
    echo "store maruanbottoni/${_DS} present"
  fi
done

phase data
NEED=$(wc -l < /content/fullrecipe_volumes.txt)
HAVE=$(ls /content/data/train 2>/dev/null | grep -c '\.zarr$' || true)
if [ "$HAVE" -lt "$NEED" ]; then
  mkdir -p /content/data
  if [ "$DATA_MODE" = "dataset" ]; then
    echo "staging $NEED volumes from dataset maruanbottoni/${DATA_DS} (smoke mode)"
    rm -rf /content/dl && mkdir -p /content/dl
    kaggle datasets download "maruanbottoni/${DATA_DS}" -p /content/dl --unzip
    if ls /content/dl/*.zip >/dev/null 2>&1; then
      ( cd /content/data && unzip -q -o /content/dl/*.zip )
    elif [ -d /content/dl/train ]; then
      cp -r /content/dl/train /content/data/
    fi
    rm -rf /content/dl
  else
    ZIP=/content/biohub-cell-tracking-during-development.zip
    if [ ! -f "$ZIP" ]; then
      echo "downloading competition zip (~81 GB) -- expect 20-40 min"
      df -h /content | tail -1
      kaggle competitions download -c biohub-cell-tracking-during-development -p /content
    fi
    du -h "$ZIP" | head -1
    # ONE unzip pass, then per-volume retry + verify (row 54: bulk unzip
    # silently skipped a volume twice; it surfaced minutes into GPU time)
    PATTERNS=""
    while read -r V; do PATTERNS="$PATTERNS train/$V.zarr/* train/$V.geff/*"; done < /content/fullrecipe_volumes.txt
    ( cd /content/data && unzip -q -o "$ZIP" $PATTERNS || true )
    while read -r V; do
      if [ ! -d "/content/data/train/$V.zarr" ] || [ ! -d "/content/data/train/$V.geff" ]; then
        echo "retrying $V"
        ( cd /content/data && unzip -q -o "$ZIP" "train/$V.zarr/*" "train/$V.geff/*" || true )
      fi
    done < /content/fullrecipe_volumes.txt
    rm -f "$ZIP"          # reclaim 81 GB before training touches the disk
    df -h /content | tail -1
  fi
fi
ls /content/data/train | grep -c '\.zarr$'

# Verify EVERY split volume present BEFORE GPU spend; verify HOLDOUT absent —
# a staged holdout volume would silently destroy the only thing the run
# exists to produce (row 90's quarantine).
if ! python - << 'EOF'
import json, os, sys
sp = json.load(open('/content/fullrecipe_splits.json'))[0]
stem = lambda v: v[:-5] if v.endswith('.zarr') else v
need = [stem(v) for k in ('train', 'test') for v in sp[k]]
missing = [n for n in need
           if not os.path.isdir(f'/content/data/train/{n}.zarr')
           or not os.path.isdir(f'/content/data/train/{n}.geff')]
if missing:
    print(f'FATAL: {len(missing)} split volumes missing: {missing[:8]}')
    sys.exit(1)
leaked = [stem(v) for v in sp.get('_holdout', [])
          if os.path.isdir(f"/content/data/train/{stem(v)}.zarr")]
if leaked:
    print(f'FATAL: {len(leaked)} HOLDOUT volumes are staged: {leaked[:8]}')
    print('The holdout must never reach the VM -- refusing to train.')
    sys.exit(1)
print(f'verified {len(need)} split volumes staged, 0 holdout volumes present')
EOF
then beacon_state "data-FATAL" "split volumes missing or holdout leaked"; sleep 20; exit 1; fi

phase repo
if [ ! -d /content/repo ]; then
  kaggle datasets download pilkwang/biohub-tracking-support-pack-50ep-v1 -p /content/sp --unzip
  cp -r /content/sp/repo /content/repo
  rm -rf /content/sp
fi
python /content/patcher.py /content/repo/scripts/train_unet_transformer.py

phase ckpt-pull
OUT=/content/repo/weights/${METHOD}/split_0
mkdir -p "$OUT"
rm -rf /content/ckpt
if kaggle datasets download "maruanbottoni/${CKPT_DS}" -p /content/ckpt --unzip 2>/dev/null; then
  find /content/ckpt -name 'checkpoint_last.pth' -exec cp {} "$OUT/" \;
  find /content/ckpt -name 'edge_predictor_best.pth' -exec cp {} "$OUT/" \;
  ls -la "$OUT" || true
else
  echo "no checkpoint store yet -- fresh run"
fi

phase train
( LASTMD5=""
  while true; do
    sleep 300
    ( B=/content/beacon_staging; rm -rf "$B" && mkdir -p "$B"
      { echo "$(hostname) train-tail $(date -u +%H:%M:%S)"; tail -40 /content/train.log 2>/dev/null | tr -d "\r"; } > "$B/beacon.txt"
      printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$BEACON_DS" "$BEACON_DS" > "$B/dataset-metadata.json"
      kaggle datasets version -p "$B" -m "train-tail $(date -u +%H:%M)" >/dev/null 2>&1 || true )
    CK="$OUT/checkpoint_last.pth"
    if [ -f "$CK" ]; then
      MD5=$(md5sum "$CK" | cut -d' ' -f1)
      if [ "$MD5" != "$LASTMD5" ]; then
        PU=/content/frckpt_push; rm -rf "$PU" && mkdir -p "$PU"
        cp "$CK" "$PU/" && cp "$OUT/edge_predictor_best.pth" "$PU/" 2>/dev/null || true
        cp "$OUT/history.jsonl" "$PU/" 2>/dev/null || true
        grep -E "Epoch +[0-9]+/" /content/train.log 2>/dev/null | tr -d "\r" > "$PU/epoch_history.txt" || true
        printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$CKPT_DS" "$CKPT_DS" > "$PU/dataset-metadata.json"
        kaggle datasets version -p "$PU" -m "live push $(date -u +%H:%M)" >/dev/null 2>&1 && LASTMD5="$MD5" && echo "[ckpt-push] $(date -u +%H:%M)" >> /content/push.log
      fi
    fi
  done
) &
PUSHER_PID=$!
kill $HB_PID 2>/dev/null || true   # the pusher beacons from here on
# DONE-aware trap: the smoke run (08-24) proved that any beacon pushed AFTER
# the STATE=done one masks it (latest version wins), leaving the orchestrator
# polling forever — the exact class of bug this v2 exists to kill. The trap is
# the LAST push, so it must re-assert done rather than stamp runner-exit on it.
trap 'kill $PUSHER_PID 2>/dev/null || true; kill $HB_PID 2>/dev/null || true;
      if [ -f /content/DONE ]; then beacon_state "done" "$(cat /content/DONE)";
      else beacon_state "runner-exit" "trap"; fi; sleep 20' EXIT
cd /content/repo
PYTHONPATH=src python scripts/train_unet_transformer.py \
  --data-dir /content/data/train \
  --splits /content/fullrecipe_splits.json --split 0 \
  --method "$METHOD" --window-size "$WINDOW" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers 4 --single-gpu \
  --seed "$SEED" --patience "$PATIENCE" \
  --max-minutes "$MAX_MINUTES" 2>&1 | tee -a /content/train.log

phase ckpt-push
kill $PUSHER_PID 2>/dev/null || true
ST=/content/frckpt_staging; rm -rf "$ST" && mkdir -p "$ST"
cp "$OUT/checkpoint_last.pth" "$ST/" 2>/dev/null || true
cp "$OUT/edge_predictor_best.pth" "$ST/" 2>/dev/null || true
cp "$OUT/history.jsonl" "$ST/" 2>/dev/null || true
tail -80 /content/train.log > "$ST/last_session_log.txt"
printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$CKPT_DS" "$CKPT_DS" > "$ST/dataset-metadata.json"
MSG=$(python - << 'EOF'
import torch
ck = torch.load('/content/frckpt_staging/checkpoint_last.pth', map_location='cpu', weights_only=False)
print(f"epoch {ck['epoch']} best {ck.get('best_score', 0):.4f} since_best {ck.get('since_best', '?')} ({len(ck.get('history', []))} history rows)")
EOF
)
kaggle datasets version -p "$ST" -m "$MSG"
echo "$MSG" > /content/DONE
echo done > /content/PHASE     # file only — a phase PUSH here would mask STATE=done
beacon_state "done" "$MSG"
