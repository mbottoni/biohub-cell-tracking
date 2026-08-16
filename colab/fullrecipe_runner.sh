#!/bin/bash
# FULL-RECIPE session runner (PHASE 3). Launched by the local orchestrator via
# `colab exec`. Adapted from w3_runner.sh, which is proven end-to-end (§50.3).
#
# WHAT CHANGED vs the w3 probe, and why:
#   * DATA. The probe pulled a pre-packed 60-volume Kaggle dataset. We need 132
#     volumes (108 GB) and the pack does not exist, so this stages straight from
#     the 81 GB competition zip and deletes it after extraction. MEASURED disk
#     budget (Phase 0, on a real runtime): /content, /kaggle and /tmp are all the
#     SAME 206 GB overlay -- /kaggle/input is read-only -- so the peak is
#     81 (zip) + 108 (volumes) = 189 GB, leaving ~17 GB. After the zip is deleted
#     there is ~98 GB free for checkpoints. Do NOT raise the volume count without
#     re-doing this arithmetic.
#   * SPLIT. /content/fullrecipe_splits.json, whose HOLDOUT (evalset-24) is
#     absent from train AND test AND from the staged volume list, so the model
#     cannot see it even by accident. That quarantine is the entire point of the
#     run (row 90: local can currently only reject, never confirm).
#   * WINDOW=2. §50 killed window-3: +0.0021 for 1.6x the per-epoch cost.
#
# Beacons the orchestrator polls: /content/PHASE, /content/train.log, /content/DONE
set -euo pipefail
cd /content
exec 9>/content/runner.lock
flock -n 9 || { echo "another runner instance holds the lock -- exiting"; exit 0; }

MAX_MINUTES="${MAX_MINUTES:-600}"
BATCH="${BATCH:-8}"
WINDOW="${WINDOW:-2}"
METHOD="${METHOD:-fullrecipe}"
CKPT_DS="${CKPT_DS:-biohub-fullrecipe-ckpt}"
BEACON_DS="${BEACON_DS:-biohub-fullrecipe-beacon}"
EPOCHS="${EPOCHS:-120}"

phase() {
  echo "$1" > /content/PHASE; echo "== $1 == $(date -u +%H:%M:%S)"
  ( B=/content/beacon_staging; rm -rf "$B" && mkdir -p "$B"
    echo "$(hostname) $1 $(date -u +%H:%M:%S)" > "$B/beacon.txt"
    printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$BEACON_DS" "$BEACON_DS" > "$B/dataset-metadata.json"
    kaggle datasets version -p "$B" -m "$1 $(date -u +%H:%M)" >/dev/null 2>&1 || true ) &
}

# ---- STAGING HEARTBEAT -------------------------------------------------
# The w3 probe staged a pre-packed subset in ~7 min, so a beacon per phase was
# enough. We download 81 GB, and the `data` phase can run 45 min with no phase
# transition -- during which the orchestrator sees a stale beacon and kills a
# perfectly healthy session. That is exactly the failure the orchestrator's own
# comment records ("executed a healthy L4 five minutes before epoch 12 banked").
# So: an explicit heartbeat every 5 min for the whole staging period, retired
# once the training pusher takes over.
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

phase data
NEED=$(wc -l < /content/fullrecipe_volumes.txt)
HAVE=$(ls /content/data/train 2>/dev/null | grep -c '\.zarr$' || true)
if [ "$HAVE" -lt "$NEED" ]; then
  mkdir -p /content/data
  ZIP=/content/biohub-cell-tracking-during-development.zip
  if [ ! -f "$ZIP" ]; then
    echo "downloading competition zip (~81 GB) -- expect 20-40 min"
    df -h /content | tail -1
    kaggle competitions download -c biohub-cell-tracking-during-development -p /content
  fi
  du -h "$ZIP" | head -1
  # ONE unzip pass over the whole pattern list: the 81 GB central directory is
  # scanned once instead of 132 times. Row 54's lesson is the verify AFTER --
  # a bulk pattern list silently skipped a volume twice, and it was only found
  # when training crashed at test-load, minutes into GPU time.
  PATTERNS=""
  while read -r V; do PATTERNS="$PATTERNS train/$V.zarr/* train/$V.geff/*"; done < /content/fullrecipe_volumes.txt
  ( cd /content/data && unzip -q -o "$ZIP" $PATTERNS || true )
  # Individually retry anything the bulk pass missed.
  while read -r V; do
    if [ ! -d "/content/data/train/$V.zarr" ] || [ ! -d "/content/data/train/$V.geff" ]; then
      echo "retrying $V"
      ( cd /content/data && unzip -q -o "$ZIP" "train/$V.zarr/*" "train/$V.geff/*" || true )
    fi
  done < /content/fullrecipe_volumes.txt
  rm -f "$ZIP"          # reclaim 81 GB before training touches the disk
  df -h /content | tail -1
fi
ls /content/data/train | grep -c '\.zarr$'

# Verify EVERY split volume is present BEFORE spending GPU time, and verify the
# HOLDOUT is absent -- a staged holdout volume would silently destroy the only
# thing this run exists to produce.
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
then phase "data-FATAL"; sleep 20; exit 1; fi

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
trap 'kill $PUSHER_PID 2>/dev/null || true; kill $HB_PID 2>/dev/null || true; phase "runner-exit"; sleep 20' EXIT
cd /content/repo
PYTHONPATH=src python scripts/train_unet_transformer.py \
  --data-dir /content/data/train \
  --splits /content/fullrecipe_splits.json --split 0 \
  --method "$METHOD" --window-size "$WINDOW" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers 4 --single-gpu \
  --max-minutes "$MAX_MINUTES" --patience 10 2>&1 | tee -a /content/train.log

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
print(f"epoch {ck['epoch']} best {ck.get('best_score', 0):.4f} ({len(ck.get('history', []))} history rows)")
EOF
)
kaggle datasets version -p "$ST" -m "$MSG"
echo "$MSG" > /content/DONE
phase done
