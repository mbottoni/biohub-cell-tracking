#!/bin/bash
# Headless window-3 probe session runner (STRATEGY 49, colab-cli mode).
# Uploaded and launched by the local orchestrator via `colab exec`; expects
# KAGGLE_API_TOKEN in the environment and these files already uploaded:
#   /content/patcher.py            resume/budget/patience trainer patch
#   /content/w3_probe_splits.json  golden-8-quarantined splits
# Writes progress beacons the orchestrator polls:
#   /content/PHASE          current phase name
#   /content/train.log      trainer stdout
#   /content/DONE           written after the checkpoint upload succeeds
set -euo pipefail
cd /content
# Single-instance lock: a doubled launch once raced the data phase and deleted
# the in-flight download (rm of "stale" zips from instance B killed A's file).
exec 9>/content/runner.lock
flock -n 9 || { echo "another runner instance holds the lock -- exiting"; exit 0; }
MAX_MINUTES="${MAX_MINUTES:-150}"
BATCH="${BATCH:-8}"

phase() {
  echo "$1" > /content/PHASE; echo "== $1 =="
  # Out-of-band beacon: ssh/kernel channels die within minutes on these
  # runtimes; the only always-working reporting path is the Kaggle API outward.
  ( B=/content/beacon_staging; rm -rf "$B" && mkdir -p "$B"
    echo "$(hostname) $1 $(date -u +%H:%M:%S)" > "$B/beacon.txt"
    printf '{"id": "maruanbottoni/biohub-w3-beacon", "title": "biohub-w3-beacon", "licenses": [{"name": "CC0-1.0"}]}' > "$B/dataset-metadata.json"
    kaggle datasets version -p "$B" -m "$1 $(date -u +%H:%M)" >/dev/null 2>&1 || true ) &
}

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
if [ ! -d /content/data/train ]; then
  mkdir -p /content/data
  # Row 54: pre-packed 60-volume subset (~7 min total staging) instead of the
  # 81 GB competition zip -- CLI runtimes die at ~60-90 min, so staging must
  # leave room for 2-3 banked epochs.
  rm -f /content/data/w3subset.zip*
  kaggle datasets download maruanbottoni/biohub-w3-subset -p /content/data
  ( cd /content/data && unzip -qo "$(ls *.zip | head -1)" && rm -f *.zip )
fi
ls /content/data/train | grep -c ".zarr$"

phase repo
if [ ! -d /content/repo ]; then
  kaggle datasets download pilkwang/biohub-tracking-support-pack-50ep-v1 -p /content/sp --unzip
  cp -r /content/sp/repo /content/repo
fi
python /content/patcher.py /content/repo/scripts/train_unet_transformer.py

phase ckpt-pull
OUT=/content/repo/weights/w3probe/split_0
mkdir -p "$OUT"
rm -rf /content/ckpt
if kaggle datasets download maruanbottoni/biohub-w3-ckpt -p /content/ckpt --unzip; then
  find /content/ckpt -name 'checkpoint_last.pth' -exec cp {} "$OUT/" \;
  find /content/ckpt -name 'edge_predictor_best.pth' -exec cp {} "$OUT/" \;
  ls -la "$OUT" || true
fi

phase train
# Continuous checkpoint push: version the store whenever checkpoint_last changes,
# so a recycled runtime costs at most one epoch (first session died holding 2).
(
  LASTMD5=""
  while true; do
    sleep 300
    # Remote log pipe: ssh dies within minutes, so ship the train.log tail out
    # through the beacon dataset every cycle -- the only always-working channel.
    ( B=/content/beacon_staging; rm -rf "$B" && mkdir -p "$B"
      { echo "$(hostname) train-tail $(date -u +%H:%M:%S)"; tail -40 /content/train.log 2>/dev/null | tr -d "\r"; } > "$B/beacon.txt"
      printf '{"id": "maruanbottoni/biohub-w3-beacon", "title": "biohub-w3-beacon", "licenses": [{"name": "CC0-1.0"}]}' > "$B/dataset-metadata.json"
      kaggle datasets version -p "$B" -m "train-tail $(date -u +%H:%M)" >/dev/null 2>&1 || true )
    CK=/content/repo/weights/w3probe/split_0/checkpoint_last.pth
    if [ -f "$CK" ]; then
      MD5=$(md5sum "$CK" | cut -d' ' -f1)
      if [ "$MD5" != "$LASTMD5" ]; then
        PU=/content/w3ckpt_push
        rm -rf "$PU" && mkdir -p "$PU"
        cp "$CK" "$PU/" && cp /content/repo/weights/w3probe/split_0/edge_predictor_best.pth "$PU/" 2>/dev/null || true
        cp /content/repo/weights/w3probe/split_0/config.json "$PU/" 2>/dev/null || true
        printf '{"id": "maruanbottoni/biohub-w3-ckpt", "title": "biohub-w3-ckpt", "licenses": [{"name": "CC0-1.0"}]}' > "$PU/dataset-metadata.json"
        kaggle datasets version -p "$PU" -m "live push $(date -u +%H:%M)" >/dev/null 2>&1 && LASTMD5="$MD5" && echo "[ckpt-push] uploaded $(date -u +%H:%M)" >> /content/push.log
      fi
    fi
  done
) &
PUSHER_PID=$!
cd /content/repo
PYTHONPATH=src python scripts/train_unet_transformer.py \
  --data-dir /content/data/train \
  --splits /content/w3_probe_splits.json --split 0 \
  --method w3probe --window-size 3 \
  --epochs 60 --batch-size "$BATCH" --num-workers 4 --single-gpu \
  --max-minutes "$MAX_MINUTES" --patience 10 2>&1 | tee -a /content/train.log

phase ckpt-push
kill $PUSHER_PID 2>/dev/null || true
ST=/content/w3ckpt_staging
rm -rf "$ST" && mkdir -p "$ST"
cp "$OUT/checkpoint_last.pth" "$ST/" 2>/dev/null || true
cp "$OUT/edge_predictor_best.pth" "$ST/" 2>/dev/null || true
cp "$OUT/config.json" "$ST/" 2>/dev/null || true
tail -50 /content/train.log > "$ST/last_session_log.txt"
cat > "$ST/dataset-metadata.json" << 'EOF'
{"id": "maruanbottoni/biohub-w3-ckpt", "title": "biohub-w3-ckpt",
 "licenses": [{"name": "CC0-1.0"}]}
EOF
MSG=$(python - << 'EOF'
import torch
ck = torch.load('/content/w3ckpt_staging/checkpoint_last.pth', map_location='cpu', weights_only=False)
print(f"epoch {ck['epoch']} best {ck.get('best_score', 0):.4f} (colab-cli A100)")
EOF
)
kaggle datasets version -p "$ST" -m "$MSG"
echo "$MSG" > /content/DONE
phase done
