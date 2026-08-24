#!/bin/bash
# END-TO-END SMOKE for the funded retrain (PLAN 3.1e). Repo rule: smoke before
# you spend — this exercises the ENTIRE v2 path on ~$0.10 of T4 time before the
# ~1,600-unit A100 launch:
#
#   session create (T4) -> file uploads (incl. token file — the new CLI has no
#   exec --env) -> detached launch -> staging 3 volumes from the pre-packed
#   biohub-frsmoke-data Kaggle dataset (per-file competition download 404s,
#   measured 08-24; this script builds the dataset from the extSSD copy on
#   first use) -> contamination guard -> trainer patch (6 anchors incl.
#   --seed) -> 2 real epochs -> checkpoint + beacon pushed to Kaggle ->
#   beacon-only completion detection -> checkpoint downloaded BOTH ways
#   (kaggle store + colab download) -> session stopped.
#
# What it deliberately does NOT cover: the 81 GB zip staging path (proven in
# row 91; a smoke should not spend 40 min re-proving it), A100 batch-8 memory
# headroom, and multi-session resume. Timings measured here are T4-on-3-volumes
# and must NOT be used to size the full run — use row 91's banked history
# (1218 s/epoch, A100 batch 8, 132 volumes).
#
#   scripts/cloud/colab/smoke.sh
set -u
CB="${CB:-$HOME/.local/bin/colab}"
REPO=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
KAGGLE="$REPO/.venv/bin/kaggle"
PY="$REPO/.venv/bin/python"
D=$REPO/scripts/cloud/colab
SNAME="${SNAME:-frsmoke$(date +%H%M)}"
CKPT_DS=biohub-frsmoke-ckpt
BEACON_DS=biohub-frsmoke-beacon
DATA_DS=biohub-frsmoke-data
WORK=$(mktemp -d)
T_START=$(date +%s)

log() { echo "[smoke $(date +%H:%M:%S)] $*"; }
elapsed() { echo "$(( ($(date +%s) - T_START) / 60 ))m$(( ($(date +%s) - T_START) % 60 ))s"; }

cleanup() {
  log "cleanup: stopping $SNAME (idle sessions burn units)"
  $CB stop -s "$SNAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- 1. smoke split: 2 train + 1 test volumes FROM the guarded split ------
# (subset of the real train/test lists, real holdout carried for the leak gate)
$PY - "$WORK" << 'EOF'
import json, sys, pathlib
w = pathlib.Path(sys.argv[1])
sp = json.load(open('/Users/maruanottoni/home/kaggle/biohub-cell-tracking/configs/fullrecipe_splits.json'))[0]
smoke = [{
    "train": sp["train"][:2],
    "test": sp["test"][:1],
    "_note": "SMOKE subset of fullrecipe_splits.json — same quarantine, 3 volumes",
    "_holdout": sp["_holdout"],
}]
(w / "smoke_splits.json").write_text(json.dumps(smoke, indent=1))
vols = [v[:-5] for v in smoke[0]["train"] + smoke[0]["test"]]
(w / "smoke_volumes.txt").write_text("".join(v + "\n" for v in vols))
print("smoke volumes:", vols)
EOF

cat > "$WORK/job_env" << EOF
MAX_MINUTES=100000
BATCH=2
WINDOW=2
METHOD=fullrecipe
CKPT_DS=$CKPT_DS
BEACON_DS=$BEACON_DS
EPOCHS=2
SEED=1337
PATIENCE=10
DATA_MODE=dataset
DATA_DS=$DATA_DS
EOF

# ---- 1b. smoke data dataset (built once from the extSSD copy; ~2.5 GB) ----
if $KAGGLE datasets files "maruanbottoni/${DATA_DS}" >/dev/null 2>&1; then
  log "smoke data dataset maruanbottoni/${DATA_DS} already exists"
else
  log "building maruanbottoni/${DATA_DS} from /Volumes/extSSD (one-time, ~2.5 GB upload)"
  RAW=/Volumes/extSSD/kaggle/biohub/raw/train
  PK="$WORK/pack/train"; mkdir -p "$PK"
  while read -r V; do
    [ -d "$RAW/$V.zarr" ] && [ -d "$RAW/$V.geff" ] || { log "FATAL: $V missing on extSSD"; exit 1; }
    cp -r "$RAW/$V.zarr" "$RAW/$V.geff" "$PK/"
  done < "$WORK/smoke_volumes.txt"
  ( cd "$WORK/pack" && zip -q -r -0 "$WORK/smoke_data.zip" train )
  ST="$WORK/ds_staging"; mkdir -p "$ST"; mv "$WORK/smoke_data.zip" "$ST/"
  printf '{"id": "maruanbottoni/%s", "title": "%s", "licenses": [{"name": "CC0-1.0"}]}' "$DATA_DS" "$DATA_DS" > "$ST/dataset-metadata.json"
  $KAGGLE datasets create -p "$ST" --dir-mode skip || { log "dataset create FAILED"; exit 1; }
  log "waiting for Kaggle to ingest the dataset"
  for i in $(seq 1 30); do
    sleep 30
    SZ=$($KAGGLE datasets files "maruanbottoni/${DATA_DS}" 2>/dev/null | awk '$1=="smoke_data.zip" {print $2}')
    if [ -n "${SZ:-}" ] && [ "$SZ" -gt 0 ] 2>/dev/null; then log "ingested (smoke_data.zip $SZ bytes)"; break; fi
    [ "$i" = 30 ] && { log "FATAL: dataset never ingested (row 58b: audit the PUBLISHED listing)"; exit 1; }
  done
  rm -rf "$WORK/pack" "$ST"
fi

# ---- 2. session ----------------------------------------------------------
log "creating T4 session $SNAME"
$CB new --gpu T4 -s "$SNAME" || { log "session create FAILED"; exit 1; }
log "session up at $(elapsed)"

# ---- 3. payload ----------------------------------------------------------
log "uploading payload"
$CB upload "$D/fullrecipe_runner.sh"                          /content/fullrecipe_runner.sh   -s "$SNAME"
$CB upload "$REPO/scripts/local/patch_fullrecipe_trainer.py"  /content/patcher.py             -s "$SNAME"
$CB upload "$WORK/smoke_splits.json"                          /content/fullrecipe_splits.json -s "$SNAME"
$CB upload "$WORK/smoke_volumes.txt"                          /content/fullrecipe_volumes.txt -s "$SNAME"
$CB upload "$WORK/job_env"                                    /content/job_env                -s "$SNAME"
$CB upload "$HOME/.kaggle/access_token"                       /content/kaggle_access_token    -s "$SNAME"

# ---- 4. detached launch --------------------------------------------------
log "launching runner (detached)"
$CB exec -s "$SNAME" -f "$D/launch_runner.py" --timeout 120 || { log "exec FAILED"; exit 1; }
log "runner launched at $(elapsed); polling BEACON ONLY from here (no live channel)"

# ---- 5. beacon-only poll -------------------------------------------------
DEADLINE=$(( $(date +%s) + 75*60 ))
DONE=""
while [ $(date +%s) -lt $DEADLINE ]; do
  sleep 120
  rm -rf "$WORK/b" && mkdir -p "$WORK/b"
  $KAGGLE datasets download "maruanbottoni/${BEACON_DS}" -f beacon.txt -p "$WORK/b" >/dev/null 2>&1 || true
  BLINE=$(head -1 "$WORK/b/beacon.txt" 2>/dev/null || echo "(no beacon yet)")
  log "beacon: $BLINE"
  if grep -q "data-FATAL" "$WORK/b/beacon.txt" 2>/dev/null; then
    log "FATAL beacon:"; sed 's/^/    /' "$WORK/b/beacon.txt"; exit 1
  fi
  # STATE=done is the contract; the first-line " done " fallback covers a
  # masked terminal beacon (the failure mode the 08-24 smoke exposed)
  if grep -q "^STATE=done" "$WORK/b/beacon.txt" 2>/dev/null \
     || head -1 "$WORK/b/beacon.txt" 2>/dev/null | grep -qE " done "; then
    log "runner DONE at $(elapsed): $(grep '^MSG=' "$WORK/b/beacon.txt" || head -1 "$WORK/b/beacon.txt")"
    DONE=1; break
  fi
done
[ -n "$DONE" ] || { log "TIMEOUT: no STATE=done within 75 min"; exit 1; }

# ---- 6. verify the checkpoint, both retrieval paths ----------------------
sleep 60   # final ckpt push
log "verifying ckpt store"
$KAGGLE datasets files "maruanbottoni/${CKPT_DS}"
rm -rf "$WORK/ck" && $KAGGLE datasets download "maruanbottoni/${CKPT_DS}" -p "$WORK/ck" --unzip >/dev/null 2>&1
$PY - "$WORK/ck/checkpoint_last.pth" << 'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
h = ck.get('history', [])
print(f"KAGGLE-PATH CHECKPOINT OK: epoch {ck['epoch']}, best {ck.get('best_score',0):.4f}, "
      f"since_best {ck.get('since_best','?')}, {len(h)} history rows")
for r in h:
    print(f"  epoch {r['epoch']}: train {r['train_s']}s test {r['test_s']}s score {r['score']:.4f}")
EOF
log "downloading the same checkpoint via colab download (second retrieval path)"
$CB download /content/repo/weights/fullrecipe/split_0/checkpoint_last.pth "$WORK/ck_direct.pth" -s "$SNAME" \
  && ls -la "$WORK/ck_direct.pth" \
  || log "colab download failed (non-fatal: the Kaggle store path is the production channel)"

# ---- 7. teardown (also in trap) ------------------------------------------
cleanup
trap - EXIT
$CB sessions
log "SMOKE COMPLETE at $(elapsed); artifacts in $WORK"
