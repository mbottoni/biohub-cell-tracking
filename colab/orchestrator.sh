#!/bin/bash
# Perpetual session orchestrator for the w3 probe (row 54/55).
# Colab CLI runtimes die at 20-90 min regardless of activity; this loop makes
# that irrelevant: provision -> launch runner -> poll -> respawn on death,
# until the banked checkpoint reaches TARGET_EPOCH or the trainer early-stops.
# Run on the Mac as a background task; all state lives in the Kaggle ckpt store.
set -u
CB=/Users/maruanottoni/home/kaggle/biohub-cell-tracking/.venv_colab/bin/colab
REPO=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
KEY=~/.ssh/id_ed25519_colab
TOKEN=$(cat ~/.kaggle/access_token)
TARGET_EPOCH=${TARGET_EPOCH:-28}
N=0

log() { echo "[orch $(date +%H:%M)] $*"; }

while true; do
  N=$((N+1))
  SNAME="w3auto$N"

  # ---- termination check: what does the ckpt store hold?
  EPOCH=$(cd /tmp && rm -rf orch_ck && ${REPO}/.venv/bin/kaggle datasets download \
      maruanbottoni/biohub-w3-ckpt -p orch_ck --unzip >/dev/null 2>&1 && \
      ${REPO}/.venv/bin/python -c "
import torch
ck = torch.load('/tmp/orch_ck/checkpoint_last.pth', map_location='cpu', weights_only=False)
print(ck['epoch'])" 2>/dev/null)
  log "ckpt store at epoch ${EPOCH:-unknown} (target ${TARGET_EPOCH})"
  if [ -n "${EPOCH:-}" ] && [ "$EPOCH" -ge "$TARGET_EPOCH" ] 2>/dev/null; then
    log "TARGET REACHED at epoch $EPOCH — orchestrator done"
    exit 0
  fi

  # ---- provision: A100 preferred, L4 fallback
  GPU=A100
  if ! $CB new --gpu A100 -s "$SNAME" 2>/dev/null | grep -q READY; then
    GPU=L4
    if ! $CB new --gpu L4 -s "$SNAME" 2>/dev/null | grep -q READY; then
      log "no GPU available — retrying in 10 min"
      sleep 600
      continue
    fi
  fi
  BATCH=$([ "$GPU" = "A100" ] && echo 8 || echo 4)
  log "session $SNAME up ($GPU)"

  # ---- payload + launch
  $CB upload "$REPO/colab/w3_runner.sh" /content/w3_runner.sh -s "$SNAME" >/dev/null 2>&1
  $CB upload "$REPO/scripts/local/patch_fullrecipe_trainer.py" /content/patcher.py -s "$SNAME" >/dev/null 2>&1
  $CB upload "$REPO/configs/w3_probe_splits48.json" /content/w3_probe_splits.json -s "$SNAME" >/dev/null 2>&1
  printf 'import subprocess, os\nenv = dict(os.environ)\nenv["KAGGLE_API_TOKEN"] = os.environ["KT"]\nenv["MAX_MINUTES"] = "100000"\nenv["BATCH"] = "%s"\nwith open("/content/runner.log", "a") as f:\n    subprocess.run(["bash", "/content/w3_runner.sh"], env=env, stdout=f, stderr=subprocess.STDOUT)\nprint("runner finished")\n' "$BATCH" \
    | $CB exec -s "$SNAME" --env "KT=$TOKEN" --timeout 10 >/dev/null 2>&1
  log "runner launched on $SNAME (batch $BATCH)"

  # ---- poll until STORE-STALE or done. Row 55: ssh dying does NOT mean the
  # runtime died -- sessions train on for hours after the tunnel 404s, and the
  # live pusher keeps banking. The only honest death signal is checkpoint-store
  # staleness. Never `colab stop` an unreachable session; it may be training.
  STALE_MIN=40
  while true; do
    sleep 600
    FDATE=$(${REPO}/.venv/bin/kaggle datasets files maruanbottoni/biohub-w3-ckpt 2>/dev/null | grep checkpoint_last | awk '{print $(NF-1), $NF}')
    NOWE=$(date -u +%s)
    FE=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "${FDATE%%.*}" +%s 2>/dev/null || echo 0)
    AGEMIN=$(( (NOWE - FE) / 60 ))
    OUT=$(echo 'grep -cE "Epoch +[0-9]+/" /content/train.log 2>/dev/null; grep -E "EARLY STOP" /content/train.log 2>/dev/null | tail -1; test -f /content/DONE && echo DONE' \
      | timeout 200 $CB ssh -s "$SNAME" --identity $KEY 2>/dev/null)
    log "$SNAME: store-age ${AGEMIN}min | ssh: ${OUT:-(unreachable — may still be training)}"
    if echo "$OUT" | grep -qE "EARLY STOP|^DONE"; then
      log "trainer finished on $SNAME — waiting for final push"
      sleep 300
      $CB stop -s "$SNAME" >/dev/null 2>&1
      log "ORCHESTRATOR DONE"
      exit 0
    fi
    if [ "$AGEMIN" -gt "$STALE_MIN" ] && [ "$FE" -ne 0 ]; then
      log "$SNAME: store stale ${AGEMIN}min — presuming real death, spawning replacement"
      $CB stop -s "$SNAME" >/dev/null 2>&1
      break
    fi
  done
  sleep 30
done
