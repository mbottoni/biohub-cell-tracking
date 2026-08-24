#!/bin/bash
# FULL-RECIPE perpetual orchestrator v2 (PLAN 3.1e). Supersedes
# colab/fullrecipe_orchestrator.sh, kept as the row-91 record.
#
# THE FIX THIS FILE EXISTS FOR (row 91, ~100 min wasted per useless epoch):
# v1 detected trainer completion by grepping train.log OVER SSH. The ssh/exec
# channel is unreachable while the trainer holds it, so after early stop the
# orchestrator saw a stale checkpoint, respawned, and each fresh session
# re-downloaded the 81 GB zip to train ONE epoch that instantly re-stopped.
# v2 reads EVERY signal from the two Kaggle stores and never opens a channel
# to a running VM:
#   * checkpoint store: epoch  -> target reached?
#                       since_best vs PATIENCE -> trainer early-stopped?
#   * beacon store:     STATE=done / data-FATAL / heartbeat freshness
# (The new colab CLI has no ssh subcommand and no exec --env — the design and
# the tool now agree.)
#
# Run per seed, e.g.:
#   SEED=1337 CKPT_DS=biohub-fr400-s1-ckpt BEACON_DS=biohub-fr400-s1-beacon \
#     SPREFIX=fr400s1 EPOCHS=400 PATIENCE=40 \
#     nohup scripts/cloud/colab/fullrecipe_orchestrator.sh > /tmp/fr400s1_orch.log 2>&1 &
set -u
CB="${CB:-$HOME/.local/bin/colab}"
REPO=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
KAGGLE="$REPO/.venv/bin/kaggle"
PY="$REPO/.venv/bin/python"
TOKEN_FILE="$HOME/.kaggle/access_token"

EPOCHS=${EPOCHS:-400}
TARGET_EPOCH=$((EPOCHS - 1))        # trainer epochs are 0-indexed
PATIENCE=${PATIENCE:-40}            # must match the runner's PATIENCE
SEED=${SEED:-1337}
CKPT_DS=${CKPT_DS:-biohub-fr400-s1-ckpt}
BEACON_DS=${BEACON_DS:-biohub-fr400-s1-beacon}
SPREFIX=${SPREFIX:-fr400s1}
WINDOW=${WINDOW:-2}
METHOD=${METHOD:-fullrecipe}
DATA_MODE=${DATA_MODE:-zip}
N=0

log() { echo "[orch $(date +%H:%M)] $*"; }

age_of() {  # newest file date in a dataset -> age in minutes (9999 on parse failure)
  local NOWE D E
  NOWE=$(date -u +%s)
  D=$($KAGGLE datasets files "$1" 2>/dev/null | awk 'NR>2 {print $(NF-1), $NF}' | sort | tail -1)
  E=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "${D%%.*}" +%s 2>/dev/null || echo 0)
  if [ "$E" -eq 0 ] || [ "$E" -gt "$NOWE" ]; then echo 9999; else echo $(( (NOWE - E) / 60 )); fi
}

fetch_beacon() {  # -> /tmp/orch_beacon/beacon.txt (empty file on failure)
  rm -rf /tmp/orch_beacon && mkdir -p /tmp/orch_beacon
  $KAGGLE datasets download "maruanbottoni/${BEACON_DS}" -f beacon.txt -p /tmp/orch_beacon >/dev/null 2>&1 || true
  [ -f /tmp/orch_beacon/beacon.txt ] || : > /tmp/orch_beacon/beacon.txt
}

while true; do
  N=$((N+1))
  SNAME="${SPREFIX}n${N}"

  # ---- TERMINATION CHECK — from the banked checkpoint, never the VM ------
  rm -rf /tmp/orch_ck
  STATE=$($KAGGLE datasets download "maruanbottoni/${CKPT_DS}" -p /tmp/orch_ck --unzip >/dev/null 2>&1 && \
    $PY - << 'EOF' 2>/dev/null
import torch
ck = torch.load('/tmp/orch_ck/checkpoint_last.pth', map_location='cpu', weights_only=False)
print(int(ck['epoch']), int(ck.get('since_best', 0)), f"{ck.get('best_score', 0):.4f}")
EOF
  )
  EPOCH=$(echo "$STATE" | awk '{print $1}')
  SINCE=$(echo "$STATE" | awk '{print $2}')
  BEST=$(echo "$STATE" | awk '{print $3}')
  log "ckpt store: epoch ${EPOCH:-none} since_best ${SINCE:-?} best ${BEST:-?} (target ${TARGET_EPOCH}, patience ${PATIENCE})"
  if [ -n "${EPOCH:-}" ] && [ "$EPOCH" -ge "$TARGET_EPOCH" ] 2>/dev/null; then
    log "TARGET REACHED at epoch $EPOCH — orchestrator done"
    exit 0
  fi
  if [ -n "${SINCE:-}" ] && [ "$PATIENCE" -gt 0 ] && [ "$SINCE" -ge "$PATIENCE" ] 2>/dev/null; then
    log "EARLY STOP banked (since_best $SINCE >= patience $PATIENCE, best $BEST) — orchestrator done"
    exit 0
  fi

  # ---- fatal guard: a data-FATAL beacon means config is broken — do NOT
  # crash-loop sessions against it (row 58b burned ~8-10 units doing that)
  fetch_beacon
  if grep -q "data-FATAL" /tmp/orch_beacon/beacon.txt && [ "$(age_of maruanbottoni/${BEACON_DS})" -lt 120 ]; then
    log "FATAL beacon (fresh): $(head -3 /tmp/orch_beacon/beacon.txt | tr '\n' ' | ')"
    log "fix the split/staging config before relaunching — exiting"
    exit 2
  fi

  # ---- adopt guard: never run two trainers against one ckpt store --------
  while true; do
    LIFE=$(age_of "maruanbottoni/${CKPT_DS}")
    B=$(age_of "maruanbottoni/${BEACON_DS}")
    [ "$B" -lt "$LIFE" ] && LIFE=$B
    [ "$LIFE" -gt 45 ] && break
    log "life signal ${LIFE}min old — an unlisted session may be alive, adopting (waiting)"
    sleep 600
  done

  # ---- provision: A100 preferred, L4 fallback ----------------------------
  GPU=A100
  if ! $CB new --gpu A100 -s "$SNAME" 2>&1 | grep -qiE "ready|created|assigned"; then
    GPU=L4
    if ! $CB new --gpu L4 -s "$SNAME" 2>&1 | grep -qiE "ready|created|assigned"; then
      log "no GPU available — retrying in 10 min"
      sleep 600
      continue
    fi
  fi
  BATCH=$([ "$GPU" = "A100" ] && echo 8 || echo 4)
  # Tier-aware patience for life signals (v1's measured numbers)
  STALE_MIN=$([ "$GPU" = "A100" ] && echo 45 || echo 60)
  GRACE_MIN=$([ "$GPU" = "A100" ] && echo 130 || echo 170)
  log "session $SNAME up ($GPU, batch $BATCH, stale>${STALE_MIN}m grace ${GRACE_MIN}m)"

  # ---- payload + detached launch ----------------------------------------
  JOBENV=$(mktemp)
  cat > "$JOBENV" << EOF
MAX_MINUTES=100000
BATCH=$BATCH
WINDOW=$WINDOW
METHOD=$METHOD
CKPT_DS=$CKPT_DS
BEACON_DS=$BEACON_DS
EPOCHS=$EPOCHS
SEED=$SEED
PATIENCE=$PATIENCE
DATA_MODE=$DATA_MODE
EOF
  D=$REPO/scripts/cloud/colab
  $CB upload "$D/fullrecipe_runner.sh"                     /content/fullrecipe_runner.sh   -s "$SNAME" >/dev/null
  $CB upload "$REPO/scripts/local/patch_fullrecipe_trainer.py" /content/patcher.py         -s "$SNAME" >/dev/null
  $CB upload "$REPO/configs/fullrecipe_splits.json"        /content/fullrecipe_splits.json -s "$SNAME" >/dev/null
  $CB upload "$REPO/configs/fullrecipe_volumes.txt"        /content/fullrecipe_volumes.txt -s "$SNAME" >/dev/null
  $CB upload "$JOBENV"                                     /content/job_env                -s "$SNAME" >/dev/null
  $CB upload "$TOKEN_FILE"                                 /content/kaggle_access_token    -s "$SNAME" >/dev/null
  rm -f "$JOBENV"
  $CB exec -s "$SNAME" -f "$D/launch_runner.py" --timeout 120
  log "runner launched detached on $SNAME"
  SESSION_START=$(date -u +%s)

  # ---- poll: STORES ONLY. ssh does not exist here, on purpose. -----------
  while true; do
    sleep 600
    AGEMIN=$(age_of "maruanbottoni/${CKPT_DS}")
    BEACONMIN=$(age_of "maruanbottoni/${BEACON_DS}")
    fetch_beacon
    BLINE=$(head -1 /tmp/orch_beacon/beacon.txt 2>/dev/null)
    SESSMIN=$(( ($(date -u +%s) - SESSION_START) / 60 ))
    log "$SNAME (${SESSMIN}m): ckpt-age ${AGEMIN}m beacon-age ${BEACONMIN}m | $BLINE"

    if grep -q "data-FATAL" /tmp/orch_beacon/beacon.txt && [ "$BEACONMIN" -lt 60 ]; then
      log "FATAL beacon — stopping $SNAME and exiting (fix config first)"
      $CB stop -s "$SNAME" >/dev/null 2>&1
      exit 2
    fi
    # terminal only if the done-beacon is FRESH — a stale STATE=done from a
    # previous session must not stop a session that is still staging
    if { grep -q "^STATE=done" /tmp/orch_beacon/beacon.txt \
         || head -1 /tmp/orch_beacon/beacon.txt | grep -qE " done "; } \
       && [ "$BEACONMIN" -lt 30 ] && [ "$SESSMIN" -gt 5 ]; then
      log "runner reports done: $(grep '^MSG=' /tmp/orch_beacon/beacon.txt | head -1)"
      log "waiting 5 min for the final ckpt push, then stopping $SNAME"
      sleep 300
      $CB stop -s "$SNAME" >/dev/null 2>&1
      break   # loop top re-reads the ckpt store and decides exit vs respawn
    fi
    LIFE=$AGEMIN
    [ "$BEACONMIN" -lt "$LIFE" ] && LIFE=$BEACONMIN
    if [ "$LIFE" -gt "$STALE_MIN" ] && [ "$SESSMIN" -gt "$GRACE_MIN" ]; then
      log "$SNAME: no life signal for ${LIFE}min (ckpt ${AGEMIN} / beacon ${BEACONMIN}) — respawning"
      $CB stop -s "$SNAME" >/dev/null 2>&1
      break
    fi
  done
  sleep 30
done
