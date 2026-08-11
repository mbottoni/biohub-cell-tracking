#!/bin/bash
# Fire the pub162_stack submission the moment the weekly GPU quota frees.
#
# Why this exists: the config is measured and the kernel is built, but
# `kaggle kernels push` rejects with "Maximum weekly GPU quota of 30.00 hours
# reached" (dettrain rows 42-50 spent the week on 08-08). No commit => no
# submission, and the 5/day submission budget has been idle since 08-07. The
# reset boundary is not exposed by the API, so poll rather than guess it.
#
# Sequence, once push succeeds: wait for the commit run to COMPLETE, pull its
# output, REFUSE to submit unless submission.csv is present and non-trivial,
# then submit once. A sentinel file makes the submit step exactly-once even if
# this script is restarted.
#
#   nohup bash scripts/autosubmit_stack.sh >> <log> 2>&1 &
set -u
REPO=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$REPO/.venv/bin/kaggle
COMP=biohub-cell-tracking-during-development
KERNEL=maruanbottoni/biohub-pub162-stack
SENTINEL=$REPO/.autosubmit_stack.done
OUTDIR=/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad/stack_out
MSG='pub162_stack: ILP disappearance 1.5->2.0 + gap-close clamp LIFTED with max_gap 3. Local official evalset-24: 0.9099 base -> 0.9119 (disapp) -> 0.9135 (+gap3). Gain is detection-FN recovery: node recall 0.9829->0.9878, division FP 37->18. PRE-REGISTERED: expect ~0.917 at row-38 half-strength translation, ~0.9186 at full; flat kills the stack.'

log() { echo "[autosubmit $(date -u +%H:%M)] $*"; }

if [ -f "$SENTINEL" ]; then log "sentinel present — already submitted, exiting"; exit 0; fi

log "waiting for GPU quota; polling push every 20 min"
while true; do
  OUT=$($K kernels push -p "$REPO/kernels/pub162_stack" 2>&1)
  if echo "$OUT" | grep -qi "quota"; then
    sleep 1200
    continue
  fi
  if echo "$OUT" | grep -qi "error"; then
    log "PUSH ERROR (not quota): $OUT"
    sleep 1200
    continue
  fi
  log "PUSH ACCEPTED: $OUT"
  break
done

# ---- commit run: ~21 min for this notebook on the dev-4 sample
log "waiting for the commit run to finish"
for _ in $(seq 1 90); do
  sleep 60
  ST=$($K kernels status "$KERNEL" 2>&1)
  case "$ST" in
    *COMPLETE*) log "commit COMPLETE"; break ;;
    *ERROR*|*CANCEL*) log "FATAL: commit failed — $ST"; exit 1 ;;
  esac
done

# ---- gate: never submit an artifact we have not inspected
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
$K kernels output "$KERNEL" -p "$OUTDIR" >/dev/null 2>&1
CSV="$OUTDIR/submission.csv"
if [ ! -f "$CSV" ]; then log "FATAL: no submission.csv in kernel output — NOT submitting"; exit 1; fi
ROWS=$(wc -l < "$CSV")
if [ "$ROWS" -lt 1000 ]; then log "FATAL: submission.csv has only $ROWS rows — NOT submitting"; exit 1; fi
log "output gate PASS: $ROWS rows"

VER=$($K kernels list -m --page-size 100 2>/dev/null | grep -c "biohub-pub162-stack")
log "submitting version 1"
$K competitions submit "$COMP" -k "$KERNEL" -v 1 -f submission.csv -m "$MSG" 2>&1 | tee -a /dev/stderr
touch "$SENTINEL"
log "SUBMITTED — sentinel written. Scoring rerun on hidden test takes ~4-6 h."
