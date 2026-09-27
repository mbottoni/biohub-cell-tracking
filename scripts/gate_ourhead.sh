#!/bin/bash
# Row 175: x138_ourhead with a visible-4 REJECT gate before submitting. Our head shifts z by
# +0.585 um on average (author's: +0.19); row 160 measured a GLOBAL bias correction at -0.005.
# Visible-4 volumes are EXCLUDED from our head's training, so they are an honest check of that
# failure mode. Submit iff visible-4 >= 0.9150 (author's own run, head trained possibly ON these
# volumes: 0.9180). ERROR -> no submit.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; PY=$ROOT/.venv/bin/python; COMP=biohub-cell-tracking-during-development
W=$ROOT/logs/wave; LOG=$W/runner.log; k=pub947x_x138_ourhead; ref=maruanbottoni/biohub-pub947x-x138-ourhead
log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') [gate_ourhead] $*" >> "$LOG"; }
until "$K" datasets files maruanbottoni/biohub-v1284-head-ours 2>/dev/null | grep -q v1284_head.pt; do sleep 60; done
log "dataset ready; queueing $k (nosubmit, gated)"
echo "$k nosubmit" >> "$W/queue.txt"
until [ -f "$W/$k.done" ] || [ -f "$W/$k.failed" ]; do sleep 120; done
[ -f "$W/$k.failed" ] && { log "$k FAILED -- no submit"; exit 1; }
v=$(cat "$W/$k.version"); O=$W/${k}_out; mkdir -p "$O"
for i in 1 2 3; do "$K" kernels output "$ref" -p "$O" >/dev/null 2>&1; [ -f "$O/submission.csv" ] && break; done
sc=$($PY "$ROOT/scripts/local/visible4_calibration.py" score "$O/submission.csv" 2>&1 | tail -1)
v4=$(echo "$sc" | sed -n 's/.*score=\([0-9.]*\).*/\1/p'); log "$k v$v visible-4: $sc"
if $PY -c "import sys; sys.exit(0 if float('${v4:-0}') >= 0.9150 else 1)"; then
  sub=$("$K" competitions submit "$COMP" -k "$ref" -v "$v" -f submission.csv -m "$(cat "$ROOT/kernels/$k/SUBMIT_MSG.txt") Visible-4 (honest: excluded from our head's training) = $v4 vs author-head run 0.9180." 2>&1)
  log "$k: SUBMITTED: $sub"
else
  log "$k: NOT submitted -- visible-4 $v4 < 0.9150 (row-160 failure mode)"
fi
