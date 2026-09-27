#!/bin/bash
# Final-days wave runner (rows 172+). Operator-authorised 09-27: "try the approaches end to
# end, leading to a real submission always".
#
# Queue: logs/wave/queue.txt, one line per kernel dir under kernels/:  <kernel_dir> <submit|nosubmit>
# Lines may be appended while the runner is alive; it re-reads the queue every cycle.
# For each entry: push when a GPU slot is free (retry on the 2-session cap), record the pushed
# version, wait for COMPLETE, then submit THAT version once with kernels/<k>/SUBMIT_MSG.txt.
# ERROR/CANCEL is logged and never submitted. Markers live in logs/wave/<k>.{version,submitted,failed,done}.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; COMP=biohub-cell-tracking-during-development
W=$ROOT/logs/wave; LOG=$W/runner.log; Q=$W/queue.txt
mkdir -p "$W"; touch "$Q"
log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') $*" >> "$LOG"; }
ref(){ echo "maruanbottoni/biohub-${1//_/-}"; }
log "=== wave runner start"
while :; do
  while read -r k mode; do
    [ -z "${k:-}" ] && continue
    case "$k" in \#*) continue;; esac
    [ -f "$W/$k.done" ] || [ -f "$W/$k.failed" ] && continue
    if [ ! -f "$W/$k.version" ]; then
      out=$(cd "$ROOT/kernels/$k" && "$K" kernels push -p . 2>&1)
      if echo "$out" | grep -q "successfully pushed"; then
        v=$(echo "$out" | sed -n 's/.*Kernel version \([0-9][0-9]*\).*/\1/p'); echo "${v:-1}" > "$W/$k.version"
        log "$k: PUSHED v${v:-1}"
      else
        log "$k: push deferred: $(echo "$out" | tail -1)"
      fi
      continue
    fi
    st=$("$K" kernels status "$(ref "$k")" 2>&1)
    case "$st" in
      *COMPLETE*)
        v=$(cat "$W/$k.version")
        if [ "$mode" = submit ]; then
          sub=$("$K" competitions submit "$COMP" -k "$(ref "$k")" -v "$v" -f submission.csv \
                -m "$(cat "$ROOT/kernels/$k/SUBMIT_MSG.txt")" 2>&1)
          log "$k: COMPLETE v$v -> submit: $sub"
        else
          log "$k: COMPLETE v$v (nosubmit)"
        fi
        touch "$W/$k.done";;
      *ERROR*|*CANCEL*) log "$k: FAILED: $st"; touch "$W/$k.failed";;
    esac
  done < "$Q"
  sleep 240
done
