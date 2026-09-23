#!/bin/bash
# Watch the two x138 kernels (row 167) and submit each exactly once on COMPLETE.
# Pushed v1 of both on 09-23. ERROR/CANCEL is logged and NOT submitted.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; COMP=biohub-cell-tracking-during-development
STATE=$ROOT/logs/x138_state; LOG=$ROOT/logs/autosubmit_x138.log
mkdir -p "$STATE"; log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') $*" >> "$LOG"; }
msg_for(){ case "$1" in
  pub947x_x138) echo "pub947x_x138 (rows 166/167): anvithpothula/biohub-x138 VERBATIM (author rank 52 @0.956; #1 of the competition scoreDescending kernel sort, own score unreadable). Different substrate from our 947 line (harmonic-v3) + flow-prior relink, readmit, sub-threshold gapfill. CONTROL arm for x138_densad. PRE-REGISTERED: >=0.949 = public x138 beats our anchor -> new anchor; 0.948 = ties densad on a DIFFERENT substrate -> decorrelated aggressive final-2; <=0.947 = sort-order lead was not a score lead, drop.";;
  pub947x_x138_densad) echo "pub947x_x138_densad (row 167): x138 + our densad density-adaptive relink table (row 127, the only LB-confirmed gain on the 947 line), ONE mechanism vs pub947x_x138. Not a clean transplant: in flow mode tight_um lands on the flow SEED gate, relaxed/bonus act as in densad; local smoke: 23/978 edges change on a synthetic middle-density volume. READ JOINTLY with the control: combo > x138 = densad transfers to the flow relinker -> anchor; combo = x138 = inert on this substrate; combo < x138 = densad fights the flow prior -> drop it.";;
esac; }
log "=== watcher start"
while :; do
  pending=0
  for k in pub947x_x138 pub947x_x138_densad; do
    [ -f "$STATE/$k.submitted" ] || [ -f "$STATE/$k.failed" ] && continue
    pending=1; ref=maruanbottoni/biohub-${k//_/-}
    st=$("$K" kernels status "$ref" 2>&1)
    case "$st" in
      *COMPLETE*) sub=$("$K" competitions submit "$COMP" -k "$ref" -v 1 -f submission.csv -m "$(msg_for "$k")" 2>&1)
                  log "$k: COMPLETE -> submit: $sub"; touch "$STATE/$k.submitted";;
      *ERROR*|*CANCEL*) log "$k: FAILED: $st"; touch "$STATE/$k.failed";;
      *) ;;
    esac
  done
  [ $pending = 0 ] && { log "=== all done"; exit 0; }
  sleep 300
done
