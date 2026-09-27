#!/bin/bash
# Watch the two x138 kernels (row 167) and submit each exactly once on COMPLETE.
# v1 of both ERRORED 09-23 (private V1284 head, row 169). v2 pushed 09-27 ~15:40Z with the
# now-public head dataset attached (row 170). ERROR/CANCEL is logged and NOT submitted.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; COMP=biohub-cell-tracking-during-development
STATE=$ROOT/logs/x138_state_v2; LOG=$ROOT/logs/autosubmit_x138.log
mkdir -p "$STATE"; log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') $*" >> "$LOG"; }
msg_for(){ case "$1" in
  pub947x_x138) echo "pub947x_x138 v2 (rows 166-171): anvithpothula 'Biohub 0.953 LB ORIGINAL' VERBATIM (byte-identical to x138) + its now-public V1284 coordinate head (anvithpothula/biohub-v1284-head-s075). 397-team 0.953 shelf; our anchor densad 0.948. CONTROL arm for x138_densad. PRE-REGISTERED: 0.953 = port reproduces -> new anchor + safe final-2; >=0.954 = lucky tie-break on hidden, same; <=0.952 = port does not reproduce under our account -> investigate before final-2.";;
  pub947x_x138_densad) echo "pub947x_x138_densad v2 (rows 167/170): x138 0.953 ORIGINAL + V1284 head + our densad density-adaptive relink table (row 127, the only LB-confirmed gain on the 947 line), ONE mechanism vs pub947x_x138. Not a clean transplant: in flow mode tight_um lands on the flow SEED gate, relaxed/bonus act as in densad; local smoke: 23/978 edges change on a synthetic middle-density volume. READ JOINTLY with the control: combo > x138 = densad transfers to the flow relinker -> anchor; combo = x138 = inert on this substrate; combo < x138 = densad fights the flow prior -> drop it.";;
esac; }
log "=== watcher start"
while :; do
  pending=0
  for k in pub947x_x138 pub947x_x138_densad; do
    [ -f "$STATE/$k.submitted" ] || [ -f "$STATE/$k.failed" ] && continue
    pending=1; ref=maruanbottoni/biohub-${k//_/-}
    st=$("$K" kernels status "$ref" 2>&1)
    case "$st" in
      *COMPLETE*) sub=$("$K" competitions submit "$COMP" -k "$ref" -v 2 -f submission.csv -m "$(msg_for "$k")" 2>&1)
                  log "$k: COMPLETE -> submit: $sub"; touch "$STATE/$k.submitted";;
      *ERROR*|*CANCEL*) log "$k: FAILED: $st"; touch "$STATE/$k.failed";;
      *) ;;
    esac
  done
  [ $pending = 0 ] && { log "=== all done"; exit 0; }
  sleep 300
done
