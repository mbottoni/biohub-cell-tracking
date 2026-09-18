#!/bin/bash
# Unattended push -> wait COMPLETE -> submit-once orchestrator for the pub947 wave (row 123).
#
# Four kernels, one mechanism each, all vs the zhincez-verified 0.947 base
# (same pilkwang weights as our 0.923; see row 122):
#   port    - verbatim port (control / new safe base)
#   dcthr15 - IDEA-1 gradient: DC safe-div threshold 0.20->0.15 (+ guard expected-value)
#   relink6 - IDEA-2 stranded adj lever: MOTION_RELINK_LEARNED_BONUS 1.0->6.0 (row 73)
#   capup   - IDEA-3 recall probe: SAFE_DIV_GLOBAL_FRAC_CAP 0.00375->0.0076 (caps bind, row 111)
#
# GPU concurrency is 2: kernels are pushed opportunistically as slots free.
# Each kernel is submitted exactly once (marker files in $STATE). ERROR/CANCEL logs and skips.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle
COMP=biohub-cell-tracking-during-development
STATE=$ROOT/logs/wave3_state
LOG=$ROOT/logs/autosubmit_wave3.log
mkdir -p "$STATE"
log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

KERNELS=(pub947_hidense pub947_det096 pub947_sectta1)
ref_for(){ case "$1" in
  pub947_hidense) echo maruanbottoni/biohub-pub947-hidense;;
  pub947_det096) echo maruanbottoni/biohub-pub947-det096;;
  pub947_sectta1) echo maruanbottoni/biohub-pub947-sectta1;;
esac; }
msg_for(){ case "$1" in
  pub947_hidense) echo "pub947_hidense WAVE-3 EDGES (rows 130/131): density-table high-group learned_bonus 1.0->3.0, ONE knob vs densad 0.948 (56313419). Rationale: densad proved density-conditioned relink pays; the HIGH-density group is nearly unserved (bonus 1.0 = baseline) and dense volumes dominate the edge-weighted scorer average (row 101). PRE-REGISTERED: >=0.949 = high-density relink pays, sweep 6.0 next + condition more knobs on density; 0.948 flat = high group is relink-saturated; <=0.947 = dense volumes punish relink aggression, keep 1.0."
;;
  pub947_det096) echo "pub947_det096 WAVE-3 EDGES (rows 110/130): BIOHUB_DET_THRESHOLD 0.965->0.96 on the densad base (guard expected-value updated), ONE knob vs 56313419 (0.948). Rationale: hiranorm's LB-logged mechanism -- probability-averaged dual detection kills one-sided detections; their node-count-parity lowering scored holdout +0.0185; never LB-tested on any 947-family base. Node-ratio is SIGNED (under-detection bonus) so this is the conservative step, not max recall. PRE-REGISTERED: >=0.949 = detection parity pays, probe 0.9625/0.955 bracket next; 0.948 flat = 0.965 already at parity; <=0.947 = added detections are net noise on hidden, close the knob."
;;
  pub947_sectta1) echo "pub947_sectta1 WAVE-3 EDGES (rows 122/130): BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT 0.75->1.0 (guard updated), ONE knob vs 56313419 (0.948). Rationale: the 0.947 author's last two paid steps were on this TTA axis and they stopped at 0.75 without a published reason; primary uses full 8-view mean already. PRE-REGISTERED: >=0.949 = full secondary spatial ensembling pays; 0.948 flat = 0.75 was optimal-enough; <=0.947 = the 0.25 original-view residue was load-bearing (consensus diversity), restore."
;;
esac; }

log "=== orchestrator start; pushed markers: $(ls $STATE 2>/dev/null | tr '\n' ' ')"
while :; do
  alldone=1
  for k in "${KERNELS[@]}"; do
    [ -f "$STATE/$k.submitted" ] || [ -f "$STATE/$k.failed" ] && continue
    alldone=0
    ref=$(ref_for "$k")
    if [ ! -f "$STATE/$k.pushed" ]; then
      out=$(cd "$ROOT/kernels/$k" && "$K" kernels push -p . 2>&1)
      if echo "$out" | grep -q "successfully pushed"; then
        ver=$(echo "$out" | sed -n 's/.*Kernel version \([0-9][0-9]*\).*/\1/p'); : "${ver:=1}"
        echo "$ver" > "$STATE/$k.version"
        log "$k: PUSHED v$ver ($out)"; touch "$STATE/$k.pushed"
      else
        log "$k: push blocked: $out"
      fi
      continue
    fi
    st=$("$K" kernels status "$ref" 2>&1)
    case "$st" in
      *COMPLETE*)
        log "$k: COMPLETE -> submitting"
        v=$(cat "$STATE/$k.version" 2>/dev/null || echo 1)
        sub=$("$K" competitions submit "$COMP" -k "$ref" -v "$v" -f submission.csv -m "$(msg_for "$k")" 2>&1)
        log "$k: submit result: $sub"
        if echo "$sub" | grep -qi "error\|400\|denied"; then touch "$STATE/$k.failed"; else touch "$STATE/$k.submitted"; fi
        ;;
      *ERROR*|*CANCEL*)
        log "$k: run FAILED ($st) -- no submission"; touch "$STATE/$k.failed" ;;
      *)
        : ;; # RUNNING/QUEUED
    esac
  done
  [ "$alldone" = 1 ] && { log "=== all kernels terminal; exiting"; break; }
  sleep 90
done
