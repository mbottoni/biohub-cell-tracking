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
STATE=$ROOT/logs/pub947_state
LOG=$ROOT/logs/autosubmit_pub947.log
mkdir -p "$STATE"
log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

KERNELS=(pub947_port pub947_dcthr15 pub947_relink6 pub947_capup)
ref_for(){ case "$1" in
  pub947_port) echo maruanbottoni/biohub-pub947-port;;
  pub947_dcthr15) echo maruanbottoni/biohub-pub947-dcthr15;;
  pub947_relink6) echo maruanbottoni/biohub-pub947-relink6;;
  pub947_capup) echo maruanbottoni/biohub-pub947-capup;;
esac; }
msg_for(){ case "$1" in
  pub947_port) echo "pub947_port (row 122/123): zhincez's LB-verified 0.947 runnable (their sub 56207450) VERBATIM under our account -- reyhanksatria 0.933->0.947 progression on the SAME pilkwang weights as our 0.923 (primary 12f6881e / secondary 9bac2fa0 / DC 8040999a). PRE-REGISTERED: expect 0.947 (new best, safe final-2 pick); <=0.945 = port does not transfer for us.";;
  pub947_dcthr15) echo "pub947_dcthr15 IDEA-1 (row 123): DeepCenter safe-div threshold 0.20->0.15, ONE knob vs pub947_port (in-notebook config guard expected-value updated to match -- guard passing proves the knob reached the code). Rationale: the author's own last paid step was 0.25->0.20 alongside DC-TTA; the gradient direction is live. PRE-REGISTERED: >=0.948 gradient continues (sweep 0.10 next); 0.947 flat = exhausted at 0.20; <=0.946 = 0.20 is the optimum, close the knob.";;
  pub947_relink6) echo "pub947_relink6 IDEA-2 (row 123): MOTION_RELINK_LEARNED_BONUS 1.0->6.0, ONE knob vs pub947_port. Rationale: row 73 measured +0.0026 adj (our best-ever association delta) on pub162, cancelled there by division tax; this substrate's division machinery (DC-TTA veto + symmetry gate) may have lifted the tax. NOTE: the in-kernel held-out ppsweep has its own bonus125 variant and may out-select 6.0 internally. PRE-REGISTERED: >=0.948 = stranded lever unblocked (retest DET parity next); 0.947 flat = inert or internally rejected; <=0.946 = still division-taxed, close.";;
  pub947_capup) echo "pub947_capup IDEA-3 (row 123): SAFE_DIV_GLOBAL_FRAC_CAP 0.00375->0.0076 (= the frame cap), ONE knob vs pub947_port. Rationale: caps measurably bind (row 111: clipped at 128 on our graft); this admits the DC-TTA-vetoed division tail beyond the global cap. PRE-REGISTERED: >=0.948 = caps were clipping REAL divisions -> fund the appearance-confirm build (div CNN AUC 0.863 + volume-drop signature); 0.947 flat = cap not binding on hidden; <=0.946 = the admitted tail is FP-heavy -> precision-first, cap stays.";;
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
