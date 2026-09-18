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
STATE=$ROOT/logs/wave4_state
LOG=$ROOT/logs/autosubmit_wave4.log
mkdir -p "$STATE"
log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

KERNELS=(pub947_consensus pub947_ourseed)
ref_for(){ case "$1" in
  pub947_consensus) echo maruanbottoni/biohub-pub947-consensus;;
  pub947_ourseed) echo maruanbottoni/biohub-pub947-ourseed;;
esac; }
msg_for(){ case "$1" in
  pub947_consensus) echo "pub947_consensus WAVE-4 ENSEMBLE (row 136): drop-only multi-postproc edge consensus on the densad 0.948 base -- ONE inference pass, 5 postproc variants (base/gap45/mintrack7/dcsd25/gap2step40) via the notebook's own pp_apply machinery with shared frame+DC caches; the base graph is pruned of nodes/edges reproduced by <3 of 5 variants (canonical keys make gap-close synthetic nodes vote-safe). Variance-reduction play: even at flat public this is the private-split-robust aggressive final-2 candidate (622-team public tie will scatter on private). PRE-REGISTERED: >=0.949 consensus denoises hidden edges -> sweep k and the variant set; 0.948 flat = keep as final-2 robustness candidate; <=0.947 = the pruned marginal edges were real, drop."
;;
  pub947_ourseed) echo "pub947_ourseed WAVE-4 DECORRELATION (rows 118/130/136): the SECONDARY seed swapped from pilkwang 314159 (memorised train, sha 9bac2fa0) to OUR honest-split retrain ep16 (guarded 120/12/24 split, sel 0.9550, sha eed647f2, drop-in verified: 136 tensors identical keys+shapes; dataset biohub-ourseed-s1). ONE mechanism vs densad 0.948. Tests whether a genuinely DECORRELATED secondary beats a correlated one on hidden -- the only play attacking adj 0.912 structurally (row 130). PRE-REGISTERED: >=0.949 = decorrelation pays -> try primary-swap and true 3-branch next; 0.948 flat = our 16-epoch seed matches their 400-epoch one on hidden (notable in itself); <=0.946 = our seed too weak as drop-in, revert."
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
