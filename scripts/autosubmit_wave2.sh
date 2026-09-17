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
STATE=$ROOT/logs/wave2_state
LOG=$ROOT/logs/autosubmit_wave2.log
mkdir -p "$STATE"
log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

KERNELS=(pub947_divoff pub948h_port pub947_divnet pub947_densad)
ref_for(){ case "$1" in
  pub947_divoff) echo maruanbottoni/biohub-pub947-divoff;;
  pub948h_port) echo maruanbottoni/biohub-pub948h-port;;
  pub947_divnet) echo maruanbottoni/biohub-pub947-divnet;;
  pub947_densad) echo maruanbottoni/biohub-pub947-densad;;
esac; }
msg_for(){ case "$1" in
  pub947_divoff) echo "pub947_divoff PROBE (rows 100/103/106 protocol, row 128 wave): OUTPUT_SAFE_DIVISIONS=0 on the 0.947 base, ONE change vs 56232696. READ: div_J_hidden ~= 10 x (0.947 - LB - coupling), coupling unmeasured on this branch (~+0.001 on pub920). PRE-REGISTERED: <=0.912 -> div_J >= ~0.33, division term near-saturated, the 0.023 gap is EDGES -> fund edges lane; ~0.922 -> div_J ~= 0.24 (917-family level), moderate div headroom; >=0.930 -> div_J <= ~0.16, the 0.947 line still under-serves divisions on hidden -> divisions are THE lane.";;
  pub948h_port) echo "pub948h_port VERIFY (row 127): haideptry 0.948+ claim ported verbatim (their DivNet gate is a PROVEN double no-op -- wrong arch via strict=False + 6-D tensor crash swallowed by except, row 128 -- so any gain is density-adaptive relink alone). PRE-REGISTERED: >=0.948 = claim real, density-adaptive pays -> our densad graft on the stronger reyhan base should pay too; 0.947 = flat (their base is 0.946-era + inert divnet); <=0.946 = the claim is public-LB marketing, ignore the branch.";;
  pub947_divnet) echo "pub947_divnet ROW-126 COMBO (row 128): giorgosi DivNet CORRECTLY reconstructed (strict=True load, marker-last/mean-pool-4/per-crop-norm/InstanceNorm, local GT AUC 0.78 vs manifest OOF 0.845) as RANK-based confirm -- safe-div forks scored, bottom 40 pct vetoed -- PLUS global div cap 0.00375->0.0076. Justified by capup FLAT + dcthr NEGATIVE: hidden divisions are confirmation-precision-limited, not quota-limited. Fail-loud: kernel raises if ckpt does not strict-load. PRE-REGISTERED: >=0.949 combo pays -> sweep veto_frac; 0.948 = cap-neutral confirm gain; 0.947 flat = reconstructed ranking inert on hidden; <=0.946 = mis-ranks on hidden, drop the reconstruction.";;
  pub947_densad) echo "pub947_densad (row 127): haideptry density-adaptive motion-relink table VERBATIM on the reyhan 0.947 base (their claimed mechanism on the stronger substrate), ONE mechanism vs 56232696. Grouping by measured avg nodes/frame (<120 low /<400 mid />=400 high) -> per-group tight/relaxed/velocity/bonus (6.0 only on middle). Refines row 124: uniform bonus 6.0 lost -0.001; conditioned should not. PRE-REGISTERED: >=0.948 = density conditioning pays on the strong base; 0.947 flat = inert (their gain was base-specific or nonexistent); <=0.946 = the table hurts the stronger base -> their claim was noise.";;
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
