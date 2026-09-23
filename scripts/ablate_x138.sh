#!/bin/bash
# x138 ablation runner (row 169). Pushes each arm when a GPU slot frees, and on COMPLETE
# pulls the commit output, verifies the ablated mechanism is really off (guard against
# silent no-ops), and scores visible-4. All arms run with the author's PRIVATE V1284 head
# OFF (mode 'zero') -- the published x138 cannot run outside its author's account.
# Submits ONLY pub947x_nh, and only if it beats the 947 line's best-ever visible-4
# (relink6 0.8982, row 154) AND flow/readmit/gapfill all fired in its log.
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; PY=$ROOT/.venv/bin/python
STATE=$ROOT/logs/x138_abl; LOG=$ROOT/logs/ablate_x138.log; RES=$ROOT/logs/x138_abl/results.tsv
mkdir -p "$STATE"; log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') $*" >> "$LOG"; }
ARMS=(pub947x_nh pub947x_nh_sub pub947x_nh_noflow pub947x_nh_noreadmit pub947x_nh_nogapfill)
COMP=biohub-cell-tracking-during-development
ref(){ echo "maruanbottoni/biohub-${1//_/-}"; }
finish(){ # $1 kernel
  local k=$1 d=$STATE/$1/out; mkdir -p "$d"
  "$K" kernels output "$(ref "$k")" -p "$d" >/dev/null 2>&1
  local lg; lg=$(ls "$d"/*.log 2>/dev/null | head -1)
  local ra gf ff
  ra=$(grep -c "readmitted [0-9]* discarded" "$lg" 2>/dev/null); gf=$(grep -c "gap filler: +" "$lg" 2>/dev/null)
  ff=$($PY -c "import pandas as p;d=p.read_csv('$d/run_stats.csv');print(int(d['motion_relink_flow_frames'].sum()) if 'motion_relink_flow_frames' in d else 0)" 2>/dev/null)
  local sc; sc=$($PY "$ROOT/scripts/local/visible4_calibration.py" score "$d/submission.csv" 2>&1 | tail -1)
  printf "%s\treadmit_lines=%s\tgapfill_lines=%s\tflow_frames=%s\t%s\n" "$k" "$ra" "$gf" "$ff" "$sc" >> "$RES"
  log "$k: DONE readmit=$ra gapfill=$gf flow=$ff | $sc"
  if [ "$k" = pub947x_nh ]; then
    local v4; v4=$(echo "$sc" | sed -n 's/.*score=\([0-9.]*\).*/\1/p')
    if [ "${ra:-0}" -gt 0 ] && [ "${gf:-0}" -gt 0 ] && [ "${ff:-0}" -gt 0 ] && \
       $PY -c "import sys; sys.exit(0 if float('${v4:-0}') > 0.8982 else 1)"; then
      sub=$("$K" competitions submit "$COMP" -k "$(ref "$k")" -v 1 -f submission.csv -m "pub947x_nh (rows 167/169): anvithpothula/biohub-x138 in its only PORTABLE form -- the author's private V1284 coordinate head OFF (mode zero); flow-prior relink + readmit + sub-threshold gapfill ON on the harmonic-v3 substrate. Visible-4 $v4 vs 947-line best 0.8982 (author's run WITH head 0.9180). PRE-REGISTERED: >=0.949 = x138 mechanisms transfer without the head -> new anchor, sweep its knobs; 0.948 = ties densad on a different substrate -> aggressive final-2; <=0.947 = the visible-4 gain was the head or visible-set tuning, drop." 2>&1)
      log "$k: SUBMITTED: $sub"
    else
      log "$k: NOT submitted (v4=$v4 readmit=$ra gapfill=$gf flow=$ff) -- gate failed"
    fi
  fi
}
log "=== ablation runner start"
while :; do
  pending=0
  for k in "${ARMS[@]}"; do
    [ -f "$STATE/$k.done" ] || [ -f "$STATE/$k.failed" ] && continue
    pending=1
    if [[ " ${ARMS[*]} " == *" $k "* ]] && [ ! -f "$STATE/$k.pushed" ]; then
      out=$(cd "$ROOT/kernels/$k" && "$K" kernels push -p . 2>&1)
      if echo "$out" | grep -q "successfully pushed"; then touch "$STATE/$k.pushed"; log "$k: PUSHED"
      else log "$k: push deferred: $(echo "$out" | tail -1)"; fi
      continue
    fi
    st=$("$K" kernels status "$(ref "$k")" 2>&1)
    case "$st" in
      *COMPLETE*) finish "$k"; touch "$STATE/$k.done";;
      *ERROR*|*CANCEL*) log "$k: FAILED: $st"; touch "$STATE/$k.failed";;
    esac
  done
  [ $pending = 0 ] && { log "=== all done"; exit 0; }
  sleep 300
done
