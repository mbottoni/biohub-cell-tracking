#!/bin/bash
# Row 174 finisher. Waits for the capture kernel (pub947x_v1284cap) to finish, pulls OUR V1284
# head + training report, gates it, publishes it as a PRIVATE dataset, waits until Kaggle has
# processed the dataset, and appends the submission kernel to the wave queue.
# Gate: held-out-by-movie residual must improve by >= 5% (author: -10.4% on 20 movies).
set -u
ROOT=/Users/maruanottoni/home/kaggle/biohub-cell-tracking
K=$ROOT/.venv/bin/kaggle; PY=$ROOT/.venv/bin/python
W=$ROOT/logs/wave; LOG=$W/runner.log; D=$W/ourhead_ds
log(){ echo "$(date -u '+%m-%d %H:%M:%SZ') [finish_ourhead] $*" >> "$LOG"; }
log "waiting for pub947x_v1284cap"
until [ -f "$W/pub947x_v1284cap.done" ] || [ -f "$W/pub947x_v1284cap.failed" ]; do sleep 120; done
[ -f "$W/pub947x_v1284cap.failed" ] && { log "capture kernel FAILED -- stop"; exit 1; }
mkdir -p "$W/cap_out" "$D"
"$K" kernels output maruanbottoni/biohub-pub947x-v1284cap -p "$W/cap_out" >/dev/null 2>&1
H=$(find "$W/cap_out" -name v1284_head.pt | head -1); R=$(find "$W/cap_out" -name v1284_train_report.json | head -1)
[ -n "$H" ] && [ -n "$R" ] || { log "no head/report in capture output -- stop"; exit 1; }
rel=$($PY -c "import json;print(json.load(open('$R'))['heldout_rel'])")
pairs=$($PY -c "import json;print(json.load(open('$R'))['pairs'])")
log "capture done: pairs=$pairs heldout_rel=$rel"
$PY -c "import sys; sys.exit(0 if float('$rel') <= -0.05 else 1)" || { log "GATE FAILED (heldout_rel=$rel > -0.05) -- not publishing, not submitting"; exit 1; }
cp "$H" "$D/v1284_head.pt"
cat > "$D/dataset-metadata.json" <<EOF
{"title": "biohub v1284 head ours", "id": "maruanbottoni/biohub-v1284-head-ours", "licenses": [{"name": "CC0-1.0"}]}
EOF
out=$("$K" datasets create -p "$D" 2>&1); log "dataset create: $(echo "$out" | tail -1)"
until "$K" datasets files maruanbottoni/biohub-v1284-head-ours 2>/dev/null | grep -q v1284_head.pt; do sleep 60; done
log "dataset ready; queueing pub947x_x138_ourhead"
echo "pub947x_x138_ourhead submit" >> "$W/queue.txt"
