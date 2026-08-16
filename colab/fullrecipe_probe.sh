#!/bin/bash
# PHASE 0 of the full-recipe retrain — the feasibility spike, run INSIDE Colab.
#
# WHY: the plan's single biggest unknown is DISK. 199 volumes uncompressed is
# ~158 GB (measured on the extSSD); the competition zip is far smaller because
# light-sheet volumes are mostly background. The w3 probe staged 59 volumes and
# worked, so we know >=50 GB was available then -- but we have never measured
# the ceiling, and the whole training-set size depends on it.
#
# This spike answers, before any real credit burn:
#   1. how much disk does an A100 runtime actually give us?
#   2. how big is one volume once extracted?
#   3. therefore: how many volumes can we train on?
#   4. does Kaggle auth + competition download work from inside Colab, and how fast?
#   5. what GPU/driver/torch do we actually get?
#
# Deliberately does NOT train. Repo rule: smoke before you spend.
set -u
say() { echo "[probe $(date -u +%H:%M:%S)] $*"; }

say "=============================================================="
say "PHASE 0 — full-recipe feasibility spike"
say "=============================================================="

say ""
say "--- 1. RUNTIME ---------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || say "no nvidia-smi (CPU runtime?)"
say "cpus       : $(nproc)"
say "ram        : $(free -g | awk '/^Mem:/{print $2" GB"}')"
python3 -c "import torch;print(f'[probe] torch      : {torch.__version__} cuda={torch.cuda.is_available()}')" 2>/dev/null

say ""
say "--- 2. DISK (the number that sizes the whole run) -----------"
df -h /content | tail -1 | awk '{print "[probe] /content   : size "$2"  used "$3"  AVAIL "$4}'
AVAIL_GB=$(df -BG /content | tail -1 | awk '{gsub("G","",$4); print $4}')
say "avail_gb   : $AVAIL_GB"

say ""
say "--- 3. KAGGLE AUTH -----------------------------------------"
pip -q install -U kaggle >/dev/null 2>&1
mkdir -p /root/.kaggle
echo -n "$KT" > /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token
kaggle competitions list -s biohub 2>&1 | head -3 | sed 's/^/[probe] /'

say ""
say "--- 4. COMPETITION DOWNLOAD --------------------------------"
cd /content
T0=$(date +%s)
kaggle competitions download -c biohub-cell-tracking-during-development -p /content 2>&1 | tail -2 | sed 's/^/[probe] /'
T1=$(date +%s)
ZIP=/content/biohub-cell-tracking-during-development.zip
if [ ! -f "$ZIP" ]; then say "FATAL: zip not present after download"; exit 1; fi
ZGB=$(du -m "$ZIP" | awk '{printf "%.2f", $1/1024}')
say "zip_size   : ${ZGB} GB"
say "dl_seconds : $((T1-T0))  (~$(python3 -c "print(f'{$ZGB*1024/max(1,$T1-$T0):.0f}')") MB/s)"

say ""
say "--- 5. PER-VOLUME EXTRACTED SIZE ---------------------------"
# Extract a few TRAIN volumes and measure. Volume names come from the staged
# list so we measure exactly what we will train on.
mkdir -p /content/probe_extract
head -3 /content/fullrecipe_volumes.txt > /content/probe3.txt
while read -r V; do
  ( cd /content/probe_extract && unzip -q "$ZIP" "train/$V.zarr/*" "train/$V.geff/*" 2>/dev/null || true )
  SZ=$(du -sm "/content/probe_extract/train/$V.zarr" 2>/dev/null | awk '{print $1}')
  say "  $V : ${SZ:-MISSING} MB"
done < /content/probe3.txt
TOT=$(du -sm /content/probe_extract 2>/dev/null | awk '{print $1}')
N=$(wc -l < /content/probe3.txt)
PERVOL=$(python3 -c "print(f'{$TOT/$N:.0f}')")
say "mean_per_volume_mb : $PERVOL"

say ""
say "--- 6. VERDICT ---------------------------------------------"
python3 - << PYEOF
avail = $AVAIL_GB
zip_gb = $ZGB
per_mb = $PERVOL
# The zip must stay on disk during extraction, so budget for it plus headroom
# for checkpoints, the repo, pip caches and the trainer's own scratch.
headroom = 12
usable = avail - zip_gb - headroom
fits = int(usable * 1024 / per_mb)
print(f"[probe] avail {avail} GB - zip {zip_gb:.1f} GB - headroom {headroom} GB = {usable:.1f} GB usable")
print(f"[probe] per volume ~{per_mb} MB  =>  MAX VOLUMES THAT FIT: {fits}")
print(f"[probe] all 199 volumes would need ~{199*per_mb/1024:.0f} GB")
print(f"[probe] RECOMMENDED --train-cap : {max(0, fits-12)}   (fits minus the 12 selection volumes)")
if fits < 60:
    print("[probe] ⚠ BELOW the 59-volume w3 probe — streaming or sharding would be required")
elif fits < 172:
    print("[probe] ⚠ cannot hold all 172 staged volumes; train on the cap above")
else:
    print("[probe] ✅ the full 172-volume staged set fits")
PYEOF

say ""
say "--- 7. DONE ------------------------------------------------"
say "probe complete; no training performed, no checkpoints written"
