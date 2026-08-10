#!/bin/bash
# One-time: build the biohub-w3-subset dataset (60 volumes) for fast session staging.
set -euo pipefail
cd /content
phase() { echo "$1" > /content/PHASE; echo "== $1 =="; }

phase deps
pip -q install -U kaggle >/dev/null

phase auth
mkdir -p /root/.kaggle
echo -n "$KAGGLE_API_TOKEN" > /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token

phase download
rm -f biohub-cell-tracking-during-development.zip*
kaggle competitions download -c biohub-cell-tracking-during-development -p /content

phase extract
mkdir -p /content/subset
PATTERNS=""
while read -r V; do
  PATTERNS="$PATTERNS train/$V.zarr/* train/$V.geff/*"
done < /content/w3_subset_volumes.txt
cd /content/subset
unzip -q /content/biohub-cell-tracking-during-development.zip $PATTERNS || true
# VERIFY every volume landed; retry individually (the bulk pattern list silently
# skipped 6bba_fc83837d on the first pack -- a selection volume, which crashed
# every training session at test-load).
while read -r V; do
  for EXT in zarr geff; do
    if [ ! -d "train/$V.$EXT" ]; then
      echo "retrying missing train/$V.$EXT"
      unzip -q /content/biohub-cell-tracking-during-development.zip "train/$V.$EXT/*"
    fi
    [ -d "train/$V.$EXT" ] || { echo "FATAL: train/$V.$EXT unextractable"; exit 1; }
  done
done < /content/w3_subset_volumes.txt
echo "verified $(ls train | grep -c '.zarr$') zarr / $(ls train | grep -c '.geff$') geff"

phase zip
cd /content/subset
zip -q -r -0 /content/w3subset.zip train
ls -la /content/w3subset.zip

phase upload
ST=/content/subset_staging
mkdir -p "$ST"
mv /content/w3subset.zip "$ST/"
printf '{"id": "maruanbottoni/biohub-w3-subset", "title": "biohub-w3-subset", "licenses": [{"name": "CC0-1.0"}]}' > "$ST/dataset-metadata.json"
kaggle datasets version -p "$ST" -m "re-pack with per-volume verification" || kaggle datasets create -p "$ST" --dir-mode skip
echo done > /content/DONE
phase done
