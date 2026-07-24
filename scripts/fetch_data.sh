#!/usr/bin/env bash
# Download the competition data into data/raw/.
# Requires a valid ~/.kaggle/kaggle.json (kaggle.com/settings -> API -> Create New Token).
set -euo pipefail

COMP="biohub-cell-tracking-during-development"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw"

if ! kaggle competitions list -s "$COMP" >/dev/null 2>&1; then
    echo "Kaggle auth failed. Regenerate the token at kaggle.com/settings -> API," >&2
    echo "save it to ~/.kaggle/kaggle.json, then: chmod 600 ~/.kaggle/kaggle.json" >&2
    exit 1
fi

mkdir -p "$DEST"

echo "== files available =="
kaggle competitions files "$COMP" -v

echo "== downloading to $DEST =="
kaggle competitions download -c "$COMP" -p "$DEST"

echo "== unzipping =="
for z in "$DEST"/*.zip; do
    [ -e "$z" ] || continue
    unzip -q -o "$z" -d "$DEST" && rm "$z"
done

echo "done. tree:"
find "$DEST" -maxdepth 2 | head -50
