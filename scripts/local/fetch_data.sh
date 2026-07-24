#!/usr/bin/env bash
# Download the competition data into data/raw/.
#
# Auth: a bearer token in KAGGLE_API_TOKEN (kaggle.com/settings -> API -> Create New Token).
# Put it in a gitignored .env at the repo root: echo 'export KAGGLE_API_TOKEN=KGAT_...' > .env
# NOTE: this needs Kaggle CLI >= 2.x. The legacy 1.5.13 only speaks username/key basic auth
# and will 401 against a KGAT_ token no matter how it is supplied.
set -euo pipefail

COMP="biohub-cell-tracking-during-development"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/data/raw"

# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

KAGGLE="$ROOT/.venv/bin/kaggle"
[ -x "$KAGGLE" ] || KAGGLE="kaggle"

if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    echo "KAGGLE_API_TOKEN is unset. Create a token at kaggle.com/settings -> API," >&2
    echo "then: echo 'export KAGGLE_API_TOKEN=KGAT_...' > $ROOT/.env && chmod 600 $ROOT/.env" >&2
    exit 1
fi

if ! "$KAGGLE" competitions files "$COMP" >/dev/null 2>&1; then
    echo "Kaggle auth failed. Check the token is current and the CLI is >= 2.x:" >&2
    echo "  $KAGGLE --version" >&2
    exit 1
fi

mkdir -p "$DEST"

echo "== downloading to $DEST (~8 GB) =="
"$KAGGLE" competitions download -c "$COMP" -p "$DEST"

echo "== unzipping =="
for z in "$DEST"/*.zip; do
    [ -e "$z" ] || continue
    unzip -q -o "$z" -d "$DEST" && rm "$z"
done

echo "done:"
find "$DEST" -maxdepth 2 -not -path '*/.*' | head -40
