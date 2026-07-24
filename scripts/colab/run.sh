#!/usr/bin/env bash
# Train the detector on a Colab GPU via the official colab CLI, headless.
#
# `colab run` provisions a fresh GPU VM, runs vm_job.py on it (pull ctrack + data from Kaggle,
# train the fold, push weights back to Kaggle), retrieves outputs, and tears the VM down.
#
# ONE-TIME AUTH (you must do this interactively — it's your Google/Colab login):
#   colab sessions          # prints an OAuth URL; approve in a browser, paste the code back
#
# Then, driven entirely from scripts:
#   scripts/colab/run.sh [GPU] [FOLD] [EPOCHS]
#   e.g.  scripts/colab/run.sh A100 0 25
#
# The Kaggle token comes from .env (KAGGLE_API_TOKEN) and is forwarded to vm_job as an argument
# (expanded from the env var, so it isn't a literal in shell history). Weights land on Kaggle as
# maruanbottoni/biohub-detector-weights, which the submission kernel attaches.
set -euo pipefail

GPU="${1:-A100}"
FOLD="${2:-0}"
EPOCHS="${3:-25}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN in .env}"

if ! colab sessions >/dev/null 2>&1; then
    echo "colab CLI is not authenticated. Run this once interactively, then re-run:" >&2
    echo "    colab sessions   # approve the OAuth URL and paste the code" >&2
    exit 1
fi

echo ">> colab run on $GPU: fold $FOLD, $EPOCHS epochs"
colab run --gpu "$GPU" "$ROOT/scripts/colab/vm_job.py" \
    "$KAGGLE_API_TOKEN" --fold "$FOLD" --epochs "$EPOCHS"

echo ">> done — weights on Kaggle: maruanbottoni/biohub-detector-weights"
echo "   attach it to kernel_infer and submit."
