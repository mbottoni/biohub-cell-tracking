#!/usr/bin/env bash
# Rebuild the offline wheels for the no-internet submission kernel.
# Kaggle image is Python 3.12 / manylinux2014 x86_64.
set -euo pipefail
cd "$(dirname "$0")"
/bin/rm -f wheels/*.whl
python -m pip download tracksdata geff pyscipopt \
  --dest wheels --platform manylinux2014_x86_64 \
  --python-version 312 --abi cp312 --implementation cp --only-binary=:all:
echo "wheels: $(ls wheels/*.whl | wc -l)"
