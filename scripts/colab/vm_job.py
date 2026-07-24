#!/usr/bin/env python3
"""All-in-one Colab VM job: pull code+data from Kaggle, train the detector, push weights back.

Meant to run on a fresh Colab GPU VM via the official colab CLI:

    colab run --gpu A100 scripts/colab/vm_job.py            # ephemeral: provision, run, teardown

The Kaggle token is read from /root/.kaggle/kaggle_token (upload it first) or the
KAGGLE_API_TOKEN env var — never passed on the command line. Config via env:
FOLD, EPOCHS, BATCH, BASE, N_VOLUMES.

This flattens colab/train_detector_colab.ipynb into a script so it runs headless. Weights are
uploaded to the `biohub-detector-weights` Kaggle dataset, which the submission kernel attaches.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMP = "biohub-cell-tracking-during-development"
WORK = Path("/content") if Path("/content").exists() else Path("/workspace")

_ap = argparse.ArgumentParser()
_ap.add_argument("token", nargs="?", help="Kaggle KGAT token (or set KAGGLE_API_TOKEN / upload a file)")
_ap.add_argument("--fold", type=int, default=int(os.environ.get("FOLD", 0)))
_ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 25)))
_ap.add_argument("--base", type=int, default=int(os.environ.get("BASE", 32)))
_ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 8)))
_ap.add_argument("--n-volumes", type=int, default=int(os.environ.get("N_VOLUMES", 150)))
ARGS, _ = _ap.parse_known_args()
N_VOLUMES = ARGS.n_volumes


def log(*a):
    print(*a, flush=True)


def token() -> str:
    tok = ARGS.token or os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        for p in ("/root/.kaggle/kaggle_token", str(Path.home() / ".kaggle/kaggle_token")):
            if Path(p).exists():
                tok = Path(p).read_text().strip()
                break
    if not tok:
        raise SystemExit("no Kaggle token (pass as arg, set KAGGLE_API_TOKEN, or upload a file)")
    return tok


TOK = token()
HDR = {"Authorization": "Bearer " + TOK, "User-Agent": "colab-vm"}


def api(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=120).read())


def get_bytes(url, retries=6):
    for a in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=300).read()
        except Exception:
            time.sleep(5 * (a + 1))
    raise RuntimeError("download failed: " + url)


# ---- deps ----
log("== installing deps ==")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr", "scipy"], check=False)

# ---- ctrack package ----
log("== fetching ctrack ==")
ctrack_zip = get_bytes(
    f"https://www.kaggle.com/api/v1/datasets/download/maruanbottoni/ctrack-src"
)
(WORK / "ctrack_src.zip").write_bytes(ctrack_zip)
import shutil
import zipfile

pkg_root = WORK / "pkg"
if pkg_root.exists():
    shutil.rmtree(pkg_root)
(pkg_root / "ctrack").mkdir(parents=True)
with zipfile.ZipFile(WORK / "ctrack_src.zip") as zf:
    zf.extractall(WORK / "_ctrack_raw")
# the dataset holds the package contents at the root -> materialise pkg/ctrack/
raw = WORK / "_ctrack_raw"
src = next((p.parent for p in raw.glob("**/detect/unet3d.py")), raw)
for item in src.iterdir():
    dst = pkg_root / "ctrack" / item.name
    (shutil.copytree if item.is_dir() else shutil.copy)(item, dst)
sys.path.insert(0, str(pkg_root))
os.environ["CTRACK_SRC"] = str(pkg_root)
import ctrack  # noqa: F401

log("ctrack ready")

# ---- data ----
log(f"== downloading {N_VOLUMES} volumes ==")
files, pt, page = [], None, 0
while True:
    u = f"https://www.kaggle.com/api/v1/competitions/data/list/{COMP}?pageSize=200"
    if pt:
        u += "&pageToken=" + urllib.parse.quote(pt)
    for a in range(6):
        try:
            d = api(u)
            break
        except Exception:
            time.sleep(5 * (a + 1))
    files += d.get("files", [])
    pt = d.get("nextPageTokenNullable")
    page += 1
    if page % 10 == 0:
        log(f"  listing... page {page}, {len(files)} files")
    if not pt or not d.get("files"):
        break
    time.sleep(1)

stores = collections.defaultdict(list)
for f in files:
    m = re.match(r"train/([^/]+)\.(zarr|geff)/", f["name"])
    if m:
        stores[m.group(1)].append(f["name"])
chosen = sorted(stores)[:N_VOLUMES]
want = sorted({n for s in chosen for n in stores[s]})
data_root = WORK / "data"


def dl(name):
    dest = data_root / name
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMP}/" + urllib.parse.quote(name)
    dest.write_bytes(get_bytes(url))


t0 = time.time()
log(f"  downloading {len(want)} files ...")
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, _ in enumerate(ex.map(dl, want)):
        if (i + 1) % 100 == 0:  # frequent heartbeat so the CLI output stream never stalls
            log(f"  {i+1}/{len(want)} files  {time.time()-t0:.0f}s")
log(f"downloaded {len(chosen)} volumes in {time.time()-t0:.0f}s")

# ---- train ----
# The training loop lives in the ctrack package (downloaded above), so vm_job.py is fully
# self-contained for `colab exec -f` / `colab run` — no sibling files needed.
log("== training ==")
out_dir = WORK / "out"
from ctrack.detect.train import train_fold

train_fold(
    data_root / "train", data_root / "train",
    fold=ARGS.fold, epochs=ARGS.epochs, batch=ARGS.batch, base=ARGS.base,
    train_stride=int(os.environ.get("TRAIN_STRIDE", 2)),
    out_dir=out_dir,
)

# ---- upload weights ----
log("== uploading weights to Kaggle ==")
out = WORK / "out"
wdir = WORK / "weights_out"
wdir.mkdir(exist_ok=True)
for p in out.glob("unet_fold*.pt"):
    shutil.copy(p, wdir / p.name)
json.dump(
    {"title": "biohub-detector-weights", "id": "maruanbottoni/biohub-detector-weights",
     "licenses": [{"name": "CC0-1.0"}]},
    open(wdir / "dataset-metadata.json", "w"),
)
os.environ["KAGGLE_API_TOKEN"] = TOK
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)
exists = subprocess.run(["kaggle", "datasets", "status", "maruanbottoni/biohub-detector-weights"],
                        capture_output=True, text=True).returncode == 0
cmd = (["kaggle", "datasets", "version", "-p", str(wdir), "-m", "colab weights", "--dir-mode", "zip"]
       if exists else ["kaggle", "datasets", "create", "-p", str(wdir), "--dir-mode", "zip"])
r = subprocess.run(cmd, capture_output=True, text=True)
log(r.stdout or r.stderr)
log("DONE — weights on Kaggle: maruanbottoni/biohub-detector-weights")
