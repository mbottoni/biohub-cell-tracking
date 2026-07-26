"""Step-1 de-risk: can train_unet_transformer.py fine-tune on a Kaggle kernel?
Loads pretrained UNet, runs ~20 iters on real GT, saves. Fast smoke test — no pseudo-labels yet."""
import json, os, subprocess, sys, shutil
from pathlib import Path

pred = next(Path("/kaggle/input").glob("**/scripts/train_unet_transformer.py"))
SRC_REPO = pred.parents[1]
REPO = Path("/kaggle/working/repo")
if not REPO.exists(): shutil.copytree(SRC_REPO, REPO)
SCRIPTS, SRCP = REPO/"scripts", REPO/"src"
WHEELS = next(Path("/kaggle/input").glob("**/wheels"), None)
UNET_W = next(Path("/kaggle/input").glob("**/unet_transformer/split_0/checkpoint_last.pth"))
TRAIN = next(p for p in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
                         Path("/kaggle/input/biohub-cell-tracking-during-development/train")] if p.exists())
print("repo",REPO,"| unet_w",UNET_W,"| train",TRAIN, flush=True)
if WHEELS:
    subprocess.run([sys.executable,"-m","pip","install","--no-index",f"--find-links={WHEELS}",
                    "zarr","geff","numcodecs","donfig","rustworkx","blosc2","imagecodecs"],check=False)

# tiny split: 4 train, 2 test (all 6bba, not in REP-VAL)
TRAINV=["6bba_05b6850b","6bba_05db0fb1","6bba_0a0e5f3c","6bba_0c582fdc"]
TESTV=["6bba_1a2b3c4d","6bba_debd7bfa"]
# keep only volumes that actually exist
have=set(p.name[:-5] for p in TRAIN.glob("*.zarr"))
TRAINV=[v for v in TRAINV if v in have][:4] or [v for v in sorted(have) if v.startswith("6bba")][:4]
TESTV=[v for v in sorted(have) if v.startswith("6bba") and v not in TRAINV][:2]
Path("/kaggle/working/sp.json").write_text(json.dumps([{"train":TRAINV,"test":TESTV}]))
print("train",TRAINV,"test",TESTV, flush=True)

env=os.environ.copy(); env["PYTHONPATH"]=f"{SRCP}:{SCRIPTS}"; env["BIOHUB_DATA_DIR"]=str(TRAIN)
cmd=[sys.executable,str(SCRIPTS/"train_unet_transformer.py"),"--data-dir",str(TRAIN),
     "--splits","/kaggle/working/sp.json","--split","0","--unet-weights",str(UNET_W),
     "--epochs","1","--max-iters","20","--single-gpu","--num-workers","2","--batch-size","4"]
print("CMD:"," ".join(cmd), flush=True)
r=subprocess.run(cmd,env=env,cwd=str(SCRIPTS),capture_output=True,text=True)
print("STDOUT tail:\n",r.stdout[-2500:], flush=True)
print("STDERR tail:\n",r.stderr[-1500:], flush=True)
print("RETURN CODE:",r.returncode, flush=True)
print("SMOKE OK" if r.returncode==0 else "SMOKE FAILED", flush=True)
