"""Apply a global coordinate shift to the submission and re-score. Pure post-proc."""
import csv, sys, subprocess
from pathlib import Path
REPO=Path("/Users/maruanottoni/home/kaggle/biohub-cell-tracking")
SRC=Path.home()/"biohub_replay947_runs/ds_ctl/submission.csv"
SC=(1.625,0.40625,0.40625)
def run(dz_um,dy_um,dx_um,tag,rounded):
    out=Path(f"/tmp/shift_{tag}.csv")
    dz,dy,dx = dz_um/SC[0], dy_um/SC[1], dx_um/SC[2]
    with SRC.open() as f, out.open("w",newline="") as g:
        rd=csv.DictReader(f); w=csv.DictWriter(g,fieldnames=rd.fieldnames); w.writeheader()
        for r in rd:
            if r["row_type"]=="node":
                z=float(r["z"])+dz; y=float(r["y"])+dy; x=float(r["x"])+dx
                if rounded: z,y,x=round(z),round(y),round(x)
                r["z"],r["y"],r["x"]=f"{z:.4f}",f"{y:.4f}",f"{x:.4f}"
            w.writerow(r)
    p=subprocess.run([str(REPO/".venv/bin/python"),str(REPO/"scripts/local/score_official.py"),str(out)],
                     capture_output=True,text=True)
    line=[l for l in p.stdout.splitlines() if l.startswith("n=")]
    print(f"  {tag:22s} {'int' if rounded else 'float':>5}  {line[0] if line else 'SCORER FAIL'}")
print("baseline + candidate global shifts (pred := pred + offset, offset = -mean residual):")
run(0,0,0,"none_float",False)
run(0.9602,0.3038,0.1565,"full_correction",False)
run(0.9602,0,0,"z_only",False)
run(0.4801,0.1519,0.0783,"half_correction",False)
run(1.9204,0.6076,0.3130,"double",False)
run(0.9602,0.3038,0.1565,"full_correction",True)
