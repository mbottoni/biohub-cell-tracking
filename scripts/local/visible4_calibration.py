#!/usr/bin/env python
"""VISIBLE-4 CALIBRATION — the first measured local->hidden transfer function (row 154).

WHY THIS EXISTS. Every local verdict in this project has been governed by one rule:
"local may REJECT, never CONFIRM" (row 90). That rule was earned from ONE sign
inversion (+0.0036 local -> -0.003 LB) and has never been re-measured on the 947
line. It is a scalar prior with no error bars.

We can do better for free. The 4 `test/` volumes Kaggle shows us are byte-identical
to 4 TRAIN volumes, so their GT is on disk (row 29, 08-04). Every commit run we have
ever paid for wrote a submission.csv over exactly those 4 volumes, and `kaggle kernels
output` retrieves it after the fact at zero GPU cost and zero submission slots. We
hold ~12 configs whose HIDDEN LB score is already known. Scoring their visible-4
outputs locally therefore yields ~12 PAIRED (local, hidden) points — a calibration
curve, not a prior.

WHAT IT MEASURED (2026-09-20, 11 configs with known LB, row 154):
    Spearman(visible-4, LB) = +0.48; pairwise concordance 29/43 = 67% on pairs whose
    LB actually differs. Better than a coin flip, far from a confirm instrument.
    Level: visible-4 mean 0.8949 vs LB mean 0.9468 = 0.052 PESSIMISTIC — the OPPOSITE
    sign to holdout6 (+0.0045 optimistic, row 143).

THE CAUSE OF THE LEVEL GAP, AND THE TRAP IT SETS. The 4 visible volumes hold 3 GT
divisions between them and the pipeline matches NONE of them: division_jaccard is
EXACTLY 0.0000 for all 13 configs scored, including the divoff ablation. So:

  * visible-4 measures the EDGE term only. Its score is adj_edge_jaccard.
  * divoff reads 0.8931 here vs port 0.8932 (delta -0.0001) and 0.911 vs 0.947 on
    hidden (delta -0.036). A probe that cost a submission slot and redirected the
    whole campaign (row 130) is INVISIBLE on this instrument.
  * Any division mechanism scored here will read flat for structural reasons, not
    because it is inert. Do not gate division work on the visible-4.

The two local instruments also DISAGREE IN SIGN on the config that was in flight when
this was built: densad_tuned reads -0.0008 vs densad on visible-4 and +0.0013 on
holdout6 (row 147). Two instruments, opposite signs, both sub-floor. That is the
honest width of local uncertainty on this line.

HOW TO USE IT. As a SECOND reject gate with a different failure mode from holdout6,
never as a confirm. holdout6 is 6 volumes routed 2 low / 4 middle by the density table
and contains no high-density volume; the visible-4 contains 6bba_05db0fb1 (~700
nodes/frame), the only HIGH-group volume we own and the only genuinely
linking-limited one (edge_J 0.8359 at node_recall 0.9910, STRATEGY 18). A mechanism
that touches the high-density branch is untestable on holdout6 and testable here.

USAGE
    # score one already-downloaded commit output
    .venv/bin/python scripts/local/visible4_calibration.py score <dir-with-submission.csv>

    # rebuild the whole calibration table (downloads ~12 kernel outputs, ~5 min)
    .venv/bin/python scripts/local/visible4_calibration.py table --out-dir /tmp/kout

Downloads use `kaggle kernels output <user>/<kernel>`; they cost no GPU quota and no
submission slot. Scoring uses the official scorer, which evaluates only datasets
present in BOTH the prediction and the GT dir, so pointing it at the full 199-volume
GT tree automatically restricts it to the 4 visible volumes.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv/bin/python"
KAGGLE = REPO / ".venv/bin/kaggle"
SCORER = REPO / "scripts/local/score_official.py"

VISIBLE_4 = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

# kernel slug -> (label, known public LB). None = not yet scored.
KERNELS = {
    "biohub-pub947-port": ("port", 0.947),
    "biohub-pub947-dcthr15": ("dcthr15", 0.946),
    "biohub-pub947-relink6": ("relink6", 0.946),
    "biohub-pub947-capup": ("capup", 0.947),
    "biohub-pub947-divoff": ("divoff", 0.911),
    "biohub-pub948h-port": ("948h_port", 0.945),
    "biohub-pub947-divnet": ("divnet", 0.944),
    "biohub-pub947-densad": ("densad", 0.948),
    "biohub-pub947-det096": ("det096", 0.948),
    "biohub-pub947-hidense": ("hidense", 0.948),
    "biohub-pub947-sectta1": ("sectta1", 0.948),
    "biohub-pub947-consensus": ("consensus", 0.948),
    "biohub-pub947-densad-tuned": ("densad_tuned", None),
}
USER = "maruanbottoni"


def score(submission: Path) -> dict:
    """Run the official scorer on a submission.csv; return the parsed summary line."""
    r = subprocess.run([str(PY), str(SCORER), str(submission)],
                       capture_output=True, text=True, cwd=REPO)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("n=")), "")
    if not line:
        print(r.stdout[-1500:], r.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"scorer produced no summary for {submission}")
    out = {"raw": line}
    for key in ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard", "node_recall"):
        m = re.search(rf"\b{key}=([0-9.]+)", line)
        if m:
            out[key] = float(m.group(1))
    return out


def spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def table(out_dir: Path, skip_download: bool) -> None:
    rows = []
    for slug, (label, lb) in KERNELS.items():
        dest = out_dir / label
        sub = dest / "submission.csv"
        if not sub.exists() and not skip_download:
            print(f"[dl] {slug}", flush=True)
            subprocess.run([str(KAGGLE), "kernels", "output", f"{USER}/{slug}",
                            "-p", str(dest)], capture_output=True, text=True)
        if not sub.exists():
            print(f"[skip] {label}: no submission.csv")
            continue
        s = score(sub)
        rows.append((label, s["score"], s.get("division_jaccard"), lb))
        print(f"{label:14s} visible4={s['score']:.4f}  div_J={s.get('division_jaccard'):.4f}  LB={lb}")

    paired = [(l, v, lb) for l, v, _, lb in rows if lb is not None]
    if len(paired) >= 3:
        xs = [p[1] for p in paired]
        ys = [p[2] for p in paired]
        conc = disc = 0
        for i in range(len(paired)):
            for j in range(i + 1, len(paired)):
                dl, dh = xs[i] - xs[j], ys[i] - ys[j]
                if abs(dh) < 1e-9:
                    continue
                conc += dl * dh > 0
                disc += dl * dh <= 0
        print(f"\nSpearman(visible-4, LB) = {spearman(xs, ys):+.3f} over {len(paired)} configs")
        print(f"pairwise concordance on differing-LB pairs: {conc}/{conc + disc} = "
              f"{conc / max(conc + disc, 1):.0%} (coin flip = 50%)")
        print(f"level: visible-4 mean {sum(xs)/len(xs):.4f} vs LB mean {sum(ys)/len(ys):.4f} "
              f"=> {sum(ys)/len(ys) - sum(xs)/len(xs):+.4f} (division term is 0 locally)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("score")
    s1.add_argument("path", help="directory holding submission.csv, or the csv itself")
    s2 = sub.add_parser("table")
    s2.add_argument("--out-dir", default="/tmp/kout")
    s2.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    if a.cmd == "score":
        p = Path(a.path)
        p = p if p.is_file() else p / "submission.csv"
        print(score(p)["raw"])
    else:
        table(Path(a.out_dir), a.skip_download)


if __name__ == "__main__":
    main()
