"""Free diagnostic: how many epochs was the PRIMARY model trained for?

Our pipeline runs the support-pack checkpoint as primary and blends the
seed314159 checkpoint in as secondary (edge weight 0.15, detection weight
0.475).  The secondary's shipped history.csv shows 400 epochs on all 199
volumes, validation_score 0.978 and still climbing at the end.  The support
pack is *named* "50ep".  If the primary really is 50 epochs (validation ~0.946)
then we are running the weaker model as primary and down-weighting the
stronger one -- a config error worth ~0.03, not a modelling gap.

CPU only, no GPU, no submission.
"""
import json
from pathlib import Path

for root in sorted(Path("/kaggle/input").glob("*")):
    print("=" * 70)
    print("DATASET:", root.name)
    # ARTIFACT_MANIFEST.json first: dataset *slugs* lie. The support pack slug
    # says "50ep" while its manifest says 400ep-snapshot, and an earlier version
    # of this script did not glob for that filename -- so it "answered" the
    # epoch-count question from a file it never read.
    for pat in ("**/ARTIFACT_MANIFEST.json", "**/history.csv", "**/training_config.json",
                "**/config.json", "**/SNAPSHOT_MANIFEST.json"):
        for p in sorted(root.glob(pat))[:6]:
            rel = p.relative_to(root)
            if p.name == "history.csv":
                lines = p.read_text().splitlines()
                hdr = lines[0].split(",")
                print(f"\n  {rel}  -> {len(lines)-1} epochs recorded")
                def col(row, name):
                    try: return row[hdr.index(name)]
                    except Exception: return "?"
                for tag, row in (("first", lines[1]), ("last", lines[-1])):
                    r = row.split(",")
                    print(f"     {tag:<5} epoch={col(r,'epoch'):<5} val_score={col(r,'validation_score')[:8]:<9}"
                          f" val_recall={col(r,'validation_recall')[:8]:<9} best={col(r,'best_score')[:8]}")
                best = max(lines[1:], key=lambda l: float(l.split(",")[hdr.index("validation_score")]))
                rb = best.split(",")
                print(f"     BEST  epoch={col(rb,'epoch')} val_score={col(rb,'validation_score')[:8]}")
            else:
                try:
                    d = json.loads(p.read_text())
                    keep = {k: v for k, v in d.items()
                            if any(s in k for s in ("epoch", "method", "seed", "datasets", "lr", "learning"))}
                    print(f"  {rel}: {keep if keep else d}")
                except Exception as e:
                    print(f"  {rel}: <{e}>")
    wts = sorted(root.glob("**/*.pth"))
    if wts:
        print("\n  checkpoints:")
        for w in wts[:8]:
            print(f"     {w.relative_to(root)}  {w.stat().st_size/1e6:.1f} MB")
