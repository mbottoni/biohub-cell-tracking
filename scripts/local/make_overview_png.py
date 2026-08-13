#!/usr/bin/env python3
"""One-page explainer of the current Biohub cell-tracking solution.

Layout rule learned the hard way: every artist stays INSIDE its axes limits and
the figure is saved at a fixed size (no bbox_inches="tight") -- text placed
outside the axes silently expands a tight bbox into a 7000px-wide canvas.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito: colourblind-safe by published design (the skill's validator script
# was unavailable, so the palette is safe by construction, not by eye).
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#9AA0A6"
INK, INK2, INK3 = "#1A1C1E", "#44474B", "#6E7276"
SURF, PANEL, LINE = "#FFFFFF", "#F5F6F7", "#E2E5E8"

fig = plt.figure(figsize=(13.2, 9.0), dpi=150, facecolor=SURF)
gs = fig.add_gridspec(3, 2, height_ratios=[0.60, 0.74, 1.0], width_ratios=[1.52, 1.0],
                      left=0.05, right=0.965, top=0.900, bottom=0.125,
                      hspace=0.40, wspace=0.11)

fig.text(0.05, 0.968, "Biohub cell tracking — how our solution works, and where the points are",
         fontsize=17.5, fontweight="bold", color=INK, va="top")
fig.text(0.05, 0.936, "Measured on 24 held-out volumes with the competition's own scorer   ·   "
                      "leaderboard 0.915 confirmed   ·   target 0.949",
         fontsize=10.5, color=INK3, va="top")

# ─────────────────────────── Panel A: the pipeline ───────────────────────────
axA = fig.add_subplot(gs[0, :]); axA.axis("off")
axA.set_xlim(0, 100); axA.set_ylim(-11, 23)
axA.text(0, 22, "THE PIPELINE", fontsize=10, fontweight="bold", color=INK2, va="top")

steps = [
    ("3D video\nover time", "input", GREY),
    ("Detector\nfinds nuclei", "public UNet, frozen", GREY),
    ("Edge model\nlinks frames", "transformer + TTA", GREY),
    ("ILP solver\nglobal linking", "our change ①", BLUE),
    ("Post-process\ngap repair", "our change ②", ORANGE),
    ("Submission", "tracks", GREY),
]
w, gap = 14.2, 2.96
for i, (title, sub, col) in enumerate(steps):
    x = i * (w + gap)
    hot = col in (BLUE, ORANGE)
    axA.add_patch(FancyBboxPatch((x, 4), w, 13.5, boxstyle="round,pad=0.3,rounding_size=1.1",
                                 linewidth=1.9 if hot else 1.1,
                                 edgecolor=col if hot else "#CBCFD3",
                                 facecolor=(col + "1A") if hot else PANEL))
    axA.text(x + w / 2, 12.6, title, ha="center", va="center", fontsize=9.5, color=INK,
             fontweight="bold" if hot else "normal", linespacing=1.35)
    axA.text(x + w / 2, 7.0, sub, ha="center", va="center", fontsize=7.8,
             color=col if hot else INK3, fontweight="bold" if hot else "normal")
    if i < len(steps) - 1:
        axA.add_patch(FancyArrowPatch((x + w + 0.4, 10.7), (x + w + gap - 0.4, 10.7),
                                      arrowstyle="-|>", mutation_scale=11,
                                      linewidth=1.3, color="#B9BDC2"))

axA.text(3 * (w + gap) + w / 2, 0.4, "① ending a track costs more\n1.5 → 2.0     +0.0020",
         ha="center", va="top", fontsize=8.4, color=BLUE, linespacing=1.5)
axA.text(4 * (w + gap) + w / 2, 0.4, "② bridge 3-frame gaps, not 1\na hard-coded cap     +0.0016",
         ha="center", va="top", fontsize=8.4, color=ORANGE, linespacing=1.5)
axA.text(50, -8.6, "The two only work together — change ② on its own is worth nothing (0.9098 vs a 0.9099 baseline)",
         ha="center", va="center", fontsize=8.8, color=INK3, style="italic")

# ─────────────────── Panel B: what the score is made of ──────────────────────
axB = fig.add_subplot(gs[1, :]); axB.set_facecolor(SURF)
axB.set_xlim(0.86, 1.013); axB.set_ylim(-1.35, 1.62); axB.set_yticks([])
axB.text(0.862, 1.34, "WHAT THE SCORE IS MADE OF", fontsize=10, fontweight="bold", color=INK2)
axB.text(0.862, 0.93, "score  =  following cells correctly   +   0.1 × spotting divisions",
         fontsize=11.5, color=INK)

assoc, earned, unclaimed = 0.9065, 0.0070, 0.0930
axB.barh([-0.15], [assoc], color=BLUE, height=0.5, zorder=3)
axB.barh([-0.15], [earned], left=assoc, color=ORANGE, height=0.5, zorder=3)
axB.barh([-0.15], [unclaimed], left=assoc + earned, color=SURF, height=0.5,
         edgecolor=ORANGE, hatch="////", linewidth=1.3, zorder=3)

axB.text(0.8825, -0.15, "following cells   0.9065", ha="left", va="center",
         fontsize=9.8, color="white", fontweight="bold", zorder=4)
axB.text(0.9855, -0.15, "divisions left on the table   0.0930",
         ha="center", va="center", fontsize=9.6, color=VERM, fontweight="bold", zorder=4,
         bbox=dict(boxstyle="round,pad=0.30", facecolor=SURF, edgecolor="none"))
axB.annotate("divisions we actually get:  0.0070",
             xy=(assoc + earned / 2, 0.13), xytext=(0.8635, 0.62),
             fontsize=8.7, color=INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.9,
                             connectionstyle="angle,angleA=0,angleB=90,rad=3"))

axB.axvline(0.9135, color=INK, linewidth=1.6, ymin=0.30, ymax=0.66, zorder=5)
axB.text(0.9135, -0.50, "we are here\n0.9135", ha="center", va="top", fontsize=8.7,
         color=INK, fontweight="bold", linespacing=1.4)
axB.axvline(0.949, color=VERM, linewidth=1.6, linestyle=(0, (4, 2)), ymin=0.30, ymax=0.60, zorder=5)
axB.text(0.949, 0.30, "target  0.949", ha="center", va="bottom", fontsize=8.7,
         color=VERM, fontweight="bold")

axB.set_xticks([])
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.spines["bottom"].set_visible(False)
axB.grid(axis="x", color="#EDEFF1", linewidth=0.8, zorder=0)
axB.text(0.862, -1.22, "Every dead end we chased for weeks lived in the blue bar, worth about 0.002 at a time.  "
                       "The hatched block is 13× larger — and nobody had looked at it.",
         fontsize=8.9, color=INK3, va="center")

# ─────────── Panel C left: what finding divisions would be worth ─────────────
axC = fig.add_subplot(gs[2, 0]); axC.set_facecolor(SURF)
axC.text(0.0, 1.20, "WHAT FINDING THEM WOULD BE WORTH", fontsize=10, fontweight="bold",
         color=INK2, transform=axC.transAxes)

labels = ["today\n3 of 25", "10 found\n10 wrong", "15 found\n10 wrong", "20 found\n5 wrong"]
scores = [0.9135, 0.9351, 0.9494, 0.9732]
bars = axC.bar(range(4), scores, color=[GREY, GREEN, GREEN, GREEN], width=0.54, zorder=3)
for i, (b, s) in enumerate(zip(bars, scores)):
    axC.text(b.get_x() + b.get_width() / 2, s + 0.0022, f"{s:.4f}", ha="center",
             fontsize=9.5, color=INK, fontweight="bold" if i == 2 else "normal")
axC.axhline(0.949, color=VERM, linewidth=1.5, linestyle=(0, (4, 2)), zorder=4)
axC.text(-0.42, 0.9504, "target", fontsize=8.5, color=VERM, va="bottom", fontweight="bold")

axC.set_xticks(range(4)); axC.set_xticklabels(labels, fontsize=8.8, color=INK2, linespacing=1.5)
axC.set_ylim(0.90, 0.988); axC.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98])
axC.tick_params(axis="both", colors=INK3, labelsize=8.5, length=0)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axC.spines[s].set_color(LINE)
axC.grid(axis="y", color="#EDEFF1", linewidth=0.8, zorder=0)

# ─────────────── Panel C right: why this is reachable ────────────────────────
axD = fig.add_subplot(gs[2, 1]); axD.axis("off")
axD.set_xlim(0, 1); axD.set_ylim(0, 1)
axD.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.90,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             linewidth=1.1, edgecolor=LINE, facecolor=PANEL))
axD.text(0.075, 0.855, "WHY THIS IS REACHABLE", fontsize=10, fontweight="bold", color=INK2)
axD.text(0.075, 0.755,
         "Of the 25 real divisions, 16 already have\n"
         "the mother AND both daughters sitting in\n"
         "our own output.\n\n"
         "We see the cells — we just never say\n"
         "\"that one split\". A labelling failure,\n"
         "not a detection failure.\n\n"
         "Geometry never separated them. Judging\n"
         "by how a dividing nucleus LOOKS scores\n"
         "0.845, where 0.5 is a coin flip.",
         fontsize=8.6, color=INK2, va="top", linespacing=1.52)
fig.text(0.635, 0.082, "Next: train that classifier, then re-test end-to-end\nbefore spending a submission.",
         fontsize=8.6, color=GREEN, va="top", linespacing=1.5, fontweight="bold")

fig.text(0.05, 0.022,
         "These 24 volumes are training volumes the public model has already seen, so the absolute numbers are optimistic. "
         "What is trustworthy is the\ncomparison between configurations, measured on identical machinery.",
         fontsize=7.9, color=INK3, va="bottom", linespacing=1.6)

out = ("/private/tmp/claude-501/-Users-maruanottoni-home-kaggle-biohub-cell-tracking/"
       "746fd85d-a0d8-47d3-b5cf-be4bb88f1e9a/scratchpad/solution_overview.png")
fig.savefig(out, dpi=150, facecolor=SURF)      # fixed canvas; no tight bbox
print("wrote", out)
