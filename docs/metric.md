# The metric, decoded

Source: `repo/src/biohub_tracking/metrics.py` from the public Kaggle dataset
`pilkwang/biohub-tracking-support-pack-50ep-v1`, which the top public notebooks use for local
validation.

> **Caveat, and it matters:** this is a *community reimplementation*, not the official scorer.
> Top teams calibrate it against the leaderboard so it is probably faithful, but we must verify
> it ourselves against a real LB score before trusting it for tuning. Until then, treat the
> structure below as high-confidence and the exact constants as provisional.

## The formula

```
total_node_ratio  = (N_pred − N_total) / N_total          # N_total = estimated_number_of_nodes
edge_jaccard      = |matched edges| / (|GT edges| + |valid pred edges| − |matched edges|)
adj_edge_jaccard  = max(0, edge_jaccard · (1 − α · total_node_ratio))     # α = 0.1
score             = adj_edge_jaccard + 0.1 · division_jaccard
```

Everything hinges on what "valid pred edge" means.

## Only edges touching annotated cells count

From `_evaluate_matched_graph`:

```python
out_valid = gt out_degree > 0        # per GT node
in_valid  = gt in_degree  > 0
# propagated onto predicted nodes via their matched GT node; False if unmatched
pred_valid = out_valid(source) | in_valid(target)
```

A predicted edge enters the denominator **only if at least one endpoint matched a ground-truth
node that itself has an edge**. Predicted edges between two unannotated cells are invisible to
the metric — neither TP nor FP.

### Consequence: this is not a "track every cell" problem

The ground truth is astonishingly sparse. For `44b6_0113de3b`:

| | |
|---|---|
| Annotated nodes | **52** |
| Annotated edges | **50** |
| Tracks (roots) | **2** |
| Divisions | **0** |
| `estimated_number_of_nodes` | **25,755** |
| **Annotation density** | **0.2%** |

So the objective is really: **for the few dozen hand-traced cells in each volume, reproduce
their trajectory edges exactly — and don't hang spurious edges off those particular nodes.**
Detecting the other ~25,700 cells is neither rewarded nor punished at the edge level.

Practical measurements from that volume (useful for setting linking radii):

- inter-frame displacement: **median 2.88 µm, p90 4.67 µm, max 7.81 µm**
  (the public baselines' `EDGE_MAX_DISTANCE_UM = 10.0` is generous but reasonable)
- annotated `t` spans 0–75 of 100 frames, with gaps — **the GT tracks themselves are not
  contiguous**, which interacts with gap-closing behaviour

## The node-count term is unclamped — and that's exploitable

```python
adj_edge_jaccard = max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
```

`max(0, ...)` bounds it below. **Nothing bounds it above.** If you predict *fewer* nodes than
`estimated_number_of_nodes`, `total_node_ratio` goes negative and the multiplier exceeds 1:

| N_pred / N_total | total_node_ratio | multiplier |
|---|---|---|
| 1.5 | +0.50 | ×0.95 |
| 1.0 | 0.00 | ×1.00 |
| 0.8 | −0.20 | **×1.02** |
| 0.5 | −0.50 | **×1.05** |
| 0.2 | −0.80 | **×1.08** |

There is a genuine optimum here: detect **selectively** — few but high-precision nodes — and
you get a free multiplier, so long as you still detect the specific cells that happen to be
annotated. This very likely explains the public `POINT_THRESHOLD = 0.955`, which is startlingly
high for a detection task and otherwise looks like a mistake.

**This tradeoff curve is directly optimizable offline** and is our Phase 2 priority. It is not
the published hub hack — it's a separate, quantified lever the field appears to be tuning only
by blind LB probing at 5 submissions/day.

## The division term is worth 0.1

`score = adj_edge_jaccard + 0.1 · division_jaccard`. Against a public LB around 0.897 that term
is roughly **11% of the total score**, and GT divisions are rare (zero in the volume we
inspected). A rare-event Jaccard is high variance, which is presumably what the public
"fake forks" augmentation manipulates — 32 injected 2-child nodes at impossible coordinates.

Whether those forks register as division FPs depends on whether unmatched predicted nodes are
excluded from the division counts, the same way they are for edges. **Not yet verified — read
`division_metrics.py` and test empirically.**

## What to do with this

1. Reimplement the formula in `ctrack.eval` and verify against a real LB score.
2. Sweep the detection threshold against the *joint* objective (edge Jaccard × node-count
   multiplier), not against detection F1.
3. Treat divisions as a first-class target worth ~0.1, not an ILP byproduct.
4. Optimize for *edge correctness on annotated cells*, which is a much narrower and more
   tractable target than "track all 25,755 cells".

## Open questions

- [ ] Verify this reimplementation against an actual leaderboard score
- [ ] Are division FP/FN also restricted to matched nodes? (`division_metrics.py`)
- [ ] How is node matching done — nearest neighbour, and within what radius?
- [ ] Distribution of GT node/edge/division counts across all 199 train volumes
- [ ] Is `estimated_number_of_nodes` per-volume constant-ish, or does it vary a lot?
