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

### Consequence: only the annotated subgraph is scored

> **CORRECTION (measured across all 199 volumes).** An earlier version of this document said
> the GT was "0.2% dense — a couple of hand-traced trajectories per volume". That came from
> hand-inspecting one volume, `44b6_0113de3b`, which turns out to sit at the **4th percentile**
> of annotation density. It was a near-minimum outlier and I generalized from it. The real
> numbers are below. The *structural* conclusions (only annotated-subgraph edges count; the
> node budget is unclamped above) were unaffected, but the "few dozen cells" framing was wrong.

Across all 199 train volumes (`docs/gt_stats.csv`, via `scripts/analyze_gt.py`):

| | mean | median | min | max |
|---|---|---|---|---|
| Annotated nodes | 670 | — | 52 | 1,950 |
| Annotated edges | 648 | — | 50 | 1,879 |
| Tracks per volume | 22 | — | 2 | 79 |
| Divisions per volume | 0.76 | — | 0 | 5 |
| **Annotation density** | **6.1%** | **3.6%** | 0.13% | 20.2% |
| `estimated_number_of_nodes` | 23,744 | 17,909 | 3,783 | 78,644 |

Totals: **133,318 annotated nodes, 128,883 edges, 151 divisions.**

So it is a real tracking problem — hundreds of nodes and ~22 tracks per volume — but still only
a **6% sample** of the cells present. The objective stands: reproduce the annotated
trajectories, and hang nothing spurious off annotated nodes. Detecting the other 94% is
neither rewarded nor punished at the edge level.

Measured across all volumes (per-volume aggregates, then averaged):

- inter-frame displacement: **median 1.95 µm, p90 3.92 µm**
- but **80 of 199 volumes contain a GT step longer than 10 µm**, and 31 exceed 15 µm (max
  60.8 µm). The public `EDGE_MAX_DISTANCE_UM = 10.0` therefore **discards real links in ~40% of
  volumes**. The p99 of per-volume p90 is 7.7 µm, so this is a tail effect — but it is a free
  source of edge FNs and the cap is worth sweeping.
- **every GT edge is dt=1**, confirmed across all 199 volumes (`edge_dt_max` is 1 everywhere).

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

## The division term is worth 0.1, and it does engage

`score = adj_edge_jaccard + 0.1 · division_jaccard`. Against a public LB around 0.897 that term
is roughly **11% of the total score**.

Measured: **151 divisions across 199 volumes; 87 volumes (43.7%) contain at least one**, up to
5 in a single volume. (The one volume I inspected by hand had zero, which briefly suggested
the term never engages — it does.) At **0.0012 divisions per annotated edge** this is a
rare-event Jaccard, so expect it to be high variance and easy to move in either direction.

Divisions are concentrated in `6bba` (125 of 151, 0.98/volume) versus `44b6` (26, 0.37/volume).

**Verified in `division_metrics.py`:** division FP counts only predicted divisions **whose node
matched a GT node** (`count_matched_pred_divisions` requires the matched GT node to have
≥1 child). Unmatched predicted divisions are excluded entirely — which is why the public
"fake forks" augmentation, placing 2-child nodes at impossible coordinates, is a **free no-op**
on this term rather than a penalty. `tests/test_biohub_metric.py` pins this.

What the hub node actually buys is still undetermined; the hub→root edges *should* count as
edge FPs wherever a root matched a GT node with an in-edge. That's an experiment to run once
the scorer is LB-validated, not something to settle by reading.

## What to do with this

1. Reimplement the formula in `ctrack.eval` and verify against a real LB score.
2. Sweep the detection threshold against the *joint* objective (edge Jaccard × node-count
   multiplier), not against detection F1.
3. Treat divisions as a first-class target worth ~0.1, not an ILP byproduct.
4. Optimize for *edge correctness on annotated cells*, which is a much narrower and more
   tractable target than "track all 25,755 cells".

## `estimated_number_of_nodes` varies 20×, and we can't read it at test time

`N_total` ranges from **3,783 to 78,644** across volumes (median 17,909) — a 20× spread, even
though every volume is the same 64×256×256 shape. So `total_node_ratio` is a *per-volume*
quantity, and a single global detection threshold produces wildly different multipliers: a
volume with N_total=3,783 gets heavily penalized for the same detection count that earns a
bonus on a volume with N_total=78,644.

The catch: **`estimated_number_of_nodes` lives in the geff metadata, which test volumes do not
have.** We cannot read N_total at inference. But we don't need to — the multiplier depends on
the *ratio*. Since N_total tracks the true cell count, keeping a fixed *fraction* of detections
(say the top 50% by confidence) yields ratio ≈ −0.5 and a ×1.05 multiplier regardless of the
absolute count. The tradeoff against lost edges is exactly what `run_baseline.py --sweep`
measures.

## Open questions

- [ ] **Verify this reimplementation against an actual leaderboard score** (still the gate)
- [x] Are division FP/FN restricted to matched nodes? — **yes**, verified and tested
- [x] Node matching — centroid distance ≤ 7.0 µm, anisotropy-scaled, optimal per frame
- [x] GT node/edge/division distribution across all 199 volumes — see above
- [x] Does `estimated_number_of_nodes` vary? — **20×**, see above
- [ ] What does the hub node in the public augmentation actually buy?
