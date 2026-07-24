# Dataset facts

Established 2026-07-23 from the Kaggle file listing (`data/filelist.json`, 24,886 files,
87.6 GB) plus byte-level probes of individual files. Everything here is verified, not assumed.

## Shape of the corpus

| | |
|---|---|
| Total | **24,886 files / 87.6 GB** |
| Stores | **204** — 199 `train/`, 4 `test/`, plus `sample_submission.csv` |
| Volume shape | **`(100, 64, 256, 256)`** = (T, Z, Y, X), `uint16` |
| Chunking | `[1, 64, 256, 256]` — **one chunk per timepoint**, blosc2-compressed |
| Voxel scale | `[1.0, 1.625, 0.40625, 0.40625]` (T, Z, Y, X); µm for space |
| Physical extent | 64×1.625 = 104 µm (Z), 256×0.40625 = 104 µm (Y, X) → a **104 µm cube** |
| Per store | ~838 MB uncompressed, ~0.55 GB on disk, 102 zarr files (+21 geff files in train) |
| Embryo prefixes | `6bba` (128 train crops), `44b6` (71 train crops) — only **two source embryos** |

These are **small crops, not whole embryos**. 100 frames of a 104 µm cube each. That makes
local training genuinely feasible — no need for the multi-TB streaming machinery I assumed in
v1 of the plan.

Group-level `zarr.json` carries `image_statistics.quantiles` for the volume — use these for
normalization rather than recomputing per volume:

```
0.0: 15   0.001: 26.2   0.01: 38   0.1: 75
0.9: 497  0.99: 1478    0.999: 2145   1.0: 4319
```

Note the long right tail (p99.9 = 2145 vs p90 = 497) — percentile clipping matters.

## The 4 test stores are byte-identical to 4 train stores

```
TEST IDS      = 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1
OVERLAP train∩test = all four
```

Verified for `44b6_0113de3b`:

| | train | test |
|---|---|---|
| `zarr.json` md5 | `c987c2e8…` | `c987c2e8…` (identical) |
| array shape | `[100,64,256,256]` | `[100,64,256,256]` |
| chunk `0/c/0/0/0/0` md5 | `45430355d75d604f24655bab2c766a49` | **identical**, 4,469,062 bytes |

`train/` versions additionally ship a `.geff` ground-truth graph (21 files: node ids, `t/z/y/x`
props, edge ids). `test/` versions ship the `.zarr` only.

### What this means

**The visible `test/` directory is a development sample, not the scored test set.** This is a
kernels-only competition (`isKernelsSubmissionsOnly: true`), so at submission rerun Kaggle
swaps `test/` for the real, hidden data. If the public LB were scored on these four, everyone
would trivially get 1.0 by reading the train geff — the public LB sits at ~0.897, so it isn't.

**How to exploit it legitimately:** we hold exact ground truth for the four volumes that flow
through the *actual submission code path*. That makes them the ideal fixture for verifying our
local metric implementation end-to-end — same loader, same inference, same CSV writer, known
answer. Use them to validate the harness.

**Do not tune on them.** Four crops from two embryos, all also present in train, is not a
validation set. Model selection uses a held-out split of the 199 train stores, grouped so the
two embryo prefixes don't leak across folds.

## Ground truth format (`.geff`)

Zarr-backed graph:

```
<id>.geff/
  nodes/ids/            node identifiers
  nodes/props/t|z|y|x/  per-node coordinates
  edges/ids/            parent -> child pairs
```

Read with the `geff` package; manipulate with `tracksdata`. `GeffMetadata.extra` carries
`estimated_number_of_nodes`, which the competition metric normalizes against — the annotation
is **partial**, which is the root cause of the metric exploit described in PLAN.md §2.

## Ground truth, measured across all 199 volumes

`scripts/analyze_gt.py` → `docs/gt_stats.csv`.

| | mean | median | min | max |
|---|---|---|---|---|
| Annotated nodes | 670 | — | 52 | 1,950 |
| Annotated edges | 648 | — | 50 | 1,879 |
| Tracks | 22 | — | 2 | 79 |
| Divisions | 0.76 | — | 0 | 5 |
| Annotation density | **6.1%** | 3.6% | 0.13% | 20.2% |
| `estimated_number_of_nodes` | 23,744 | 17,909 | 3,783 | 78,644 |

Totals: 133,318 nodes, 128,883 edges, **151 divisions in 87 of 199 volumes (43.7%)**.

### The two embryos are wildly unbalanced

| embryo | volumes | nodes | edges | **share of all edges** | divisions | mean density |
|---|---|---|---|---|---|---|
| `6bba` | 128 | 113,121 | 109,057 | **84.6%** | 125 | 8.98% |
| `44b6` | 71 | 20,197 | 19,826 | 15.4% | 26 | 0.99% |

`6bba` is **9× denser** and carries 85% of all annotated edges. Because the metric
micro-averages (counts summed, then Jaccard), **the score is effectively the `6bba` score**.

This is a serious problem for leave-one-embryo-out CV: the fold validating on `44b6` measures
performance on 15% of the signal at a tenth the density, and the fold validating on `6bba`
trains on a very unrepresentative 71 volumes. Neither fold resembles the real distribution.
Weight experiments accordingly and do not read the two folds as interchangeable.

## CV: within-embryo kfold is valid

`scripts/check_crop_overlap.py` over all 199 volumes (2026-07-24): same-embryo first-frame
correlation (median +0.011, p95 +0.44, max +0.89) is indistinguishable from different-embryo
(median +0.017, p95 +0.44, max +0.85); no exact duplicates, no pair above 0.9. So same-embryo
crops are **spatially independent**, and `ctrack.data.splits.kfold` (splitting within embryos)
does not leak. **We get 5 folds, not just 2.** Still stratify by embryo so each fold sees both
— and remember the score is `6bba`-dominated (85% of edges), so weight accordingly.

## Open questions

- [x] Do the 199 train crops overlap spatially? — **no**, kfold is valid (above)
- [x] How partial is the annotation? — 6.1% mean, 0.13–20.2% range
- [x] Division rate — 151 total, 43.7% of volumes have ≥1
- [x] Inter-frame displacement — median 1.95 µm, p90 3.92 µm, but 80/199 volumes contain a
      step >10 µm (max 60.8), so the public 10 µm linking cap drops real links
