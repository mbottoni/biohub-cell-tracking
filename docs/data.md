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

## Open questions

- [ ] How partial is the annotation? (nodes in geff vs `estimated_number_of_nodes`)
- [ ] Division rate per volume
- [ ] Do the 199 train crops overlap spatially, or tile the embryo disjointly? Overlap would
      leak across CV folds and must be handled by grouping.
- [ ] Cell count per frame and typical inter-frame displacement (sets the linking radius)
