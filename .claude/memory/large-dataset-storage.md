---
name: large-dataset-storage
description: Internal disk is tight; large Kaggle datasets go on the external SSD at /Volumes/extSSD
metadata: 
  node_type: memory
  type: project
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-24T00:46:56.123Z
---

The Mac's internal disk runs near capacity (~90% used, ~100 GB free as of 2026-07-23), so
datasets over ~20 GB do not fit alongside their own unzipped contents. Put them on the
external SSD, mounted at **`/Volumes/extSSD`** (note: `/Volumes`, not `/volume`), which had
~670 GB free. It is exFAT — fine for huge single files, no symlinks or POSIX permissions.

**Why:** the Biohub cell-tracking competition archive is 87.6 GB; a naive `kaggle competitions
download` to the repo would have filled the internal disk with no room to extract.

**How to apply:** before starting any multi-GB download, check `df -h` and stage it under
`/Volumes/extSSD/kaggle/<comp>/`. Keep the repo on the internal disk and point at the data via
a path/env var (`CTRACK_DATA` in [[biohub-cell-tracking-comp]]).
