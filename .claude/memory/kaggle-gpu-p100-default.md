---
name: kaggle-gpu-p100-default
description: Kaggle GPU kernels default to P100 (sm_60) which the image torch rejects; force T4
metadata: 
  node_type: memory
  type: reference
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-25T03:34:46.354Z
---

Kaggle notebook/kernel GPU runs default to **`NvidiaTeslaP100`** (compute sm_60), but the
current Kaggle image's **PyTorch only supports sm_70+** — so any torch `.cuda()` call dies with
`CUDA error: no kernel image is available for execution on the device`. This silently broke
every GPU kernel run (training and inference) until diagnosed.

**Fix:** request a T4 explicitly. Valid accelerator values: `NvidiaTeslaP100` (default),
`NvidiaTeslaT4`, `TpuV6E8`.
- In `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"`
- On push: `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`

Bake it into the metadata (not just the push flag) so the **submission rerun** — Kaggle
re-executes the kernel on hidden data at submit time — also lands a T4, or the submission fails
the same way. See [[biohub-cell-tracking-comp]] and [[kaggle-api-auth]].
