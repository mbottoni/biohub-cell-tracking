---
name: kaggle-api-auth
description: Kaggle auth uses a KGAT_ bearer token and needs CLI >=2.x; the legacy kaggle.json 401s
metadata: 
  node_type: memory
  type: reference
  originSessionId: 04028120-dfc0-408d-9b71-734ce309387d
  modified: 2026-07-24T00:47:04.604Z
---

Kaggle API access for this user is a **bearer token** (`KAGGLE_API_TOKEN=KGAT_...`), not the
legacy `~/.kaggle/kaggle.json` username/key pair — that file returns
`401 {"code":401,"message":"Unauthenticated"}` and regenerating it does not help.

The token also requires **Kaggle CLI ≥ 2.x**. Homebrew's `kaggle` was 1.5.13, which only
speaks basic auth and 401s against a `KGAT_` token no matter how it is supplied. Install a
current CLI into the project venv (`uv pip install -U kaggle`) and use that binary.

Raw API calls work with `Authorization: Bearer <token>`; basic auth and `X-Kaggle-ApiToken`
both fail.

**How to apply:** store the token in a gitignored `.env` at the repo root (mode 600), source
it, and call the venv's kaggle binary. Listing thousands of competition files will hit
`429 Too Many Requests` — page with ~1s sleeps and exponential backoff. See
[[biohub-cell-tracking-comp]].
