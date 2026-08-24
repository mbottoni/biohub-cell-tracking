#!/usr/bin/env python3
"""Exec payload: start fullrecipe_runner.sh on the VM and RETURN IMMEDIATELY.

The new colab CLI (~/.local/bin/colab) has no `exec --env`, and holding the
exec channel open for a multi-hour run is exactly the fragility rows 55/91
document (the channel dies while the runtime keeps training). So this payload:

  1. reads non-secret config from /content/job_env (uploaded per-session),
  2. reads the Kaggle token from /content/kaggle_access_token (uploaded file —
     never on a command line, never printed),
  3. Popen-detaches the runner with its own session id, then exits.

After this returns, the ONLY channels to the run are the beacon and checkpoint
Kaggle datasets — which is the design, not a limitation (PLAN 3.1e: read the
stop signal from the checkpoint beacon, never ssh).
"""
import os
import subprocess

env = dict(os.environ)
if os.path.exists("/content/job_env"):
    for line in open("/content/job_env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
env["KAGGLE_API_TOKEN"] = open("/content/kaggle_access_token").read().strip()

log = open("/content/runner.log", "a")
p = subprocess.Popen(["bash", "/content/fullrecipe_runner.sh"],
                     env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
print(f"runner launched detached (pid {p.pid}); config keys: "
      + ",".join(k for k in ("EPOCHS", "BATCH", "SEED", "PATIENCE", "DATA_MODE",
                             "CKPT_DS", "BEACON_DS", "METHOD", "WINDOW")
                 if k in env))
