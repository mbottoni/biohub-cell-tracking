#!/usr/bin/env python
"""Row 178 decision script: turns pending LB reads into 09-28 submissions by PRE-REGISTERED rules.

Operator-authorised 09-27 ("go ahead with the plan, end to end, leading to submissions").
Appends to logs/wave/queue.txt, which scripts/wave_runner.sh executes (push on free slot,
submit the pushed version on COMPLETE). Every decision is logged to logs/wave/runner.log.

  FT    : at/after 2026-09-28 00:00Z queue pub947x_x138_densad_ft (submit). Waiting for the UTC
          day boundary so it lands on 09-28's allowance, not the exhausted 09-27 one.
  DC    : when x138 (56616473) and dc20 (56613666) are scored: dc20 > x138 -> queue
          pub947x_x138_densad_dc20 (submit); otherwise skip. Equal at 3 decimals = no evidence.
  HEAD  : when pub947x_v1284cap120 is done: pull its head, gate held-out-by-movie <= -5%,
          publish maruanbottoni/biohub-v1284-head-ours120, and once ourhead (56616700) and x138
          are scored with ourhead >= x138: queue pub947x_x138_densad_ourhead120 (nosubmit), score
          its visible-4, submit iff >= 0.9150 (same reject gate as row 175).
"""
import csv, datetime as dt, io, json, subprocess, time
from pathlib import Path

ROOT = Path('/Users/maruanottoni/home/kaggle/biohub-cell-tracking')
K = str(ROOT / '.venv/bin/kaggle'); PY = str(ROOT / '.venv/bin/python')
COMP = 'biohub-cell-tracking-during-development'
W = ROOT / 'logs/wave'; LOG = W / 'runner.log'; Q = W / 'queue.txt'; STATE = W / 'decide_0928.json'
X138, DC20, OURHEAD = '56616473', '56613666', '56616700'


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f"{dt.datetime.utcnow():%m-%d %H:%M:%SZ} [decide] {msg}\n")


def sh(*args, timeout=900):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def scores():
    out = sh(K, 'competitions', 'submissions', '-c', COMP, '-v')
    rows = list(csv.DictReader(io.StringIO(out[out.index('ref'):] if 'ref' in out else '')))
    s = {}
    for r in rows:
        v = (r.get('publicScore') or '').strip()
        if v:
            s[r['ref']] = float(v)
    return s


def enqueue(line):
    if line not in Q.read_text():
        with open(Q, 'a') as f:
            f.write(line + '\n')
        log(f'queued: {line}')


st = json.loads(STATE.read_text()) if STATE.exists() else {}
def done(key): return st.get(key) is not None
def mark(key, val):
    st[key] = val; STATE.write_text(json.dumps(st, indent=1))

log('=== decide_0928 start')
while not all(done(k) for k in ('ft', 'dc', 'head')):
    now = dt.datetime.utcnow()
    try:
        s = scores()
    except Exception as e:  # transient API failure: retry next cycle
        log(f'scores() failed: {e}'); s = {}

    if not done('ft') and now >= dt.datetime(2026, 9, 28, 0, 0):
        enqueue('pub947x_x138_densad_ft submit'); mark('ft', 'queued')

    if not done('dc') and X138 in s and DC20 in s:
        if s[DC20] > s[X138]:
            enqueue('pub947x_x138_densad_dc20 submit'); mark('dc', f'queued ({s[DC20]} > {s[X138]})')
        else:
            log(f'DC skip: dc20 {s[DC20]} <= x138 {s[X138]}'); mark('dc', 'skip')

    if not done('head') and (W / 'pub947x_v1284cap120.failed').exists():
        log('HEAD: capture120 FAILED -- skip'); mark('head', 'skip-failed')
    if not done('head') and (W / 'pub947x_v1284cap120.done').exists():
        if not st.get('head_ds'):
            out_dir = W / 'cap120_out'; head = None
            for _ in range(3):
                sh(K, 'kernels', 'output', 'maruanbottoni/biohub-pub947x-v1284cap120', '-p', str(out_dir), timeout=1500)
                hs = list(out_dir.rglob('v1284_out/v1284_head.pt')); rs = list(out_dir.rglob('v1284_out/v1284_train_report.json'))
                if hs and rs:
                    head, rep = hs[0], json.loads(rs[0].read_text()); break
            if head is None:
                log('HEAD: no head in capture120 output after 3 pulls -- skip'); mark('head', 'skip-nohead'); continue
            log(f"HEAD: capture120 pairs={rep['pairs']} movies={rep['movies']} heldout_rel={rep['heldout_rel']:+.3f}")
            if rep['heldout_rel'] > -0.05:
                log('HEAD: gate failed (> -5%) -- skip'); mark('head', 'skip-gate'); continue
            ds = W / 'ourhead120_ds'; ds.mkdir(exist_ok=True)
            (ds / 'v1284_head.pt').write_bytes(head.read_bytes())
            (ds / 'dataset-metadata.json').write_text(json.dumps({'title': 'biohub v1284 head ours120',
                'id': 'maruanbottoni/biohub-v1284-head-ours120', 'licenses': [{'name': 'CC0-1.0'}]}))
            log('HEAD: dataset create: ' + sh(K, 'datasets', 'create', '-p', str(ds)).strip().splitlines()[-1])
            mark('head_ds', 'created')
        if st.get('head_ds') == 'created' and 'v1284_head.pt' in sh(K, 'datasets', 'files', 'maruanbottoni/biohub-v1284-head-ours120'):
            mark('head_ds', 'ready'); log('HEAD: dataset ready')
        if st.get('head_ds') == 'ready' and X138 in s and OURHEAD in s and not st.get('head_queued'):
            if s[OURHEAD] >= s[X138]:
                enqueue('pub947x_x138_densad_ourhead120 nosubmit'); mark('head_queued', f'{s[OURHEAD]} >= {s[X138]}')
            else:
                log(f'HEAD skip: ourhead {s[OURHEAD]} < x138 {s[X138]}'); mark('head', 'skip-lb')
        k = 'pub947x_x138_densad_ourhead120'
        if st.get('head_queued') and (W / f'{k}.failed').exists():
            log(f'HEAD: {k} FAILED'); mark('head', 'failed')
        if st.get('head_queued') and (W / f'{k}.done').exists() and not done('head'):
            o = W / f'{k}_out'; v = (W / f'{k}.version').read_text().strip()
            for _ in range(3):
                sh(K, 'kernels', 'output', f'maruanbottoni/biohub-{k.replace("_", "-")}', '-p', str(o), timeout=1500)
                if (o / 'submission.csv').exists(): break
            line = sh(PY, str(ROOT / 'scripts/local/visible4_calibration.py'), 'score', str(o / 'submission.csv')).strip().splitlines()[-1]
            v4 = float(line.split('score=')[1].split()[0]) if 'score=' in line else 0.0
            log(f'HEAD: {k} v{v} visible-4 {line}')
            if v4 >= 0.9150:
                msg = (ROOT / f'kernels/{k}/SUBMIT_MSG.txt').read_text().strip() + f' Visible-4 {v4:.4f}.'
                log('HEAD: SUBMIT: ' + sh(K, 'competitions', 'submit', COMP, '-k', f'maruanbottoni/biohub-{k.replace("_", "-")}',
                                          '-v', v, '-f', 'submission.csv', '-m', msg).strip())
                mark('head', 'submitted')
            else:
                log(f'HEAD: NOT submitted, visible-4 {v4} < 0.9150'); mark('head', 'skip-v4')
    time.sleep(600)
log('=== decide_0928 all decisions made: ' + json.dumps(st))
