#!/usr/bin/env python3
"""Live progress board for the patch_p64 catalog."""
from __future__ import annotations
import json, time
from pathlib import Path
import sys
import yaml

ROOT = Path('/ml2/music/sliders-conceptmod')
sys.path.insert(0, str(ROOT))
from scripts.yue2_training_dashboard import dashboard_html, publish_metrics

EXP = ROOT / 'analysis/yue2_patch64_catalog_20260918'
LISTEN = ROOT / 'eval/listen/yue2-patch64-catalog-20260918'
PROMPTS = ROOT / 'analysis/yue2_uni16_1200_20260917/prompts'
STEPS = 1600

def label(key):
    path = PROMPTS / f'{key}-train.yaml'
    if not path.exists():
        return key
    return yaml.safe_load(path.read_text()).get('plus_label') or key

def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)

def chart_page(key, lab, note=''):
    banner = ''
    if note:
        banner = (
            '<p style="background:#3c3021;color:#f4c27a;padding:12px 14px;border-radius:8px">'
            f'{note}</p>'
        )
    return (
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        f'<title>{lab} · patch_p64</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}a{color:#a9d7ff}</style>'
        f'<p><a href="../">← catalog</a></p><h1>{lab}</h1>'
        '<p>patch_p64 ParticleGAN seed-bank · 1600 steps. The cosine chart is “Direction alignment”, scaled from −1 to 1.</p>'
        + banner
        + dashboard_html('particle_bridge')
    )

def tick():
    exp = json.loads((EXP/'experiment.json').read_text())
    state = json.loads((EXP/'state.json').read_text()) if (EXP/'state.json').exists() else {}
    active = {v: int(k) for k, v in (state.get('active') or {}).items() if v}
    results = {r['id']: r for r in state.get('results') or []}
    jobs = []
    for job in exp['jobs']:
        key = job['id']
        run = EXP/'runs'/key
        st = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        pr = json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
        dest = LISTEN/key
        if list(run.glob('updates-from-*.jsonl')):
            try:
                publish_metrics(run, dest)
            except (OSError, ValueError) as exc:
                print(f'metrics {key}: {exc}', flush=True)
            note = ''
            if status_hint := (st.get('abort_reason') or ''):
                note = (
                    'Stopped by the queue rule, not a cliff. Cosine never locked near 1; '
                    f'it was still wandering around 0.2–0.5 when this fired: {status_hint}. '
                    'On the −1…1 cosine chart that is a small step at the right edge, not a crash from 0.98.'
                )
            write(dest/'index.html', chart_page(key, label(key), note))
        status = st.get('status') or ('queued' if key not in active else 'starting')
        if key in results and results[key].get('reason','').startswith('collapse'):
            status = 'collapsed_abort'
        if key in active and status in ('queued', 'starting', None):
            status = 'training'
        jobs.append(dict(
            id=key, label=label(key),
            gpu=active.get(key, job.get('gpu')),
            status=status,
            completed=int(st.get('completed') or pr.get('step') or 0),
            total=STEPS,
            stage=st.get('abort_reason') or status,
            metrics={k: pr.get(k) for k in ('loss','g_adv','d_loss','particle_vic','grad_norm','cos_pos','noise_std','step_seconds')},
        ))
    workers = []
    for gpu in (0, 1):
        name = (state.get('active') or {}).get(str(gpu))
        workers.append(dict(gpu=gpu, status='training' if name else 'idle', job=name))
    write(LISTEN/'queue-state.json', json.dumps(dict(
        updated=time.time(), phase=state.get('phase'), jobs=jobs, workers=workers,
        pending=state.get('pending') or [],
    ), indent=2)+'\n')

def main():
    LISTEN.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            tick()
        except Exception as exc:
            print(f'publish: {exc}', flush=True)
        if not __import__('subprocess').run(['systemctl','--user','is-active','--quiet','music-yue2-patch64-catalog']).returncode == 0:
            # one last publish after the trainer stops
            try: tick()
            except Exception: pass
            # keep publishing while any run dir is still being written? exit when catalog unit is dead
            break
        time.sleep(2)

if __name__ == '__main__':
    main()
