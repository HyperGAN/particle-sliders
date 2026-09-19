#!/usr/bin/env python3
"""Live progress board for edit-norm sn_mlp vs gmix."""
from __future__ import annotations
import json, time
from pathlib import Path
import sys

ROOT = Path('/ml2/music/sliders-conceptmod')
sys.path.insert(0, str(ROOT))
from scripts.yue2_training_dashboard import dashboard_html, publish_metrics

EXP = ROOT / 'analysis/yue2_editnorm_critics_20260918'
LISTEN = ROOT / 'eval/listen/yue2-editnorm-critics-20260918'
STEPS = 1600
UNIT = 'music-yue2-editnorm-critics'


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def chart_page(name, title):
    return (
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        f'<title>{title}</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}a{color:#a9d7ff}</style>'
        f'<p><a href="./">← edit-norm compare</a></p><h1>{title}</h1>'
        '<p>Paired-edit whitening · noise_start = edit_rms / 0.28 · ParticleGAN (no R1/R2).</p>'
        + dashboard_html('particle_bridge')
    )


def index_page(jobs, state):
    rows = []
    for j in jobs:
        m = j.get('metrics') or {}
        rows.append(
            f"<tr><td><a href='{j['id']}/'>{j['id']}</a></td><td>{j.get('slider')}</td>"
            f"<td>{j.get('critic')}</td><td>{j.get('status')}</td>"
            f"<td>{j.get('completed')}/{STEPS}</td>"
            f"<td>{(m.get('cos_pos') or 0):.4f}</td>"
            f"<td>{(m.get('noise_std') or 0):.4f}</td>"
            f"<td>{j.get('edit_rms') or '—'}</td>"
            f"<td>{j.get('noise_start') or '—'}</td></tr>"
        )
    return (
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        '<title>Edit-norm critics</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}'
        'table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:8px;text-align:left}'
        'a{color:#a9d7ff}</style>'
        '<h1>Edit-norm · sn_mlp vs gmix</h1>'
        f"<p>phase={state.get('phase')} · paired-edit whitening before architecture compare</p>"
        '<table><tr><th>run</th><th>slider</th><th>critic</th><th>status</th><th>step</th>'
        '<th>cos</th><th>noise</th><th>edit_rms</th><th>noise_start</th></tr>'
        + ''.join(rows) + '</table>'
    )


def tick():
    exp = json.loads((EXP / 'experiment.json').read_text())
    state = json.loads((EXP / 'state.json').read_text()) if (EXP / 'state.json').exists() else {}
    active = {v: int(k) for k, v in (state.get('active') or {}).items() if v}
    jobs = []
    for cand in exp['candidates']:
        key = cand['name']
        run = EXP / 'runs' / key
        st = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
        pr = json.loads((run / 'progress.json').read_text()) if (run / 'progress.json').exists() else {}
        audit = {}
        if (run / 'teacher-audit.json').exists():
            audit = (json.loads((run / 'teacher-audit.json').read_text()).get('normalization') or {})
        dest = LISTEN / key
        if list(run.glob('updates-from-*.jsonl')):
            try:
                publish_metrics(run, dest)
            except (OSError, ValueError) as exc:
                print(f'metrics {key}: {exc}', flush=True)
            write(dest / 'index.html', chart_page(key, key))
        status = st.get('status') or ('queued' if key not in active else 'starting')
        if key in active and status in ('queued', 'starting', None):
            status = 'training'
        jobs.append(dict(
            id=key, slider=cand.get('slider'), critic=cand.get('critic'),
            gpu=active.get(key), status=status,
            completed=int(st.get('completed') or pr.get('step') or 0),
            edit_rms=audit.get('edit_rms'), noise_start=audit.get('noise_start'),
            metrics={k: pr.get(k) for k in (
                'loss', 'g_adv', 'd_loss', 'particle_vic', 'grad_norm', 'cos_pos', 'noise_std', 'step_seconds'
            )},
        ))
    write(LISTEN / 'index.html', index_page(jobs, state))
    write(LISTEN / 'queue-state.json', json.dumps(dict(
        updated=time.time(), phase=state.get('phase'), jobs=jobs,
        pending=state.get('pending') or [], active=state.get('active') or {},
    ), indent=2) + '\n')


def main():
    LISTEN.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            tick()
        except Exception as exc:
            print(f'publish: {exc}', flush=True)
        inactive = __import__('subprocess').run(
            ['systemctl', '--user', 'is-active', '--quiet', UNIT]
        ).returncode != 0
        if inactive:
            try:
                tick()
            except Exception:
                pass
            break
        time.sleep(2)


if __name__ == '__main__':
    main()
