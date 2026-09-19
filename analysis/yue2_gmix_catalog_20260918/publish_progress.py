#!/usr/bin/env python3
"""Live board for the gmix catalog queue."""
from __future__ import annotations
import json, math, shutil, time
from pathlib import Path
import sys
import yaml

ROOT = Path('/ml2/music/sliders-conceptmod')
sys.path.insert(0, str(ROOT))
from scripts.yue2_training_dashboard import dashboard_html, publish_metrics

EXP = ROOT / 'analysis/yue2_gmix_catalog_20260918'
LISTEN = ROOT / 'eval/listen/yue2-gmix-catalog-20260918'
PROMPTS = ROOT / 'analysis/yue2_uni16_1200_20260917/prompts'
STEPS = 1600
UNIT = 'yue2-gmix-catalog'
PROBE_FIELDS = (
    'step', 'strength', 'residual_rms', 'residual_p95', 'gain_mean',
    'orthogonal_error', 'teacher_swd', 'game_swd_sigma_1', 'evaluator_auc',
)
LATEST = (
    'loss', 'g_adv', 'd_loss', 'particle_vic', 'grad_norm', 'particle_grad_norm',
    'cos_pos', 'noise_std', 'step_seconds',
)


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


def num(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def chart_page(lab):
    return (
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        f'<title>{lab} · gmix</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}a{color:#a9d7ff}</style>'
        f'<p><a href="../">← gmix catalog</a></p><h1>{lab}</h1>'
        '<p>gmix critic. Noise floor at step 1600. No cosine abort.</p>'
        + dashboard_html('particle_bridge')
    )


def publish_probe(run, dest):
    points = []
    path = run / 'probe.jsonl'
    if not path.exists():
        return None, {}, None
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            continue
        record = json.loads(line)
        if record.get('weights_kind') != 'ema':
            continue
        points.append({key: record.get(key) for key in PROBE_FIELDS})
    summary_path = run / 'probe-summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary_path.exists():
        shutil.copy2(summary_path, dest / 'probe-summary.json')
    payload = dict(
        points=points,
        stopping=summary.get('stopping') or {},
        best_step=summary.get('best_step'),
        published_at=time.time(),
    )
    write(dest / 'probe-metrics.json', json.dumps(payload, allow_nan=False) + '\n')
    latest = next((point for point in reversed(points) if point.get('strength') == 1.0), None)
    stamp = summary_path.stat().st_mtime if summary_path.exists() else path.stat().st_mtime
    return latest, payload['stopping'], stamp


def publish_board():
    src = EXP / 'board.html'
    dest = LISTEN / 'index.html'
    if not src.exists():
        return
    text = src.read_text()
    if dest.exists() and dest.read_text() == text:
        return
    write(dest, text)


def job_record(key, lab, run, *, active, pending, gpu=None):
    st = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
    pr = json.loads((run / 'progress.json').read_text()) if (run / 'progress.json').exists() else {}
    dest = LISTEN / key
    metrics_at = None
    has_metrics = False
    if list(run.glob('updates-from-*.jsonl')):
        try:
            payload = publish_metrics(run, dest)
            metrics_at = payload.get('updated_at')
            has_metrics = bool(payload.get('points'))
        except (OSError, ValueError) as exc:
            print(f'metrics {key}: {exc}', flush=True)
        write(dest / 'index.html', chart_page(lab))
        if (run / 'status.json').exists():
            (dest / 'status.json').write_text((run / 'status.json').read_text())
    latest_probe, stopping, probe_at = (None, {}, None)
    if (run / 'probe.jsonl').exists():
        try:
            latest_probe, stopping, probe_at = publish_probe(run, dest)
        except (OSError, ValueError) as exc:
            print(f'probe {key}: {exc}', flush=True)
    status = st.get('status') or ('training' if key in active else 'queued')
    if key in active and status in ('queued', 'starting', None):
        status = 'training'
    step = int(st.get('completed') or pr.get('step') or 0)
    shown_gpu = active.get(key)
    if shown_gpu is None and gpu is None:
        shown_gpu = st.get('gpu')
    return dict(
        id=key, label=lab, gpu=shown_gpu if gpu is None else gpu, status=status,
        completed=step, total=int(st.get('total') or STEPS),
        position=(pending.index(key) + 1) if key in pending else None,
        has_metrics=has_metrics, metrics_at=metrics_at,
        has_probe=latest_probe is not None, probe_at=probe_at,
        probe_status=(stopping or {}).get('status'),
        latest={name: num(pr.get(name)) for name in LATEST},
        probe=None if latest_probe is None else {name: latest_probe.get(name) for name in PROBE_FIELDS},
    )


def tick():
    exp = json.loads((EXP / 'experiment.json').read_text())
    state = json.loads((EXP / 'state.json').read_text()) if (EXP / 'state.json').exists() else {}
    active = {v: int(k) for k, v in (state.get('active') or {}).items() if v}
    pending = list(state.get('pending') or [])
    gender = ROOT / 'analysis/yue2_gender_v2_20260918/runs'
    jobs = [
        job_record('male-v2', 'Male', gender / 'male', active=active, pending=pending),
        job_record('female-v2', 'Female', gender / 'female', active=active, pending=pending),
    ]
    for job in exp['jobs']:
        jobs.append(job_record(job['id'], label(job['id']), EXP / 'runs' / job['id'], active=active, pending=pending))
    write(LISTEN / 'queue-state.json', json.dumps(dict(
        updated=time.time(), phase=state.get('phase'), jobs=jobs,
        pending=pending, held_gpus=state.get('held_gpus') or [],
        active=state.get('active') or {},
    ), allow_nan=False, indent=2) + '\n')
    publish_board()


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
        if inactive and not (EXP / 'state.json').exists():
            time.sleep(2)
            continue
        if inactive:
            try:
                tick()
            except Exception:
                pass
            phase = json.loads((EXP / 'state.json').read_text()) if (EXP / 'state.json').exists() else {}
            if phase.get('phase') in ('done', 'stopped') and not phase.get('pending') and not any((phase.get('active') or {}).values()):
                break
        time.sleep(2)


if __name__ == '__main__':
    main()
