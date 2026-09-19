#!/usr/bin/env python3
"""Live charts for the female noise-floor retrains."""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/ml2/music/sliders-conceptmod')
sys.path.insert(0, str(ROOT))
from scripts.yue2_training_dashboard import dashboard_html, publish_metrics

EXP = ROOT / 'analysis/yue2_editnorm_critics_20260918'
LISTEN = ROOT / 'eval/listen/yue2-editnorm-critics-20260918'
RUNS = (
    ('female_sn_mlp_noise1600', 'sn_mlp'),
    ('female_gmix_noise1600', 'gmix'),
)
PROBE_FIELDS = (
    'step', 'strength', 'residual_rms', 'residual_p95', 'gain_mean',
    'orthogonal_error', 'teacher_swd', 'game_swd_sigma_1', 'evaluator_auc',
)
PROBE_PANEL = '''
<section id="probe">
<style>
#probe{margin:8px 0 28px}
#probe table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
#probe td,#probe th{border-bottom:1px solid #333;padding:6px 8px;text-align:left}
#probe .muted{color:#aab7c6;font-size:14px}
</style>
<h2>Held-out probe</h2>
<p class="muted" id="probe-status">Waiting for the first EMA probe.</p>
<table>
<thead><tr><th>step</th><th>residual RMS</th><th>p95</th><th>gain</th><th>teacher SWD</th><th>game SWD σ=1</th></tr></thead>
<tbody id="probe-rows"></tbody>
</table>
<p class="muted">Strength 1.0 on the held-out bank. <a href="probe-metrics.json">probe-metrics.json</a> also has 0.25 and 0.5. evaluator_auc stays null.</p>
<script>
function fmt(v){return v==null?'—':Number(v).toFixed(3)}
async function pollProbe(){
  try{
    const response=await fetch('probe-metrics.json',{cache:'no-store'});
    if(!response.ok)throw new Error('missing');
    const data=await response.json();
    const rows=(data.points||[]).filter(point=>point.strength===1);
    const stop=data.stopping||{};
    document.getElementById('probe-status').textContent=
      'status '+(stop.status||'collecting')+' · best '+(data.best_step??'—')+' · evaluator not trained';
    document.getElementById('probe-rows').innerHTML=rows.length?rows.map(point=>
      '<tr><td>'+point.step+'</td><td>'+fmt(point.residual_rms)+'</td><td>'+fmt(point.residual_p95)+
      '</td><td>'+fmt(point.gain_mean)+'</td><td>'+fmt(point.teacher_swd)+'</td><td>'+fmt(point.game_swd_sigma_1)+'</td></tr>'
    ).join(''):'<tr><td colspan="6">No EMA probe yet.</td></tr>';
  }catch(error){
    document.getElementById('probe-status').textContent='Probe metrics not published yet.';
  }
  setTimeout(pollProbe,2000);
}
pollProbe();
</script>
</section>
'''
CHART = (
    '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
    '<title>__TITLE__</title>'
    '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}a{color:#a9d7ff}</style>'
    '<p><a href="../">← edit-norm compare</a></p><h1>__TITLE__</h1>'
    '<p>Female retrain. Noise floor at step 1600. ParticleGAN, no cosine abort.</p>'
    + dashboard_html('particle_bridge')
    + PROBE_PANEL
)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def publish_probe(run, dest):
    points = []
    path = run / 'probe.jsonl'
    if path.exists():
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
    latest = next((point for point in reversed(points) if point['strength'] == 1.0), None)
    return latest, payload['stopping']


def tick():
    rows = []
    alive = False
    for key, critic in RUNS:
        run = EXP / 'runs' / key
        dest = LISTEN / key
        publish_metrics(run, dest)
        latest, stopping = publish_probe(run, dest)
        write(dest / 'index.html', CHART.replace('__TITLE__', key))
        status_path = run / 'status.json'
        if status_path.exists():
            shutil.copy2(status_path, dest / 'status.json')
        st = json.loads(status_path.read_text()) if status_path.exists() else {}
        pr = json.loads((run / 'progress.json').read_text()) if (run / 'progress.json').exists() else {}
        status = st.get('status') or 'training'
        if status not in ('complete', 'failed', 'paused', 'collapsed_abort'):
            alive = True
        step = st.get('completed') or pr.get('step') or 0
        cos = float(pr.get('cos_pos') or 0)
        noise = float(pr.get('noise_std') or 0)
        def cell(point, name):
            if not point or point.get(name) is None:
                return '—'
            return f'{float(point[name]):.3f}'
        rows.append(
            '<tr><td><a href="{key}/">{key}</a></td><td>female</td><td>{critic}</td>'
            '<td>{status}</td><td>{step}/1600</td><td>{cos:.4f}</td><td>{noise:.4f}</td>'
            '<td>{rms}</td><td>{gain}</td><td>{swd}</td><td>{stop}</td></tr>'.format(
                key=key, critic=critic, status=status, step=step, cos=cos, noise=noise,
                rms=cell(latest, 'residual_rms'), gain=cell(latest, 'gain_mean'),
                swd=cell(latest, 'teacher_swd'), stop=(stopping or {}).get('status') or '—',
            )
        )
    rows.append(
        '<tr><td><a href="metal_sn_mlp_w128_l1/samples/">metal_sn_mlp samples</a></td>'
        '<td>metal</td><td>sn_mlp</td><td>listening</td><td>1600/1600</td><td></td><td></td>'
        '<td>—</td><td>—</td><td>—</td><td>—</td></tr>'
    )
    rows.append(
        '<tr><td><a href="metal_gmix_t8_w48_l1/samples/">metal_gmix samples</a></td>'
        '<td>metal</td><td>gmix</td><td>listening</td><td>1600/1600</td><td></td><td></td>'
        '<td>—</td><td>—</td><td>—</td><td>—</td></tr>'
    )
    write(
        LISTEN / 'index.html',
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        '<title>Edit-norm critics</title>'
        '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}'
        'table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:8px;text-align:left}a{color:#a9d7ff}</style>'
        '<h1>Edit-norm · female retrain</h1>'
        '<p>Noise reaches the floor at step 1600. Charts refresh every second. '
        '<a href="../yue2-gmix-catalog-20260918/">gmix catalog queue</a></p>'
        '<table><tr><th>run</th><th>slider</th><th>critic</th><th>status</th><th>step</th><th>cos</th><th>noise</th>'
        '<th>residual RMS</th><th>gain</th><th>teacher SWD</th><th>probe</th></tr>'
        + ''.join(rows) + '</table>',
    )
    return alive


def main():
    while tick():
        time.sleep(2)
    tick()


if __name__ == '__main__':
    main()
