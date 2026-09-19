#!/usr/bin/env python3
"""One on/off listening page for finished and in-progress gmix sliders.

`publish` rewrites manifest.json for the static page.
`sample` renders missing off and on clips, then renders a newer checkpoint
when one is saved. Finished runs go first.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path('/ml2/music/sliders-conceptmod')
LISTEN_ROOT = ROOT / 'eval/listen'
PAGE = LISTEN_ROOT / 'yue2-onoff-20260918'
PROMPTS = ROOT / 'analysis/yue2_uni16_1200_20260917/prompts'
CATALOG = ROOT / 'analysis/yue2_gmix_catalog_20260918'
GENDER = ROOT / 'analysis/yue2_gender_v2_20260918'
PY = '/ml2/music/.cache/yue2-test-env/bin/python'
SEEDS = (1709, 2903)
ROWS = (0, 1)

EXISTING = {
    'pop': LISTEN_ROOT / 'yue2-gmix-catalog-20260918/pop/samples',
    'metal': LISTEN_ROOT / 'yue2-editnorm-critics-20260918/metal_gmix_t8_w48_l1/samples',
}

PAGE_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>On / off</title>
<style>
  body{margin:0;background:#14171b;color:#eee;font:16px/1.4 system-ui}
  main{max-width:980px;margin:0 auto;padding:28px 18px 80px}
  h1{font-size:28px;margin:0 0 6px}
  .sub{color:#a8b0b8;margin:0 0 22px}
  article{border:1px solid #3c4550;border-radius:12px;padding:16px 16px 8px;margin:0 0 16px;background:#1c2127}
  article h2{margin:0;font-size:20px}
  .meta{color:#a8b0b8;margin:4px 0 12px}
  .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px}
  .clip{background:#12161a;border-radius:8px;padding:10px 12px}
  .clip b{display:block;margin-bottom:6px}
  audio{width:100%;height:36px}
  .wait{color:#8b949e;margin:8px 0}
  h3{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#8b949e;margin:22px 0 8px}
  @media(max-width:700px){.pair{grid-template-columns:1fr}}
</style>
<main>
  <h1>On / off</h1>
  <p class="sub">Same prompt and seed. Off is scale 0. On is scale 1. For Female and Male, that off caption is the other gender. Finished runs use the last checkpoint. A run still training uses the latest save, then replaces it when a newer one is ready.</p>
  <div id="app"></div>
</main>
<script>
const app = document.querySelector('#app');
const nodes = new Map();

function clip(id, src, label) {
  let box = nodes.get(id);
  if (!box) {
    box = document.createElement('div');
    box.className = 'clip';
    box.innerHTML = '<b></b>';
    nodes.set(id, box);
  }
  box.querySelector('b').textContent = label;
  const audio = box.querySelector('audio');
  if (src && (!audio || audio.getAttribute('src') !== src)) {
    const wait = box.querySelector('.wait');
    if (wait) wait.remove();
    const next = document.createElement('audio');
    next.controls = true;
    next.preload = 'none';
    next.src = src;
    if (audio) audio.replaceWith(next);
    else box.appendChild(next);
  } else if (!src && !audio && !box.querySelector('.wait')) {
    const wait = document.createElement('p');
    wait.className = 'wait';
    wait.textContent = 'Rendering';
    box.appendChild(wait);
  }
  return box;
}

function draw(data) {
  const groups = [
    ['Done', data.runs.filter(run => run.state === 'complete')],
    ['Still training', data.runs.filter(run => run.state !== 'complete')],
  ];
  app.replaceChildren();
  for (const [title, runs] of groups) {
    if (!runs.length) continue;
    const heading = document.createElement('h3');
    heading.textContent = title;
    app.appendChild(heading);
    for (const run of runs) {
      const card = document.createElement('article');
      const h2 = document.createElement('h2');
      h2.textContent = run.label;
      const meta = document.createElement('p');
      meta.className = 'meta';
      meta.textContent = run.note;
      card.append(h2, meta);
      for (const row of run.rows) {
        const pair = document.createElement('div');
        pair.className = 'pair';
        const key = run.id + '-' + row.row + '-' + row.seed;
        pair.append(
          clip(key + '-off', row.off, (run.off_label || 'Off') + ' · prompt ' + (row.row + 1) + ' · seed ' + row.seed),
          clip(key + '-on', row.on, 'On · prompt ' + (row.row + 1) + ' · seed ' + row.seed),
        );
        card.appendChild(pair);
      }
      app.appendChild(card);
    }
  }
}

async function tick() {
  try {
    const response = await fetch('manifest.json', {cache: 'no-store'});
    if (response.ok) draw(await response.json());
  } catch (error) {}
}
tick();
setInterval(tick, 3000);
</script>
"""


def read_json(path):
    return json.loads(Path(path).read_text())


def ema_weights(run_dir):
    found = [path for path in run_dir.glob('*_last.safetensors') if not path.name.endswith('_live_last.safetensors')]
    if len(found) != 1:
        raise RuntimeError(f'Expected one EMA checkpoint in {run_dir}, found {found}')
    return found[0]


def ckpt_step(weights):
    return int(read_json(weights.with_suffix('.json'))['step'])


def url_for(path):
    relative = Path(path).resolve().relative_to(LISTEN_ROOT.resolve())
    return '/' + relative.as_posix()


def clips_ready(directory):
    return all(
        (directory / f'row-{row}-seed-{seed}' / take / 'audio.flac').exists()
        for row in ROWS for seed in SEEDS for take in ('off', 'metal')
    )


# Scale 0 stays the base model. For these sliders the base caption is the other gender.
OPPOSITE = {
    'female': ('male-eval.yaml', 'Off · male'),
    'female-h13': ('male-eval.yaml', 'Off · male'),
    'male-h13': ('female-eval.yaml', 'Off · female'),
}


def opposite_prompts(slider_id, target_path):
    spec = OPPOSITE.get(slider_id)
    if spec is None:
        return target_path, None
    other_name, label = spec
    target = yaml.safe_load(Path(target_path).read_text())
    other = yaml.safe_load((PROMPTS / other_name).read_text())
    if len(target['rows']) != len(other['rows']):
        raise RuntimeError(f'{slider_id} and {other_name} have different row counts')
    for row, opposite in zip(target['rows'], other['rows']):
        if row['lyrics'] != opposite['lyrics']:
            raise RuntimeError(f'{slider_id} lyrics do not match {other_name}')
        row['neutral'] = opposite['positive']
    dest = PAGE / slider_id / 'prompts-off-opposite.yaml'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(target, sort_keys=False, allow_unicode=True))
    return dest, label


def jobs():
    found = []
    experiment = read_json(CATALOG / 'experiment.json')
    for spec in experiment['jobs']:
        run_dir = CATALOG / 'runs' / spec['id']
        status_path = run_dir / 'status.json'
        if not status_path.exists():
            continue
        status = read_json(status_path)
        if status.get('status') not in ('training', 'complete', 'probing'):
            continue
        weights = ema_weights(run_dir)
        found.append({
            'id': spec['id'],
            'label': read_label(PROMPTS / f"{spec['id']}-eval.yaml", spec['id']),
            'state': 'complete' if status.get('status') == 'complete' else 'training',
            'live_step': int(status.get('completed') or 0),
            'ckpt_step': ckpt_step(weights),
            'weights': weights,
            'prompts': PROMPTS / f"{spec['id']}-eval.yaml",
            'off_label': None,
            'existing': EXISTING.get(spec['id']),
        })
    for spec in (
        {'id': 'female', 'label': 'Female', 'run': 'female', 'prompts': 'female-eval.yaml'},
        {'id': 'female-h13', 'label': 'Female h13', 'run': 'female_h13', 'prompts': 'female-eval.yaml'},
        {'id': 'male-h13', 'label': 'Male h13', 'run': 'male_h13', 'prompts': 'male-eval.yaml'},
    ):
        run_dir = GENDER / 'runs' / spec['run']
        status_path = run_dir / 'status.json'
        if not status_path.exists():
            continue
        status = read_json(status_path)
        if status.get('status') != 'complete':
            continue
        weights = ema_weights(run_dir)
        prompts, off_label = opposite_prompts(spec['id'], PROMPTS / spec['prompts'])
        found.append({
            'id': spec['id'],
            'label': spec['label'],
            'state': 'complete',
            'live_step': int(status.get('completed') or 0),
            'ckpt_step': ckpt_step(weights),
            'weights': weights,
            'prompts': prompts,
            'off_label': off_label,
            'existing': None,
        })
    metal = EXISTING['metal']
    if metal.exists():
        found.append({
            'id': 'metal',
            'label': 'Metal',
            'state': 'complete',
            'live_step': 1600,
            'ckpt_step': 1600,
            'weights': None,
            'prompts': PROMPTS / 'metal-eval.yaml',
            'off_label': None,
            'existing': metal,
        })
    found.sort(key=lambda job: (job['state'] != 'complete', job['label'].lower()))
    return found


def read_label(path, fallback):
    for line in path.read_text().splitlines():
        if line.startswith('plus_label:'):
            return line.split(':', 1)[1].strip().strip("'\"")
    return fallback


def audio_dir(job):
    if job['existing']:
        return job['existing']
    name = f"step-{job['ckpt_step']}"
    if job.get('off_label'):
        name += '-off-opposite'
    return PAGE / job['id'] / name


def manifest():
    runs = []
    for job in jobs():
        directory = audio_dir(job)
        rows = []
        for row in ROWS:
            for seed in SEEDS:
                off = directory / f'row-{row}-seed-{seed}' / 'off' / 'audio.flac'
                on = directory / f'row-{row}-seed-{seed}' / 'metal' / 'audio.flac'
                rows.append({
                    'row': row,
                    'seed': seed,
                    'off': url_for(off) if off.exists() else None,
                    'on': url_for(on) if on.exists() else None,
                })
        if job['state'] == 'complete':
            note = f"Finished · checkpoint {job['ckpt_step']}"
        else:
            note = f"Training step {job['live_step']} · listening to checkpoint {job['ckpt_step']}"
        if job.get('off_label'):
            note += '. Off uses the other gender caption. On is that caption with the slider at 1.'
        runs.append({
            'id': job['id'], 'label': job['label'], 'state': job['state'], 'note': note,
            'off_label': job.get('off_label'), 'rows': rows,
        })
    return {'runs': runs}


def publish_once():
    PAGE.mkdir(parents=True, exist_ok=True)
    html_path = PAGE / 'index.html'
    if not html_path.exists() or html_path.read_text() != PAGE_HTML:
        html_path.write_text(PAGE_HTML)
    payload = json.dumps(manifest(), indent=2)
    manifest_path = PAGE / 'manifest.json'
    if not manifest_path.exists() or manifest_path.read_text() != payload:
        manifest_path.write_text(payload)


def publish_loop():
    while True:
        publish_once()
        time.sleep(2)


def render(job):
    dest = audio_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        PY, '-u', str(ROOT / 'scripts/evaluate_yue2_arm_b.py'),
        '--recipe', 'particle_bridge',
        '--takes', 'off', 'metal',
        '--weights', str(job['weights']),
        '--prompts_file', str(job['prompts']),
        '--output_dir', str(dest),
        '--seeds', '1709', '2903',
        '--max_tokens', '500',
    ]
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(ROOT),
        'CUDA_VISIBLE_DEVICES': '0',
        'HF_HUB_OFFLINE': '1',
        'HF_HOME': '/ml2/music/.cache/huggingface',
    })
    print(f"render {job['id']} step {job['ckpt_step']}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def sample_loop():
    while True:
        pending = []
        for job in jobs():
            if job['existing'] or job['weights'] is None:
                continue
            if job['state'] != 'complete':
                continue
            dest = audio_dir(job)
            if clips_ready(dest):
                continue
            pending.append(job)
        pending.sort(key=lambda job: (job['state'] != 'complete', -job['ckpt_step']))
        if not pending:
            time.sleep(20)
            continue
        render(pending[0])


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ('publish', 'sample'):
        raise SystemExit('usage: onoff_listen.py publish|sample')
    if sys.argv[1] == 'publish':
        publish_loop()
    else:
        sample_loop()
