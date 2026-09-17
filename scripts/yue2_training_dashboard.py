#!/usr/bin/env python3
"""Publish existing YuE2 training logs for the live listening-page charts."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import time

FIELDS = ('loss', 'g_adv', 'd_loss', 'd_pen', 'grad_norm',
          'cos_pos', 'step_seconds')


def dashboard_html(recipe='unipolar_gan'):
    document = (Path(__file__).parent / 'assets/yue2-training-dashboard.html').read_text()
    if recipe == 'gan_plus_neu':
        document = document.replace('The generator trains +1 only, using the adversarial loss averaged across four shuffled prompt rows.',
            'The GAN judges +1 and 0 separately using a scale-conditioned discriminator, averaging across both endpoints and four shuffled prompt rows. The exact-zero endpoint contributes a constant log(2) and no adapter gradient.')
        document = document.replace('GAN (+1)', 'GAN (+/0)').replace('Global L2 norm before per-element clipping at 1.',
            'Global L2 norm before the optimizer update; no gradient clipping.')
    elif recipe == 'particle_bridge':
        document = document.replace('The generator trains +1 only, using the adversarial loss averaged across four shuffled prompt rows.',
            'The generator trains +1 with the paired-error GAN and particle VIC. D and G use separate batches of 64 prompt/noise pairs. All branches use routed particles. Listening weights use EMA 0.995; curves show live training.')
        document = document.replace('Global L2 norm before per-element clipping at 1.',
            'Global L2 norm before the optimizer update; no gradient clipping.')
        document = document.replace('mean softplus(D(real) − D(fake))</code>',
            'mean softplus(D(real) − D(fake)) + particle VIC</code>')
        document = document.replace('Real samples are raw metal-caption deltas.',
            'Real = noise; fake = the same noise + normalized student − normalized positive target.')
        document = document.replace('in teacher-RMS coordinates, with coefficient 1.',
            'in normalized error coordinates, every fourth update with ×4 weighting. Noise decreases from 1 to 0.03 over 8,000 updates; all three learning rates stay constant.')
        document = document.replace('No ending, pole-MSE, feature-matching, lyric-hold or other auxiliary losses.',
            'VIC acts only on the particle cloud. No ending, pole-MSE, feature-matching or lyric-hold loss.')
        document = document.replace("note:'The entire objective is the unipolar GAN loss.',series:[['g_adv','GAN (+1)','#8bbfff']]",
            "note:'Live G total = paired-error GAN + particle VIC. EMA is used only for exported listening weights.',series:[['loss','G total','#eeeeee'],['g_adv','GAN','#8bbfff'],['particle_vic','Particle VIC','#78dfc0']]")
        document = document.replace("  let points = [], data = null", """  specs.push(
    {title:'Particle gradients',note:'GAN-only cloud gradient is measured before adding VIC; total includes VIC.',series:[['particle_gan_grad_norm','GAN only','#8bbfff'],['particle_grad_norm','With VIC','#cba8ff']]},
    {title:'Critic noise',note:'Geometric noise schedule; this is not learning-rate decay.',series:[['noise_std','Noise standard deviation','#78dfc0']]}
  );
  let points = [], data = null""")
    return document


def publish_metrics(run: Path, output: Path):
    """Read completed lines only; a resumed log supersedes its abandoned tail."""
    points = {}
    updated = None
    logs = sorted(run.glob('updates-from-*.jsonl'), key=lambda p: int(p.stem.rsplit('-', 1)[1]))
    for path in logs:
        start = int(path.name.split('-')[2])
        points = {step: point for step, point in points.items() if step <= start}
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue  # The trainer may still be writing this line.
            record = json.loads(line)
            point = {'step': int(record['step'])}
            for key in FIELDS:
                value = float(record[key])
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite {key} at update {point["step"]}')
                point[key] = value
            for key in ('particle_vic','particle_grad_norm','particle_gan_grad_norm','noise_std','g_lr','d_lr','particle_lr'):
                if key in record:
                    value=float(record[key])
                    if not math.isfinite(value):raise ValueError(f'Non-finite {key}')
                    point[key]=value
            points[point['step']] = point
        updated = path.stat().st_mtime
    status = run / 'status.json'
    training = json.loads(status.read_text()) if status.exists() else {}
    payload = dict(points=[points[k] for k in sorted(points)], training=training,
                   updated_at=updated, published_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'training-metrics.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')) + '\n')
    temporary.replace(target)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.metrics-publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            payload = publish_metrics(args.save_dir, args.output_dir)
            if not args.watch or payload['training'].get('status') in {'complete', 'paused', 'failed'}:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
