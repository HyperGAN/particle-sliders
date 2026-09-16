#!/usr/bin/env python3
"""Publish existing YuE2 training logs for the live listening-page charts."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import time

FIELDS = ('loss', 'g_adv', 'pole', 'end', 'd_loss', 'd_pen', 'grad_norm',
          'cos_pos', 'cos_neg', 'step_seconds')


def dashboard_html():
    return (Path(__file__).parent / 'assets/yue2-training-dashboard.html').read_text()


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
