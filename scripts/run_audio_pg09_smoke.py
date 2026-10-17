"""One bounded lease, three sequential owned workers, exact1+2vs3 check."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args, worker_args = parser.parse_known_args()
    if any(flag in worker_args for flag in ('--until', '--resume', '--out')):
        parser.error('supervisor owns until/resume/out worker arguments')
    worker = Path(__file__).with_name('qualify_audio_pg09_live.py')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    full, first, resumed = [args.out / name for name in ('full3', 'first1', 'resumed3')]
    stages = [('full3', ['--until', '3', '--out', str(full)]),
              ('first1', ['--until', '1', '--out', str(first)]),
              ('resumed3', ['--until', '3', '--out', str(resumed), '--resume', str(first / 'checkpoint.pt')])]
    for name, extra in stages:
        remaining = 900 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('900-second total smoke lease budget exhausted')
        with (args.out / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(worker), *worker_args, *extra],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
    a = torch.load(full / 'qualification.pt', map_location='cpu', weights_only=True)
    b = torch.load(resumed / 'qualification.pt', map_location='cpu', weights_only=True)
    matched = equal(a, b)
    receipt = dict(status='exact_smoke_passed' if matched else 'exact_smoke_failed',
        exact_full_state_metrics_probes_export=matched, elapsed_seconds=time.monotonic() - started,
        maximum_wall_seconds=900, worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        worker_arguments=worker_args, pretrained_quality_qualified=False)
    (args.out / 'comparison.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not matched:
        raise ValueError('fresh-process model/optimizer/RNG/loss/probe/export state differs')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
