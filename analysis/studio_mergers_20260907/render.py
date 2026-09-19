"""Run the isolated dense-merger screen on physical GPU 1, preserving every attempt."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['LATHE_SLIDER_APPLY'] = 'merge'
os.environ['HF_HUB_OFFLINE'] = '1'

import argparse
import fcntl
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import APP_ROOT, WORK, sha

OUTPUT = APP_ROOT/'sliders-conceptmod/eval/listen/studio-mergers-20260907'


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify(spec):
    from app import sliders
    checks = {**spec['source_sha256'], str(sliders.REGISTRY_PATH): spec['registry_sha256'],
              spec['fixture']['path']: spec['fixture']['sha256']}
    for item in spec['checkpoints']:
        checks[item['weights']] = item['sha256']
        checks[str(sliders._sidecar_path(Path(item['weights'])))] = item['sidecar_sha256']
    for path, expected in checks.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    checked_manifests = set()
    for job in spec['jobs']:
        if job['method'] in ('ties','knots_ties'):
            for comp in job['lora_components']:
                if sha(comp['weights']) != comp['sha256']:
                    raise ValueError('Manifest changed')
                manifest = json.loads(Path(comp['weights']).read_text())
                if sliders.resolve(job['sliders'], host_energy=job['energy']) != manifest['source_components']:
                    raise ValueError('Dense source components changed')
                if comp['sha256'] not in checked_manifests:
                    for module in manifest['modules']:
                        if sha(module['path']) != module['sha256']:
                            raise ValueError('Dense artifact changed: '+module['name'])
                    checked_manifests.add(comp['sha256'])
            continue
        if sliders.resolve(job['sliders'], host_energy=job['energy']) != job['lora_components']:
            raise ValueError(f'Resolved components changed: {job["id"]}')


def update_page(state):
    # Reporting is independent of frozen inference code; a page failure must
    # not throw away a completed render or stop remaining jobs.
    try:
        from report import build
        build(state, OUTPUT)
    except Exception:
        logging.exception('Listening page refresh failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (WORK/'render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads((WORK/'screen.json').read_text())
    verify(spec)
    processes = subprocess.check_output([
        'nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
    ], text=True).strip()
    if processes:
        raise RuntimeError(f'Physical GPU 1 is occupied by PID(s): {processes}')
    import numpy as np
    import soundfile as sf
    import torch
    from app import generator
    from dense_runtime import install
    apply_events = install(generator)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    state_path = WORK/'renders.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        schema=1, screen_sha256=sha(WORK/'screen.json'), status='running',
        physical_gpu=1, output=str(OUTPUT), records={})
    if state['screen_sha256'] != sha(WORK/'screen.json'):
        raise ValueError('Screen changed since first attempt')
    state['renderer_sha256'] = sha(__file__)
    state['status'] = 'running'
    state['pid'] = os.getpid()
    state['packages'] = dict(torch=torch.__version__)
    write(state_path, state)
    update_page(state)
    for job in spec['jobs']:
        record = state['records'].get(job['id'])
        if record and record['status'] == 'complete':
            for name in ('audio', 'excerpt'):
                if sha(record[name]) != record[name+'_sha256']:
                    raise ValueError(f'Completed audio changed: {record[name]}')
            continue
        if record and record['status'] == 'error' and not args.retry_errors:
            continue
        previous = [] if record is None else record.get('previous_attempts', []) + [
            {k:v for k,v in record.items() if k != 'previous_attempts'}]
        attempt = len(previous)+1
        dest = OUTPUT/f'{job["id"]}-attempt{attempt}.wav'
        excerpt = OUTPUT/f'{job["id"]}-attempt{attempt}-20s.wav'
        record = dict(id=job['id'], label=job['label'], status='running', attempt=attempt,
                      previous_attempts=previous, audio=str(dest), excerpt=str(excerpt),
                      started=time.time(), stage='starting')
        state['records'][job['id']] = record
        state['current'] = job['id']
        write(state_path, state)
        update_page(state)
        print(f'START {job["id"]} {job["label"]}', flush=True)
        last_update = [0.]

        def progress(phase, current, total):
            now = time.time()
            if now-last_update[0] >= 10 or current == total:
                record.update(stage=phase, progress_current=current, progress_total=total)
                write(state_path, state)
                last_update[0] = now

        def status(message, current, total):
            record['stage'] = message
            write(state_path, state)

        try:
            apply_events.clear()
            result = generator.generate(
                caption=job['caption'], lyrics=job['lyrics'], duration=job['requested_duration'],
                dest_wav=dest, device='cuda:0', on_progress=progress, on_status=status,
                lora_components=job['lora_components'], seed=job['seed'])
            data, rate = sf.read(dest, dtype='float32', always_2d=True)
            finite = bool(np.isfinite(data).all())
            if not finite or not len(data):
                raise ValueError('Empty or nonfinite output')
            sf.write(excerpt, data[:20*rate], rate, subtype='PCM_16')
            record.update(status='complete', result=result, elapsed=time.time()-record['started'],
                          audio_sha256=sha(dest), excerpt_sha256=sha(excerpt),
                          inspection=dict(seconds=len(data)/rate,
                                          rms=float(np.sqrt(np.mean(data.astype('float64')**2))),
                                          peak=float(np.abs(data).max()),
                                          clipped_fraction=float(np.mean(np.abs(data) >= .999)),
                                          short=len(data) < 20*rate, finite=finite),
                          apply_mode=generator._apply_mode('cuda:0'),
                          apply_events=list(apply_events))
            print(f'DONE {job["id"]} {record["elapsed"]:.1f}s; audio {len(data)/rate:.2f}s', flush=True)
        except Exception as exc:
            record.update(status='error', error=str(exc), traceback=traceback.format_exc(),
                          elapsed=time.time()-record['started'])
            print(f'ERROR {job["id"]}: {exc}', flush=True)
            if 'CUDA' in str(exc) or 'cuda' in str(exc):
                state['status'] = 'interrupted_cuda_error'
                write(state_path, state)
                update_page(state)
                raise
        write(state_path, state)
        update_page(state)
    state['status'] = ('complete' if all(r['status'] == 'complete' for r in state['records'].values())
                       and len(state['records']) == len(spec['jobs']) else 'complete_with_errors')
    state['current'] = None
    state['finished'] = time.time()
    write(state_path, state)
    update_page(state)
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
