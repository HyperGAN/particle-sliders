"""Resource controller for frozen convergence fixtures and comparison rules.

The user reassigned music work to GPU 0. The original setup, scoring, 8-to-16
prompt expansion and decision functions are reused with a capacity queue.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.gan_bcap import convergence_audio as study
GPU = os.environ.get('MUSIC_GAN_GPU', '0')
if GPU not in ('0', '1'):
    raise ValueError('MUSIC_GAN_GPU must identify physical GPU 0 or 1')


def wait_for_capacity():
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 28672:
            return free
        study.write(study.WORK / 'status.json', dict(status='waiting',
            phase=f'Waiting for GPU {GPU} render capacity', free_mib=free,
            required_free_mib=28672, catalog='paused',
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Convergence audio queued on GPU {GPU}: {free} MiB free', flush=True)
        time.sleep(10)


def render(arguments, log):
    wait_for_capacity()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('a') as handle:
        subprocess.run(arguments, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    manifest = study.setup()
    study.write(study.WORK / 'resource-scheduler.json', dict(version=2, gpu=GPU,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_study_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
        change='User GPU reassignment; wait for capacity instead of requiring an empty GPU.',
        required_free_mib=28672, other_processes_terminated=0,
        mathematical_protocol_unchanged=True, prompt_ranges=[[0, 8], [8, 16]],
        reused_functions=['setup', 'score', 'evaluate'],
        partial_render_archive=str(study.WORK / 'gpu-reassignment.json')))
    with (study.WORK / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = ROOT / 'analysis/gan_bcap/convergence_20260905/status.json'
        while json.loads(pending.read_text()).get('status') == 'running':
            study.write(study.WORK / 'status.json', dict(status='waiting',
                phase='Waiting for unilateral probes to finish'))
            time.sleep(10)
        with ThreadPoolExecutor(max_workers=1) as cpu:
            for start, end in [(0, 8), (8, 16)]:
                futures = []
                for i in range(start, end):
                    out = ROOT / 'eval/listen' / f'gan-convergence-prompt-{i:02d}-20260905'
                    weights = [r['path'] for r in manifest['checkpoints']]
                    study.write(study.WORK / 'status.json', dict(status='running',
                        phase=f'Rendering prompt {i+1}/{end} on GPU {GPU}',
                        updated_utc=datetime.now(timezone.utc).isoformat()))
                    render([sys.executable, '-u', str(ROOT / 'analysis/gan_bcap/render_steps.py'),
                        '--weights', *weights, '--out', str(out), '--prompts', manifest['fixtures'][i]['path'],
                        '--seeds', *map(str, study.SEEDS), '--duration', '20'], study.WORK / f'render-{i:02d}.log')
                    futures.append(cpu.submit(study.score, out, study.WORK / f'score-{i:02d}.json'))
                study.write(study.WORK / 'status.json', dict(status='running',
                    phase=f'Finishing CPU measurements for {end} prompts'))
                for future in futures:
                    future.result()
                result = study.evaluate(manifest, end)
                if not any(r['decision'] == 'evaluate_more' for r in result['comparisons']):
                    break
        study.write(study.WORK / 'status.json', dict(status='complete',
            phase='Prospective paired comparison complete', result=str(study.WORK / f'comparison-{end:02d}.json'),
            catalog='paused', precision_exhausted=any(r['decision'] == 'evaluate_more' for r in result['comparisons'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        study.write(study.WORK / 'status.json', dict(status='failed', error=str(error), catalog='paused'))
        raise
