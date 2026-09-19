"""Run bounded, restartable convergence diagnostics on the user's GPU 1."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'analysis/gan_bcap/convergence_20260905'


def save(state):
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    p = WORK/'status.json'
    t = p.with_suffix('.json.tmp')
    t.write_text(json.dumps(state, indent=2)+'\n')
    t.replace(p)


def main():
    WORK.mkdir(exist_ok=True)
    with (WORK/'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = ROOT/'models/gan-bcap-repair'
        original = model/'smoke-steps600-s7-20260904'
        bounded = model/'smoke-stepcap2-only-1350-s7-20260905'
        jobs = [
            ('original600', original/'smoke-steps600-s7-20260904_state.pt', .25),
            ('bounded1350', bounded/'smoke-stepcap2-only-1350-s7-20260905_state.pt', .25),
            ('early300', original/'smoke-steps600-s7-20260904_step300_state.pt', .25),
            ('failed900', model/'smoke-steps900-s7-20260904/smoke-steps900-s7-20260904_state.pt', .25),
            ('original600-lr1', original/'smoke-steps600-s7-20260904_state.pt', 1.),
            ('bounded1350-lr1', bounded/'smoke-stepcap2-only-1350-s7-20260905_state.pt', 1.),
        ]
        state = dict(status='running', purpose='Convergence criteria research; slider catalog remains paused', jobs={})
        for name, source, rate in jobs:
            out = WORK/name
            result = out/'result.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':
                state['jobs'][name] = dict(status='complete', result=str(result));save(state);continue
            if (out/'probe_train.jsonl').exists():
                # Preserve interrupted diagnostic artifacts; use a fresh attempt.
                attempt = 1
                while (WORK/f'{name}-attempt{attempt}').exists():
                    attempt += 1
                out = WORK/f'{name}-attempt{attempt}'
                result = out/'result.json'
            state['phase'] = name
            state['jobs'][name] = dict(status='running', directory=str(out))
            save(state)
            used = int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if used > 512:
                raise RuntimeError(f'GPU 1 is unexpectedly occupied: {used} MiB')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', HF_HOME='/ml2/music/.cache/huggingface',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (WORK/f'{out.name}.log').open('a') as log:
                command = [sys.executable,'-u',str(ROOT/'analysis/gan_bcap/convergence_probe.py'),
                           '--state',str(source),'--out',str(out),'--g-lr-scale',str(rate)]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid;save(state)
                while child.poll() is None:
                    save(state);time.sleep(10)
            state.pop('child_pid', None)
            if child.returncode:
                state['status']='failed';state['jobs'][name]['status']='failed';save(state)
                raise RuntimeError(f'{name} failed: see {log.name}')
            state['jobs'][name] = dict(status='complete', result=str(result));save(state)
        state['status']='complete';state['phase']='All unilateral probes complete';save(state)


if __name__ == '__main__':
    main()
