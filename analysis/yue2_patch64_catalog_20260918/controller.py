#!/usr/bin/env python3
"""Dual-GPU catalog: every slider on the patch_p64 ParticleGAN formula."""
from __future__ import annotations
import json, os, signal, subprocess, time
from pathlib import Path

ROOT = Path('/ml2/music/sliders-conceptmod')
EXP = ROOT / 'analysis/yue2_patch64_catalog_20260918'
PYBIN = '/ml2/music/.cache/yue2-test-env/bin/python'
STEPS = 1600
WARMUP = 500
COS_ABORT = 0.5

def load_jobs():
    return json.loads((EXP/'experiment.json').read_text())['jobs']

def write_state(**kw):
    path = EXP/'state.json'
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kw); cur['updated'] = time.time()
    path.write_text(json.dumps(cur, indent=2)+'\n')

def argv_for(job):
    run = EXP/'runs'/job['id']
    run.mkdir(parents=True, exist_ok=True)
    return [
        PYBIN, '-u', str(ROOT/'conceptmod/textsliders/train_lora_yue2_arm_b.py'),
        '--recipe', 'particle_bridge',
        '--name', job['name'],
        '--prompts_file', job['train_prompts'],
        '--save_dir', str(run),
        '--steps', str(STEPS), '--until', str(STEPS),
        '--save_every', '100', '--seed', '7', '--device', 'cuda:0',
        '--sample_seeds', '128', '--history_tokens', '32', '--adv_batch', '8',
        '--critic', 'patch', '--critic_patch', '64',
        '--critic_width', '48', '--critic_layers', '1', '--critic_heads', '4',
    ], run

def collapsed(run: Path):
    p = run/'progress.json'
    if not p.exists():
        return False, ''
    j = json.loads(p.read_text())
    step = int(j.get('step') or 0)
    if step < WARMUP:
        return False, ''
    cos = float(j.get('cos_pos') or 0)
    if cos < COS_ABORT:
        return True, f'cos_pos={cos:.3f}<{COS_ABORT} @ {step}'
    return False, ''

def snapshot(job, run, abort=''):
    st = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    pr = json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
    return dict(id=job['id'], status=st.get('status'), completed=st.get('completed'),
                step=pr.get('step'), cos_pos=pr.get('cos_pos'), abort=abort or st.get('abort_reason'))

class _Adopted:
    """Popen-shaped handle for a trainer we did not spawn."""
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            return 0
        except PermissionError:
            return None
    def send_signal(self, sig):
        os.kill(self.pid, sig)
    def kill(self):
        os.kill(self.pid, 9)
    def wait(self, timeout=None):
        end = time.time() + (timeout or 1e9)
        while time.time() < end:
            if self.poll() is not None:
                return 0
            time.sleep(0.5)
        raise subprocess.TimeoutExpired(cmd='adopted', timeout=timeout)


def live_trainers():
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            cmd = [c.decode() for c in open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0') if c]
            if not any(c.endswith('train_lora_yue2_arm_b.py') for c in cmd):
                continue
            if '--save_dir' not in cmd:
                continue
            save = Path(cmd[cmd.index('--save_dir') + 1])
            if save.parent != EXP / 'runs':
                continue
            envd = {}
            for item in open(f'/proc/{pid}/environ', 'rb').read().split(b'\0'):
                if b'=' in item:
                    key, value = item.decode(errors='replace').split('=', 1)
                    envd[key] = value
            gpu = int(envd.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
            yield gpu, int(pid), save.name, save
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            continue


def main():
    jobs = load_jobs()
    by_id = {job['id']: job for job in jobs}
    results = []
    procs = {0: None, 1: None}
    adopted = set()
    for gpu, pid, name, save in live_trainers():
        job = by_id.get(name)
        if job is None or procs.get(gpu) is not None:
            continue
        log = open(EXP / 'logs' / f'{name}.log', 'a', buffering=1)
        procs[gpu] = dict(proc=_Adopted(pid), job=job, run=save, logf=log, gpu=gpu, adopted=True)
        adopted.add(name)
        print(f'adopted {name} pid={pid} gpu={gpu}', flush=True)
    retry, rest = [], []
    for job in jobs:
        if job['id'] in adopted:
            continue
        run = EXP / 'runs' / job['id']
        status_path = run / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        completed = int(status.get('completed') or 0)
        if status.get('status') == 'complete' and completed >= STEPS:
            results.append(dict(snapshot(job, run), reason='complete'))
            continue
        if status.get('status') == 'collapsed_abort' and completed < WARMUP:
            status.update(status='retry_queued', abort_reason=None,
                          note=f'early stop at {completed} ignored; resume after warmup {WARMUP}')
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            retry.append(job)
            continue
        rest.append(job)
    pending = retry + rest
    stop = False
    def on_sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    def launch(gpu, job):
        cmd, run = argv_for(job)
        log = EXP/'logs'/f"{job['id']}.log"
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HUB_OFFLINE='1', HF_HOME='/ml2/music/.cache/huggingface')
        lf = open(log, 'a', buffering=1)
        lf.write(f'\n===== launch gpu={gpu} {time.ctime()} =====\n')
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        return dict(proc=proc, job=job, run=run, logf=lf, gpu=gpu)

    def finish(gpu, reason):
        info = procs[gpu]
        if not info:
            return
        proc = info['proc']
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=30)
        run = info['run']
        if reason.startswith('collapse'):
            st_path = run/'status.json'
            st = json.loads(st_path.read_text()) if st_path.exists() else {}
            st.update(status='collapsed_abort', abort_reason=reason)
            st_path.write_text(json.dumps(st, indent=2)+'\n')
        results.append(dict(snapshot(info['job'], run, reason), gpu=gpu, reason=reason))
        try: info['logf'].close()
        except Exception: pass
        procs[gpu] = None

    write_state(phase='running', pending=[j['id'] for j in pending], results=results)
    while not stop:
        for gpu in (0, 1):
            if procs[gpu] is None and pending:
                job = pending.pop(0)
                # keep gpu assignment from job if we want sticky; fill whichever is free
                procs[gpu] = launch(gpu, job)
        if all(v is None for v in procs.values()) and not pending:
            break
        for gpu, info in list(procs.items()):
            if not info:
                continue
            code = info['proc'].poll()
            dead, why = collapsed(info['run'])
            if dead:
                finish(gpu, f'collapse:{why}'); continue
            if code is not None:
                finish(gpu, 'complete' if code == 0 else f'exit_{code}')
        write_state(phase='running',
                    pending=[j['id'] for j in pending],
                    active={str(g): (procs[g]['job']['id'] if procs[g] else None) for g in procs},
                    live={str(g): snapshot(procs[g]['job'], procs[g]['run']) if procs[g] else None for g in procs},
                    results=results)
        time.sleep(15)
    for gpu in (0, 1):
        if procs[gpu]:
            if stop:
                try: procs[gpu]['logf'].close()
                except Exception: pass
                print(f"detach gpu={gpu} {procs[gpu]['job']['id']}", flush=True)
                procs[gpu] = None
            else:
                finish(gpu, 'done')
    write_state(phase='stopped' if stop else 'done',
                pending=[j['id'] for j in pending], results=results,
                active={str(g): None for g in procs})

if __name__ == '__main__':
    main()
