#!/usr/bin/env python3
"""Durable two-GPU queue around the existing YuE2 particle campaign.

This schedules independent sliders. It does not implement a training update.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_identity(pid):
    """PID plus Linux start time, so a reused PID cannot block a queue."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start=stat[19]) if stat[0] != 'Z' else None
    except FileNotFoundError:
        return None


def campaign_argv(manifest, job):
    return [manifest['python'], '-u', str(ROOT/'scripts/train_yue2_arm_b_campaign.py'),
        '--recipe', 'particle_bridge', '--gpu', str(job['gpu']), '--seed', '7',
        '--steps', str(manifest['steps']), '--name', job['name'],
        '--save_dir', job['save_dir'], '--output_dir', job['output_dir'],
        '--prompts_file', job['train_prompts'], '--eval_prompts_file', job['eval_prompts'],
        '--include_canary', '--hidden_diagnostics']


def verify(manifest):
    if manifest['source_root'] != str(ROOT):
        raise ValueError('Queue belongs to a different frozen source directory')
    for name, expected in manifest['files'].items():
        if sha(name) != expected:
            raise ValueError(f'Frozen queue input changed: {name}')
    if manifest['steps'] != 1200 or len(manifest['jobs']) != 16:
        raise ValueError('Expected the approved 16-slider, 1200-update queue')
    for job in manifest['jobs']:
        if job['gpu'] not in (0, 1) or job['argv'] != campaign_argv(manifest, job):
            raise ValueError('Queue command or GPU assignment changed')


def prepare(args):
    import yaml
    from types import SimpleNamespace
    from conceptmod.textsliders.train_lora_yue2_arm_b import source_hashes
    from conceptmod.textsliders.yue2_arm_b import load_prompts
    from conceptmod.textsliders.yue2_backend import YuE2Backend
    from yue2.storage import resolve_model
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    sys.path.insert(0, str(args.registry.parent.parent))
    from app.rewriter import _artist_name_hit

    target = args.queue_dir/'manifest.json'
    if target.exists():
        verify(read(target)); print('Existing queue verified'); return
    registry = read(args.registry)
    active = {item['id']: item for item in registry['sliders']}
    order = ['female', 'metal'] + [key for key in active if key not in ('female', 'metal')]
    if len(active) != 16 or set(order) != set(active):
        raise ValueError('Expected the current 16-slider MiniMax catalog')
    path = resolve_model('m-a-p/YuE2-3B', local_files_only=True)
    tokenizer = YuE2TextTokenizer(path/'qwen.tiktoken')
    backend = SimpleNamespace(tokenizer=tokenizer)
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_root=str(ROOT), training_reference='2067705', python=sys.executable,
        steps=1200, seed=7, recipe='particle_bridge',
        learning_rates=dict(generator=.0006, discriminator=.0009, particles=.006),
        noise_decay_steps=8000, merge_to_trainer=False,
        registry_sha256=sha(args.registry), gpus=[0, 1],
        authorization='User requested all sliders to 1200 on both GPUs; preserve studio queue.',
        eval_rows_from_minimax=[2, 3], eval_seeds=[1709, 2903],
        eval_scales=[0, .5, 1], negative_canary=-1,
        min_free_mib=10000, output_root=str(output), gpu_lock_root=str(args.registry.parent.parent),
        wait_for={'0': [], '1': []}, jobs=[], files={})
    for pid in args.wait_gpu1_pid:
        identity = process_identity(pid)
        if identity: manifest['wait_for']['1'].append(identity)
    for index, key in enumerate(order):
        item = active[key]
        weights = Path(registry['root'])/item['components'][0]['weights']
        sidecar = read(weights.with_suffix('.json'))
        source_train = Path(sidecar['prompts_file'])
        source_eval = source_train.with_name(f'{key}-eval.yaml')
        job = dict(id=key, label=item['label_plus'], gpu=index % 2,
            name=f'{key}-yue2-particle-1200-s7',
            save_dir=str(args.models_root.resolve()/key), output_dir=str(output/key),
            minimax_weights=str(weights), minimax_sidecar_sha256=sha(weights.with_suffix('.json')),
            resume_from=0)
        if key == 'metal':
            job.update(name=args.metal_run.name, save_dir=str(args.metal_run.resolve()), resume_from=600)
            state = read(args.metal_run/'manifest.json')
            if state['settings']['sources'] != source_hashes():
                raise ValueError('Metal training source differs; use its identical frozen trainer')
            source_train = ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml'
            source_eval = ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml'
        audit = {}
        for split, source in [('train', source_train), ('eval', source_eval)]:
            original = yaml.safe_load(source.read_text())
            rows = original['rows']
            indices = list(range(len(rows))) if split == 'train' or key == 'metal' else [2, 3]
            # Preserve caption/lyric text; only remove legacy unused fields.
            selected = [{field: rows[i][field] for field in ('neutral', 'positive', 'lyrics')}
                        for i in indices]
            for row in selected:
                if _artist_name_hit('', row['neutral']+'\n'+row['positive'], row['lyrics']):
                    raise ValueError(f'Named reference in {key}/{split}; do not train it')
            document = dict(plus_label=job['label'], zero_label='Off', recommended_range=[0., 1.], rows=selected)
            destination = args.queue_dir/'prompts'/f'{key}-{split}.yaml'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100))
            checked, _ = load_prompts(destination)
            lengths = [len(YuE2Backend.prefix(backend, row[field], row['lyrics']))
                       for row in checked for field in ('neutral', 'positive')]
            if max(lengths) > 1024: raise ValueError(f'{key}/{split} exceeds trainer context')
            job[split+'_prompts'] = str(destination.resolve())
            audit[split] = dict(source=str(source), source_sha256=sha(source), source_rows=indices,
                rows=len(selected), max_tokens=max(lengths), names_checked=True)
            manifest['files'][str(destination.resolve())] = sha(destination)
        train_rows, train_meta = load_prompts(job['train_prompts'])
        eval_rows, _ = load_prompts(job['eval_prompts'])
        if {r['lyrics'] for r in train_rows} & {r['lyrics'] for r in eval_rows}:
            raise ValueError(f'Overlapping lyrics for {key}')
        if len(train_rows) != 4 or len(eval_rows) != 2:
            raise ValueError('Expected four train and two held-out rows')
        if key == 'metal' and (state['settings']['rows'] != train_rows or state['settings']['metadata'] != train_meta):
            raise ValueError('Metal prompts changed; cannot resume')
        job['prompt_audit'] = audit
        job['argv'] = campaign_argv(manifest, job)
        manifest['jobs'].append(job)
    for name, digest in source_hashes().items(): manifest['files'][str(ROOT/name)] = digest
    for name in ['scripts/queue_yue2_particle_catalog.py', 'scripts/train_yue2_arm_b_campaign.py',
                 'scripts/evaluate_yue2_arm_b.py', 'scripts/yue2_training_dashboard.py',
                 'scripts/assets/yue2-training-dashboard.html']:
        manifest['files'][str(ROOT/name)] = sha(ROOT/name)
    verify(manifest); write(target, manifest)
    write(output/'commands.json', manifest)
    for job in manifest['jobs']:
        write(args.queue_dir/'jobs'/f"{job['id']}.json", dict(status='queued', id=job['id'], gpu=job['gpu']))
        from scripts.evaluate_yue2_arm_b import page
        rows, meta = load_prompts(job['eval_prompts'])
        dest = Path(job['output_dir']); dest.mkdir(parents=True, exist_ok=True)
        page(dest, rows, [1709, 2903], 'particle_bridge', True, job['label'])
        write(dest/'training-metrics.json', dict(points=[], training=dict(status='queued', total=1200),
            updated_at=None, published_at=time.time()))
        write(dest/'status.json', dict(stage='Queued for 1200 updates'))
    publish(args.queue_dir, manifest)
    print(json.dumps(dict(learning_rates=manifest['learning_rates'], jobs=[
        {k:j[k] for k in ('id','gpu','resume_from','argv')} for j in manifest['jobs']]), indent=2))


def completed(manifest, job):
    training = read(Path(job['save_dir'])/'status.json', {})
    result = read(Path(job['output_dir'])/'status.json', {})
    return (training.get('completed') == manifest['steps'] and
            result.get('stage') == 'Training and held-out comparisons complete')


def worker(args, manifest):
    stop = False; child = None
    def interrupted(*_):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    worker_file = args.queue_dir/f'worker-{args.gpu}.json'
    def status(phase, **extra): write(worker_file, dict(gpu=args.gpu, status=phase, pid=os.getpid(), updated=time.time(), **extra))
    with (args.queue_dir/f'worker-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        waits = manifest['wait_for'][str(args.gpu)]
        while any(process_identity(p['pid']) == p for p in waits) and not stop:
            status('waiting_previous_renders'); time.sleep(2)
        for job in manifest['jobs']:
            if stop: break
            if job['gpu'] != args.gpu: continue
            record_file = args.queue_dir/'jobs'/f"{job['id']}.json"
            prior = read(record_file, {})
            if prior.get('status') == 'complete' and completed(manifest, job): continue
            if prior.get('status') == 'failed': continue  # Retain failures for inspection.
            verify(manifest)
            lease_path = Path(manifest['gpu_lock_root'])/f'.music-gpu-{args.gpu}.lock'
            with lease_path.open('a') as lease:
                while not stop:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB); break
                    except BlockingIOError:
                        status('waiting_gpu_lock', job=job['id']); time.sleep(2)
                while not stop:
                    free = int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                        '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
                    if free >= manifest['min_free_mib']: break
                    status('waiting_memory', job=job['id'], free_mib=free); time.sleep(5)
                if stop: break
                log_path = args.queue_dir/'logs'/f"{job['id']}.log"; log_path.parent.mkdir(exist_ok=True)
                status('running', job=job['id'])
                write(record_file, dict(status='running', gpu=args.gpu, id=job['id'], started=time.time(), argv=job['argv']))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
                    HF_HOME=os.environ.get('HF_HOME','/ml2/music/.cache/huggingface'), PYTHONPATH=str(ROOT),
                    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                env.pop('TRANSFORMERS_CACHE', None)
                with log_path.open('a') as log:
                    # An orphaned campaign retains the leases until it exits.
                    child = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, pass_fds=(lock.fileno(), lease.fileno()))
                    while child.poll() is None: time.sleep(1)
                phase = 'paused' if stop else 'complete' if child.returncode == 0 and completed(manifest, job) else 'failed'
                write(record_file, dict(status=phase, gpu=args.gpu, id=job['id'], finished=time.time(),
                    returncode=child.returncode, log=str(log_path)))
                child = None
        status('paused' if stop else 'finished')


def publish(queue_dir, manifest):
    jobs = []
    for job in manifest['jobs']:
        state = read(queue_dir/'jobs'/f"{job['id']}.json", dict(status='queued'))
        train = read(Path(job['save_dir'])/'status.json', {})
        progress = read(Path(job['save_dir'])/'progress.json', {})
        display = read(Path(job['output_dir'])/'status.json', {})
        jobs.append(dict(id=job['id'], label=job['label'], gpu=job['gpu'], status=state['status'],
            completed=train.get('completed', job['resume_from']), total=manifest['steps'],
            stage=display.get('stage','Queued'), metrics={k:progress.get(k) for k in
                ('g_adv','particle_vic','grad_norm','cos_pos','step_seconds')}))
    write(Path(manifest['output_root'])/'queue-state.json', dict(updated=time.time(), jobs=jobs,
        workers=[read(queue_dir/f'worker-{gpu}.json', dict(gpu=gpu,status='not_started')) for gpu in (0,1)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','worker','publish'])
    parser.add_argument('--queue_dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=[0,1])
    parser.add_argument('--registry', type=Path, default=Path('/ml2/music/app/sliders.json'))
    parser.add_argument('--models_root', type=Path)
    parser.add_argument('--output_root', type=Path)
    parser.add_argument('--metal_run', type=Path)
    parser.add_argument('--wait_gpu1_pid', type=int, nargs='*', default=[])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(); args.queue_dir=args.queue_dir.resolve(); args.queue_dir.mkdir(parents=True,exist_ok=True)
    if args.mode == 'prepare':
        if not all((args.models_root,args.output_root,args.metal_run)): parser.error('Preparation needs model/output roots and metal run')
        prepare(args); return
    manifest=read(args.queue_dir/'manifest.json'); verify(manifest)
    if args.mode == 'worker':
        if args.gpu is None: parser.error('Worker needs --gpu')
        worker(args, manifest)
    else:
        with (args.queue_dir/'publisher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                publish(args.queue_dir, manifest)
                if not args.watch: break
                time.sleep(2)


if __name__ == '__main__': main()
