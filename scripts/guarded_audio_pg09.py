"""Root-owned, single-card guardian for one bounded three-child audio smoke.

The fixed source-bound supervisor and its three sequential children share one
owned process group. No continuation, renderer, queue, retry or package writes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import guarded_pg09_canary as resources

ROOT=Path('/mnt/nvme/sliders/qualification')
EXECUTABLE='/ml2/sliders/.venv-yue2-retrain/bin/python'
SUPERVISOR='/mnt/nvme/sliders/audio-adapters-pg09-20261017/scripts/run_audio_pg09_smoke.py'
WORKER='/mnt/nvme/sliders/audio-adapters-pg09-20261017/scripts/qualify_audio_pg09_live.py'
UUID='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
RESIDENTS={3196}
LEASES=[str(ROOT/'admission-gpu-1.lock'),'/mnt/nvme/sliders/locks/gpu-1.lock']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,data):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def argument(command,flag):
    if command.count(flag)!=1:
        raise ValueError('exactly one bound argument required: '+flag)
    return command[command.index(flag)+1]


def validate(card):
    if (card.get('schema')!='root-audio-smoke-admission-v1' or
            card.get('status')!='REVIEWED_BOUNDED_AUDIO_SMOKE' or
            card.get('physical_gpu')!=1 or card.get('uuid')!=UUID or
            card.get('lease_paths')!=LEASES or card.get('family')!='yue2' or
            card.get('compute_seconds')!=720 or card.get('shutdown_grace_seconds')!=180 or
            card.get('maximum_wall_seconds')!=900):
        raise ValueError('audio smoke scope/card differs')
    sources=card['source_pins']
    required=[str(Path(__file__).resolve()),str(Path(resources.__file__).resolve()),SUPERVISOR,WORKER,
        str(ROOT/'ops-sana-kernel-fixed/kernel_gpu_guard.py'),str(ROOT/'ops-music3/music3_resources.py')]
    if any(p not in sources for p in required):
        raise ValueError('all executed guardian/worker/resource sources must be bound')
    for path,expected in sources.items():
        if sha(path)!=expected:
            raise ValueError('reviewed audio source changed: '+path)
    for path,expected in card['input_pins'].items():
        if sha(path)!=expected:
            raise ValueError('reviewed audio input changed: '+path)
    command=card['command']
    if command[:2]!=[EXECUTABLE,SUPERVISOR] or '--fixture' in command or '--resume' in command or '--until' in command:
        raise ValueError('guardian requires fixed native serial supervisor')
    fixed={'--family':'yue2','--device':'cuda:0','--max-seq-len':'512',
           '--pg-source-profile':'alz-continuous-v1','--penalty-input-domain':'continuous',
           '--total-steps':'20','--preset':'bispf_optimized','--seed':'7','--minimum-free-gib':'16',
           '--admission-id':card['admission_id'],'--out':card['output_directory']}
    for flag,value in fixed.items():
        if argument(command,flag)!=value:
            raise ValueError('fixed native smoke argument differs: '+flag)
    output=Path(card['output_directory'])
    if output.parent!=ROOT/'runs/audio-pg09-reviewed-20261017' or output.name!='yue2-smoke':
        raise ValueError('audio output must be the new isolated smoke directory')
    env=card['environment']
    if (env.get('CUDA_VISIBLE_DEVICES')!='1' or env.get('HF_HUB_OFFLINE')!='1' or
            env.get('TRANSFORMERS_OFFLINE')!='1' or env.get('OMP_NUM_THREADS')!='1' or
            env.get('MKL_NUM_THREADS')!='1' or env.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or
            any('site-packages' in p for p in env.get('PYTHONPATH','').split(os.pathsep))):
        raise ValueError('native audio environment/source isolation differs')
    return output


def owned_gpu_processes(pids,group):
    owned=[]
    for pid in pids-RESIDENTS:
        try:
            pgid=os.getpgid(pid)
        except ProcessLookupError:
            continue
        if pgid!=group:
            raise RuntimeError('unowned GPU process observed: '+str(pid))
        owned.append(dict(pid=pid,pgid=pgid))
    if len(owned)>1:
        raise RuntimeError('more than one concurrent CUDA child in bounded smoke')
    return owned


def guardian(path):
    card=json.loads(path.read_text())
    output=validate(card)
    if output.exists():
        raise FileExistsError('audio smoke output exists; no automatic retry')
    admission=output.parent/'yue2-smoke-root-admission'
    if admission.exists():
        raise FileExistsError('audio admission outcome exists; no automatic retry')
    if (not os.path.ismount('/mnt/nvme') or subprocess.check_output(
            ['findmnt','-no','SOURCE','--target','/mnt/nvme'],text=True).strip()!='/dev/nvme0n1p2'):
        raise RuntimeError('expected NVMe missing')
    hold=json.loads((ROOT/'MSE-DISPATCH-HOLD.json').read_text())
    if hold.get('mse_only_training_allowed') is not False or hold.get('automatic_mse_restart_or_dispatch_allowed') is not False:
        raise RuntimeError('historical MSE hold changed')
    leases=[]
    for name in LEASES:
        lease=Path(name).open('a')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        leases.append(lease)
    resources.GPU='1'
    resources.UUID=UUID
    resources.RESIDENTS=RESIDENTS
    initial=resources.status()
    if initial['temperature_c']>=76 or initial['gpu_free_mib']<16384 or resources.processes()-RESIDENTS:
        raise RuntimeError('GPU1 fresh memory/cooling/resident admission failed')
    start=time.time()
    resources.kernel_check(start)
    admission.mkdir(parents=True)
    receipt=dict(schema=card['schema'],card_sha256=sha(path),admission_id=card['admission_id'],
        command=card['command'],uuid=UUID,physical_gpu=1,lease_paths=LEASES,initial=initial,
        status='admitted',seen_cuda_children=[],maximum_wall_seconds=900,quality_qualified=False)
    process=None
    try:
        with (admission/'supervisor.log').open('x') as log:
            process=subprocess.Popen(card['command'],env=dict(os.environ,**card['environment']),
                cwd='/mnt/nvme/sliders/audio-adapters-pg09-20261017',stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            receipt.update(status='running',supervisor_pid=process.pid,owned_pgid=process.pid)
            write(admission/'receipt.json',receipt)
            while process.poll() is None:
                if time.time()-start>=720:
                    raise RuntimeError('720-second compute deadline reserves180s for safe shutdown')
                current=resources.status()
                children=owned_gpu_processes(resources.processes(),process.pid)
                for child in children:
                    if child not in receipt['seen_cuda_children']:
                        receipt['seen_cuda_children'].append(child)
                resources.kernel_check(start)
                with (admission/'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(elapsed_seconds=time.time()-start,children=children,**current))+'\n')
                time.sleep(2)
            if process.returncode:
                raise RuntimeError('audio supervisor failed; no retry')
            resources.kernel_check(start)
            proof=json.loads((output/'comparison.json').read_text())
            if proof.get('status')!='exact_smoke_passed' or not proof.get('exact_full_state_metrics_probes_export'):
                raise RuntimeError('exact three-child smoke proof missing')
            for stage in ('full3','first1','resumed3'):
                result=json.loads((output/stage/'progress.json').read_text())
                if (result.get('status')!='bounded_training_software_checks_passed' or
                        not result.get('actual_bank_weights_changed') or
                        not result.get('frozen_base_unchanged') or
                        not result.get('export_replay_exact') or not result.get('zero_replay_exact')):
                    raise RuntimeError('native stage learning/reload/base proof incomplete')
            receipt.update(status='completed',proof=proof)
    except BaseException as error:
        if process is not None:
            # The owned group may outlive a failed supervisor; never signal residents.
            try:
                os.killpg(process.pid,signal.SIGINT)
            except ProcessLookupError:
                pass
            deadline=time.time()+180
            while time.time()<deadline:
                try:
                    os.killpg(process.pid,0)
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                os.killpg(process.pid,signal.SIGKILL)
            if process.poll() is None:
                process.wait(timeout=10)
        receipt.update(status='failed_no_retry',error=repr(error))
        raise
    finally:
        receipt.update(elapsed_seconds=time.time()-start,exit_code=process.returncode if process else None)
        write(admission/'receipt.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',type=Path,required=True)
    guardian(parser.parse_args().card)
