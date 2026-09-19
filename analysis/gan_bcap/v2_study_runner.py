"""Resource-only wrapper for the locked v2 experiment protocol.

Permit coexistence with other jobs on the assigned GPU when at least 32 GiB
remains free for training, or 28 GiB for the measured 24 GiB renderer.
No other process is stopped. Experiment arms, data, optimizer
settings and scoring remain those in the unchanged, declared study harness.
"""
from __future__ import annotations

from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.gan_bcap import v2_study as study
from conceptmod.textsliders.gan_v2.data import sha

GPU=os.environ.get('MUSIC_GAN_GPU','0')
if GPU not in ('0','1'):raise ValueError('MUSIC_GAN_GPU must identify physical GPU 0 or 1')

def command(arguments,log,*,gpu):
    if gpu:
        required=(28 if str(arguments[0])=='analysis/gan_bcap/render_v2.py' else 32)*1024
        start=time.monotonic()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=required:break
            if time.monotonic()-start>1800:raise RuntimeError(f'GPU {GPU} has less than {required} MiB available after waiting; no other process was stopped')
            study.write(study.WORK/'gpu-wait.json',dict(status='waiting',gpu=GPU,free_mib=free,required_mib=required,
                updated_utc=datetime.now(timezone.utc).isoformat()))
            print(f'Waiting for GPU {GPU} capacity: {free} MiB free, need {required} MiB',flush=True)
            time.sleep(10)
        study.write(study.WORK/'gpu-wait.json',dict(status='available_at_launch',gpu=GPU,free_mib=free,required_mib=required))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU if gpu else '',HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with Path(log).open('a') as handle:
        subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    study.write(study.WORK/'scheduler-amendment.json',dict(version=3,source_sha256=sha(__file__),gpu=GPU,
        locked_harness_sha256=sha(study.__file__),
        change='GPU preflight checks available capacity instead of requiring exclusive occupancy; mathematical and evaluation protocol unchanged.',
        required_free_mib=dict(training=32768,rendering=28672),
        observed_renderer_used_mib=24076,maximum_wait_seconds=1800,other_processes_terminated=0,
        previous_amendment=str(study.WORK/'scheduler-amendment-initial.json'),
        reassignment=str(study.WORK/'gpu-reassignment.json')))
    study.command=command
    study.main()
