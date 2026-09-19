"""Resource-only wrapper for the locked v2 experiment protocol.

Permit coexistence with small jobs on the assigned GPU when at least 32 GiB
remains free. No other process is stopped. Experiment arms, data, optimizer
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


def command(arguments,log,*,gpu):
    if gpu:
        start=time.monotonic()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=32*1024:break
            if time.monotonic()-start>1800:raise RuntimeError('GPU 1 has less than 32 GiB available after waiting; no process was stopped')
            study.write(study.WORK/'gpu-wait.json',dict(status='waiting',free_mib=free,required_mib=32*1024,
                updated_utc=datetime.now(timezone.utc).isoformat()))
            print(f'Waiting for GPU 1 capacity: {free} MiB free, need 32768 MiB',flush=True)
            time.sleep(10)
        study.write(study.WORK/'gpu-wait.json',dict(status='available_at_launch',free_mib=free,required_mib=32*1024))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if gpu else '',HF_HUB_OFFLINE='1',
        HF_HOME='/ml2/music/.cache/huggingface',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with Path(log).open('a') as handle:
        subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    study.write(study.WORK/'scheduler-amendment.json',dict(version=1,source_sha256=sha(__file__),
        locked_harness_sha256=sha(study.__file__),
        change='GPU preflight checks available capacity instead of requiring exclusive occupancy; mathematical and evaluation protocol unchanged.',
        required_free_mib=32768,maximum_wait_seconds=1800,no_processes_terminated=True))
    study.command=command
    study.main()
