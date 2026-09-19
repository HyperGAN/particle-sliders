"""Bounded capture and gradient prerequisites in separate GPU-1 processes."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
import signal
import time
from .core import read,write,immutable,sha,Store


def run(home,folder,observation):
    home=Path(home).resolve();folder=Path(folder).resolve()
    sources=[Path(__file__).with_name(n) for n in ('acoustic_audit_sequence.py','capture_acoustic_tail.py','audit_acoustic_gradient.py','acoustic_tail.py','ce_wave_gradient.py','merged_forward.py','audit_ce_gradient_v3.py')]
    protocol=dict(hypothesis='Direct CE gradients through a separate acoustic attention LoRA may improve waveform quality while retaining the fixed language-model plan.',
        phase='engineering prerequisites, no training candidate',physical_gpu=1,reference_observation=str(Path(observation).resolve()),
        budget=dict(new_engineering_clips=1,discarded_optimizer_updates=1,max_seconds=3600),
        source_hashes={str(p.resolve()):sha(p) for p in sources},production_changes=False)
    immutable(folder/'sequence-protocol.json',protocol);Store(home).event('acoustic_host_audits_started',protocol=str(folder/'sequence-protocol.json'))
    Store(home).status('acoustic_host_audits',research_complete=False,folder=str(folder))
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel);began=time.monotonic()
    for name,args in [('capture_acoustic_tail',['--observation',str(Path(observation).resolve())]),('audit_acoustic_gradient',[])]:
        process=subprocess.Popen([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.'+name,'--home',str(home),'--folder',str(folder),*args],start_new_session=True)
        try:code=process.wait(timeout=max(1.,3600-(time.monotonic()-began)))
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=180)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        if code:raise RuntimeError(f'Acoustic prerequisite {name} failed with exit {code}; original evidence retained')
    Store(home).event('acoustic_host_audits_finished',passed=True,new_engineering_clips=1,discarded_optimizer_updates=1)
    Store(home).status('acoustic_audits_passed_training_recipe_required',research_complete=False,folder=str(folder))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--observation',required=True);a=p.parse_args()
    run(a.home,a.folder,a.observation)
