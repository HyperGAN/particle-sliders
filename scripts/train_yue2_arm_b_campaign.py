#!/usr/bin/env python3
"""Train the unipolar YuE2 GAN recipe, then render held-out comparisons."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.train_lora_yue2_fresh import write_json
from conceptmod.textsliders.yue2_arm_b import load_prompts
from scripts.evaluate_yue2_arm_b import page
from scripts.yue2_training_dashboard import publish_metrics


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',choices=['unipolar_gan','gan_plus_neu','particle_bridge'],default='unipolar_gan')
    p.add_argument('--propose_only_c9_g4x',action='store_true')
    p.add_argument('--propose_only_lr_scale',type=float)
    p.add_argument('--include_canary',action='store_true')
    p.add_argument('--hidden_diagnostics',action='store_true')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--save_dir',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--name',default='metal-yue2-arm-b')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--gpu',default='1')
    p.add_argument('--prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml')
    p.add_argument('--eval_prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml')
    a=p.parse_args(argv)
    if a.recipe=='particle_bridge' and (a.propose_only_c9_g4x or a.propose_only_lr_scale is not None):
        p.error('Particle bridge pins the reference constant learning rates')
    if a.propose_only_c9_g4x and a.recipe!='unipolar_gan':p.error('c9_g4x requires --recipe unipolar_gan')
    if a.propose_only_lr_scale is not None and (not 0<a.propose_only_lr_scale<=1 or a.propose_only_c9_g4x):
        p.error('Native LR scale must be in (0, 1] and cannot combine with c9_g4x')
    a.save_dir=a.save_dir.resolve();a.output_dir=a.output_dir.resolve()
    a.save_dir.mkdir(parents=True,exist_ok=True);a.output_dir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HF_HOME=os.getenv('HF_HOME','/ml2/music/.cache/huggingface'),
        HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('TRANSFORMERS_CACHE',None)
    rows,meta=load_prompts(a.eval_prompts_file);page(a.output_dir,rows,[1709,2903],a.recipe,a.include_canary,meta.get('plus_label','Metal'))
    train_rows,_=load_prompts(a.prompts_file)
    if {r['lyrics'] for r in train_rows}&{r['lyrics'] for r in rows}:raise ValueError('Held-out lyrics overlap training')
    stop=False
    def interrupted(*_):
        nonlocal stop
        stop=True
    def status(stage,**extra):
        write_json(a.output_dir/'status.json',dict(stage=stage,pid=os.getpid(),**extra))
        try:publish_metrics(a.save_dir,a.output_dir)
        except (OSError,ValueError) as exc:
            print(f'Chart metrics unavailable: {exc}',file=sys.stderr,flush=True)
    def command(cmd,label):
        if stop:raise InterruptedError('Campaign stopped')
        with (a.save_dir/f'{label}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            sent_stop=False
            while child.poll() is None:
                if stop and not sent_stop:child.terminate();sent_stop=True
                state=a.save_dir/'status.json'
                progress=json.loads(state.read_text()) if state.exists() else {}
                status('Stopping after the current update' if stop else
                    f'Training {progress.get("completed",0)}/{a.steps}' if label=='training' else 'Rendering held-out comparisons',
                    child_pid=child.pid,training=progress)
                time.sleep(1)
        if stop:raise InterruptedError('Campaign stopped')
        if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}; see {a.save_dir}/{label}.log')
    with (a.save_dir/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            command([sys.executable,'-u',str(ROOT/'conceptmod/textsliders/train_lora_yue2_arm_b.py'),
                '--recipe',a.recipe,'--save_dir',str(a.save_dir),'--name',a.name,'--steps',str(a.steps),
                '--seed',str(a.seed),'--device','cuda:0','--prompts_file',str(a.prompts_file)]
                +(['--propose_only_c9_g4x'] if a.propose_only_c9_g4x else [])
                +(['--propose_only_lr_scale',str(a.propose_only_lr_scale)] if a.propose_only_lr_scale is not None else []),'training')
            command([sys.executable,'-u',str(ROOT/'scripts/evaluate_yue2_arm_b.py'),
                '--recipe',a.recipe,
                '--weights',str(a.save_dir/f'{a.name}_last.safetensors'),'--prompts_file',str(a.eval_prompts_file),
                '--output_dir',str(a.output_dir)]+(['--include_canary'] if a.include_canary else [])
                +(['--hidden_diagnostics'] if a.hidden_diagnostics else []),'evaluation')
            status('Training and held-out comparisons complete')
        except InterruptedError as exc:status('Stopped',message=str(exc))
        except BaseException as exc:
            status('Failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)

if __name__=='__main__':main()
