#!/usr/bin/env python3
"""Two-GPU YuE2 critic search transferred from the Music 3 particle game.

The search transfers the formulation and critic architecture, never model
weights.  It screens all candidates, extends the strongest training locks,
renders finalists on held-out prompts, then builds an enjoyment-gated board.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = '/ml2/music/.cache/yue2-test-env/bin/python'
TRAIN_PROMPTS = ROOT / 'conceptmod/textsliders/data/prompts-yue2-metal-arm-b.yaml'
EVAL_PROMPTS = ROOT / 'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml'
MUSIC3_REFERENCE = ROOT / 'analysis/music3_critic_sweep_bal_20260917'

# MLP is the accepted YuE2 control. Mix is the Music 3 transformer finalist.
# The remaining cells ask whether its nonlocal attention can transfer without
# the fat H→T·W stem that can memorize four caption coordinates.
CANDIDATES = [
    dict(name='mlp', critic='mlp'),
    dict(name='music3_mix_t16_w128_l2', critic='mix', tokens=16, width=128, layers=2, heads=4, score_bound=8.),
    dict(name='bottleneck_t8_w48_l1', critic='bottleneck', tokens=8, width=48, layers=1, heads=4, score_bound=8.),
    dict(name='bottleneck_t16_w48_l1', critic='bottleneck', tokens=16, width=48, layers=1, heads=4, score_bound=8.),
    dict(name='hybrid_t8_w48_l1', critic='hybrid', tokens=8, width=48, layers=1, heads=4, score_bound=8.),
    dict(name='lowrank_r64_t8_w48_l1', critic='lowrank', tokens=8, width=48, layers=1, heads=4, rank=64, score_bound=8.),
    dict(name='bquery_t8_q4_w48_l1', critic='bquery', tokens=8, queries=4, width=48, layers=1, heads=4, score_bound=8.),
]


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def critic_args(candidate: dict) -> list[str]:
    args = ['--critic', candidate['critic']]
    for key in ('tokens','queries','width','layers','heads','rank','score_bound'):
        if key in candidate:
            args += ['--critic_' + key, str(candidate[key])]
    return args


def environment(gpu: int) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
        HF_HOME=os.environ.get('HF_HOME','/ml2/music/.cache/huggingface'),
        PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('TRANSFORMERS_CACHE', None)
    return env


def completed(run: Path) -> int:
    path = run / 'status.json'
    return int(json.loads(path.read_text()).get('completed', 0)) if path.exists() else 0


def training_score(run: Path) -> float:
    from scripts.music_architecture_scoreboard import training_rows, training_score
    return float(training_score(training_rows(run)).get('training_fitness', -1e9))


def train_command(candidate: dict, run: Path, steps: int) -> list[str]:
    return [PYTHON, '-u', str(ROOT/'conceptmod/textsliders/train_lora_yue2_arm_b.py'),
        '--recipe','particle_bridge','--name',f"metal-yue2-{candidate['name']}",
        '--prompts_file',str(TRAIN_PROMPTS),'--save_dir',str(run),
        '--steps',str(steps),'--until',str(steps),'--save_every','100',
        '--seed','7','--device','cuda:0'] + critic_args(candidate)


def eval_command(candidate: dict, run: Path, output: Path) -> list[str]:
    weights = run / f"metal-yue2-{candidate['name']}_last.safetensors"
    return [PYTHON, '-u', str(ROOT/'scripts/evaluate_yue2_arm_b.py'),
        '--recipe','particle_bridge','--weights',str(weights),
        '--prompts_file',str(EVAL_PROMPTS),'--output_dir',str(output),
        '--seeds','1709','2903','--max_tokens','500','--hidden_diagnostics']


def run_batch(root: Path, jobs: list[tuple[dict,int]], steps: int, phase: str):
    processes = []
    for candidate, gpu in jobs:
        run = root/'runs'/candidate['name']; run.mkdir(parents=True, exist_ok=True)
        if completed(run) >= steps:
            continue
        lock_path = ROOT.parent/f'.music-gpu-{gpu}.lock'
        lock = lock_path.open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
        log_path = root/'logs'/f"{candidate['name']}-{phase}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open('a')
        process = subprocess.Popen(train_command(candidate, run, steps), cwd=ROOT,
            env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT,
            pass_fds=(lock.fileno(),))
        processes.append((candidate,gpu,process,handle,lock))
    for candidate,gpu,process,handle,lock in processes:
        code = process.wait(); handle.close(); lock.close()
        if code:
            raise RuntimeError(f"{candidate['name']} on GPU {gpu} exited {code}")


def train_phase(root: Path, candidates: list[dict], steps: int, gpus: list[int], phase: str):
    pending = list(candidates)
    while pending:
        batch = [(pending.pop(0), gpu) for gpu in gpus if pending]
        run_batch(root, batch, steps, phase)
        provisional_board(root, candidates)


def render_finalists(root: Path, finalists: list[dict], gpus: list[int]):
    processes = []
    for index,candidate in enumerate(finalists):
        gpu = gpus[index % len(gpus)]
        output = root/'listens'/candidate['name']
        status = output/'status.json'
        if status.exists() and json.loads(status.read_text()).get('stage') == 'Training and matched rendering complete':
            continue
        lock = (ROOT.parent/f'.music-gpu-{gpu}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
        log_path=root/'logs'/f"{candidate['name']}-render.log"; handle=log_path.open('a')
        process=subprocess.Popen(eval_command(candidate,root/'runs'/candidate['name'],output),
            cwd=ROOT,env=environment(gpu),stdout=handle,stderr=subprocess.STDOUT,
            pass_fds=(lock.fileno(),))
        processes.append((candidate,gpu,process,handle,lock))
    for candidate,gpu,process,handle,lock in processes:
        code=process.wait();handle.close();lock.close()
        if code:raise RuntimeError(f"render {candidate['name']} on GPU {gpu} exited {code}")


def provisional_board(root: Path, candidates: list[dict]):
    try:
        from scripts.music_architecture_scoreboard import compile_board, markdown
        board=compile_board(root,MUSIC3_REFERENCE,root/'quality-cache','cpu',False)
        write(root/'scoreboard.json',board);(root/'scoreboard.md').write_text(markdown(board))
        write(root/'state.json',dict(updated=time.time(),candidates=candidates,winner=board['winner'],rows=board['rows']))
    except Exception as exc:
        # Never stall the dual-GPU screen on a board compile/import hiccup.
        write(root/'scoreboard-error.json',dict(updated=time.time(),error=str(exc)))
        print(f'provisional_board: {exc}', flush=True)


def final_board(root: Path):
    command=[PYTHON,'-u',str(ROOT/'scripts/music_architecture_scoreboard.py'),
        '--search_root',str(root),'--music3_root',str(MUSIC3_REFERENCE),
        '--cache',str(root/'quality-cache'),'--device','cuda:0','--measure_audio']
    # Quality models use one GPU after all dual-card training/rendering ends.
    subprocess.run(command,cwd=ROOT,env=environment(1),check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'analysis/yue2_critic_search_20260917')
    parser.add_argument('--screen_steps',type=int,default=600)
    parser.add_argument('--final_steps',type=int,default=1200)
    parser.add_argument('--top_k',type=int,default=2)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or any(g not in (0,1) for g in args.gpus):
        parser.error('Use distinct physical GPUs 0 and/or 1')
    if not 10<=args.screen_steps<=args.final_steps or not 1<=args.top_k<=len(CANDIDATES):
        parser.error('Invalid budgets')
    args.root=args.root.resolve();(args.root/'runs').mkdir(parents=True,exist_ok=True)
    (args.root/'logs').mkdir(exist_ok=True);(args.root/'listens').mkdir(exist_ok=True)
    write(args.root/'search.json',dict(schema=1,transfer='formulation_and_critic_not_weights',
        gpus=args.gpus,screen_steps=args.screen_steps,final_steps=args.final_steps,
        top_k=args.top_k,candidates=CANDIDATES,quality_gate='matched enjoyment/production delta >= -0.2'))
    train_phase(args.root,CANDIDATES,args.screen_steps,args.gpus,'screen')
    finalists=sorted(CANDIDATES,key=lambda c:training_score(args.root/'runs'/c['name']),reverse=True)[:args.top_k]
    train_phase(args.root,finalists,args.final_steps,args.gpus,'final')
    render_finalists(args.root,finalists,args.gpus)
    final_board(args.root)


if __name__=='__main__':main()
