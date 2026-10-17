"""Bounded CPU qualification of TinyQwen through the selected shared PG09 core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--core',type=Path,required=True,
                        help='immutable checkout containing packages/particle-sliders-core')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--until',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.until<=6:
        parser.error('qualification is bounded to six updates')
    core_src=args.core.resolve()/'packages/particle-sliders-core/src'
    module=core_src/'particle_sliders/pg09.py'
    if not module.is_file():
        parser.error('selected checkout does not contain shared ConditionalPG09')
    digest=hashlib.sha256(module.read_bytes()).hexdigest()
    # Select the core before importing any native adapter or torch code. The
    # historical core embedded in this checkout does not expose PG09.
    sys.path[:0]=[str(core_src),str(ROOT)]
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from particle_sliders import pg09
    from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
    from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun
    if Path(pg09.__file__).resolve()!=module:
        raise RuntimeError('shared core import differs from selected checkout')
    torch.set_num_threads(1)
    torch.manual_seed(91)
    backend=TinyLLMBackend(dummy=True)
    run=PG09TinyRun(backend,[dict(neutral='a quiet song',positive='an intense metal song'),
                           dict(neutral='a calm melody',positive='an aggressive metal melody')],
                   total_steps=6,batch_size=4)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/'training.pt'
    if checkpoint.exists():
        run.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    while run.game.completed_steps<args.until:
        print(json.dumps(run.step()),flush=True)
        torch.save(run.state_dict(),checkpoint)
    if hashlib.sha256(module.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('selected shared core changed during qualification')
    run.export(args.output/'tiny-pg09.safetensors')
    (args.output/'receipt.json').write_text(json.dumps(dict(protocol=run.signature['protocol'],
        completed_steps=run.game.completed_steps,pg_version='0.9.0',
        game_class=type(run.game).__module__+'.'+type(run.game).__name__,
        core_module=str(module),core_sha256=digest,
        game_config=run.game.state_dict()['config'],
        resolved_recipe=run.signature['recipe'],checkpoint=str(checkpoint),dummy=True,
        cuda_initialized=torch.cuda.is_initialized(),quality_qualified=False),indent=2)+'\n')


if __name__=='__main__':
    main()
