"""Bounded CPU qualification through the installed shared PG09 package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--core',type=Path,
                        help='retired checkout injection; install the pinned package instead')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--until',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.until<=6:
        parser.error('qualification is bounded to six updates')
    if args.core is not None:
        parser.error('--core checkout injection is retired; install the pinned shared package')
    sys.path.insert(0,str(ROOT))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from particle_sliders import pg09
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun
    from importlib.metadata import distribution
    installed=distribution('particle-sliders-core')
    module=Path(pg09.__file__).resolve()
    if (installed.version!='0.6.0' or module!=Path(installed.locate_file('particle_sliders/pg09.py')).resolve()):
        raise RuntimeError('shared core must come from the installed pinned wheel, not a source checkout')
    from conceptmod.textsliders.text_package import require_text_package
    package_identity=require_text_package()
    digest=hashlib.sha256(module.read_bytes()).hexdigest()
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
        completed_steps=run.game.completed_steps,pg_version=pg09.PG_VERSION,core_version=installed.version,
        game_class=type(run.game).__module__+'.'+type(run.game).__name__,
        core_module=str(module),core_sha256=digest,
        package_identity=package_identity,
        game_config=run.game.state_dict()['config'],
        resolved_recipe=run.signature['recipe'],checkpoint=str(checkpoint),dummy=True,
        cuda_initialized=torch.cuda.is_initialized(),quality_qualified=False),indent=2)+'\n')


if __name__=='__main__':
    main()
