"""Bounded CPU milestone for the actual TinyQwen adapter port, no Hub or GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch  # noqa: E402
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend  # noqa: E402
from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--until',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.until<=6:
        parser.error('qualification is bounded to six updates')
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
    run.export(args.output/'tiny-pg09.safetensors')
    (args.output/'receipt.json').write_text(json.dumps(dict(protocol=run.signature['protocol'],
        completed_steps=run.game.completed_steps,pg_version='0.9.0',
        resolved_recipe=run.signature['recipe'],checkpoint=str(checkpoint),dummy=True,
        cuda_initialized=torch.cuda.is_initialized(),quality_qualified=False),indent=2)+'\n')


if __name__=='__main__':
    main()
