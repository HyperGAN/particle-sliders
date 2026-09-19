"""Render only the explicitly requested small checkpoint comparison."""
import argparse
from pathlib import Path
import torch

from ..reward_search.renderer import SearchRenderer
from ..reward_sliders.evaluate import component
from ..reward_sliders.reward import CEReward
from ..reward_sliders.specs import RewardSpec,read_json,write_json,sha
from ..reward_sliders.experiment import verify


def render(run,stage,gpu):
    run=Path(run);jobs=read_json(run/'jobs'/f'{stage}-gpu{gpu}.json')
    if not jobs:return
    m=read_json(run/'manifest.json');verify(m);renderer=SearchRenderer(run/'renders'/stage,m,gpu)
    scorer=CEReward(RewardSpec(**m['reward_spec']))
    try:
        for job in jobs:
            treatment=job['arm'];extra=None
            if treatment.get('checkpoint'):
                assert sha(treatment['checkpoint'])==treatment['checkpoint_sha256']
                extra=component(treatment['checkpoint'],treatment['coefficient'])
            row=renderer.observe(job['family'],job['seed'],treatment,scorer,extra=extra)
            if row['status']!='complete':raise RuntimeError('Failed audio retained without reroll: '+row['id'])
    finally:
        renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
        exact=all(torch.equal(module.weight.detach().cpu(),base)
                  for module,base in renderer.host._merge_state(renderer.device).pristine.items())
        write_json(run/'audit'/f'{stage}-gpu{gpu}-off.json',dict(exact=exact))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--stage',required=True);args=p.parse_args();render(args.run_dir,args.stage,args.gpu)
