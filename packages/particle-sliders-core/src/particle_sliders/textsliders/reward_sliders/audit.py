"""Numerical live-host audit of ordinary LoRA merge and exact Off restoration."""
from pathlib import Path
import sys
import torch

from .specs import WORKSPACE, sha


@torch.no_grad()
def audit_merged_adapter(pipe, device, components, reward_component):
    sys.path.insert(0,str(WORKSPACE))
    from app import generator
    from safetensors.torch import load_file
    tensors={c['weights']:load_file(c['weights'],device='cpu') for c in components}
    reward=tensors[reward_component['weights']]
    names=sorted(k[:-len('.lora_down.weight')] for k in reward if k.endswith('.lora_down.weight'))
    if len(names)!=144 or len(reward)!=432:
        raise ValueError('Expected 144 complete rank-8 projection triplets')
    network=generator._slider_network(pipe,device,reward_component,attach=False)
    generator._merge_sliders(pipe,device,components)
    state=generator._merge_state(device)
    projections=[]
    try:
        for name in names:
            host=network.hosts[name]
            base=state.pristine[host]
            total=torch.zeros_like(base,dtype=torch.float32)
            for comp in components:
                data=tensors[comp['weights']]
                a,b=data[name+'.lora_down.weight'],data[name+'.lora_up.weight']
                alpha=float(data[name+'.alpha'])
                if a.shape[0]!=8 or b.shape[1]!=8 or alpha!=8.:
                    raise ValueError('Actual checkpoint rank/alpha differs from the declared contract')
                total += (b.float()@a.float())*(comp['multiplier']*alpha/a.shape[0])
            expected=(base.float()+total).to(base.dtype)
            actual=host.weight.detach().cpu()
            error=float((actual.float()-expected.float()).abs().max())
            projections.append(dict(name=name,shape=list(actual.shape),max_error=error,
                                    effective_delta_l2=float(total.norm())))
            if not torch.equal(actual,expected):
                raise ValueError(f'Full-delta merge differs from the explicit checkpoint formula at {name}')
    finally:
        generator._merge_sliders(pipe,device,[])
    restored=all(torch.equal(module.weight.detach().cpu(),base) for module,base in state.pristine.items())
    if not restored:
        raise ValueError('Reward Off failed to restore exact base weights')
    return dict(passed=True,formula='W = cast_bf16(W_base.float32 + sum multiplier * alpha/rank * B.float32 @ A.float32)',
                checkpoint_sha256=sha(reward_component['weights']),rank=8,alpha=8,modules=144,tensors=432,
                exact_off_restoration=True,projections=projections)
