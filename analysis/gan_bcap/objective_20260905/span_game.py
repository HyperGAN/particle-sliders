"""Conditional saved-span canary: cap vs persistent damping vs one metric.

The generator consists of directly movable hidden values, so this can falsify
an objective but cannot establish LoRA attainability or musical quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from conceptmod.textsliders.gan_v2.critic import SpanCritic,conditional_cap
from conceptmod.textsliders.lm_adv import rp_d_loss,rp_g_loss
from .conditional import ConditionalMetric,conditional_energy


def run(seed,arm,steps,device):
    torch.manual_seed(seed)
    cache=torch.load(Path(__file__).parents[1]/'critic_failure_probe_20260904.pt',weights_only=True)
    # Small slice of genuine saved LM states. Position masks are retained.
    real=cache[0]['real'][:,:,:64].to(device)
    context=cache[0]['fake'][:,:,:64].to(device).detach()
    mask=cache[0]['mask'].to(device)
    fake=torch.nn.Parameter(torch.zeros_like(real))
    learned=arm=='conditional_energy'
    if learned:
        d=ConditionalMetric(64,hidden=32).to(device)
        scale=float(real[mask].square().mean().sqrt())
    else:
        d=SpanCritic(64,width=32,layers=1,heads=4,conditioned=True,ordered=True,normalized_features=False).to(device)
        d.calibrate_input_scale(real,mask)
    og=torch.optim.Adam([fake],lr=.01,betas=(0.,.999))
    od=torch.optim.Adam(d.parameters(),lr=.00075,betas=(0.,.999))
    history=[];start=time.monotonic()
    for step in range(steps):
        d.requires_grad_(True);od.zero_grad(set_to_none=True)
        if learned:
            ld=-conditional_energy(d,fake.detach()[mask],real[mask],context[mask],scale=scale)
        else:
            penalty,stats=conditional_cap(d,real,fake.detach(),mask,context,
                coefficient=1. if arm in ('b_cap','b_cap_fm') else float(arm.split('_')[-1]),
                kappa=1. if arm in ('b_cap','b_cap_fm') else 0.)
            ld=rp_d_loss(d(real,mask,context),d(fake.detach(),mask,context))+penalty
        ld.backward();od.step()
        if learned:d.project()
        d.requires_grad_(False);og.zero_grad(set_to_none=True)
        if learned:
            lg=conditional_energy(d,fake[mask],real[mask],context[mask],scale=scale)
        else:
            with torch.no_grad():rl=d(real,mask,context)
            lg=rp_g_loss(d(fake,mask,context),rl)
            if arm=='b_cap_fm':
                with torch.no_grad():rf=d.features(real,mask,context).mean(0)
                lg=lg+F.mse_loss(d.features(fake,mask,context).mean(0),rf)
        lg.backward()
        if not torch.isfinite(fake.grad).all():raise FloatingPointError('Nonfinite span gradient')
        og.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            with torch.no_grad():
                error=(fake-real)[mask]
                record=dict(step=step+1,nrmse=float(error.norm()/real[mask].norm()),
                    magnitude=float(fake[mask].norm()/real[mask].norm()),
                    d_loss=float(ld.detach()),g_loss=float(lg.detach()))
                history.append(record)
    return dict(seed=seed,arm=arm,seconds=time.monotonic()-start,history=history,final=history[-1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,23,101])
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('span-results.json'))
    args=p.parse_args();torch.set_num_threads(1)
    runs=[]
    for seed in args.seeds:
        for arm in ['b_cap_fm','b_cap','r1r2_0.02','r1r2_0.1','r1r2_1.0','conditional_energy']:
            result=run(seed,arm,args.steps,torch.device(args.device));runs.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
            args.out.write_text(json.dumps(dict(runs=runs,steps=args.steps),indent=2)+'\n')


if __name__=='__main__':main()
