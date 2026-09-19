"""Matched small-model audit of GAN and fixed distribution objectives.

Imports the pinned local ParticleGAN models, without modifying that checkout.
Explicitly varies only named arguments; live and EMA results are separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from .distribution import energy_distance, rbf_mmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('/tmp/opencode/ParticleGAN'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--objective', choices=['b_cap','r1r2','energy','mmd','learned_energy','learned_mmd'], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=7000)
    p.add_argument('--horizon', type=int, default=7000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.0006)
    p.add_argument('--schedule', choices=['delayed','constant'], default='delayed')
    p.add_argument('--prior', choices=['particles','gaussian'], default='particles')
    p.add_argument('--vicreg', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--spectral-limit', type=float, default=1.)
    p.add_argument('--energy-power', type=float, default=1.)
    p.add_argument('--eval-every', type=int, default=1000)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh output directory')
    args.out.mkdir(parents=True)
    sys.path.insert(0, str(args.reference))
    from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator, sample_100gaussians
    from lib.particle_prior import ParticlePrior
    from lib.vicreg_loss import VICRegLikeLoss
    from lib.grad_regularizers import GradRegularizer
    from lib.toy_metrics import per_mode_core_ratio
    from lib.gan_loss import GANLoss
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0')
    prior = ParticlePrior(20000, 4, learnable=args.prior=='particles').to(device)
    g = SimpleMLPGenerator().to(device)
    d = SimpleMLPDiscriminator().to(device)
    for module in list(g.modules()) + list(d.modules()):
        if isinstance(module,torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
    @torch.no_grad()
    def project_critic():
        for module in d.modules():
            if isinstance(module,torch.nn.Linear):
                norm = torch.linalg.matrix_norm(module.weight,ord=2)
                module.weight.div_((norm/args.spectral_limit).clamp_min(1.))
    def learned_objective(fake,real):
        # Keep the raw coordinates: a learned projection alone is not
        # characteristic. The scalar learned feature has the same Fourier
        # input as the reference and globally bounded linear operator norms.
        f = torch.cat([fake,d(fake)[:,None]],dim=-1)
        r = torch.cat([real,d(real)[:,None]],dim=-1)
        return energy_distance(f,r,power=args.energy_power) if args.objective=='learned_energy' else rbf_mmd(f,r)
    if args.objective.startswith('learned_'): project_critic()
    eg, ep = copy.deepcopy(g), copy.deepcopy(prior)
    eg.requires_grad_(False); ep.requires_grad_(False)
    og = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0., .999))
    od = torch.optim.Adam(d.parameters(), lr=1.5*args.lr, betas=(0., .999))
    op = torch.optim.Adam(prior.parameters(), lr=10*args.lr, betas=(0., .999)) if args.prior=='particles' else None
    optimizers = [(og, args.lr), (od, 1.5*args.lr)] + ([(op,10*args.lr)] if op else [])
    reg = GradRegularizer('b_cap' if args.objective=='b_cap' else 'a_r1r2',
                         coeff=1. if args.objective=='b_cap' else .02)
    vicreg = VICRegLikeLoss()
    gan = GANLoss(loss_type='logistic', mode='rp')
    axis = torch.arange(10, device=device) - 4.5
    centers = torch.cartesian_prod(axis, axis)
    eval_rng = torch.Generator(device=device).manual_seed(9001)
    eval_idx = torch.randint(0,20000,(20000,),device=device,generator=eval_rng)
    initial_prior = prior.z.detach().clone()
    start = time.monotonic()
    invocation = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    invocation['reference_commit'] = subprocess.check_output(['git','-C',str(args.reference),'rev-parse','HEAD'],text=True).strip()
    invocation['source_sha256'] = {str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'distribution.py':hashlib.sha256(Path(__file__).with_name('distribution.py').read_bytes()).hexdigest()}
    (args.out/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    log = (args.out/'progress.jsonl').open('w')
    @torch.no_grad()
    def evaluate(step, losses):
        result = dict(step=step,seconds=time.monotonic()-start,losses=losses)
        for name, model, cloud in [('live',g,prior),('ema',eg,ep)]:
            samples = model(cloud.z[eval_idx])
            distances = torch.cdist(samples,centers)
            nearest, assignment = distances.min(-1)
            counts = torch.bincount(assignment[nearest<=.09],minlength=100)
            metrics = dict(modes=int((counts>=10).sum()),hq=float((nearest<=.09).float().mean()),
                tail10=float((nearest>.3).float().mean()),rms=float(samples.square().mean().sqrt()))
            if step == args.steps:
                metrics['core'] = per_mode_core_ratio(samples,std=.03)
                torch.save(samples.cpu(),args.out/f'{name}_samples.pt')
            result[name] = metrics
        log.write(json.dumps(result,allow_nan=False)+'\n'); log.flush()
        print(json.dumps(result,allow_nan=False),flush=True)
    evaluate(0,{})
    for step in range(args.steps):
        phase = min(1.,max(0.,(step/max(args.horizon-1,1)-.6)/.4))
        rate = 1. if args.schedule=='constant' else .05+.95*.5*(1+math.cos(math.pi*phase))
        for optimizer, base in optimizers:
            optimizer.param_groups[0]['lr'] = rate*base
        values = {}
        if args.objective in ('b_cap','r1r2'):
            d.requires_grad_(True)
            real = sample_100gaussians(args.batch,device)
            with torch.no_grad():
                z,_ = prior.sample(args.batch)
                fake = g(z)
            penalty, _ = reg.penalty(d,real,fake,step)
            ld = gan.d_loss(d(real),d(fake)) + penalty
            od.zero_grad(set_to_none=True); ld.backward(); od.step()
            d.requires_grad_(False)
            values.update(d=float(ld.detach()),penalty=float(penalty.detach()))
        elif args.objective.startswith('learned_'):
            d.requires_grad_(True)
            real = sample_100gaussians(args.batch,device)
            with torch.no_grad():
                z,_ = prior.sample(args.batch)
                fake = g(z)
            ld = -learned_objective(fake,real)
            od.zero_grad(set_to_none=True); ld.backward(); od.step()
            project_critic()
            d.requires_grad_(False)
            values.update(d=float(ld.detach()))
        z,idx = prior.sample(args.batch)
        fake = g(z)
        real = sample_100gaussians(args.batch,device)
        if args.objective in ('b_cap','r1r2'):
            loss = gan.g_loss(d(fake),d(real))
        elif args.objective=='energy':
            loss = energy_distance(fake,real,power=args.energy_power)
        elif args.objective.startswith('learned_'):
            loss = learned_objective(fake,real)
        else:
            loss = rbf_mmd(fake,real)
        regularizer = vicreg(prior.z[torch.unique(idx)]) if op and args.vicreg else fake.new_zeros(())
        total = loss + regularizer
        if not torch.isfinite(total):
            raise FloatingPointError(f'Nonfinite objective at {step}')
        og.zero_grad(set_to_none=True)
        if op: op.zero_grad(set_to_none=True)
        total.backward()
        if not all(torch.isfinite(v.grad).all() for v in list(g.parameters())+list(prior.parameters()) if v.grad is not None):
            raise FloatingPointError(f'Nonfinite gradient at {step}')
        og.step()
        if op: op.step()
        with torch.no_grad():
            for a,b in zip(eg.parameters(),g.parameters()): a.lerp_(b,.005)
            ep.z.lerp_(prior.z,.005)
        values.update(g=float(loss.detach()),vicreg=float(regularizer.detach()),lr_scale=rate)
        if (step+1)%args.eval_every==0 or step+1==args.steps:
            evaluate(step+1,values)
    torch.save(dict(g=g.state_dict(),d=d.state_dict(),prior=prior.state_dict(),
        ema_g=eg.state_dict(),ema_prior=ep.state_dict(),g_optimizer=og.state_dict(),
        d_optimizer=od.state_dict(),prior_optimizer=None if op is None else op.state_dict(),
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=args.steps),args.out/'state.pt')
    (args.out/'completion.json').write_text(json.dumps(dict(status='budget_complete',
        moved_particles=int(((prior.z-initial_prior).norm(dim=-1)>0).sum()),seconds=time.monotonic()-start))+'\n')
    log.close()


if __name__=='__main__':
    main()
