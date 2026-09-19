"""One conditional minimax energy objective with a constrained metric.

The raw-coordinate skip makes the learned map injective. Exact spectral
projection constrains the learned metric, rather than penalizing a different
scalar classifier. This is a research candidate, not a production default.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class ConditionalMetric(nn.Module):
    def __init__(self, width, hidden=128):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(2*width,hidden),nn.LeakyReLU(.2),
            nn.Linear(hidden,hidden),nn.LeakyReLU(.2),nn.Linear(hidden,1))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.project()

    @torch.no_grad()
    def project(self):
        norms = []
        for module in self.modules():
            if isinstance(module,nn.Linear):
                norm = torch.linalg.matrix_norm(module.weight,ord=2)
                module.weight.div_(norm.clamp_min(1.))
                norms.append(float(norm.clamp_max(1.)))
        return norms

    def forward(self, value, condition):
        if condition.requires_grad or value.shape != condition.shape:
            raise ValueError('Condition must be fixed and aligned')
        context = condition / condition.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        return self.net(torch.cat([value,context],-1)/math.sqrt(self.width))


def conditional_energy(metric, fake, real, condition, *, scale, smoothing=.1):
    """Energy distance between singleton distributions at each condition.

    Metric space: T_phi(c,x)=[x/sqrt(H), D_phi(c,x)], where x is measured
    in fixed teacher-RMS units. Raw RMS error is retained in every direction.
    G minimizes and D maximizes THIS SAME expression. There is no FM or
    logit-ranking auxiliary loss and no discriminator input-gradient penalty.
    """
    if fake.shape!=real.shape or fake.shape!=condition.shape or fake.numel()==0:
        raise ValueError('Conditional target geometry differs')
    if not math.isfinite(smoothing) or smoothing<=0:
        raise ValueError('Invalid smoothing')
    s = torch.as_tensor(scale,device=fake.device,dtype=fake.dtype)
    if s.requires_grad or s.numel()!=1 or not torch.isfinite(s) or s<=0:
        raise ValueError('Calibration must be fixed, finite, scalar and positive')
    x,y = fake/s,real.detach()/s
    raw = (x-y).square().mean(-1)
    learned = (metric(x,condition)-metric(y,condition)).square().squeeze(-1)
    r2 = raw+learned
    return (2*r2/((r2+smoothing**2).sqrt()+smoothing)).mean()


@torch.no_grad()
def backtrack(parameters, before, proposed, evaluate, initial_loss, *, fractions=(1.,.5,.25,.125,.0625,.03125)):
    """Accept only actual decrease of the fixed current-batch scalar loss.

    Adam's proposal is separate from this line search. Caller must restore
    the optimizer state if every proposal is rejected. No nonfinite proposal
    is evaluated. The guarantee concerns this deterministic batch/critic.
    """
    if not math.isfinite(initial_loss):
        raise FloatingPointError('Nonfinite starting objective')
    if not (len(parameters)==len(before)==len(proposed)) or not fractions or any(not 0<f<=1 for f in fractions):
        raise ValueError('Invalid line-search inputs')
    if not all(torch.isfinite(p).all() for p in proposed):
        for p,old in zip(parameters,before): p.copy_(old)
        return dict(accepted=False,fraction=0.,loss=initial_loss,trials=0)
    for trial,fraction in enumerate(fractions,1):
        for p,old,new in zip(parameters,before,proposed): p.copy_(old+fraction*(new-old))
        try:
            loss = float(evaluate())
        except Exception:
            for p,old in zip(parameters,before): p.copy_(old)
            raise
        if math.isfinite(loss) and loss<initial_loss:
            return dict(accepted=True,fraction=fraction,loss=loss,trials=trial)
    for p,old in zip(parameters,before): p.copy_(old)
    return dict(accepted=False,fraction=0.,loss=initial_loss,trials=len(fractions))
