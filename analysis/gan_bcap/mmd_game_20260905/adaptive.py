"""Transactional adaptive proposals for the unchanged paired MMD objective."""
from __future__ import annotations

import copy
import math

import torch


def restore_research(source, network, optimizer, *, restore_rng=True):
    if source.get('schema') not in ('conditional-energy-research-1', 'mmd-adaptive-1'):
        raise ValueError('Expected a complete research state')
    if source.get('metric') is not None or source.get('d_optimizer') is not None:
        raise ValueError('This continuation requires the fixed kernel without a critic')
    if source['history'] and source['history'][-1]['step'] != source['step']:
        raise ValueError('History and completed attempt count differ')
    network.load_state_dict(source['network'], strict=True)
    optimizer.load_state_dict(source['optimizer'])
    if restore_rng:
        torch.set_rng_state(source['torch_rng'])
        if source['cuda_rng']:
            if len(source['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('Visible GPU topology differs from the source')
            torch.cuda.set_rng_state_all(source['cuda_rng'])
    return copy.deepcopy(source.get('proposal', dict(scale=1/32, consecutive_rejections=0)))


@torch.no_grad()
def adaptive_step(parameters, optimizer, evaluate, initial_loss, proposal, *, trials=8):
    """Search Adam, then normalized negative gradient; roll back rejected moments.

    Fractions adapt across attempts. The fallback uses the initial Adam trial's
    L2 step norm, preserving a meaningful scale while changing the direction.
    Every acceptance requires measured strict decrease, even with bf16 forwards.
    """
    if not math.isfinite(initial_loss) or trials < 1 or not 0 < proposal['scale'] <= 1:
        raise ValueError('Invalid starting loss or search configuration')
    before = [p.detach().clone() for p in parameters]
    gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
    if not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Nonfinite gradient')
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    records = []
    accepted = False
    method, fraction, loss = 'none', 0., initial_loss

    def reset():
        for p, old in zip(parameters, before):
            p.copy_(old)

    def search(direction, name, start):
        nonlocal accepted, method, fraction, loss
        if not all(torch.isfinite(d).all() for d in direction):
            records.append(dict(method=name, fraction=0., loss=None, reason='nonfinite proposal'))
            return
        for index in range(trials):
            scale = start * 2.**-index
            for p, old, delta in zip(parameters, before, direction):
                p.copy_(old + scale*delta)
            value = float(evaluate())
            records.append(dict(method=name, fraction=scale,
                                loss=value if math.isfinite(value) else None))
            if math.isfinite(value) and value < initial_loss:
                accepted, method, fraction, loss = True, name, scale, value
                return
        reset()

    try:
        optimizer.step()
        direction = [p.detach()-old for p, old in zip(parameters, before)]
        adam_norm = math.sqrt(sum(float(d.double().square().sum()) for d in direction))
        search(direction, 'adam', proposal['scale'])
        if not accepted:
            optimizer.load_state_dict(saved_optimizer)
            gradient_norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
            if gradient_norm > 0 and math.isfinite(adam_norm) and adam_norm > 0:
                direction = [-g*(adam_norm/gradient_norm) for g in gradients]
                search(direction, 'gradient', proposal['scale'])
        if not accepted:
            reset()
            optimizer.load_state_dict(saved_optimizer)
    except Exception:
        reset()
        optimizer.load_state_dict(saved_optimizer)
        raise

    movement = math.sqrt(sum(float((p.detach().double()-old.double()).square().sum())
                             for p, old in zip(parameters, before)))
    next_scale = min(1., max(2.**-20, fraction*1.25 if accepted else proposal['scale']*.25))
    proposal.update(scale=next_scale, consecutive_rejections=0 if accepted else proposal['consecutive_rejections']+1)
    return dict(accepted=accepted, method=method, fraction=fraction, loss=loss,
                trials=len(records), evaluations=records, parameter_step=movement,
                full_adam_norm=adam_norm, next_scale=next_scale)
