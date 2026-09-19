"""Directional probes that release each forward graph before evaluating the next row."""
from __future__ import annotations

import copy
import math

import torch


def directional_probe(parameters, losses, *, lengths=(.1, .01, .001, .0001)):
    """Compare autograd with two-sided finite differences; leave parameters/RNG intact.

    ``losses`` yields additive scalar losses, allowing one row's graph at a time.
    The direction is the full-batch normalized negative gradient. Actual rounded
    displacement is recorded, so an unchanged float32 parameter is not a step.
    """
    parameters = list(parameters)
    before = [p.detach().clone() for p in parameters]
    old_grad = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def value(grad_enabled=False):
        with torch.set_grad_enabled(grad_enabled):
            total = 0.
            iterator = iter(losses())
            while True:
                try:
                    loss = next(iterator)
                except StopIteration:
                    break
                total += float(loss.detach())
                del loss
            return total

    try:
        repeated = [value() for _ in range(3)]
        grad_forward = [value(True) for _ in range(2)]
        for p in parameters:
            p.grad = None
        starting = 0.
        with torch.enable_grad():
            for loss in losses():
                starting += float(loss.detach())
                loss.backward()
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('Probe requires a finite nonzero gradient')
        direction = [-g / norm for g in gradients]
        trials = []
        for length in lengths:
            pair = {}
            for sign in (-1, 1):
                with torch.no_grad():
                    for p, old, d in zip(parameters, before, direction):
                        p.copy_(old + sign * length * d)
                actual = math.sqrt(sum(float((p.detach().double()-old.double()).square().sum())
                                       for p, old in zip(parameters, before)))
                measured = value()
                pair[str(sign)] = dict(loss=measured, change=measured-repeated[0], parameter_l2=actual)
            trials.append(dict(length=length, sides=pair,
                               finite_difference=(pair['1']['loss']-pair['-1']['loss'])/(2*length),
                               predicted_derivative=-norm))
        return dict(no_grad_repeats=repeated, grad_forward_repeats=grad_forward,
                    backward_loss=starting, gradient_norm=norm, trials=trials)
    finally:
        with torch.no_grad():
            for p, old, grad in zip(parameters, before, old_grad):
                p.copy_(old)
                p.grad = grad
        torch.set_rng_state(rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
