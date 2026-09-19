"""Precision diagnostics and fixed-history teacher preparation for round two."""
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
            return sum(float(loss.detach()) for loss in losses())

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


@torch.no_grad()
def float32_teachers(lm, tokenizer, network, prepared, rows, device):
    """Recompute fixed teachers, preserving all cached bf16 history values exactly.

    Base model tensors have been widened from bf16 to float32, so this is a
    change of arithmetic, not loading a different pretrained model checkpoint.
    """
    from conceptmod.textsliders import train_lm_slider_music3 as legacy
    from analysis.gan_bcap.objective_20260905.live import state_sequence
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CFG_TOKEN_ID

    legacy._set_scale(network, 0.)
    result = []
    for original, source in zip(prepared, rows):
        row = copy.deepcopy(original)
        row['prompt_embeds'] = original['prompt_embeds'].float()
        row['frame_embeds'] = None if original['frame_embeds'] is None else original['frame_embeds'].float()
        positive_tokens, mask = legacy._tokenize(tokenizer,
            legacy._assemble(source['positive'], source['lyrics']), device)
        neutral_tokens, neutral_mask = legacy._tokenize(tokenizer,
            legacy._assemble(source.get('neutral') or source['target'], source['lyrics']), device)
        ns, ps = legacy._assert_lyric_span(neutral_tokens, neutral_mask, positive_tokens, mask,
            tokenizer, source['lyrics'], where='float32 fixed histories')
        def embeds(tokens):
            unconditional = tokens.clone()
            unconditional[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
            return lm.model.embed_tokens(torch.cat([tokens, unconditional], 0)).cpu()
        if not torch.equal(embeds(neutral_tokens), row['prompt_embeds']):
            raise ValueError('Neutral token embeddings differ from the frozen inputs')
        if not torch.equal(ns.repeat(2, 1).cpu(), row['span_mask']):
            raise ValueError('Frozen lyric positions changed')
        teacher = dict(row, prompt_embeds=embeds(positive_tokens), span_mask=ps.repeat(2, 1).cpu())
        neutral = state_sequence(lm, row, device).cpu()
        positive = state_sequence(lm, teacher, device).cpu()
        row['energy_neutral'] = neutral
        row['energy_target'] = positive-neutral
        if positive.shape != original['energy_target'].shape:
            raise ValueError('Teacher positions changed')
        result.append(row)
    legacy._set_scale(network, 1.)
    return result
