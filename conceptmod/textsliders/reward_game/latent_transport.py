"""Bounded latent guidance and transport of a captured flow path.

This is supervised velocity distillation, not a generator reward derivative.
"""
import torch


def overlap_frames(capture, index):
    from diffusers.modular_pipelines.minimax_music3.denoise import _OVERLAP_LATENT_LENGTH
    if index == 0:
        return 0
    previous = capture['chunks'][index-1]['latent'].shape[-1]
    start = max(0, previous-2*_OVERLAP_LATENT_LENGTH)
    end = max(start, previous-_OVERLAP_LATENT_LENGTH)
    return min(end-start, capture['chunks'][index]['latent'].shape[-1])


def bounded_step(reference, current, gradient, overlap, step=.005, radius=.025):
    """Normalize ascent per chunk, project, then quantize for the real vocoder."""
    if not all(torch.isfinite(x).all() for x in (reference, current, gradient)):
        raise ValueError('Nonfinite latent guidance input')
    reference = reference.float()
    gradient = gradient.float().clone()
    gradient[..., :overlap] = 0
    norm = gradient.norm()
    if not norm > 0:
        raise ValueError('Missing free-latent reward gradient')
    scale = reference.norm().clamp_min(1e-12)
    delta = current.float()-reference + step*scale*gradient/norm
    delta[..., :overlap] = 0
    delta *= (radius*scale/delta.norm().clamp_min(1e-12)).clamp(max=1.)
    result = (reference+delta).to(current.dtype)
    result[..., :overlap] = reference[..., :overlap].to(result)
    relative = float((result.float()-reference).norm()/scale)
    if relative > .03:
        raise ValueError('Quantized target exceeds the predeclared latent radius')
    return result.detach(), relative


def schedule(capture, device='cpu'):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(capture['scheduler_config'])
    scheduler.set_timesteps(sigmas=np.linspace(1., 1./capture['steps'], capture['steps']), device=device)
    if capture['guidance_config']['guidance_rescale'] != 0.:
        raise ValueError('Transport assumes affine CFG without rescaling')
    return scheduler


def transported(branch, delta, sigma, start, end, device):
    """x'(sigma)=x(sigma)+progress*delta, v'=v+delta/(end-start)."""
    span = float(end-start)
    if span == 0:
        raise ValueError('Degenerate flow interval')
    latent = branch['latent'].to(device)
    displacement = delta.to(device).float()
    shifted = (latent.float()+float((sigma-start)/span)*displacement).to(latent.dtype)
    velocity = branch['velocity'].to(device).float()+displacement/span
    return dict(hidden_states=shifted, timestep=branch['timestep'].to(device),
                encoder_hidden_states=branch['condition'].to(device), return_dict=False), velocity


def velocity_loss(prediction, target, reference_velocity):
    return (prediction.float()-target.float()).square().mean()/reference_velocity.float().square().mean().clamp_min(1e-8)
