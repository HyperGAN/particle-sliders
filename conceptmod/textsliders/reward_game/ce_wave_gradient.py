"""Frozen first-20-second CE and its waveform derivative on CPU.

This differentiates the exact scalar preprocessing; it is not itself evidence
of a useful trainable generator path or improved music.
"""
import torch
from .audit_ce_gradient_v3 import input_gradients


def value_and_gradient(scorer,waveform,rate):
    from torchaudio.functional import resample
    # Public input is [1, channels, samples], matching the vocoder.
    if waveform.ndim!=3 or waveform.shape[0]!=1 or waveform.shape[-1]<20*rate:raise ValueError('Need a valid complete 20-second waveform')
    source=waveform[0,:,:20*rate].T.detach().float().cpu().requires_grad_(True)
    if not torch.isfinite(source).all():raise ValueError('Nonfinite CE input')
    rms=source.double().square().mean().sqrt()
    if float(rms)<1e-8:raise ValueError('Silent CE input')
    normalized=source*(.1/rms).float();mono=normalized.mean(1)
    if scorer.model is None:scorer.load()
    model=scorer.model.requires_grad_(False)
    with input_gradients(model):
        values=[]
        for start in (0,10*rate):
            wav=resample(mono[start:start+10*rate],rate,16000)[None,None]
            output=model(dict(wav=wav,mask=torch.ones_like(wav,dtype=torch.bool)))['CE']
            values.append(output*model.target_transform['CE']['std']+model.target_transform['CE']['mean'])
        ce=torch.stack(values).mean();gradient,=torch.autograd.grad(ce,source)
    if not torch.isfinite(gradient).all() or not gradient.abs().sum():raise ValueError('Missing finite CE waveform gradient')
    result=torch.zeros_like(waveform,device='cpu',dtype=torch.float32)
    result[0,:,:20*rate]=gradient.T
    return float(ce.detach()),result
