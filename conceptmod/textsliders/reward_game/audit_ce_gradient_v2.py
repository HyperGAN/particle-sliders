"""Existing-audio CPU audit of CE input gradients with unchanged eval values."""
from contextlib import contextmanager
from pathlib import Path
import time

from .core import DEFAULT_HOME,ROOT,read,immutable,sha


@contextmanager
def input_gradients(model):
    # Only the root flags control AesMultiOutput's explicit grad-mode guards.
    # Children remain in eval, so dropout and feature extraction do not change.
    previous=(model.training,model.freeze_encoder,model.wavlm_model.feature_grad_mult)
    try:
        model.training=True;model.freeze_encoder=False;model.wavlm_model.feature_grad_mult=1.
        yield
    finally:
        model.training,model.freeze_encoder,model.wavlm_model.feature_grad_mult=previous


def audit(home=DEFAULT_HOME):
    import numpy as np
    import torch
    import soundfile as sf
    from torchaudio.functional import resample
    from ..reward_sliders.reward import CEReward
    from ..reward_sliders.specs import RewardSpec
    from .setup import verify
    home=Path(home);game,_=verify(home);folder=home/'audit/ce-input-gradient-v2'
    row=read(ROOT/'analysis/reward_search_20260908/stages/training-capture/observations/search-train-04-s3301-off.json')
    if sha(row['audio'])!=row['audio_sha256']:raise ValueError('Training waveform changed')
    immutable(folder/'protocol.json',dict(audio_sha256=row['audio_sha256'],physical_gpu=None,new_rendered_clips=0,
        source_sha256=sha(__file__),reward_spec=game['reward_spec'],scalar_tolerance=1e-5,
        finite_difference_epsilons=[1e-5,1e-6,1e-7,3e-8],finite_difference_relative_tolerance=.15,
        success_rule='Two adjacent smaller perturbations must each agree with the analytical derivative within 15 percent'))
    torch.set_num_threads(4);began=time.monotonic();scorer=CEReward(RewardSpec(**game['reward_spec']));scorer.load()
    model=scorer.model.requires_grad_(False);data,rate=sf.read(row['audio'],dtype='float32',always_2d=True)
    source=data[:20*rate];wave=torch.from_numpy(source.copy()).requires_grad_(True)
    def normalized(wave):
        rms=wave.double().square().mean().sqrt()
        return wave*(.1/rms).float()
    def value(wave):
        mono=normalized(wave).mean(1);values=[]
        for start in (0,10*rate):
            wav=resample(mono[start:start+10*rate],rate,16000)[None,None]
            output=model(dict(wav=wav,mask=torch.ones_like(wav,dtype=torch.bool)))['CE']
            values.append(output*model.target_transform['CE']['std']+model.target_transform['CE']['mean'])
        return torch.stack(values).mean()
    expected=(source*(.1/float(np.sqrt(np.mean(source.astype('float64')**2))))).astype('float32')
    normalization_exact=np.array_equal(normalized(wave).detach().numpy(),expected)
    normal=float(value(wave).detach())
    with input_gradients(model):
        differentiable=value(wave);differentiable.backward()
    gradient=wave.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum()==0:
        raise RuntimeError('No finite nonzero CE input gradient')
    direction=gradient/gradient.square().mean().sqrt()
    predicted=float((gradient*direction).sum());checks=[]
    for epsilon in (1e-5,1e-6,1e-7,3e-8):
        plus=float(value(wave.detach()+epsilon*direction));minus=float(value(wave.detach()-epsilon*direction))
        measured=(plus-minus)/(2*epsilon);relative=abs(measured-predicted)/max(abs(predicted),1e-12)
        checks.append(dict(epsilon=epsilon,plus=plus,minus=minus,measured=measured,relative_error=relative))
    converged=any(checks[i]['relative_error']<.15 and checks[i+1]['relative_error']<.15 for i in (1,2))
    exact_restoration=not model.training and model.freeze_encoder and all(not m.training for m in model.modules())
    scalar_error=abs(float(differentiable.detach())-row['reward']['scalar'])
    passed=normalization_exact and scalar_error<1e-5 and abs(normal-float(differentiable.detach()))<1e-5 and converged and exact_restoration
    result=dict(passed=passed,normalization_exact=normalization_exact,cached_ce=row['reward']['scalar'],
        eval_ce=normal,differentiable_ce=float(differentiable.detach()),scalar_error=scalar_error,
        gradient_absolute_sum=float(gradient.abs().sum()),predicted_directional_derivative=predicted,
        finite_difference_checks=checks,finite_difference_converged=converged,
        all_model_parameter_gradients_absent=all(p.grad is None for p in model.parameters()),eval_flags_restored=exact_restoration,
        seconds=time.monotonic()-began,new_rendered_clips=0,optimizer_updates=0,
        interpretation='Numerical feasibility only. No waveform or checkpoint is exported and no music-improvement claim is made.')
    immutable(folder/'result.json',result);print(result,flush=True)


if __name__=='__main__':audit()
