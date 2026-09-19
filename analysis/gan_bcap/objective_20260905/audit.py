"""Executable counterexamples to transferring the Gaussian stability claim."""
from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn.functional as F

from conceptmod.textsliders.gan_v2.critic import SpanCritic
from .distribution import paired_energy, energy_distance


def dirac():
    eta = .2
    result = {}
    for damping in (0., .02, 1.):
        # Local negative gradient field in (generator point, critic slope).
        j = torch.tensor([[0., .5], [-.5, -2*damping]], dtype=torch.float64)
        simultaneous = torch.eye(2) + eta*j
        alternating = torch.tensor([[1-eta*eta/4, eta/2*(1-2*eta*damping)],
                                     [-eta/2, 1-2*eta*damping]],dtype=torch.float64)
        extragradient = torch.eye(2) + eta*j + eta*eta*(j@j)
        result[str(damping)] = {name:float(torch.linalg.eigvals(a).abs().max())
            for name,a in [('simultaneous',simultaneous),('alternating',alternating),('extragradient',extragradient)]}
    theta = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    phi = torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    cap = F.relu(phi.abs()-1).square()
    g = F.softplus(-phi*theta)
    result['exact_match_nonstationary'] = dict(cap=float(cap.detach()),
        generator_gradient=float(torch.autograd.grad(g,theta)[0]))
    return result


def adam():
    result = {}
    for scale in (1., .01, 1e-4):
        p = torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        opt = torch.optim.Adam([p],lr=.0005,betas=(0.,.999))
        p.grad = torch.tensor(scale,dtype=torch.float64)
        opt.step()
        result[str(scale)] = float(p.detach().abs())
    def clipped_field(x):
        gradient = torch.stack([2*x[0],4*x[1]])
        return gradient/gradient.norm()
    j = torch.autograd.functional.jacobian(clipped_field,torch.tensor([1.,1.],dtype=torch.float64))
    result['clipped_field_cross_derivative_difference'] = float(j[0,1]-j[1,0])
    return result


def critic_nullspace():
    torch.manual_seed(71)
    critic = SpanCritic(4096,conditioned=False,ordered=False,normalized_features=False).eval()
    real = torch.randn(4,7,4096)
    critic.calibrate_input_scale(real)
    w = critic.proj.weight.detach().double()
    vector = torch.randn(4096,dtype=torch.float64)
    vector -= w.T @ torch.linalg.solve(w@w.T,w@vector)
    vector = vector/vector.square().mean().sqrt()
    fake = real+10*vector.float()
    with torch.no_grad():
        score_difference = (critic(real)-critic(fake)).abs().max()
        feature_error = F.mse_loss(critic.features(real),critic.features(fake))
    return dict(input_width=4096,projection_width=128,minimum_nullity=4096-128,
        hidden_rmse=float((real-fake).square().mean().sqrt()),
        score_max_difference=float(score_difference),fm_mse=float(feature_error),
        paired_energy=float(paired_energy(fake,real)))


def empirical_scores():
    real = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    swapped = real.flip(0).clone().requires_grad_(True)
    exact = real.clone().requires_grad_(True)
    loss = paired_energy(exact,real)
    exact_gradient = torch.autograd.grad(loss,exact)[0]
    return dict(marginal_swap_energy=float(energy_distance(swapped,real,unbiased=False).detach()),
        conditional_swap_energy=float(paired_energy(swapped,real).detach()),
        exact_value=float(loss.detach()),exact_gradient=float(exact_gradient.norm()),
        collapsed_fake_energy=float(energy_distance(torch.zeros_like(real),real,unbiased=False)))


def saved_coordinate_geometry():
    root=Path(__file__).resolve().parents[3]
    cached=torch.load(root/'analysis/gan_bcap/critic_failure_probe_20260904.pt',map_location='cpu',weights_only=True)[0]
    source=torch.load(root/'models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_state.pt',
        map_location='cpu',weights_only=True)
    d=SpanCritic(4096,conditioned=False,ordered=False,normalized_features=False).eval()
    d.load_state_dict(source['modules']['critic'],strict=True)
    d.requires_grad_(False)
    x=cached['fake'].clone().requires_grad_(True)
    mask=cached['mask']
    gradient=torch.autograd.grad(d(x,mask).sum(),x)[0]
    n=mask.sum(-1)*x.shape[-1]
    calibrated_l2_gradient=gradient.flatten(1).norm(dim=-1)*d.input_scale
    teacher=cached['real']*mask[...,None]
    teacher_l2=(teacher/d.input_scale).flatten(1).norm(dim=-1)
    return dict(valid_coordinates=n.tolist(),input_scale=float(d.input_scale),
        teacher_calibrated_l2=teacher_l2.tolist(),
        current_cap_coordinate_norm=calibrated_l2_gradient.tolist(),
        rms_cost_dual_norm=(n.float().sqrt()*calibrated_l2_gradient).tolist(),
        interpretation='The dual norm for RMS displacement is sqrt(number of valid coordinates) times the L2 gradient norm. A threshold in one geometry is not the same threshold in the other.')


def main():
    torch.set_num_threads(2)
    output = dict(dirac=dirac(),optimizer=adam(),projection=critic_nullspace(),scores=empirical_scores(),
        saved_coordinate_geometry=saved_coordinate_geometry())
    target = Path(__file__).with_name('counterexamples.json')
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
