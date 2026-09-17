"""Routed particle GAN: equations, native gradient path, deployment and resume."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from conceptmod.textsliders import particle_bridge_gan as pg
from conceptmod.textsliders import yue2_particle_bridge as native
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args, train, run_recipe


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.bridge = pg.RoutedMLP(3, 3)

    def forward(self, x):
        return self.bridge(x, self.particles)


def test_routing_matches_explicit_reference_and_has_no_batch_dependency():
    torch.manual_seed(4); net = Toy().double(); x = torch.randn(5, 3, dtype=torch.float64)
    q = net.bridge.router(x)
    z = (q @ net.particles.T / 2.).softmax(-1) @ net.particles
    expected = net.bridge.net(torch.cat((x, z), -1))
    torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(net(x), torch.cat([net(a[None]) for a in x]), rtol=1e-12, atol=1e-12)
    grad = torch.autograd.grad(net(x).square().sum(), net.particles)[0]
    assert grad.norm() > 0 and torch.isfinite(grad).all()


def test_vic_is_only_sample_variance_hinge_and_off_diagonal_covariance():
    torch.manual_seed(7); z = torch.randn(64, 4, dtype=torch.float64, requires_grad=True)
    centered = z - z.mean(0)
    covariance = centered.T @ centered / 63
    expected = (1 - (z.var(0, unbiased=True) + 1e-4).sqrt()).clamp_min(0).mean()
    expected += sum(covariance[i,j].square() for i in range(4) for j in range(4) if i != j) / 4
    torch.testing.assert_close(pg.particle_vic(z), expected, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(pg.particle_vic(z), z)[0]
    b = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_schedule_and_separate_recoverable_streams():
    assert pg.noise_std(0) == 1 and pg.noise_std(8000) == .03 and pg.noise_std(16000) == .03
    assert pg.noise_std(600) == pytest.approx(.768748168503285)
    a = pg.BridgeSampler(4, 7); b = pg.BridgeSampler(4, 9)
    state = a.state_dict(); b.load_state_dict(state)
    ai, an = a.batch(3, 'cpu', .2); bi, bn = b.batch(3, 'cpu', .2)
    assert torch.equal(ai, bi) and torch.equal(an, bn)
    gi, gn = a.batch(3, 'cpu', .2)
    assert not torch.equal(ai, gi) and not torch.equal(an, gn)


def independent_update(net, critic, g, d, targets, x, sampler, step):
    """Ordinary full minibatches; explicit equations independent of row caching."""
    sigma = pg.noise_std(step)
    di, noise = sampler.batch(3, 'cpu', sigma)
    noise = noise.to(targets)
    target = (targets - critic.target_mean) / critic.target_std
    critic.requires_grad_(True); d.zero_grad(set_to_none=True)
    fake = noise + net(x[di]).detach() - target[di]
    cap = torch.tensor(0.)
    if step % 4 == 0:
        for value in [noise, fake]:
            leaf = value.detach().requires_grad_(True)
            derivative = torch.autograd.grad(critic(leaf).sum(), leaf, create_graph=True)[0]
            cap = cap + 2 * ((derivative.square().sum(-1) + 1e-12).sqrt() - 1).clamp_min(0).square().mean()
    dl = F.softplus(critic(fake) - critic(noise)).mean() + cap
    dl.backward(); d.step(); d.zero_grad(set_to_none=True); critic.requires_grad_(False)
    gi, noise = sampler.batch(3, 'cpu', sigma)
    noise = noise.to(targets)
    g.zero_grad(set_to_none=True)
    adv = F.softplus(critic(noise).detach() - critic(noise + net(x[gi]) - target[gi])).mean()
    z = net.particles[sampler.vic_rows(128, 'cpu')]
    cov = (z - z.mean(0)).T @ (z - z.mean(0)) / 63
    vic = (1 - (z.var(0) + 1e-4).sqrt()).clamp_min(0).mean()
    vic = vic + (cov.square().sum() - cov.diag().square().sum()) / 4
    (adv + vic).backward(); g.step()
    return float(dl.detach()), float(adv.detach()), float(cap.detach())


@pytest.mark.parametrize('step,gain', [(1, 1.), (4, 4.)])
def test_complete_update_equals_batched_reference_without_mse(monkeypatch, step, gain):
    # Double precision separates equation parity from near-zero Adam sign
    # differences caused by float32 batched versus single-row GEMMs.
    torch.manual_seed(23); a = Toy().double(); b = deepcopy(a)
    x = torch.randn(4, 3, dtype=torch.float64); targets = torch.randn(4, 3, dtype=torch.float64)
    ca, ga, da = pg.build_game(a, targets); cb, gb, db = pg.build_game(b, targets)
    ca.double(); cb.double()
    with torch.no_grad():
        for p in ca.parameters(): p.mul_(gain)
    cb.load_state_dict(ca.state_dict())
    sa = pg.BridgeSampler(4, 7); sb = pg.BridgeSampler(4, 7)
    def forbidden(*a, **k): raise AssertionError('No reconstruction loss')
    monkeypatch.setattr(F, 'mse_loss', forbidden)
    expected = independent_update(b, cb, gb, db, targets, x, sb, step)
    got = pg.update(a, ca, ga, da, targets, lambda i, phase: a(x[i:i+1]), sampler=sa, step=step)
    for key, value in zip(['d_loss','g_adv','d_pen'], expected):
        assert got[key] == pytest.approx(value, rel=2e-5, abs=2e-6)
    assert got['loss'] == got['g_adv'] + got['particle_vic']
    for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-6)
    for p, q in zip(ca.parameters(), cb.parameters()): torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-6)
    assert all(p.grad is None for p in ca.parameters())
    assert got['particle_grad_norm'] > 0
    assert got['particle_gan_grad_norm'] > 0
    assert ga.param_groups[1]['lr'] == .006


def test_new_recipe_allows_only_the_requested_particle_term():
    args = parse_args(['--recipe','particle_bridge','--save_dir','unused'])
    assert run_recipe(args, native)['vicreg_weight'] == 1
    with pytest.raises(SystemExit):
        parse_args(['--recipe','particle_bridge','--save_dir','unused','--propose_only_lr_scale','.2'])
    old = native.RECIPE['cover_weight']
    try:
        native.RECIPE['cover_weight'] = 1
        with pytest.raises(ValueError): run_recipe(args, native)
    finally: native.RECIPE['cover_weight'] = old


def test_native_shared_cloud_checkpointing_and_deployment(tmp_path):
    pytest.importorskip('yue2')
    from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider
    torch.manual_seed(19); a = YuE2Backend(dummy=True); b = YuE2Backend(dummy=True)
    b.model.load_state_dict(a.model.state_dict())
    na = native.ParticleSlider(a.model); nb = native.ParticleSlider(b.model)
    nb.load_state_dict(na.state_dict())
    # Move beyond the zero-output initialization so GAN gradients reach routers/cloud.
    with torch.no_grad():
        for adapter in na.adapters.values(): adapter.lora_up.weight.normal_(std=.01)
    nb.load_state_dict(na.state_dict())
    assert sum(p is na.particles for p in na.parameters()) == 1
    assert all(adapter._particles is na.particles for adapter in na.adapters.values())
    ids = [4, 8, 3, 7]
    base = a.hidden(ids).detach().clone()
    with na.scaled(1.):
        ordinary = a.hidden(ids, checkpointing=False); ordinary.square().mean().backward()
    with nb.scaled(1.):
        checkpointed = b.hidden(ids, checkpointing=True); checkpointed.square().mean().backward()
    torch.testing.assert_close(ordinary, checkpointed, rtol=1e-6, atol=1e-7)
    for p,q in zip(na.parameters(),nb.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=1e-7)
    assert na.particles.grad.norm()>0 and all(p.grad is None for p in a.model.parameters())
    assert torch.equal(a.hidden(ids), base)
    path = tmp_path/'routed.safetensors'; na.save(path, dict(dummy=False))
    c = YuE2Backend(dummy=True); c.model.load_state_dict(a.model.state_dict())
    restored, record = YuE2Slider.load(c.model,path)
    with restored.scaled(1.): torch.testing.assert_close(c.hidden(ids), ordinary.detach(),rtol=0,atol=0)
    assert record['format']==native.FORMAT
    assert all(adapter._particles is restored.particles for adapter in restored.adapters.values())
    assert torch.equal(c.hidden(ids),base)


def _same(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return a==b


def test_native_resume_restores_particles_ema_both_optimizers_and_all_streams(tmp_path,monkeypatch):
    pytest.importorskip('yue2')
    from conceptmod.textsliders.yue2_backend import YuE2Backend
    def forbidden(*a,**k): raise AssertionError('No audio sampling or MSE')
    monkeypatch.setattr(YuE2Backend,'continuation',forbidden); monkeypatch.setattr(F,'mse_loss',forbidden)
    prompts=tmp_path/'p.yaml';prompts.write_text(yaml.safe_dump(dict(rows=[
        dict(neutral=f'Guitar band {i}',positive=f'Heavy metal guitar band {i}',lyrics=f'[verse]\nWe carry crate {i}') for i in range(4)])))
    common=['--recipe','particle_bridge','--dummy','--steps','4','--prompts_file',str(prompts)]
    train(parse_args(common+['--save_dir',str(tmp_path/'full')]))
    train(parse_args(common+['--save_dir',str(tmp_path/'split'),'--until','2']))
    train(parse_args(common+['--save_dir',str(tmp_path/'split')]))
    a=torch.load(tmp_path/'full/state.pt',weights_only=True);b=torch.load(tmp_path/'split/state.pt',weights_only=True)
    for key in ['network','critic','ema','g_optimizer','d_optimizer','sampler','rng']:
        assert _same(a[key],b[key]),key
    assert [g['lr'] for g in a['g_optimizer']['param_groups']]==[.0006,.006]
    assert a['d_optimizer']['param_groups'][0]['lr']==.0009
    assert not torch.equal(a['ema']['particles'],a['network']['particles'])
    assert all(h['loss']==h['g_adv']+h['particle_vic'] for h in a['history'])
    assert all(h['d_rows']!=h['g_rows'] for h in a['history'])
    assert a['history'][-1]['particle_gan_grad_norm'] > 0
    from safetensors.torch import load_file
    assert _same(load_file(str(tmp_path/'full/metal-yue2-arm-b_last.safetensors')),a['ema'])
    assert _same(load_file(str(tmp_path/'full/metal-yue2-arm-b_live_last.safetensors')),a['network'])
    with pytest.raises(ValueError,match='Resume'):
        train(parse_args(['--dummy','--steps','4','--prompts_file',str(prompts),'--save_dir',str(tmp_path/'split')]))


def test_particle_dashboard_distinguishes_vic_and_live_gan():
    from scripts.yue2_training_dashboard import dashboard_html
    page=dashboard_html('particle_bridge')
    assert "['particle_vic','Particle VIC'" in page
    assert 'particle_gan_grad_norm' in page
    assert 'raw metal-caption deltas' not in page
    assert 'teacher-RMS coordinates' not in page
    assert 'or other auxiliary losses' not in page


@pytest.mark.parametrize('damage', ['shape', 'nonfinite', 'metadata'])
def test_invalid_particle_checkpoint_rejects_before_attaching(tmp_path, damage):
    pytest.importorskip('yue2')
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider, attention_targets, _Adapter
    source=YuE2Backend(dummy=True); network=native.ParticleSlider(source.model)
    path=tmp_path/'invalid.safetensors'; network.save(path,dict(dummy=False))
    with safe_open(str(path),framework='pt',device='cpu') as handle: metadata=handle.metadata()
    state=load_file(str(path))
    if damage=='shape': state['particles']=state['particles'][:127]
    elif damage=='nonfinite': state['particles'][0,0]=float('nan')
    else:
        record=json.loads(metadata['conceptmod']);record['particle_dim']=8
        metadata['conceptmod']=json.dumps(record)
    save_file(state,str(path),metadata=metadata)
    target=YuE2Backend(dummy=True)
    with pytest.raises(ValueError): YuE2Slider.load(target.model,path)
    assert all(not isinstance(getattr(m.forward,'__self__',None),_Adapter)
               for m in attention_targets(target.model).values())
