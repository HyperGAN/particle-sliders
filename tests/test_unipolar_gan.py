"""GAN-only unipolar acceptance, independent update equations, native parity."""
from copy import deepcopy
import json

import pytest
import torch
import torch.nn.functional as F
import yaml

from analysis.slider2d.plus_neu_exam import PLUS_NEU_CELLS, score_plus_neu_residual
from analysis.slider2d.unipolar_gan import NAME, Student, fit
from conceptmod.textsliders import unipolar_gan as game


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('cell', ['divergent', 'close'])
def test_gan_only_passes_uni_with_learned_neutral_origin(cell, seed, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('GAN training must not use MSE')
    monkeypatch.setattr(F, 'mse_loss', forbidden)
    before = torch.get_rng_state().clone()
    field = PLUS_NEU_CELLS[cell](seed=seed)
    residual, history = fit(field, steps=400, seed=seed)
    assert torch.equal(before, torch.get_rng_state())
    row = score_plus_neu_residual(NAME, field, residual, teacher='faithful_plus_neu', plus_neu=True)
    assert row['cover'] >= .85 and row['off_caption'] <= .05 and row['neu_hold'] >= .85, row
    assert row['hit']
    # Zero is learned on the same free-origin student, not frozen to win hold.
    assert any(h['zero_delta_norm'] > 0 for h in history)
    assert all(h['loss'] == .5 * (h['g_pos'] + h['g_zero']) == h['g_adv'] for h in history)
    # The separate bipolar report must not become another UNI acceptance gate.


def reference(network, critic, g, d, real, grad_arm='b_cap'):
    """Full-batch transcription, manual state-only b_cap; no shared losses."""
    critic.requires_grad_(True)
    d.zero_grad(set_to_none=True)
    dloss = 0.
    for s in (0., 1.):
        r = real if s else torch.zeros_like(real)
        f = network.delta(s)[None].expand_as(real).detach()
        terms = []
        for raw in (r, f):
            z = (raw / critic.input_scale).detach().requires_grad_(True)
            grad = torch.autograd.grad(critic.calibrated(z, s).sum(), z, create_graph=True)[0]
            terms.append(grad.square().sum(-1).mean() if grad_arm=='a_r1r2' else
                F.relu(torch.sqrt(grad.square().sum(-1) + 1e-12) - 1).square().mean())
        dloss = dloss + .5 * (F.softplus(critic(f, s)-critic(r, s)).mean() + .5 * sum(terms))
    dloss.backward(); d.step(); d.zero_grad(set_to_none=True); critic.requires_grad_(False)
    g.zero_grad(set_to_none=True)
    gloss = 0.
    for s in (0., 1.):
        r = real if s else torch.zeros_like(real)
        f = network.delta(s)[None].expand_as(real)
        gloss = gloss + .5 * F.softplus(critic(r, s).detach()-critic(f, s)).mean()
    gloss.backward(); g.step()
    return float(gloss.detach()), float(dloss.detach())


@pytest.mark.parametrize('gain', [.01, .8])
@pytest.mark.parametrize('grad_arm', ['b_cap', 'a_r1r2'])
def test_complete_update_matches_independent_equations(gain, grad_arm):
    torch.manual_seed(10)
    real = torch.tensor([[.2, .4, -.1], [.3, .5, -.2], [.1, .2, .3]])
    a = Student(3); b = deepcopy(a)
    ca = game.ScaleCritic(3, real, hidden=8)
    with torch.no_grad():
        for p in ca.parameters(): p.fill_(gain)
    cb = deepcopy(ca)
    def opts(net, critic):
        g = torch.optim.Adam(net.parameters(), lr=.005, betas=(0., .99))
        d = torch.optim.Adam(critic.parameters(), lr=.005, betas=(0., .99))
        for opt in (g, d): opt.param_groups[0]['initial_lr'] = .005
        return g, d
    ga, da = opts(a, ca); gb, db = opts(b, cb)
    def predict(i, s, checkpointing):
        assert a.scale == s
        return a.delta(s)[None]
    for step in (1, 2):
        expected_g, expected_d = reference(b, cb, gb, db, real, grad_arm)
        got = game.update(a, ca, ga, da, real, predict, step=step, total_steps=400, grad_arm=grad_arm)
        assert got['loss'] == pytest.approx(expected_g, rel=2e-6, abs=1e-6)
        assert got['d_loss'] == pytest.approx(expected_d, rel=2e-6, abs=1e-6)
        for x, y in zip(a.parameters(), b.parameters()): torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
        for x, y in zip(ca.parameters(), cb.parameters()): torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
        assert a.scale == 0. and all(p.grad is None for p in ca.parameters())


def rows():
    return [dict(neutral=f'Guitar band, drum pattern {i}', positive=f'Heavy metal guitar band, drum pattern {i}',
                 lyrics=f'[verse]\nWe carry crate {i}') for i in range(4)]


def test_native_checkpointed_gradients_and_exact_zero():
    pytest.importorskip('yue2')
    from conceptmod.textsliders import yue2_gan_plus_neu as native
    from conceptmod.textsliders.yue2_backend import YuE2Backend, YuE2Slider
    torch.manual_seed(19)
    a = YuE2Backend(dummy=True); b = YuE2Backend(dummy=True)
    b.model.load_state_dict(a.model.state_dict())
    fixed = native.prepare(a, rows(), {}, 512)
    na = YuE2Slider(a.model); nb = YuE2Slider(b.model); nb.load_state_dict(na.state_dict())
    ca, ga, da = native.build_game(a, na, fixed)
    cb, gb, db = native.build_game(b, nb, fixed); cb.load_state_dict(ca.state_dict())
    current = [dict(r, ids=r['prefix']) for r in fixed]
    base = {k: v.clone() for k, v in a.model.state_dict().items()}
    eager = native.update(a, na, ca, ga, da, current, step=1, total_steps=600, checkpointing=False)
    checked = native.update(b, nb, cb, gb, db, current, step=1, total_steps=600, checkpointing=True)
    assert eager['loss'] == pytest.approx(checked['loss'], rel=1e-6)
    assert eager['zero_delta_norm'] == 0. and eager['g_zero'] == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert eager['grad_norm'] > 0
    for x, y in zip(na.parameters(), nb.parameters()): torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
    assert all(torch.equal(base[k], v) for k, v in a.model.state_dict().items())
    assert all(p.grad is None for p in a.model.parameters())
    with na.scaled(0.):
        assert torch.equal(a.hidden(fixed[0]['prefix'])[:, -1].float(), fixed[0]['neutral'])


@pytest.mark.parametrize('r1r2', [False, True])
def test_native_resume_is_exact_and_schedule_horizon_is_pinned(tmp_path, monkeypatch, r1r2):
    pytest.importorskip('yue2')
    from conceptmod.textsliders.train_lora_yue2_arm_b import train, parse_args
    from conceptmod.textsliders.yue2_backend import YuE2Backend
    def no_sampling(*args, **kwargs): raise AssertionError('No unused music sampling')
    monkeypatch.setattr(YuE2Backend, 'continuation', no_sampling)
    prompts = tmp_path / 'prompts.yaml'; prompts.write_text(yaml.safe_dump(dict(rows=rows())))
    common = ['--dummy', '--recipe', 'gan_plus_neu', '--steps', '3', '--prompts_file', str(prompts)]
    if r1r2: common.append('--propose_only_r1r2')
    full, split = tmp_path/'full', tmp_path/'split'
    train(parse_args(common + ['--save_dir', str(full)]))
    train(parse_args(common + ['--save_dir', str(split), '--until', '1']))
    train(parse_args(common + ['--save_dir', str(split)]))
    a = torch.load(full/'state.pt', weights_only=True); b = torch.load(split/'state.pt', weights_only=True)
    def same(a, b):
        if torch.is_tensor(a): return torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    for key in ['network', 'critic', 'g_optimizer', 'd_optimizer', 'sampler', 'rng']:
        assert same(a[key], b[key]), key
    with pytest.raises(ValueError, match='Resume'):
        train(parse_args(common + ['--save_dir', str(split), '--steps', '4']))
    if r1r2:
        with pytest.raises(ValueError, match='Resume'):
            train(parse_args([v for v in common if v != '--propose_only_r1r2'] + ['--save_dir', str(split)]))
