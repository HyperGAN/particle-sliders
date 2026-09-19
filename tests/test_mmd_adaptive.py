import copy

import pytest
import torch

from analysis.gan_bcap.mmd_game_20260905.adaptive import adaptive_step, restore_research


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_smaller_than_old_grid_can_accept_and_continue():
    p = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    optimizer = torch.optim.AdamW([p], lr=100., betas=(0., .999), weight_decay=0.)
    proposal = dict(scale=1/32, consecutive_rejections=0)
    for _ in range(5):
        optimizer.zero_grad(); loss = p.square(); loss.backward()
        result = adaptive_step([p], optimizer, lambda: p.square(), float(loss), proposal)
        assert result['accepted'] and result['loss'] < float(loss)
    assert abs(p) < .1


def test_rejected_proposals_restore_initialized_moments_and_change_next_scale():
    p = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([p], lr=.1)
    p.square().backward(); optimizer.step()
    before, moments = p.detach().clone(), copy.deepcopy(optimizer.state_dict())
    proposal = dict(scale=1/32, consecutive_rejections=0)
    result = adaptive_step([p], optimizer, lambda: 1., 1., proposal, trials=2)
    assert not result['accepted'] and result['parameter_step'] == 0
    assert torch.equal(p, before) and equal(moments, optimizer.state_dict())
    assert proposal == dict(scale=1/128, consecutive_rejections=1)


def test_gradient_fallback_recovers_from_adam_momentum_ascent():
    p = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    optimizer = torch.optim.AdamW([p], lr=.1, betas=(.9, .999), weight_decay=0.)
    p.grad = torch.tensor(-100., dtype=p.dtype); optimizer.step()
    with torch.no_grad(): p.fill_(1.)
    optimizer.zero_grad(); p.square().backward()
    moments = copy.deepcopy(optimizer.state_dict())
    result = adaptive_step([p], optimizer, lambda: p.square(), 1., dict(scale=1., consecutive_rejections=0))
    assert result['method'] == 'gradient' and result['loss'] < 1
    assert equal(moments, optimizer.state_dict())


def test_evaluator_exception_rolls_back_parameters_and_optimizer():
    p = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([p], lr=.1)
    p.square().backward()
    before = copy.deepcopy(optimizer.state_dict())
    def fail(): raise RuntimeError('evaluation failed')
    with pytest.raises(RuntimeError, match='evaluation failed'):
        adaptive_step([p], optimizer, fail, 1., dict(scale=1., consecutive_rejections=0))
    assert p == 1 and equal(before, optimizer.state_dict())


def test_full_research_resume_reproduces_next_adaptive_update():
    torch.manual_seed(9)
    network = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01, betas=(0., .999), weight_decay=0.)
    x = torch.tensor([[1., -2.]], dtype=torch.float64)
    proposal = dict(scale=.2, consecutive_rejections=0)
    def update(net, opt, search):
        opt.zero_grad(); loss = net(x).square().mean(); loss.backward()
        return adaptive_step(list(net.parameters()), opt, lambda: net(x).square().mean(), float(loss), search)
    update(network, optimizer, proposal)
    source = copy.deepcopy(dict(schema='mmd-adaptive-1', step=721, history=[dict(step=721)],
        network=network.state_dict(), optimizer=optimizer.state_dict(), metric=None, d_optimizer=None,
        torch_rng=torch.get_rng_state(), cuda_rng=[], proposal=proposal))
    clone = torch.nn.Linear(2, 1).double()
    clone_opt = torch.optim.AdamW(clone.parameters(), lr=999.)
    restored = restore_research(source, clone, clone_opt)
    assert torch.equal(network(x), clone(x))
    expected = update(network, optimizer, proposal)
    actual = update(clone, clone_opt, restored)
    assert equal(network.state_dict(), clone.state_dict())
    assert equal(optimizer.state_dict(), clone_opt.state_dict())
    assert actual == expected and restored == proposal
    source['schema'] = 1
    with pytest.raises(ValueError, match='research state'):
        restore_research(source, clone, clone_opt)
