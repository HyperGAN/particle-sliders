"""Bounded CPU gates for paired BISPF wiring, signs, routing and continuation."""
from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from particlegan import BISPFCriticAdam, BISPFCriticPenalty, SBNPPriorRegularizer, SinkhornRelativisticLoss
from particlegan.bispf import sinkhorn_coupling

from particle_sliders import BISPFGame, GlobalMixErrorCritic, RoutedMLP


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.bridge = RoutedMLP(3, 3)
        self.dropout = nn.Dropout(.1)
        self.active = False

    @contextmanager
    def scaled(self, scale):
        self.active = scale == 1.
        try:
            yield
        finally:
            self.active = False

    def forward(self, x):
        out = self.bridge(self.dropout(x), self.particles)
        if out.requires_grad:
            # Backward, including any checkpoint replay, retains the scale.
            out.register_hook(lambda grad: grad if self.active else grad * float("nan"))
        return out


def fixture(seed=0, frozen_critic=False, **kwargs):
    data = torch.Generator().manual_seed(144)
    neutral = torch.randn(7, 3, generator=data)
    targets = neutral + torch.randn(7, 3, generator=data)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        network = Network()
        critic = GlobalMixErrorCritic(targets, neutrals=neutral, tokens=2, width=8, heads=2, score_bound=0.)
        if frozen_critic:
            critic.mix.weight.requires_grad_(False)
    game = BISPFGame(network, critic, targets, seed=seed, batch_size=4, **kwargs)
    def predict(i, phase):
        assert network.active, "both D and G must see the active adapter"
        return critic.normalize(neutral[i:i+1] + network(neutral[i:i+1]))
    return game, predict


def assert_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_public_recipe_ownership_and_updates():
    game, predict = fixture()
    assert isinstance(game.opt_d, BISPFCriticAdam)
    assert isinstance(game.penalty, BISPFCriticPenalty)
    assert isinstance(game.spread, SBNPPriorRegularizer)
    assert isinstance(game.loss, SinkhornRelativisticLoss)
    assert game.prior.z is game.network.particles
    owned = [p for g in game.opt_g.param_groups for p in g["params"]]
    assert sum(p is game.network.particles for p in owned) == 1
    assert len(owned) == len({id(p) for p in owned})
    assert game.spread.weight == .05
    assert game.opt_d.ema_critic is None
    initial = game.network.particles.detach().clone()
    rates = [group["lr"] for opt in (game.opt_g, game.opt_d) for group in opt.param_groups]
    global_rng = torch.get_rng_state().clone()
    for i in range(1, 4):
        result = game.step(i, predict)
        assert all(torch.isfinite(torch.tensor(result[k])) for k in ("loss", "g_adv", "d_loss", "prior_regularization"))
        assert result["noise_std"] == 1.
        assert len(result["g_rows"]) == 4
    assert not torch.equal(initial, game.network.particles)
    assert rates == [group["lr"] for opt in (game.opt_g, game.opt_d) for group in opt.param_groups]
    assert torch.equal(global_rng, torch.get_rng_state())


def test_sinkhorn_uses_coordinates_full_batch_and_minimizing_signs():
    game, _ = fixture()
    real = torch.tensor([[0., 0., 0.], [2., 1., 1.], [-1., 2., 0.]])
    fake = torch.tensor([[1., .5, 0.], [-1., 0., 2.], [3., 1., 0.]], requires_grad=True)
    r, f = game.critic(real), game.critic(fake)
    assert r._bispf_input is real and f._bispf_input is fake
    coupling, *_ = sinkhorn_coupling(real, fake)
    delta = r[:, None] - f[None, :]
    torch.testing.assert_close(game.loss.d_loss(r, f), (3*coupling*torch.nn.functional.softplus(-delta)).sum())
    actual = game.loss.g_loss(f, r)
    expected = (3*coupling*torch.nn.functional.softplus(delta)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, fake, retain_graph=True)[0], torch.autograd.grad(expected, fake)[0])
    # Increasing fake raw scores must decrease the minimized generator loss.
    raw = torch.tensor([-.3, .1, .2], requires_grad=True)
    value = SinkhornRelativisticLoss().g_loss(raw, torch.ones(3))
    assert (torch.autograd.grad(value, raw)[0] < 0).all()
    seen = []
    original = game.loss.g_loss
    game.loss.g_loss = lambda f, r: (seen.append((len(f), len(r))) or original(f, r))
    # This spy checks the actual runner's batch coupling, with its own network.
    def own_predict(i, phase):
        return game.network(torch.ones(1, 3) * i)
    game.step(1, own_predict)
    assert seen == [(4, 4)]


def test_exact_checkpoint_resume_with_model_noise_and_no_global_rng_dependency(tmp_path):
    game, predict = fixture()
    game.step(1, predict)
    game.step(2, predict)
    checkpoint = game.state_dict()
    torch.save(checkpoint, tmp_path / "state.pt")
    expected = game.step(3, predict)
    expected_state = game.state_dict()
    other, other_predict = fixture(seed=99)
    other.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    torch.randn(17)  # unrelated activity must not affect the resumed update
    actual = other.step(3, other_predict)
    assert_equal(actual, expected)
    assert_equal(other.state_dict(), expected_state)
    # Saved tensors are detached snapshots, not aliases of future live updates.
    assert checkpoint["completed_steps"] == 2


def test_thermal_state_and_initial_moments_survive_resume():
    game, predict = fixture()
    game.step(1, predict)
    reference = game.prior.moment_tether_mean.clone()
    game.opt_d.record._thermal_scale = .123
    state = game.state_dict()
    other, _ = fixture(seed=99)
    other.load_state_dict(state)
    assert other.opt_d.record._thermal_scale == .123
    torch.testing.assert_close(other.prior.moment_tether_mean, reference, rtol=0, atol=0)
    assert other.prior.z is other.network.particles


def test_nonreentrant_model_checkpoint_replay_keeps_adapter_scale():
    from torch.utils.checkpoint import checkpoint
    game, predict = fixture()
    other, other_predict = fixture()
    expected = game.step(1, predict)
    def checkpointed(i, phase):
        return (checkpoint(other_predict, i, phase, use_reentrant=False)
                if phase == "g" else other_predict(i, phase))
    actual = other.step(1, checkpointed)
    assert_equal(actual, expected)
    assert_equal(other.state_dict(), game.state_dict())


def test_reentrant_checkpoint_callback_keeps_connectivity_scale_and_exact_resume():
    from torch.utils.checkpoint import checkpoint

    def checkpointed(predict):
        def call(i, phase):
            if phase != "g":
                return predict(i, phase)
            # The actual routed parameters are captured by the native model,
            # as in a reentrant transformer block checkpoint.
            token = torch.ones((), requires_grad=True)
            return checkpoint(lambda _: predict(i, phase), token, use_reentrant=True)
        return call

    reference, predict = fixture()
    game, native_predict = fixture()
    assert_equal(game.step(1, checkpointed(native_predict)), reference.step(1, predict))
    # Reentrant backward accumulates parameter contributions in a different
    # order; equivalent updates can differ by one float32 rounding unit.
    for left, right in ((game.network, reference.network), (game.critic, reference.critic)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=1e-6, atol=1e-7)
    assert_equal(game.state_dict()["streams"], reference.state_dict()["streams"])
    assert_equal(game.model_cpu_rng, reference.model_cpu_rng)
    state = game.state_dict()
    expected = game.step(2, checkpointed(native_predict))
    resumed, resumed_predict = fixture(seed=99)
    resumed.load_state_dict(state)
    torch.randn(19)
    assert_equal(resumed.step(2, checkpointed(resumed_predict)), expected)
    assert_equal(resumed.state_dict(), game.state_dict())
    assert not game.network.active and not resumed.network.active


@pytest.mark.parametrize("failure", [None, "disconnected", "backward"])
def test_reentrant_connectivity_probe_retains_graph_rng_and_prior_grad_buffers(failure):
    from torch.utils.checkpoint import checkpoint
    from particle_sliders._training import require_connected_response

    weight = nn.Parameter(torch.tensor([2.]))
    unused = nn.Parameter(torch.tensor([7.]))
    value = torch.tensor([3.], requires_grad=True)
    weight.grad, unused.grad, value.grad = torch.tensor([5.]), torch.tensor([11.]), torch.tensor([13.])
    prior = [(leaf, leaf.grad, leaf.grad.clone()) for leaf in (weight, unused, value)]
    calls = 0

    def native(inputs):
        nonlocal calls
        calls += 1
        if failure == "backward" and calls > 1:
            raise RuntimeError("injected checkpoint backward failure")
        return inputs * weight

    response = checkpoint(native, value, use_reentrant=True)
    rng = torch.get_rng_state().clone()
    parameters = [weight, unused] if failure == "disconnected" else [weight]
    if failure:
        error = ValueError if failure == "disconnected" else RuntimeError
        with pytest.raises(error, match="disconnected|injected checkpoint backward failure"):
            require_connected_response(response, parameters)
    else:
        require_connected_response(response, parameters)
        require_connected_response(response, parameters)
    for leaf, gradient, expected in prior:
        assert leaf.grad is gradient
        assert torch.equal(leaf.grad, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    if failure is None:
        # The retained graph must still perform the real training backward.
        response.sum().backward()
        assert torch.equal(weight.grad, torch.tensor([8.]))
        assert torch.equal(value.grad, torch.tensor([15.]))


@pytest.mark.parametrize("disconnection", ["detached", "bank"])
def test_disconnected_native_gan_path_rejected_before_any_optimizer_update(disconnection):
    game, predict = fixture()
    before = deepcopy((game.network.state_dict(), game.critic.state_dict(),
                       game.opt_g.state_dict(), game.opt_d.state_dict()))
    seen = []

    def broken_predict(i, phase):
        seen.append(phase)
        if disconnection == "detached":
            return predict(i, phase).detach()
        # The bridge is live, but its native route reads a detached particle
        # bank. SBNP alone must not make this qualify as a particle GAN step.
        return game.network.bridge(torch.ones(1, 3) * (i + 1),
                                   game.network.particles.detach())

    with pytest.raises(ValueError, match="detached generator|routed bank is disconnected"):
        game.step(1, broken_predict)
    assert seen and set(seen) == {"g"}
    assert game.completed_steps == 0
    assert_equal(before, (game.network.state_dict(), game.critic.state_dict(),
                         game.opt_g.state_dict(), game.opt_d.state_dict()))
    assert all(p.requires_grad for p in game.critic.parameters())
    assert not game.network.active


def test_frozen_critic_parameters_and_phase_flags_survive_success_and_failure():
    game, predict = fixture(frozen_critic=True)
    frozen = game.critic.mix.weight
    before = frozen.detach().clone()
    flags = [p.requires_grad for p in game.critic.parameters()]
    assert not any(p is frozen for group in game.opt_d.param_groups for p in group["params"])
    game.step(1, predict)
    assert torch.equal(before, frozen)
    assert [p.requires_grad for p in game.critic.parameters()] == flags

    def fail_g_loss(*args):
        raise RuntimeError("injected generator failure")

    game.loss.g_loss = fail_g_loss
    with pytest.raises(RuntimeError, match="injected generator failure"):
        game.step(2, predict)
    assert [p.requires_grad for p in game.critic.parameters()] == flags
    assert torch.equal(before, frozen)
    assert not game.network.active


def test_reject_mismatch_before_update():
    game, predict = fixture()
    with pytest.raises(ValueError, match="contiguous"):
        game.step(2, predict)
    with pytest.raises(ValueError, match="predict"):
        game.step(1, lambda i, phase: torch.ones(3))
    other, _ = fixture(g_lr=.001)
    with pytest.raises(ValueError, match="incompatible"):
        other.load_state_dict(game.state_dict())
    state = deepcopy(game.state_dict())
    state["config"]["upstream_commit"] = "moving-develop"
    with pytest.raises(ValueError, match="incompatible"):
        game.load_state_dict(state)
