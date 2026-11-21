"""CPU gate for the PSR claim (arXiv:2605.03907): token-specific steering
coefficients imitate prompt steering better than the best constant coefficient.

Paper: "Steer Like the LLM: Activation Steering that Mimics Prompting"
(Heyman & Vandeputte, 2026). Its central mechanism: prompt steering does NOT
intervene uniformly -- it applies strong interventions on some tokens while
barely affecting others, so a learned per-token coefficient (PSR) beats
constant-coefficient activation steering at matched coherence.

Toy operationalization: token states h in R^12. The "prompted" target applies
a known ground-truth gain profile along a fixed direction v: gain 2.2 on the
half of tokens whose activation aligns with a probe direction u, 0.1 on the
rest. Two adapter arms are trained adversarially against identically
initialized critics. Both arms use EMPF through the shared core's
ParticleGAN recipe wrapper and its critic optimizer, critic penalty, and
relativistic loss builders:

  (a) constant arm:  h + softplus(alpha) * v, one scalar for every token;
  (b) token arm:     h + softplus(MLP(h)) * v, a coefficient per token.

Both arms are given the true direction v, isolating the coefficient claim.
The constant arm is also compared against the oracle constant (the exact
MSE-optimal scalar, computed in closed form): training must reach that
ceiling, and the token arm must beat it by a wide margin.
"""
import pytest
import torch
from torch import nn

from particle_sliders import particlegan_get_recipe


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(before)


D = 12
GAIN_HI, GAIN_LO = 2.2, 0.1


def _world(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(D, generator=g)
    v = v / v.norm()
    u = torch.randn(D, generator=g)
    u = u / u.norm()
    return g, v, u


def _gains(h, u):
    return torch.where(h @ u > 0,
                       torch.full_like(h[:, 0], GAIN_HI),
                       torch.full_like(h[:, 0], GAIN_LO))


def _target(h, v, u):
    return h + _gains(h, u)[:, None] * v[None, :]


class _ConstAdapter(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("v", v)
        self.alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, h):
        return h + torch.nn.functional.softplus(self.alpha) * self.v


class _TokenAdapter(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("v", v)
        self.net = nn.Sequential(nn.Linear(D, 32), nn.SiLU(), nn.Linear(32, 1))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.fill_(0.7)

    def forward(self, h):
        return h + torch.nn.functional.softplus(self.net(h)) * self.v


def _train_arm(kind, seed, recipe, steps=75, batch=128, n_train=4096):
    g, v, u = _world(seed)
    torch.random.default_generator.manual_seed(seed)
    base = torch.randn(n_train, D, generator=g)
    tgt = _target(base, v, u)
    # Initialize the shared critic before the adapter consumes random draws.
    critic = nn.Sequential(nn.Linear(D, 48), nn.SiLU(),
                           nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))
    adapter = _ConstAdapter(v) if kind == "const" else _TokenAdapter(v)
    opt_d = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(opt_d)
    loss = recipe.make_loss()
    opt_g = torch.optim.Adam(adapter.parameters(), lr=3e-3)
    for _ in range(steps):
        sel = torch.randint(0, n_train, (batch,), generator=g)
        h, real = base[sel], tgt[sel]
        fake = adapter(h)
        opt_d.zero_grad()
        d_loss = loss.d_loss(critic(real), critic(fake.detach()))
        (d_loss + penalty(critic, real, fake.detach())).backward()
        opt_d.step()
        opt_g.zero_grad()
        loss.g_loss(critic(adapter(h)), critic(real)).backward()
        opt_g.step()
    ge = torch.Generator().manual_seed(555 + seed)
    he = torch.randn(1024, D, generator=ge)
    te = _target(he, v, u)
    with torch.no_grad():
        mse = ((adapter(he) - te) ** 2).sum(1).mean().item()
        gs = _gains(he, u)
        oracle = ((gs - gs.mean()) ** 2).mean().item()  # ||v|| == 1
        none = (gs ** 2).mean().item()
    return mse, oracle, none


def test_token_coefficient_beats_best_constant():
    recipe = particlegan_get_recipe("empf", loss="relativistic", ema_decay=None)
    assert recipe.effective_critic_formulation == "empf"
    for seed in (0, 1, 2):
        const_mse, oracle, none = _train_arm("const", seed, recipe)
        token_mse, _, _ = _train_arm("token", seed, recipe)
        # The constant arm, trained adversarially, must reach its own
        # closed-form ceiling -- otherwise the comparison proves nothing.
        assert 0.9 * oracle <= const_mse <= 1.25 * oracle, (
            f"seed {seed}: constant arm MSE {const_mse:.3f} vs oracle {oracle:.3f}")
        # Steering at all must beat doing nothing.
        assert const_mse < none and token_mse < none
        # The paper's claim: a per-token coefficient imitates the prompted
        # states far better than ANY constant coefficient can.
        assert token_mse < 0.65 * oracle, (
            f"seed {seed}: token MSE {token_mse:.3f} vs best-constant {oracle:.3f}")
        assert token_mse < const_mse
