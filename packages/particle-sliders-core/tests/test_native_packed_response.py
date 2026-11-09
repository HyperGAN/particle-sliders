"""Full native rank3 admission; no tensor projection or reshaping to train."""
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from particle_sliders.pg095 import ConditionalPG095, validate_pg09_domain
from particle_sliders.response import ResponseCapabilities


@dataclass(frozen=True)
class Prepared:
    row: dict
    target: torch.Tensor
    neutral: torch.Tensor


class PackedBackend:
    def __init__(self):
        self.network = nn.Linear(64, 64, bias=False)

    def metadata(self):
        return dict(model='synthetic-full-packed-CPU-only', shape=[1, 7, 64])

    def generated(self, prepared):
        return prepared.neutral + self.network(prepared.row['input'])


class PackedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(64, 8)
        self.context = nn.Linear(2, 8, bias=False)
        self.position = nn.Linear(1, 8, bias=False)
        self.score = nn.Linear(8, 1)
        self.register_buffer('positions', torch.arange(7, dtype=torch.float32)[None, :, None])

    def forward(self, response, condition):
        assert response.ndim == 3 and response.shape[1:] == (7, 64)
        hidden = self.value(response) + self.context(condition)[:, None] + self.position(self.positions)
        # This is the learned critic readout, not a reduced PG response.
        return self.score(torch.nn.functional.silu(hidden)).mean((1, 2))


def game():
    with torch.random.fork_rng():
        torch.manual_seed(731)
        backend, critic = PackedBackend(), PackedCritic()
        generator = torch.Generator().manual_seed(911)
        rows = []
        for i in range(4):
            neutral = torch.randn(1, 7, 64, generator=generator)
            rows.append(Prepared(dict(input=torch.randn(1, 7, 64, generator=generator)),
                                 neutral + .1 * (i + 1), neutral))
        return ConditionalPG095(backend, rows, critic, conditions=torch.eye(4, 2),
            response_domain='continuous_packed', penalty_input_domain='auto',
            batch_size=2, total_steps=2, seed=7, particles=None)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_explicit_packed_descriptor_preserves_old_domain_rejections():
    result = validate_pg09_domain((1, 7, 64), response_domain='continuous_packed')
    assert result['resolved_penalty_domain'] == 'image'  # Official PG rank>=3 branch.
    descriptor = ResponseCapabilities('adapter_gan', 'image', 'native packed velocity',
        'continuous_packed', (1, 7, 64), '0.9.5', 'release-0.9.5')
    assert descriptor.response_shape == (1, 7, 64) and descriptor.render is False
    for domain in ('continuous_vector', 'continuous_field'):
        with pytest.raises(ValueError):
            validate_pg09_domain((1, 7, 64), response_domain=domain)
    for shape in ((1, 448), (1, 7, 8, 8), (2, 7, 64), (1, 0, 64)):
        with pytest.raises(ValueError):
            validate_pg09_domain(shape, response_domain='continuous_packed')
    with pytest.raises(ValueError):
        validate_pg09_domain((1, 7, 64), response_domain='continuous_packed',
                             penalty_input_domain='continuous')


def test_packed_penalty_matches_actual_upstream_on_unchanged_rank3():
    actual, reference = game(), game()
    generator = torch.Generator().manual_seed(819)
    real = torch.randn(2, 7, 64, generator=generator)
    fake = torch.randn(2, 7, 64, generator=generator)
    condition = torch.tensor([[1., 0.], [0., 1.]])
    # Direct official factory, with no custom law or input transform.
    upstream = reference.recipe.make_critic_penalty(reference.opt_d, collect_stats=True)
    seen = []
    handle = actual.critic.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
    try:
        with torch.random.fork_rng():
            torch.manual_seed(121)
            value = actual.penalty(actual.critic, real, fake, condition)
        with torch.random.fork_rng():
            torch.manual_seed(121)
            expected = upstream(reference.critic, real, fake, condition)
        assert torch.equal(value, expected)
        value.backward(); expected.backward()
        for parameter, original in zip(actual.critic.parameters(), reference.critic.parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.equal(parameter.grad, original.grad)
        assert seen and all(shape[1:] == (7, 64) for shape in seen)
    finally:
        handle.remove()
    probe = real.detach().clone().requires_grad_(True)
    gradients = torch.autograd.grad(actual.critic(probe, condition).sum(), probe, create_graph=True)[0]
    assert gradients.shape == real.shape and torch.all(gradients != 0)
    assert torch.isfinite(torch.autograd.grad(gradients.square().sum(), probe)[0]).all()


def test_full_rank3_updates_and_exact_full_resume_without_a_particle_bank():
    full, split = game(), game()
    original_g = deepcopy(full.network.state_dict())
    original_d = deepcopy(full.critic.state_dict())
    full.step(); full.step()
    split.step()
    resumed = game()
    resumed.load_state_dict(split.state_dict())
    resumed.step()
    equal(full.state_dict(), resumed.state_dict())
    assert full.prior is None and full.signature['generator_arm'] == 'adapter_gan'
    assert full.response_shape == (1, 7, 64)
    assert full.signature['resolved_penalty_domain'] == 'image'
    assert any(not torch.equal(v, original_g[k]) for k, v in full.network.state_dict().items())
    assert any(not torch.equal(v, original_d[k]) for k, v in full.critic.state_dict().items())
