import pytest
import torch
from conceptmod.textsliders.reward_game.latent_transport import bounded_step, transported, velocity_loss


@pytest.mark.parametrize('start,end', [(0., 1.), (1., 0.)])
def test_transported_velocity_reaches_displaced_endpoint(start, end):
    # A constant reference velocity lets the Euler identity be checked exactly,
    # including the inverted-sigma schedule used by the actual audio host.
    x = torch.tensor([[[1., 2., 3.]]])
    v = torch.tensor([[[.2, -.4, .1]]])
    delta = torch.tensor([[[.01, -.02, .03]]])
    sigmas = torch.linspace(start, end, 31)
    actual = x.clone()
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        branch = dict(latent=x+(a-start)*v, velocity=v, timestep=a[None], condition=torch.zeros(1))
        kwargs, teacher = transported(branch, delta, a, start, end, 'cpu')
        torch.testing.assert_close(actual, kwargs['hidden_states'], atol=1e-6, rtol=1e-6)
        actual = actual+(b-a)*teacher
    torch.testing.assert_close(actual, x+(end-start)*v+delta, atol=1e-6, rtol=1e-6)


def test_bounded_ascent_preserves_overlap_and_quantized_radius():
    torch.manual_seed(7)
    reference = torch.randn(1, 128, 689).bfloat16()
    current = reference.clone()
    for _ in range(20):
        current, relative = bounded_step(reference, current, torch.ones_like(current), 172)
        assert torch.equal(current[..., :172], reference[..., :172])
        assert relative <= .03
    assert relative > 0
    with pytest.raises(ValueError, match='Missing'):
        bounded_step(reference, current, torch.zeros_like(current), 172)


def test_velocity_objective_reaches_trainable_parameters():
    weight = torch.tensor(0., requires_grad=True)
    reference = torch.ones(2, 3)
    prediction = reference+weight
    loss = velocity_loss(prediction, reference+.01, reference)
    loss.backward()
    assert weight.grad < 0
