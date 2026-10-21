"""Public ParticleGAN 0.8 API and fixed-cap compatibility regression gates."""
import pytest
import torch
from particlegan import GANLoss, ParticleRegularizer, get_recipe
from particlegan.grad_regularizers import GradientPenalty as UpstreamGradientPenalty
from particle_sliders import GradRegularizer, GradientPenalty, particlegan_get_recipe, particlegan_locked_shared, winning_formulation
from particle_sliders.reference import particle_vic, rp_d_loss, rp_g_loss


def test_recipe_public_builders():
    recipe = particlegan_get_recipe("bispf")
    critic = torch.nn.Linear(3, 1)
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer)
    assert penalty.critic is critic
    assert recipe.critic_formulation == "bispf"
    assert recipe.lr_schedule == "constant"
    assert recipe.make_loss().__class__.__name__ == "SinkhornRelativisticLoss"


def test_legacy_cap_uses_upstream_kernel():
    assert GradRegularizer is GradientPenalty
    regularizer = winning_formulation().regularizer()
    assert isinstance(regularizer, UpstreamGradientPenalty)
    assert regularizer.arm == "b_cap"
    assert regularizer.lazy_k == 4
    with pytest.raises(ValueError, match="does not implement legacy"):
        GradientPenalty(norm="rms")


def test_legacy_locked_helper():
    primitives = particlegan_locked_shared()
    assert isinstance(primitives["b_cap"], UpstreamGradientPenalty)
    assert primitives["b_cap"].arm == "b_cap"
    assert isinstance(primitives["gan_loss"], GANLoss)
    assert primitives["gan_loss"].loss == "relativistic"


def test_product_losses_preserve_legacy_equations():
    real = torch.tensor([0.2, 0.8, -0.1])
    fake = torch.tensor([-0.3, 0.1, 0.4], requires_grad=True)
    expected_d = torch.nn.functional.softplus(fake-real).mean()
    expected_g = torch.nn.functional.softplus(real-fake).mean()
    torch.testing.assert_close(rp_d_loss(real, fake), expected_d)
    torch.testing.assert_close(rp_g_loss(real, fake), expected_g)
    particles = torch.randn(16, 4)
    torch.testing.assert_close(particle_vic(particles), ParticleRegularizer(weight=1.)(particles))
