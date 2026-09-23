"""CPU gates: particle-sliders-core calls ParticleGAN develop public API."""
from pathlib import Path

import particlegan
import pytest
import torch
from particlegan import (
    GANLoss,
    GradientPenalty,
    ParticleRegularizer,
    get_recipe,
    make_b_cap,
    make_gan_loss,
)
from particlegan.grad_regularizers import GradRegularizer as UpstreamGradRegularizer

from particle_sliders import (
    GradRegularizer,
    GradientPenalty as CoreGradientPenalty,
    particlegan_get_recipe,
    particlegan_locked_shared,
    winning_formulation,
)
from particle_sliders.grad_regularizers import finite_difference_norm
from particle_sliders.reference import particle_vic, rp_d_loss, rp_g_loss


def test_particlegan_package_imports():
    assert particlegan.__version__ if hasattr(particlegan, "__version__") else True
    assert callable(get_recipe)
    assert callable(make_b_cap)
    assert callable(make_gan_loss)
    recipe = get_recipe("gan")
    assert hasattr(recipe, "make_gradient_penalty")
    gp = recipe.make_gradient_penalty()
    assert isinstance(gp, GradientPenalty)


def test_grad_regularizer_alias_is_particlegan_backed():
    assert GradRegularizer is UpstreamGradRegularizer
    assert GradRegularizer is GradientPenalty
    assert CoreGradientPenalty is GradientPenalty
    stamp_reg = winning_formulation().regularizer()
    assert isinstance(stamp_reg, GradientPenalty)
    assert isinstance(stamp_reg, GradRegularizer)
    assert stamp_reg.arm == "b_cap"
    assert stamp_reg.__class__.__module__.startswith("particlegan")


def test_no_vendored_grad_regularizer_class_body():
    path = Path(__file__).resolve().parents[1] / "src" / "particle_sliders" / "grad_regularizers.py"
    text = path.read_text()
    assert "class GradRegularizer" not in text
    assert "from particlegan" in text
    assert len(text.splitlines()) < 40


def test_locked_shared_uses_particlegan_builders():
    primitives = particlegan_locked_shared()
    assert isinstance(primitives["b_cap"], GradientPenalty)
    assert isinstance(primitives["gan_loss"], GANLoss)
    # make_b_cap / make_gan_loss are the ParticleGAN lock builders
    torch.testing.assert_close(
        torch.tensor(primitives["b_cap"].coeff),
        torch.tensor(make_b_cap().coeff),
    )
    assert primitives["gan_loss"].loss_type == make_gan_loss().loss_type
    assert primitives["gan_loss"].mode == "rp"


def test_documented_get_recipe_helper():
    recipe = particlegan_get_recipe("gan")
    assert recipe.make_gradient_penalty().arm == get_recipe("gan").make_gradient_penalty().arm


def test_product_losses_delegate_to_particlegan():
    torch.manual_seed(0)
    real = torch.randn(8)
    fake = torch.randn(8)
    gan = GANLoss(loss_type="logistic", mode="rp")
    torch.testing.assert_close(rp_d_loss(real, fake), gan.d_loss(real, fake))
    torch.testing.assert_close(rp_g_loss(real, fake), gan.g_loss(fake, real))
    particles = torch.randn(16, 4)
    vic = ParticleRegularizer(weight=1.0)(particles)
    torch.testing.assert_close(particle_vic(particles), vic)


def test_finite_difference_norm_reexported():
    assert callable(finite_difference_norm)
