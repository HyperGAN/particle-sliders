"""Fixed-vector calibration regression: dimension, cap units, and opt-in scope."""
from copy import deepcopy
import pytest
import torch
from conceptmod.textsliders import yue2_arm_b as arm, yue2_gan_plus_neu as plus
from conceptmod.textsliders.train_lora_yue2_arm_b import (
    apply_run_coordinates, parse_args, run_recipe,
)
from conceptmod.textsliders.lm_adv import LMDiscriminator
from conceptmod.textsliders.unipolar_gan import ScaleCritic

@pytest.mark.parametrize('dim', [9, 2048])
@pytest.mark.parametrize('conditional', [False, True])
def test_fixed_vector_rms_preserves_magnitude_and_zero_with_dimension_independent_units(dim, conditional):
    real = torch.arange(1., dim + 1.)[None].repeat(4, 1)
    real = real * torch.tensor([.5, 1., 1.5, 2.])[:, None]
    c = ScaleCritic(dim, real, hidden=4) if conditional else LMDiscriminator(dim, hidden_dim=4, in_mode='scaled')
    rows = [dict(targets=real, neutral=torch.zeros_like(real))]
    apply_run_coordinates(c, rows, {'critic_coordinates': 'fixed_teacher_vector_rms'})
    z = real / c.input_scale
    torch.testing.assert_close(z.square().sum(-1).mean(), torch.tensor(1.))
    # A fixed scalar retains amplitude and gives a finite Jacobian at zero.
    torch.testing.assert_close(z[3], 4 * z[0])
    x = torch.zeros_like(real, requires_grad=True)
    f = c(x, 1.) if conditional else c(x)
    grad = torch.autograd.grad(f.sum(), x)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0

@pytest.mark.parametrize('game, recipe_name', [(arm, 'unipolar_gan'), (plus, 'gan_plus_neu')])
def test_coordinate_trial_changes_no_losses_or_rates_and_leaves_defaults_intact(game, recipe_name):
    before = deepcopy(game.RECIPE)
    args = parse_args(['--save_dir', 'unused', '--recipe', recipe_name, '--propose_only_vector_rms'])
    recipe = run_recipe(args, game)
    changes = {k for k in recipe if recipe.get(k) != before.get(k)}
    assert changes == {'cap_coordinates', 'critic_coordinates', 'propose_only', 'merge_to_trainer'}
    assert game.RECIPE == before
    assert recipe['merge_to_trainer'] is False
    c = LMDiscriminator(3, hidden_dim=4, in_mode='scaled', input_scale=7.)
    apply_run_coordinates(c, [], before)
    assert c.input_scale == 7.  # No recalibration in either historical path.

@pytest.mark.parametrize('value', [0., float('nan'), float('inf')])
def test_bad_teacher_fails_closed(value):
    c = LMDiscriminator(3, hidden_dim=4, in_mode='scaled')
    with pytest.raises(ValueError, match='finite, nonzero'):
        apply_run_coordinates(c, [dict(targets=torch.full((1, 3), value), neutral=torch.zeros(1, 3))],
                              {'critic_coordinates': 'fixed_teacher_vector_rms'})
