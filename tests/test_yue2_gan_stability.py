"""The discriminator must be able to damp the game below the old cap."""
from copy import deepcopy
import pytest
import torch
from conceptmod.textsliders import yue2_arm_b as arm, yue2_gan_plus_neu as plus
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args, run_recipe


def test_r1r2_supplies_curvature_in_the_cap_dead_zone():
    critic = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): critic.weight.fill_(.06)
    # At matched samples, both objectives have the same adversarial term.
    x = torch.ones(4, 1, dtype=torch.float64)
    curvature = {}
    for name in ('b_cap', 'a_r1r2'):
        reg = arm.make_regularizer(name)
        p, stats = reg.penalty(critic, x, x, step=1)
        first = torch.autograd.grad(p, critic.weight, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), critic.weight)[0]
        curvature[name] = float(second)
        assert stats['center'] == (0. if name == 'a_r1r2' else 1.)
    assert curvature['b_cap'] == 0.
    assert curvature['a_r1r2'] == pytest.approx(2.)


@pytest.mark.parametrize('game, name', [(arm, 'unipolar_gan'), (plus, 'gan_plus_neu')])
def test_only_discriminator_penalty_changes_and_recipe_is_opt_in(game, name):
    before = deepcopy(game.RECIPE)
    args = parse_args(['--save_dir', 'unused', '--recipe', name, '--propose_only_r1r2'])
    recipe = run_recipe(args, game)
    changed = {k for k in recipe if recipe.get(k) != before.get(k)}
    assert changed == {'grad_arm', 'discriminator_penalty', 'penalty_center', 'propose_only', 'merge_to_trainer'}
    assert recipe['grad_arm'] == 'a_r1r2' and recipe['merge_to_trainer'] is False
    assert game.RECIPE == before
    reg = arm.make_regularizer(recipe['grad_arm'])
    assert (reg.coeff, reg.norm, reg.method, reg.lazy_k, reg.target_anneal) == (1., 'l2', 'autograd', 1, 'none')


def test_unknown_discriminator_arm_is_rejected():
    with pytest.raises(ValueError, match='Unsupported'):
        arm.make_regularizer('f_none')
