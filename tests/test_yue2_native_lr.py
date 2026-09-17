"""Native rate transfer must preserve the GAN and survive the +/0 scheduler."""
from copy import deepcopy
import pytest
import torch
from analysis.slider2d.adv import delayed_cosine
from analysis.slider2d.unipolar_gan import Student
from conceptmod.textsliders import yue2_arm_b as arm, yue2_gan_plus_neu as plus, unipolar_gan as shared
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args, run_recipe, apply_run_lrs

@pytest.mark.parametrize('game, name', [(arm, 'unipolar_gan'), (plus, 'gan_plus_neu')])
def test_native_scale_changes_only_optimizer_rates(game, name):
    before = deepcopy(game.RECIPE)
    args = parse_args(['--save_dir','unused','--recipe',name,'--propose_only_lr_scale','.2'])
    recipe = run_recipe(args, game)
    changed = {k for k in recipe if recipe.get(k) != before.get(k)}
    assert changed == {'g_lr','d_lr','lr_scale','ablation','propose_only','merge_to_trainer'}
    assert recipe['g_lr'] == pytest.approx(before['g_lr'] / 5)
    assert recipe['d_lr'] / recipe['g_lr'] == pytest.approx(before['d_lr'] / before['g_lr'])
    assert recipe['merge_to_trainer'] is False
    assert game.RECIPE == before

@pytest.mark.parametrize('value', ['0','-1','nan','inf','2'])
def test_bad_rate_scale_is_rejected_before_loading(value):
    with pytest.raises(SystemExit):
        parse_args(['--save_dir','unused','--propose_only_lr_scale',value])

def test_rate_trials_cannot_silently_compose():
    with pytest.raises(SystemExit):
        parse_args(['--save_dir','unused','--propose_only_c9_g4x','--propose_only_lr_scale','.2'])

@pytest.mark.parametrize('step', [1, 81, 300, 600])
def test_real_plus_neu_update_does_not_reset_the_native_override(step, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('GAN-only G')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    previous = torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        net = Student(3);real = torch.tensor([[.2, .3, -.1], [.1, .4, -.2]])
        c, g, d = shared.build_game(net, real, lr=plus.RECIPE['g_lr'])
        recipe = run_recipe(parse_args(['--save_dir','unused','--recipe','gan_plus_neu','--propose_only_lr_scale','.2']), plus)
        apply_run_lrs(g, d, recipe)
        def predict(i, scale, checkpointing): return net.delta(scale)[None]
        metrics = shared.update(net,c,g,d,real,predict,step=step,total_steps=600)
        factor = delayed_cosine(step-1,total=600,delay=80,min_ratio=.05)
        for optimizer, key in [(g,'g_lr'),(d,'d_lr')]:
            assert optimizer.param_groups[0]['initial_lr'] == pytest.approx(.0001)
            assert metrics[key] == pytest.approx(.0001 * factor)
        assert metrics['loss'] == .5 * (metrics['g_pos'] + metrics['g_zero'])
    finally:
        torch.set_num_threads(previous)
