"""The explicit native LR flag must execute the winning toy's same game."""
from copy import deepcopy
import pytest
import torch
import torch.nn.functional as F

from analysis.slider2d import yue2_unipg_c as sweep
from analysis.slider2d.yue2_gan_exam import ToyBackend, ToySlider
from analysis.slider2d.plus_neu_exam import PLUS_NEU_CELLS, score_plus_neu_residual
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import yue2_arm_b as game
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args,run_recipe,apply_run_lrs


def test_flag_only_changes_recorded_lrs_and_does_not_mutate_defaults():
    before=deepcopy(game.RECIPE)
    args=parse_args(['--save_dir','unused','--propose_only_c9_g4x'])
    recipe=run_recipe(args,game)
    assert {k for k in before if before[k]!=recipe[k]}=={'g_lr','d_lr'}
    assert recipe['propose_only'] and recipe['merge_to_trainer'] is False
    assert recipe['g_lr']==sweep.ARMS['c9_g4x']['g_lr']==.002
    assert recipe['d_lr']==sweep.ARMS['c9_g4x']['d_lr']==.003
    assert game.RECIPE==before
    with pytest.raises(SystemExit):
        parse_args(['--save_dir','unused','--recipe','gan_plus_neu','--propose_only_c9_g4x'])


@pytest.mark.parametrize('key',['end_weight','pole_weight','cover_weight','fm_weight',
    'lyrichold_weight','plan_weight','anchor_weight','vicreg_weight','parts'])
def test_forbidden_generator_companions_fail_before_loading(key,monkeypatch):
    monkeypatch.setitem(game.RECIPE,key,1.)
    with pytest.raises(ValueError,match='GAN-only'):
        run_recipe(parse_args(['--save_dir','unused']),game)


@isolated_seed('seed')
def fit(cell,seed):
    field=PLUS_NEU_CELLS[cell](seed=seed)
    network=ToySlider(field.dim);backend=ToyBackend(field,network)
    rows=[]
    for i in range(field.rows):
        positive,_,neutral=field.poles(i)
        rows.append(dict(ids=[i],prefix_len=1,neutral=neutral[None],targets=positive[None]))
    critic,g,d=game.build_game(backend,network,rows)
    recipe=run_recipe(parse_args(['--save_dir','unused','--propose_only_c9_g4x']),game)
    apply_run_lrs(g,d,recipe)
    for step in range(1,601):
        order=torch.randperm(len(rows)).tolist()
        metrics=game.update(backend,network,critic,g,d,[rows[i] for i in order],step=step,checkpointing=False)
        assert metrics['loss']==metrics['g_adv']
    return score_plus_neu_residual('native_c9',field,network.snapshot(),teacher='faithful_plus_neu',plus_only=True)


@pytest.mark.parametrize('seed',[0,1,7])
@pytest.mark.parametrize('cell',['divergent','close'])
def test_actual_native_flag_passes_original_uni_gates_at_600(cell,seed,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No MSE in G')
    monkeypatch.setattr(F,'mse_loss',forbidden)
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:score=fit(cell,seed)
    finally:torch.set_num_threads(before)
    assert score['cover']>=.85 and score['off_caption']<=.05 and score['neu_hold']>=.85
