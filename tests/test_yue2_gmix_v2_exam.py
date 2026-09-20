"""Released-v2 2D arm: Hub knob pins, GAN-only fail-closed, no live flips.

Pins the propose_only particle-gmix-1600-v2 arm
(:mod:`analysis.slider2d.yue2_gmix_v2_exam`) against the Hub golden numbers
(catalog.json female record, FORMULATION.md, metal/female evidence) and guards
the live surface: Music bipolar ARM_B, the live --lm_target default, the
production YuE2 Arm B recipe, and locked AdvConfig() defaults.
"""
import pytest
import torch
from torch.nn import functional as F

from analysis.slider2d import yue2_gmix_v2_exam as v2
from analysis.slider2d.plus_neu_exam import PLUS_NEU_CELLS
from conceptmod.textsliders import particle_bridge_gan as shared


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


HUB_GOLDEN = dict(
    recipe_name='anneal-routed-particle-error-yue2-v1',
    config_sha256='1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb',
    model_glue_reference='df70ccb2ca8f532bdcc07a343fd12bec77362523',
    generator_objective='paired_error_rpgan_plus_particle_vic',
    critic='gmix',
    critic_tokens=8,
    critic_width=48,
    critic_layers=1,
    critic_heads=4,
    critic_score_bound=8.0,
    g_lr=0.0006,
    d_lr=0.0009,
    particle_lr=0.006,
    betas=(0.0, 0.999),
    schedule='constant',
    ema=0.995,
    parts=128,
    particle_dim=4,
    particle_vic_batch=64,
    particle_vic_target_std=1.0,
    particle_vic_eps=1e-4,
    vicreg_weight=1.0,
    adv_b_cap=1.0,
    adv_reg_kappa=1.0,
    penalty_lazy_k=4,
    penalty_method='autograd',
    penalty_anneal='none',
    cap_coordinates='normalized_paired_error_plus_shared_gaussian',
    target_normalization='paired_edit_per_coordinate_std_median_rms_gain',
    edit_rms_target=1.0,
    edit_noise_ratio=0.28,
    noise_start='edit_rms/edit_noise_ratio',
    noise_floor=0.03,
    noise_decay_steps=1600,
    noise_hold='edit_rms*noise_hold_ratio',
    noise_hold_ratio=1.3,
    adv_batch=8,
    sample_seeds=128,
    history_tokens=32,
    seedbank_sources=512,
    polarity='unipolar',
    lm_target='faithful_plus_neu',
    trained_scales=(1.0,),
    recommended_range=(0.0, 1.0),
    adapter_rank=8,
    adapter_alpha=8.0,
    adapter_width=48,
    router_width=16,
    adv_weight=1.0,
    propose_only=True,
)


def test_v2_spec_matches_hub_golden():
    assert v2.V2_SPEC['aux_weights'] == dict(
        anchor_weight=0.0, cover_weight=0.0, fm_weight=0.0, end_weight=0.0,
        lyrichold_weight=0.0, plan_weight=0.0, pole_weight=0.0)
    spec = {k: val for k, val in v2.V2_SPEC.items() if k != 'aux_weights'}
    assert spec == HUB_GOLDEN
    assert v2.V2_CRITIC_CONFIG == dict(tokens=8, width=48, layers=1, heads=4, score_bound=8.0)
    assert (v2.V2_NOISE_DECAY_STEPS, v2.V2_BATCH) == (1600, 8)
    assert tuple(v2.V2_LADDER) == (600, 1200, 1600, 3400)


def _toy_game(cell='divergent', seed=0):
    field = PLUS_NEU_CELLS[cell](seed=seed)
    network = v2.Student(field.dim); backend = v2.Backend(field, network)
    fixed = [dict(ids=[i], prefix_len=1, neutral=field.poles(i)[2][None],
                  targets=field.poles(i)[0][None]) for i in range(field.rows)]
    torch.manual_seed(seed + 1000)
    critic, g, d = v2.build_v2_game(backend, network, fixed)
    return field, backend, network, fixed, critic, g, d


def test_v2_game_uses_gmix_critic_paired_edit_norm_and_v2_schedule():
    _field, _backend, _network, _fixed, critic, g, d = _toy_game()
    assert isinstance(critic, shared.GlobalMixErrorCritic)
    assert (critic.tokens, critic.width) == (8, 48)
    assert len(critic.blocks) == 1 and critic.score_bound == pytest.approx(8.0)
    assert critic.normalization == 'paired_edit_per_coordinate_std_median_rms_gain'
    assert float(critic.noise_start) == pytest.approx(float(critic.edit_rms) / 0.28)
    assert int(critic.noise_decay_steps) == 1600
    assert [group['lr'] for group in g.param_groups] == pytest.approx([0.0006, 0.006])
    assert [group['lr'] for group in d.param_groups] == pytest.approx([0.0009])
    assert tuple(g.param_groups[0].get('betas', (0.0, 0.999))) == (0.0, 0.999)


def test_v2_batch_sampler_draws_eight_without_changing_global_default():
    assert shared.REFERENCE['batch_size'] == 64
    sampler = shared.BridgeSampler(3, 0, batch_size=8)
    rows, noise = sampler.batch(9, 'cpu', 0.5)
    assert len(rows) == 8 and noise.shape == (8, 9)
    rows2, _ = sampler.batch(9, 'cpu', 0.5)
    assert len(rows2) == 8 and not torch.equal(rows, rows2)
    legacy = shared.BridgeSampler(3, 0)
    assert len(legacy.batch(9, 'cpu', 0.5)[0]) == 64
    assert shared.REFERENCE['batch_size'] == 64
    state = sampler.state_dict()
    clone = shared.BridgeSampler(3, 0, batch_size=8)
    clone.load_state_dict(state)
    assert torch.equal(clone.batch(9, 'cpu', 0.5)[0], sampler.batch(9, 'cpu', 0.5)[0])
    with pytest.raises(ValueError, match='batch size'):
        shared.BridgeSampler(3, 0).load_state_dict(state)


def test_v2_update_is_gan_only_no_mse(monkeypatch):
    _field, backend, network, fixed, critic, g, d = _toy_game()
    sampler = shared.BridgeSampler(len(fixed), 0, batch_size=8)
    def forbidden(*args, **kwargs): raise AssertionError('No reconstruction loss')
    monkeypatch.setattr(F, 'mse_loss', forbidden)
    for step in (1, 2):
        metrics = v2.game.update(backend, network, critic, g, d, fixed,
                                 sampler=sampler, step=step, checkpointing=False)
        assert metrics['loss'] == pytest.approx(metrics['g_adv'] + metrics['particle_vic'])
        assert all(torch.isfinite(torch.as_tensor(metrics[key])) for key in
                   ('g_adv', 'particle_vic', 'd_loss', 'd_pen', 'noise_std'))
    # Zero-init output branch holds the cloud still on step 1; GAN gradients
    # reach it once the branch moves (as in the native preflight).
    assert metrics['particle_gan_grad_norm'] > 0


def test_v2_noise_follows_t1600_hold_schedule():
    _field, _backend, _network, _fixed, critic, _g, _d = _toy_game()
    start = float(critic.noise_start)
    hold = float(critic.edit_rms) * 1.3
    assert start > hold > 0.03
    assert shared.noise_std(0, start=start, decay_steps=1600, hold=hold) == pytest.approx(start)
    mid = shared.noise_std(800, start=start, decay_steps=1600, hold=hold)
    assert mid == pytest.approx(max(start * (0.03 / start) ** 0.5, hold))
    assert shared.noise_std(1600, start=start, decay_steps=1600, hold=hold) == pytest.approx(hold)
    assert shared.noise_std(3400, start=start, decay_steps=1600, hold=hold) == pytest.approx(hold)


def test_v2_scale_zero_exact_and_canary_unscored():
    field = PLUS_NEU_CELLS['divergent'](seed=0)
    network = v2.Student(field.dim)
    scored = v2.score(field, network)
    assert scored['uni']['hit'] is False  # starts at the base, which fails
    assert scored['uni']['neu_hold'] == pytest.approx(1.0)
    assert scored['uni']['canary']['scored'] is False
    assert scored['bipolar_hit'] is False


def test_v2_short_run_is_finite_and_reports_honestly():
    result = v2.run_cell('divergent', seed=0, steps=(2,))
    assert result['critic_kind'] == 'GlobalMixErrorCritic'
    assert len(result['checkpoints']) == 1
    checkpoint = result['checkpoints'][0]
    assert checkpoint['step'] == 2
    assert torch.isfinite(torch.as_tensor(checkpoint['metrics']['noise_std']))
    assert set(checkpoint['ema']['uni']) >= {'cover', 'off_caption', 'neu_hold', 'hit'}


def test_v2_arm_changes_no_locked_defaults_or_live_recipes():
    from analysis.slider2d.adv import AdvConfig
    from analysis.slider2d.locked_baseline_defaults import assert_advconfig_defaults_match_locked
    assert_advconfig_defaults_match_locked(AdvConfig())
    from conceptmod.textsliders import yue2_arm_b as production
    assert production.RECIPE['name'] == 'unipolar-rpgan-bcap-yue2-v3'
    assert production.RECIPE['critic_hidden'] == 256
    from conceptmod.textsliders.train_lm_slider_music3 import ARM_B, parse_args
    assert ARM_B['lm_target'] == 'faithful_guard_e'
    assert ARM_B['pole_weight'] == 1.0 and ARM_B['cover_weight'] == 1.0
    assert parse_args(['--prompts_file', 'dummy.yaml']).lm_target == 'v9'
