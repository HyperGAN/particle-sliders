from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from particlegan import get_recipe

from conceptmod.textsliders.pg09_audio import (
    AudioRow, Music3ARResponseAdapter, audio_conditions, require_audio_width,
)
from pg09_audio_cpu_fixtures import make_audio


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('family', ['yue2', 'music3'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_shared_objective_native_bank_conditioning_and_frozen_base(family, preset, monkeypatch):
    adapter, model, fixed, game = make_audio(family, preset=preset, checkpointing=True)
    owned = {id(p) for p in adapter.network.parameters()}
    base = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in owned}
    before = deepcopy(adapter.network.state_dict())
    assert adapter.particles is adapter.slider.particles
    assert adapter.particles.shape == (128, 4)
    assert sum(p is adapter.particles for g in game.opt_g.param_groups for p in g['params']) == 1
    assert asdict(game.recipe) == asdict(get_recipe(
        preset, total_steps=4, batch_size=2, num_particles=128, z_dim=4))
    assert audio_conditions(fixed).shape == (2, 36)
    assert torch.equal(audio_conditions(fixed)[0, :32], fixed[0].neutral[0])
    assert audio_conditions(fixed)[0, -2:].tolist() == [2., 1.]
    if family == 'music3':
        # Keep real Transformers integer config keys; no fixture sanitization.
        assert model.config.id2label == {0: 'LABEL_0', 1: 'LABEL_1'}

    def forbidden(*args, **kwargs):
        raise AssertionError('MSE fallback reached')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    # No positive caption reaches the student after preparation.
    original = adapter._hidden
    seen = []
    def traced(ids):
        seen.append(tuple(ids))
        return original(ids)
    adapter._hidden = traced
    for _ in range(3):
        metrics = game.step()
        assert metrics['g_loss'] == pytest.approx(metrics['g_adv'] + metrics['prior_regularization'])
        assert metrics['d_loss'] == pytest.approx(metrics['d_adv'] + metrics['d_pen'])
    assert set(seen) == {p.row.neutral_ids for p in fixed}
    assert all(torch.equal(base[n], p) for n, p in model.named_parameters() if n in base)
    changed = [name for name, p in adapter.network.state_dict().items() if not torch.equal(before[name], p)]
    assert len(changed) == len(before)
    adapter.network.zero_grad(set_to_none=True)
    value = adapter.generated(fixed[0])
    # LayerNorm makes an unweighted hidden sum constant; probe a real direction.
    (value * torch.linspace(-1, 1, value.shape[-1])).sum().backward()
    assert adapter.particles.grad is not None and adapter.particles.grad.abs().sum() > 0
    assert all(p.grad is not None for p in adapter.network.parameters())
    assert torch.equal(adapter.sample(fixed[0], scale=0), fixed[0].neutral)
    assert not torch.equal(adapter.sample(fixed[0]), fixed[0].neutral)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['yue2', 'music3'])
@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_checkpoint_resume_export_and_probe_rng(family, preset):
    adapter, model, fixed, game = make_audio(family, preset=preset, checkpointing=True)
    game.step()
    state = game.state_dict()
    rng = torch.get_rng_state().clone()
    sample = adapter.sample(fixed[0])
    assert torch.equal(rng, torch.get_rng_state())
    equal(state, game.state_dict())
    expected_metrics = game.step()
    expected = game.state_dict()
    other, _, other_fixed, other_game = make_audio(family, preset=preset, checkpointing=True)
    other_game.load_state_dict(state)
    assert torch.equal(sample, other.sample(other_fixed[0]))
    equal(expected_metrics, other_game.step())
    equal(expected, other_game.state_dict())
    # Reject changed history/teacher identity before changing any live state.
    tampered = deepcopy(state)
    tampered['config']['backend']['model_identity']['seed'] = 999
    prior = other_game.state_dict()
    with pytest.raises(ValueError):
        other_game.load_state_dict(tampered)
    equal(prior, other_game.state_dict())
    exported = game.export_state_dict()
    assert all(k.startswith('weights.') for k in exported['network'])
    other_game.load_export_state_dict(exported)
    assert torch.equal(adapter.sample(fixed[0]), other.sample(other_fixed[0]))
    with pytest.raises(RuntimeError, match='inference export'):
        other_game.step()


@pytest.mark.parametrize('family', ['yue2', 'music3'])
def test_checkpoint_forward_scale_survives_context_exit(family):
    adapter, _, fixed, _ = make_audio(family)
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(.02)
    initial = deepcopy(adapter.network.state_dict())
    losses = [adapter._response(fixed[0].row.neutral_ids, scale=s).square().mean() for s in (1., .5)]
    sum(losses).backward()
    reference = [p.grad.clone() for p in adapter.network.parameters()]
    other, _, other_fixed, _ = make_audio(family, checkpointing=True)
    other.network.load_state_dict(initial)
    losses = [other._response(other_fixed[0].row.neutral_ids, scale=s).square().mean() for s in (1., .5)]
    assert other.network.scale == 1.0
    sum(losses).backward()
    for a, p in zip(reference, other.network.parameters()):
        torch.testing.assert_close(a, p.grad, atol=0, rtol=0)


@pytest.mark.parametrize('width', [170, 2048, 2560, 4096])
def test_full_width_rejected_before_forward_or_optimizer(width):
    with pytest.raises(ValueError, match='width >=170'):
        require_audio_width(width)
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=width))
    with pytest.raises(ValueError, match='released PG09'):
        Music3ARResponseAdapter(model, None, model_identity={'fixture': 'width guard only'})


@pytest.mark.parametrize('family', ['yue2', 'music3'])
def test_preparation_snapshots_tokens_and_rejects_wrong_history(family):
    adapter, _, fixed, _ = make_audio(family)
    ids = list(fixed[0].row.neutral_ids)
    p = adapter.prepare(replace(fixed[0].row, neutral_ids=ids))
    ids[0] = 99
    assert p.row.neutral_ids == fixed[0].row.neutral_ids
    assert torch.equal(p.target, adapter._response(p.row.positive_ids, scale=0))
    assert torch.equal(p.neutral, adapter._response(p.row.neutral_ids, scale=0))
    with pytest.raises(ValueError, match='same continuation'):
        adapter.prepare(AudioRow((1, 2, 3), (4, 5, 6), history_tokens=1))
    with pytest.raises(ValueError, match='unipolar'):
        adapter.prepare(replace(fixed[0].row, scale=-1))


def test_both_native_families_and_presets_fresh_process_exact(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts' / 'pg09_audio_cpu_fixtures.py'
    full, first, resumed = [tmp_path / f'{s}.pt' for s in ('full', 'first', 'resumed')]
    for args in ([ '--steps', '3', '--output', str(full)],
                 [ '--steps', '1', '--output', str(first)],
                 [ '--steps', '2', '--state-in', str(first), '--output', str(resumed)]):
        result = subprocess.run([sys.executable, str(driver), *args], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    equal(torch.load(full, weights_only=False), torch.load(resumed, weights_only=False))
