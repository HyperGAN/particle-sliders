"""Run separately with the exact reviewed ALZ overlay and native interpreter."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from conceptmod.textsliders.pg09_audio import Music3ARResponseAdapter, require_audio_width
from pg09_audio_cpu_fixtures import make_audio
from test_pg09_audio import equal

pytestmark = pytest.mark.skipif('PG09_AUDIO_FAMILY' not in os.environ,
    reason='explicit reviewed-overlay qualification uses its own native interpreter process')
FAMILY = os.environ.get('PG09_AUDIO_FAMILY', 'music3')
WIDTH = {'yue2': 2048, 'music3': 4096}[FAMILY]
OPTIONS = dict(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_native_full_width_objective_bank_restore_and_export(preset, monkeypatch):
    adapter, model, fixed, game = make_audio(FAMILY, width=WIDTH, preset=preset,
                                            checkpointing=True, **OPTIONS)
    owned = {id(p) for p in adapter.network.parameters()}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in owned}
    initial = deepcopy(adapter.network.state_dict())
    assert fixed[0].target.shape == (1, WIDTH)
    assert game._signature['penalty_input_domain'] == 'continuous'
    assert game._signature['resolved_penalty_domain'] == 'continuous'
    assert game._signature['pg_patch_sha256'] == 'c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571'
    assert adapter.domain_gate['pg_source_profile'] == 'alz-continuous-v1'
    def forbidden(*args, **kwargs):
        raise AssertionError('MSE must not enter explicit continuous audio training')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    for _ in range(3):
        metric = game.step()
        assert metric['d_pen'] > 0
        assert metric['g_loss'] == pytest.approx(metric['g_adv'] + metric['prior_regularization'])
    assert all(not torch.equal(initial[n], p) for n, p in adapter.network.state_dict().items())
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    assert adapter.particles is adapter.slider.particles
    assert torch.equal(adapter.sample(fixed[0], scale=0), fixed[0].neutral)
    adapter.network.zero_grad(set_to_none=True)
    response = adapter.generated(fixed[0])
    (response * torch.linspace(-1, 1, WIDTH)).sum().backward()
    assert adapter.particles.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()
    state = game.state_dict()
    altered = deepcopy(state)
    altered['config']['pg_source_profile'] = 'release'
    altered['config']['penalty_input_domain'] = 'auto'
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        game.load_state_dict(altered)
    equal(state, game.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    exported = game.export_state_dict()
    other, _, other_fixed, other_game = make_audio(FAMILY, width=WIDTH, preset=preset,
                                                  checkpointing=True, **OPTIONS)
    other_game.load_export_state_dict(exported)
    assert torch.equal(adapter.sample(fixed[0]), other.sample(other_fixed[0]))


def test_release_and_mismatched_profile_still_fail_before_host_forward():
    with pytest.raises(RuntimeError, match='source profile'):
        require_audio_width(WIDTH)
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=WIDTH))
    with pytest.raises(ValueError, match='qualified pair'):
        Music3ARResponseAdapter(model, None, model_identity={'test': 'no model forward'},
                               pg_source_profile='release', penalty_input_domain='continuous')


def test_native_width_fresh_process_and_safe_checkpoint_loader(tmp_path):
    driver = Path(__file__).resolve().parents[1] / 'scripts/pg09_audio_cpu_fixtures.py'
    full, first, resumed = [tmp_path / f'{s}.pt' for s in ('full', 'first', 'resumed')]
    common = ['--family', FAMILY, '--width', str(WIDTH), '--pg-source-profile',
              'alz-continuous-v1', '--penalty-input-domain', 'continuous']
    for args in (['--steps', '3', '--output', str(full)],
                 ['--steps', '1', '--output', str(first)],
                 ['--steps', '2', '--state-in', str(first), '--output', str(resumed)]):
        result = subprocess.run([sys.executable, str(driver), *common, *args],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    equal(torch.load(full, weights_only=True), torch.load(resumed, weights_only=True))
