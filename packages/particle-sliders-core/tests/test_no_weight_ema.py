"""Raw-only admission before allocation, without native models or GPU work."""
from copy import deepcopy

import pytest
import torch

from particle_sliders.lifecycle import CheckpointError, EMA, SessionConfig, advance
from particle_sliders.weight_policy import require_raw_provenance
from test_image_lifecycle import equal, seal, session


@pytest.mark.parametrize('decay', [0., .9, .995, True, '0.9'])
def test_config_refuses_every_non_none_weight_ema(decay):
    with pytest.raises(ValueError, match='Weight EMA is disabled'):
        SessionConfig('unused', 'raw-cohort', ema_decay=decay)
    with pytest.raises(ValueError, match='Weight EMA is disabled'):
        SessionConfig('unused', 'raw-cohort', ema=decay)
    assert SessionConfig('unused', 'raw-cohort').options['ema_decay'] is None


def test_ema_compatibility_class_never_allocates_a_shadow():
    class NoState:
        def state_dict(self):
            pytest.fail('shadow allocation')
    assert EMA(NoState()).state_dict() == {'decay': None, 'values': None}
    with pytest.raises(ValueError, match='Weight EMA is disabled'):
        EMA(NoState(), .995)


def test_averaging_object_refused_before_optimizer_update():
    active = session()
    before = active.checkpoint()
    with pytest.raises(ValueError, match='EMA'):
        advance(active.game, ema=object())
    equal(before, active.checkpoint())


def test_ema_artifacts_rejected_before_any_inference_backend_preparation():
    active = session()
    packet = active.export()
    packet['payload']['selection'] = 'ema'
    seal(packet)
    before = active.checkpoint()
    with pytest.raises(CheckpointError, match='raw'):
        active.load_inference(packet)
    equal(before, active.checkpoint())
    with pytest.raises(ValueError, match='raw'):
        active.export_inference(selection='ema', specification=None)


@pytest.mark.parametrize('metadata', [
    {'weights_kind': 'ema'}, {'effective_ema': {'decay': .9}}, {'ema': .995}, {'ema_decay': .9},
])
def test_known_averaged_provenance_is_not_relabelled(metadata):
    original = deepcopy(metadata)
    with pytest.raises(ValueError, match='averaged|Averaged|EMA'):
        require_raw_provenance(metadata)
    assert metadata == original


def test_explicit_historical_live_export_keeps_its_raw_provenance():
    metadata = {'weights_kind': 'live', 'ema': .995}
    require_raw_provenance(metadata)
    assert metadata == {'weights_kind': 'live', 'ema': .995}
    assert not torch.cuda.is_initialized()


def test_live_recipes_are_raw_and_named_historical_stamp_is_retained():
    from particle_sliders.formulation import particle_gmix_1600_v2, winning_formulation
    from particle_sliders.recipe import SliderRecipe
    from particle_sliders.textsliders._slider2d.adv import AdvConfig
    assert winning_formulation().spec['ema'] is None
    assert winning_formulation().formulation_id == 'particle-gmix-1600-v2-raw'
    assert particle_gmix_1600_v2().parameters['ema'] == .995
    assert SliderRecipe().ema is None and AdvConfig().ema is None
    with pytest.raises(ValueError, match='EMA'):
        SliderRecipe.from_dict({'ema': .995})
    with pytest.raises(ValueError, match='EMA'):
        AdvConfig(ema=.995)


@pytest.mark.parametrize('module,needs_host', [
    ('tiny_llm_particle', True), ('bonsai_gguf_particle', False),
    ('music3_particle_bridge', True), ('yue2_particle_bridge', True),
])
def test_particle_loaders_reject_averaged_metadata_before_host_allocation(tmp_path, module, needs_host):
    import importlib
    import json
    from safetensors.torch import save_file
    loaded = importlib.import_module('particle_sliders.textsliders.' + module)
    name = {'tiny_llm_particle': 'TinyParticleSlider', 'bonsai_gguf_particle': 'BonsaiParticleHead',
            'music3_particle_bridge': 'ParticleNetwork', 'yue2_particle_bridge': 'ParticleSlider'}[module]
    path = tmp_path / (module + '.safetensors')
    save_file({'weight': torch.ones(1)}, path, metadata={'conceptmod': json.dumps({'weights_kind': 'ema'})})
    arguments = (None, path) if needs_host else (path,)
    with pytest.raises(ValueError, match='Averaged adapter'):
        getattr(loaded, name).load(*arguments)


def test_upstream_disabled_recipe_value_stays_zero_and_scalar_history_is_untouched():
    from particle_sliders.formulation import particlegan_get_recipe
    assert particlegan_get_recipe(ema_decay=0.0).ema_decay == 0.0
    with pytest.raises(ValueError, match='EMA'):
        particlegan_get_recipe(ema_decay=.9)
    require_raw_provenance({'ema_decay': 0.0})


def test_historical_stamp_cannot_be_used_to_enable_new_averaging():
    from particle_sliders.formulation import WinningFormulation, particle_gmix_1600_v2
    from particle_sliders.formulation_game import FormulationGame
    historical = WinningFormulation(particle_gmix_1600_v2())
    with pytest.raises(ValueError, match='EMA'):
        FormulationGame(historical, historical.as_dict())


@pytest.mark.parametrize('provenance_source', ['sidecar', 'embedded'])
def test_generic_checkpoint_rejects_averaged_artifact_provenance(tmp_path, provenance_source):
    import json
    from safetensors.torch import save_file
    from particle_sliders.weight_policy import require_raw_checkpoint
    path = tmp_path / 'ordinary.safetensors'
    record = {'effective_ema': {'decay': .995}}
    metadata = {'conceptmod': json.dumps(record)} if provenance_source == 'embedded' else None
    save_file({'weight': torch.ones(1)}, path, metadata=metadata)
    if provenance_source == 'sidecar':
        path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Averaged adapter'):
        require_raw_checkpoint(path)
    if provenance_source == 'sidecar':
        assert json.loads(path.with_suffix('.json').read_text()) == record
    assert path.exists()
