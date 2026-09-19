import pytest
from conceptmod.textsliders.reward_game import acoustic_confirmation as confirmation
from conceptmod.textsliders.reward_game import acoustic_composition as composition
from conceptmod.textsliders.reward_game.core import IntegrityError, write, read, digest


@pytest.mark.parametrize('host', ['robust_block'])
def test_acoustic_confirmation_requires_full_development_pass(tmp_path, monkeypatch, host):
    import importlib
    confirmation=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_confirmation')
    composition=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_composition')
    acoustic_evaluate=importlib.import_module('conceptmod.textsliders.reward_game.'+('block' if host=='robust_block' else host)+'_evaluate')
    monkeypatch.setattr(confirmation, 'verify', lambda home: ({}, {}))
    monkeypatch.setattr(acoustic_evaluate, 'inspect', lambda *args: dict(stage=8, valid_cases=8, advance=True))
    with pytest.raises(IntegrityError, match='strict completed'):
        confirmation.freeze(tmp_path, 'partial', tmp_path/'unused', 'fresh')
    assert not (tmp_path/'confirmation').exists()


@pytest.mark.parametrize('host', ['robust_block'])
def test_fresh_manifest_carries_separate_host_limits(tmp_path, monkeypatch, host):
    import importlib
    confirmation=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_confirmation')
    composition=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_composition')
    acoustic_evaluate=importlib.import_module('conceptmod.textsliders.reward_game.'+('block' if host=='robust_block' else host)+'_evaluate')
    from conceptmod.textsliders.reward_game.confirmation_fixtures import families
    from conceptmod.textsliders.reward_sliders import render, evaluate
    parent=tmp_path/'parent'; home=parent/(host+'-v1')
    write(home/'game.json', dict(parent_home=str(parent)))
    write(parent/('recipes/confirmation-template-'+host.replace('_','-')+'-v1.json'), dict(version=host+'-v1'))
    if host in ('robust_block',):write(parent/'audit/confirmation-intent-cache-v1.json',dict(passed=True,actual_processors_and_models_loaded=True,models=[]))
    fixtures=tmp_path/'fixtures.json'; write(fixtures, dict(families=families(), seeds=[101,103]))
    candidate=dict(path='candidate', multiplier=1., weights_sha256='candidate')
    original=dict(path='original', multiplier=1., weights_sha256='original')
    game=dict(cases=[], original_reference=dict(checkpoint='original'), duration_seconds=20.4,
              host_energy_by_kind=dict(language_model=4., transformer=3.))
    base=dict(families=[], pilot={}, style_hashes={})
    monkeypatch.setattr(confirmation, 'verify', lambda *args: (game, base))
    monkeypatch.setattr(acoustic_evaluate, 'inspect', lambda *args: dict(stage=16, valid_cases=16, advance=True, candidate_provenance=candidate))
    monkeypatch.setattr(confirmation, 'checkpoint', lambda *args: candidate)
    monkeypatch.setattr(confirmation, 'lm_checkpoint', lambda *args: original)
    monkeypatch.setattr(render, 'resolve_styles', lambda *args: [])
    monkeypatch.setattr(evaluate, 'extra_style_hashes', lambda *args: None)
    confirmation.freeze(home, 'passed', fixtures, 'fresh')
    protocol, manifest=confirmation.verify_batch(home, home/'confirmation/fresh')
    assert manifest['host_energy_by_kind']==game['host_energy_by_kind']
    assert protocol['candidate']==candidate and protocol['original']==original


@pytest.mark.parametrize('host', ['robust_block'])
def test_composition_matches_actual_studio_allocation_without_reducing_other_host(tmp_path, monkeypatch, host):
    import importlib
    confirmation=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_confirmation')
    composition=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_composition')
    from app.sliders import _apply_host_energy
    renderer_module=importlib.import_module('conceptmod.textsliders.reward_game.'+('block' if host=='robust_block' else host)+'_renderer')
    renderer_class=getattr(renderer_module, {'acoustic':'AcousticRenderer','ff':'FeedForwardRenderer','block':'AcousticBlockRenderer','robust_block':'AcousticBlockRenderer'}[host])
    from conceptmod.textsliders.reward_sliders import render, evaluate, specs
    families=[dict(family=f'f{i}', style_multipliers={'first':.8, 'second':1.2}) for i in range(8)]
    candidate=dict(path='candidate', multiplier=1.)
    protocol=dict(candidate=candidate, seeds=[13, 17])
    manifest=dict(families=families, host_energy_by_kind=dict(language_model=4., transformer=3.), style_hashes={})
    monkeypatch.setattr(composition, 'verify_batch', lambda *args: (protocol, manifest))
    monkeypatch.setattr(specs, 'validate_families', lambda *args: None)
    monkeypatch.setattr(evaluate, 'extra_style_hashes', lambda *args: None)
    def resolve(styles):
        return [dict(kind='language_model', name=k, multiplier=v, rank=8, alpha=8) for k,v in styles.items()]
    monkeypatch.setattr(render, 'resolve_styles', resolve)
    write(tmp_path/'confirmation/first/scorecard.json', dict(batch_pass=True))
    result=composition.freeze(tmp_path, 'first', 'composition')
    assert result['reduced_style_multipliers']==result['original_style_multipliers']
    frozen=read(tmp_path/'composition/composition/manifest.json')
    renderer=object.__new__(renderer_class); renderer.manifest=frozen
    extra=dict(kind='transformer', name='reward', multiplier=1., rank=8, alpha=8)
    for family in frozen['families']:
        control=renderer.components(family)
        combined=renderer.components(family, extra)
        allocated=_apply_host_energy([dict(c) for c in combined], result['host_energy_by_kind'])
        assert allocated==combined
        assert allocated[:-1]==control


@pytest.mark.parametrize('host', ['robust_block'])
def test_acoustic_replication_failure_prevents_packaging(tmp_path, monkeypatch, host):
    import importlib
    confirmation=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_confirmation')
    composition=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_composition')
    completion=importlib.import_module('conceptmod.textsliders.reward_game.'+host+'_complete_confirmation')
    def verify(home, folder):
        first=folder.name=='first'
        return dict(replication_of=None if first else 'first', candidate=dict(path='same'),
                    seeds=[1,2] if first else [3,4], cases=[dict(family='a' if first else 'b')]), {}
    monkeypatch.setattr(completion, 'verify_batch', verify)
    monkeypatch.setattr(completion, 'score', lambda home,name: dict(batch_pass=name=='first'))
    with pytest.raises(IntegrityError, match='Both fresh batches'):
        completion.complete(tmp_path, 'first', 'replication', 'composition', 'winner')
    assert not (tmp_path/'confirmed').exists()
