import pytest
from conceptmod.textsliders.reward_game.core import read, write, IntegrityError
from conceptmod.textsliders.reward_game.joint_confirmation_support import check_fresh


def test_failed_sibling_confirmation_and_composition_are_exposed(tmp_path):
    home=tmp_path/'joint-v1'
    write(home/'game.json', dict(parent_home=str(tmp_path)))
    old=dict(family='old',caption='Sparse bass and brushes.',lyrics='[instrumental]')
    new=dict(family='new',caption='Bright brass and toms.',lyrics='[opening theme]')
    for area in ('confirmation','composition'):
        folder=tmp_path/'acoustic-v1'/area/'failed'
        write(folder/'protocol.json',dict(cases=[dict(seed=101)]))
        write(folder/'manifest.json',dict(families=[old]))
    with pytest.raises(IntegrityError,match='fixture already exposed'):
        check_fresh(home,dict(families=[]),dict(cases=[]),[old],[107,109])
    with pytest.raises(IntegrityError,match='seed already exposed'):
        check_fresh(home,dict(families=[]),dict(cases=[]),[new],[101,109])
    result=check_fresh(home,dict(families=[]),dict(cases=[]),[new],[107,109])
    assert len(result['previous_protocols'])==2


def test_joint_composition_matches_native_studio_energy_allocation(tmp_path,monkeypatch):
    from app.sliders import _apply_host_energy
    from conceptmod.textsliders.reward_game import joint_composition as composition
    from conceptmod.textsliders.reward_game.joint_renderer import JointRenderer
    from conceptmod.textsliders.reward_sliders import render,evaluate,specs
    families=[dict(family=f'f{i}',style_multipliers={'first':.8,'second':1.2}) for i in range(8)]
    candidate=dict(path='candidate',multiplier=1.)
    protocol=dict(candidate=candidate,seeds=[13,17])
    manifest=dict(families=families,host_energy_by_kind=dict(language_model=4.,transformer=3.),style_hashes={})
    monkeypatch.setattr(composition,'verify_batch',lambda *args:(protocol,manifest))
    monkeypatch.setattr(specs,'validate_families',lambda *args:None)
    monkeypatch.setattr(evaluate,'extra_style_hashes',lambda *args:None)
    def resolve(styles):
        return [dict(kind='language_model',name=k,multiplier=v,rank=8,alpha=8) for k,v in styles.items()]
    monkeypatch.setattr(render,'resolve_styles',resolve)
    write(tmp_path/'confirmation/first/scorecard.json',dict(batch_pass=True))
    result=composition.freeze(tmp_path,'first','composition')
    frozen=read(tmp_path/'composition/composition/manifest.json')
    assert frozen['host_energy_by_kind']==dict(language_model=2.,transformer=1.)
    renderer=object.__new__(JointRenderer);renderer.manifest=frozen
    extra=[dict(kind=k,name=k+' reward',multiplier=1.,rank=8,alpha=8) for k in ('language_model','transformer')]
    for family,original in zip(frozen['families'],result['original_style_multipliers']):
        assert family['style_multipliers']=={k:v/2 for k,v in original.items()}
        control=renderer.components(family)
        combined=renderer.components(family,extra)
        assert _apply_host_energy([dict(c) for c in combined],result['host_energy_by_kind'])==combined
        assert combined[:-2]==control


def test_partial_development_cannot_freeze_joint_confirmation(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_game import joint_confirmation as confirmation,joint_evaluate
    monkeypatch.setattr(confirmation,'verify',lambda *args:({},{}))
    monkeypatch.setattr(joint_evaluate,'inspect',lambda *args:dict(stage=8,valid_cases=8,advance=True))
    with pytest.raises(IntegrityError,match='strict completed'):
        confirmation.freeze(tmp_path,'partial',tmp_path/'unused','fresh')
    assert not (tmp_path/'confirmation').exists()


def test_full_pass_freezes_pair_provenance_and_diagnostic_lock(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_game import joint_confirmation as confirmation,joint_evaluate
    from conceptmod.textsliders.reward_game.joint_replication_fixtures import families
    from conceptmod.textsliders.reward_sliders import render,evaluate
    home=tmp_path/'joint-v1'
    write(home/'game.json',dict(parent_home=str(tmp_path)))
    write(tmp_path/'recipes/confirmation-template-joint-v1.json',dict(version='joint-v1'))
    write(tmp_path/'audit/confirmation-intent-cache-v1.json',dict(passed=True,actual_processors_and_models_loaded=True,models=[]))
    fixtures=tmp_path/'fixtures.json';write(fixtures,dict(families=families(),seeds=[101,103]))
    candidate=dict(path='pair.json',multiplier=1.,weights_sha256='pair',components=[dict(kind='language_model'),dict(kind='transformer')])
    original=dict(path='original',multiplier=1.,weights_sha256='original')
    game=dict(cases=[],original_reference=dict(checkpoint='original'),duration_seconds=20.4,host_energy_by_kind=dict(language_model=4.,transformer=3.))
    base=dict(families=[],pilot={},style_hashes={})
    monkeypatch.setattr(confirmation,'verify',lambda *args:(game,base))
    monkeypatch.setattr(joint_evaluate,'inspect',lambda *args:dict(stage=16,valid_cases=16,advance=True,candidate_provenance=candidate))
    monkeypatch.setattr(confirmation,'checkpoint',lambda *args:candidate)
    monkeypatch.setattr(confirmation,'lm_checkpoint',lambda *args:original)
    monkeypatch.setattr(render,'resolve_styles',lambda *args:[])
    monkeypatch.setattr(evaluate,'extra_style_hashes',lambda *args:None)
    confirmation.freeze(home,'passed',fixtures,'fresh')
    p,m=confirmation.verify_batch(home,home/'confirmation/fresh')
    assert p['candidate']==candidate and p['original']==original and p['freshness']['sibling_campaigns_included']
    assert m['host_energy_by_kind']==game['host_energy_by_kind']
    write(home/'confirmation/fresh/intent/model-lock.json',dict(passed=False))
    with pytest.raises(IntegrityError,match='model lock changed'):
        confirmation.verify_batch(home,home/'confirmation/fresh')


def test_replication_failure_prevents_pair_packaging(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_game import joint_complete_confirmation as completion
    def verify(home,folder):
        first=folder.name=='first'
        return dict(replication_of=None if first else 'first',candidate=dict(path='same'),
                    seeds=[1,2] if first else [3,4],cases=[dict(family='a' if first else 'b')]),{}
    monkeypatch.setattr(completion,'verify_batch',verify)
    monkeypatch.setattr(completion,'score',lambda home,name:dict(batch_pass=name=='first'))
    with pytest.raises(IntegrityError,match='Both fresh batches'):
        completion.complete(tmp_path,'first','replication','composition','winner')
    assert not (tmp_path/'confirmed').exists()
