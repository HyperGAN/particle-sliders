from pathlib import Path
import pytest
from conceptmod.textsliders.reward_game import acoustic_confirmation as confirmation
from conceptmod.textsliders.reward_game import acoustic_composition as composition
from conceptmod.textsliders.reward_game.core import read,write,digest,sha,Store,IntegrityError
from conceptmod.textsliders.reward_game.confirmation_amendments import effective_sources


@pytest.mark.parametrize('kind', ['confirmation','composition'])
@pytest.mark.parametrize('host', ['acoustic','lm','joint','ff'])
def test_actual_renderer_observation_is_accepted_by_scoring(tmp_path,monkeypatch,kind,host):
    import torch
    import importlib
    prefix={'acoustic':'acoustic_','lm':'','joint':'joint_','ff':'ff_'}[host]
    confirmation_module=importlib.import_module('conceptmod.textsliders.reward_game.'+prefix+'confirmation')
    composition_module=importlib.import_module('conceptmod.textsliders.reward_game.'+prefix+'composition')
    from conceptmod.textsliders.reward_game.acoustic_renderer import AcousticRenderer
    from conceptmod.textsliders.reward_sliders.specs import digest as renderer_digest
    family=dict(family='fresh',split='test',caption='A steady acoustic groove.',lyrics='[instrumental theme]',style_multipliers={})
    manifest=dict(style_components={'fresh':[]},host_energy_by_kind={'language_model':4.,'transformer':3.},
                  sampler={},model_hashes={},capture_spec={})
    folder=tmp_path/kind/'batch';folder.mkdir(parents=True)
    renderer=object.__new__(AcousticRenderer);renderer.run=folder;renderer.manifest=manifest;renderer.gpu=1;renderer.ownership={}
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'synthetic CPU test')
    def generate(family,seed,audio,**kwargs):
        Path(audio).parent.mkdir(parents=True,exist_ok=True);Path(audio).write_bytes(b'synthetic fixture; no model audio')
        return dict(total_seconds=0.),None
    renderer.generate=generate
    class Scorer:
        def measure(self,path):return dict(valid=True,scalar=7.)
    names=('off','original','candidate') if kind=='confirmation' else ('style-reduced','fixed-energy')
    arms=[dict(name=n,checkpoint=None,multiplier=0.) for n in names]
    for arm in arms:
        row=renderer.observe(family,101,arm,Scorer())
        assert row['provenance']['manifest_sha256']==renderer_digest(manifest)!=digest(manifest)
    cases=[dict(id='fresh-s101',family='fresh',seed=101)]
    if kind=='confirmation':
        # Other planned cases stay unattempted; identity validation must still
        # accept all three real observe() records without rerendering anything.
        cases=[dict(id=f'f{i}-s{s}',family=f'f{i}',seed=s) for i in range(8) for s in (101,103)]
        cases[0]=dict(id='fresh-s101',family='f0',seed=101)
        for arm in arms:
            path=folder/'observations'/f"fresh-s101-{arm['name']}.json"
            r=read(path);r['family']='f0';write(path,r)
        protocol=dict(cases=cases,arms=arms,replication_of=None,candidate={'weights_sha256':'candidate'})
        monkeypatch.setattr(confirmation_module,'verify_batch',lambda *args:(protocol,manifest))
        card=confirmation_module.score(tmp_path,'batch')
    else:
        monkeypatch.setattr(composition_module,'verify',lambda *args:(dict(cases=cases,arms=arms),manifest))
        card=composition_module.score(tmp_path,'batch')
    assert card['valid_cases']==1 and card['new_clips']==len(arms)


def test_amendment_requires_ledger_and_preserved_original_source(tmp_path):
    folder=tmp_path/'confirmation/batch';old=tmp_path/'old.py';old.write_text('original')
    current=tmp_path/'current.py';current.write_text('corrected')
    observation=tmp_path/'observation.json';write(observation,dict(status='complete'))
    protocol=dict(name='batch',sources={str(current):sha(old)})
    amendment=dict(name='hash-fix',protocol_sha256=digest(protocol),rendering_changed=False,ce_scoring_changed=False,
        changes=[dict(path=str(current),previous_sha256=sha(old),current_sha256=sha(current),archived_source=str(old))],added_sources={},
        preserved_observations_and_audio={str(observation):sha(observation)})
    write(folder/'amendments/hash-fix/amendment.json',amendment);Store(tmp_path).event('unrelated')
    with pytest.raises(IntegrityError,match='ledger'):
        effective_sources(tmp_path,folder,protocol)
    Store(tmp_path).event('confirmation_amendment_frozen',batch='batch',name='hash-fix',amendment_sha256=digest(amendment))
    assert effective_sources(tmp_path,folder,protocol)[str(current)]==sha(current)
    old.write_text('lost original')
    with pytest.raises(IntegrityError,match='preserved'):
        effective_sources(tmp_path,folder,protocol)
    old.write_text('original');write(observation,dict(status='changed'))
    with pytest.raises(IntegrityError,match='completed pre-amendment'):
        effective_sources(tmp_path,folder,protocol)
