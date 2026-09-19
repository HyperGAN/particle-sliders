import pytest
from conceptmod.textsliders.reward_game.acoustic_identity import generation_identity
from conceptmod.textsliders.reward_game.acoustic_renderer import AcousticRenderer
from conceptmod.textsliders.reward_game.acoustic_artifact import STRUCTURE as ACOUSTIC
from conceptmod.textsliders.reward_game.core import STRUCTURE,IntegrityError


def test_acoustic_and_lm_host_budgets_are_separate():
    style=dict(STRUCTURE,weights='style',multiplier=2.)
    candidate=dict(structure=ACOUSTIC,multiplier=3.,weights_sha256='candidate')
    game=dict(host_energy_by_kind=dict(language_model=4.,transformer=3.),sampler={},duration_seconds=20.4)
    manifest=dict(style_hashes={'style':'stylehash'},model_hashes={},source_hashes={},style_components={'family':[style]},host_energy_by_kind=game['host_energy_by_kind'])
    case=dict(id='case',family={'family':'family'},seed=9,physical_gpu=1,style_components=[style])
    identity=generation_identity(game,manifest,case,candidate)
    assert identity['resolved_energy']==dict(language_model=2.,transformer=3.)
    renderer=object.__new__(AcousticRenderer);renderer.manifest=manifest
    extra=dict(ACOUSTIC,weights='candidate',multiplier=3.)
    assert renderer.components(case['family'],extra)==[style,extra]
    candidate['multiplier']=3.01
    with pytest.raises(IntegrityError,match='transformer energy'):generation_identity(game,manifest,case,candidate)
    with pytest.raises(ValueError,match='transformer energy'):renderer.components(case['family'],dict(extra,multiplier=3.01))


def test_acoustic_initialization_refuses_missing_actual_model_audit(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_game import acoustic_setup
    from conceptmod.textsliders.reward_game.core import write
    parent=tmp_path/'parent';write(parent/'game.json',{})
    monkeypatch.setattr(acoustic_setup,'verify',lambda home: ({},{}))
    with pytest.raises(FileNotFoundError):acoustic_setup.init(tmp_path/'new',parent,tmp_path/'missing-audit')
    assert not (tmp_path/'new/game.json').exists()


def test_real_acoustic_network_export_has_separate_checked_format(tmp_path):
    import torch
    from diffusers import MiniMaxMusic3Transformer1DModel
    from safetensors.torch import save_file
    from app.lora_runtime import LoRANetwork
    from conceptmod.textsliders.reward_game.acoustic_artifact import checkpoint
    from conceptmod.textsliders.reward_game.core import checkpoint as lm_checkpoint,write,sha
    # Real host topology, with base weights on meta; only rank-8 factors allocate.
    with torch.device('meta'):
        host=MiniMaxMusic3Transformer1DModel()
    network=LoRANetwork(host,rank=8,alpha=8.,target_replace=['MiniMaxMusic3Attention'],prefix='lora_unet',delimiter='-',train_method='full',attach=False)
    weights={k:v.detach().contiguous() for k,v in network.state_dict().items()}
    assert len(weights)==432
    path=tmp_path/'acoustic.safetensors';save_file(weights,str(path))
    prompt=tmp_path/'prompts.json';write(prompt,dict(caption='A small acoustic ensemble with a steady beat.'))
    metadata=dict(ACOUSTIC,weights_sha256=sha(path),prompts_file=str(prompt));write(path.with_suffix('.json'),metadata)
    assert checkpoint(path)['tensor_count']==432
    with pytest.raises(IntegrityError,match='LM structure'):lm_checkpoint(path,1.)
    key=next(k for k in weights if k.endswith('lora_up.weight'));weights[key][0,0]=float('nan')
    save_file(weights,str(path));metadata['weights_sha256']=sha(path);write(path.with_suffix('.json'),metadata)
    with pytest.raises(IntegrityError,match='nonfinite'):checkpoint(path)
