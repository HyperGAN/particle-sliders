from types import SimpleNamespace
import pytest
from conceptmod.textsliders.reward_game.core import STRUCTURE as LM,IntegrityError
from conceptmod.textsliders.reward_game.acoustic_artifact import STRUCTURE as TF
from conceptmod.textsliders.reward_game.joint_identity import generation_identity
from conceptmod.textsliders.reward_game.joint_renderer import JointRenderer


def test_joint_identity_budgets_both_hosts_and_tracks_each_component():
    style=dict(LM,weights='style',multiplier=2.)
    parts=[dict(structure=LM,multiplier=1.,weights_sha256='lm'),dict(structure=TF,multiplier=1.,weights_sha256='tf')]
    candidate=dict(structure={'kind':'ordinary_lora_pair','components':parts},components=parts,multiplier=1.,weights_sha256='manifest')
    game=dict(host_energy_by_kind=dict(language_model=4.,transformer=3.),sampler={},duration_seconds=20.4)
    manifest=dict(style_hashes={'style':'s'},model_hashes={},source_hashes={},style_components={'family':[style]},host_energy_by_kind=game['host_energy_by_kind'])
    case=dict(id='c',family={'family':'family'},seed=17,physical_gpu=1,style_components=[style])
    identity=generation_identity(game,manifest,case,candidate)
    assert identity['resolved_energy']==dict(language_model=3.,transformer=1.)
    renderer=object.__new__(JointRenderer);renderer.manifest=manifest
    extra=[dict(LM,weights='lm',multiplier=1.),dict(TF,weights='tf',multiplier=1.)]
    assert renderer.components(case['family'],extra)==[style,*extra]
    parts[0]['multiplier']=3.
    with pytest.raises(IntegrityError,match='language_model energy'):
        generation_identity(game,manifest,case,candidate)


def test_native_merger_loads_both_host_files_and_restores_off(tmp_path,monkeypatch):
    import torch
    from torch import nn
    from safetensors.torch import save_file
    from app import generator
    from app.lora_runtime import LoRANetwork
    class Qwen3Attention(nn.Module):
        def __init__(self):
            super().__init__();self.q_proj=nn.Linear(16,16,bias=False);self.v_proj=nn.Linear(16,16,bias=False)
    class MiniMaxMusic3Attention(nn.Module):
        def __init__(self):
            super().__init__();self.to_q=nn.Linear(16,16,bias=False);self.to_v=nn.Linear(16,16,bias=False)
    lm=nn.Sequential(Qwen3Attention()).to(torch.bfloat16)
    tf=nn.Sequential(MiniMaxMusic3Attention()).to(torch.bfloat16)
    pipe=SimpleNamespace(language_model=lm,transformer=tf);components=[];expected={};original={}
    for index,(host,structure) in enumerate(((lm,LM),(tf,TF))):
        network=LoRANetwork(host,rank=8,alpha=8.,multiplier=1.,target_replace=structure['target_replace'],prefix=structure['prefix'],delimiter='-',train_method='full',attach=False)
        with torch.no_grad():
            for lora in network.unet_loras:
                lora.lora_up.weight.fill_(.03*(index+1));lora.lora_down.weight.fill_(.02)
                module=network.hosts[lora.lora_name];original[module]=module.weight.detach().clone()
                expected[module]=(module.weight.float()+lora.lora_up.weight.float()@lora.lora_down.weight.float()).to(torch.bfloat16)
        path=tmp_path/f'{index}.safetensors';save_file(network.state_dict(),str(path))
        components.append(dict(structure,weights=str(path),mtime=path.stat().st_mtime,multiplier=1.))
    monkeypatch.setattr(generator,'_slider_nets',{});monkeypatch.setattr(generator,'_merge_states',{})
    generator._merge_sliders(pipe,'joint-cpu-test',components)
    assert len(generator._slider_nets)==2
    assert all(torch.equal(module.weight,w) for module,w in expected.items())
    generator._merge_sliders(pipe,'joint-cpu-test',[])
    assert all(torch.equal(module.weight,w) for module,w in original.items())
