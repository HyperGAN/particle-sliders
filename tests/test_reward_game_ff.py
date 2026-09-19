from types import SimpleNamespace
import pytest


def test_real_blocks_train_only_feed_forward_and_native_merge_matches(tmp_path,monkeypatch):
    import torch
    from safetensors.torch import save_file
    from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    from app import generator
    from conceptmod.textsliders.reward_game.ff_network import make,check_frozen_attention
    from conceptmod.textsliders.reward_game.ff_artifact import STRUCTURE
    from conceptmod.textsliders.reward_game.merged_forward import MergedForward
    torch.set_num_threads(2);torch.manual_seed(79)
    tf=MiniMaxMusic3Transformer1DModel(in_channels=2,condition_dim=8,num_layers=36,
        num_attention_heads=2,attention_head_dim=4,ff_inner_dim=16,rotary_dim=4,fourier_embedding_dim=4).to(torch.bfloat16).eval().requires_grad_(False)
    network=make(tf,'cpu');before={k:v.detach().clone() for k,v in network.state_dict().items()}
    args=dict(hidden_states=torch.randn(1,2,6,dtype=torch.bfloat16),encoder_hidden_states=torch.randn(1,6,8,dtype=torch.bfloat16),timestep=torch.ones(1,dtype=torch.bfloat16),return_dict=False)
    with torch.no_grad():baseline=tf(**args)[0].clone()
    wrapper=MergedForward(network).attach()
    assert torch.equal(tf(**args)[0],baseline)
    tf(**args)[0].float().square().mean().backward()
    trainable=[p for p in network.parameters() if p.requires_grad]
    assert len(trainable)==144 and any(p.grad is not None and bool(p.grad.count_nonzero()) for p in trainable)
    torch.optim.AdamW(trainable,lr=.001,weight_decay=0.).step()
    assert check_frozen_attention(network,before)
    assert any(not torch.equal(v,before[k]) for k,v in network.state_dict().items() if '-ff_' in k)
    with torch.no_grad():differentiable=tf(**args)[0].clone()
    wrapper.detach()
    path=tmp_path/'ff.safetensors';save_file(network.state_dict(),str(path))
    component=dict(STRUCTURE,weights=str(path),mtime=path.stat().st_mtime,multiplier=1.)
    monkeypatch.setattr(generator,'_slider_nets',{});monkeypatch.setattr(generator,'_merge_states',{})
    pipe=SimpleNamespace(transformer=tf)
    generator._merge_sliders(pipe,'ff-cpu-test',[component])
    with torch.no_grad():ordinary=tf(**args)[0]
    assert torch.equal(ordinary,differentiable)
    generator._merge_sliders(pipe,'ff-cpu-test',[])
    assert wrapper.base_unchanged()
    with torch.no_grad():assert torch.equal(tf(**args)[0],baseline)


def test_format_requires_zero_attention_and_exact_block_shapes(tmp_path):
    import torch
    from safetensors.torch import save_file
    from conceptmod.textsliders.reward_game.ff_artifact import STRUCTURE,checkpoint
    from conceptmod.textsliders.reward_game.core import write,sha,IntegrityError
    tensors={}
    for layer in range(36):
        dims={**{f'attn-{n}':(2048,2048) for n in ('to_q','to_k','to_v','to_out-0')},'ff_in':(16384,2048),'ff_out':(2048,8192)}
        for projection,(out_dim,in_dim) in dims.items():
            name=f'lora_unet-transformer_blocks-{layer}-{projection}'
            tensors[name+'.lora_up.weight']=torch.zeros(out_dim,8)
            tensors[name+'.lora_down.weight']=torch.zeros(8,in_dim)
            tensors[name+'.alpha']=torch.tensor(8.)
    path=tmp_path/'ff.safetensors';prompt=tmp_path/'prompts.json';write(prompt,dict(caption='Dry acoustic kit with warm bass.'))
    def save():
        save_file(tensors,str(path));write(path.with_suffix('.json'),dict(STRUCTURE,weights_sha256=sha(path),prompts_file=str(prompt)))
    save();assert checkpoint(path)['tensor_count']==648
    key='lora_unet-transformer_blocks-0-attn-to_q.lora_up.weight';tensors[key][0,0]=.01;save()
    with pytest.raises(IntegrityError,match='zero attention'):
        checkpoint(path)
