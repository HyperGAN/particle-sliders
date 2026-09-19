"""Numerical checks for an unselected, first-order training-speed prototype."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from conceptmod.textsliders.reward_game.merged_forward import MergedForward
from conceptmod.textsliders.reward_game.cached_merged_forward import CachedMergedForward


def network(seed=93, zero=False, frozen=False):
    torch.manual_seed(seed)
    result=nn.Module()
    result.unet_loras=nn.ModuleList()
    result.hosts={}
    for index in range(2):
        lora=nn.Module()
        lora.lora_name=f'projection-{index}'
        lora.scale=.75
        lora.lora_up=nn.Linear(3,12,bias=False)
        lora.lora_down=nn.Linear(12,3,bias=False)
        if zero:nn.init.zeros_(lora.lora_up.weight)
        if frozen:lora.requires_grad_(False)
        result.unet_loras.append(lora)
        result.hosts[lora.lora_name]=nn.Linear(12,12,bias=True).to(torch.bfloat16).requires_grad_(False)
    return result


def values_and_grads(net, wrapper_type, mode):
    wrapper=wrapper_type(net).attach()
    wrapper.styles([(network(101),.9),(network(112),-.3)])
    wrapper.multiplier=1.25
    xs=[];values=[];losses=[]
    try:
        for index in range(3):
            torch.manual_seed(300+index)
            x=torch.randn(2,7,12,dtype=torch.bfloat16,requires_grad=True)
            xs.append(x)
            def forward(value):
                for host in net.hosts.values():value=host(value).tanh()
                return value
            value=checkpoint(forward,x,use_reentrant=False) if mode=='checkpoint' else forward(x)
            values.append(value.detach().clone())
            loss=(value.float()*torch.randn(value.shape)).sum()
            if mode=='separate':loss.backward()
            else:losses.append(loss)
        if losses:sum(losses).backward()
        grads=[p.grad.clone() if p.grad is not None else None for p in net.parameters()]
        return values,grads,[x.grad.clone() for x in xs]
    finally:
        wrapper.detach()
        assert wrapper.base_unchanged()


@pytest.mark.parametrize('mode',['separate','summed','checkpoint'])
@pytest.mark.parametrize('zero,frozen',[(False,False),(True,False),(True,True)])
def test_cached_forward_and_per_call_gradients_match_frozen_merger(mode,zero,frozen):
    torch.set_num_threads(2)
    old=network(zero=zero,frozen=frozen);new=copy.deepcopy(old)
    expected=values_and_grads(old,MergedForward,mode)
    actual=values_and_grads(new,CachedMergedForward,mode)
    for old_group,new_group in zip(expected,actual):
        for left,right in zip(old_group,new_group):
            if left is None:assert right is None
            else:assert torch.equal(left,right),float((left-right).abs().max())


def test_cache_tracks_optimizer_state_load_replacement_styles_and_multiplier():
    torch.set_num_threads(2)
    net=network();wrapper=CachedMergedForward(net).attach()
    old_forwards={name:wrapper.original[name] for name in net.hosts}
    lora=net.unet_loras[0];host=net.hosts[lora.lora_name]
    x=torch.randn(2,12,dtype=torch.bfloat16)
    def compare():
        got=host(x).detach().clone()
        wrapper.detach()
        reference=MergedForward(net).attach()
        reference.styles([(style,scale) for style,scale in styles])
        reference.multiplier=wrapper.multiplier
        try:assert torch.equal(got,host(x).detach())
        finally:reference.detach();wrapper.attach()
    styles=[]
    try:
        one=wrapper.weight(lora,x.device);cached=wrapper.cache[lora.lora_name]['value']
        two=wrapper.weight(lora,x.device)
        assert one.data_ptr()==two.data_ptr()==cached.data_ptr()
        assert one.grad_fn is not two.grad_fn
        host(x).float().square().mean().backward()
        torch.optim.AdamW(net.parameters(),lr=.002).step()
        wrapper.weight(lora,x.device)
        assert wrapper.cache[lora.lora_name]['value'] is not cached
        compare()
        state=copy.deepcopy(net.state_dict())
        for key in state:state[key].add_(.03)
        net.load_state_dict(state)
        compare()
        # assign=True replaces Parameters and can reuse the old version number.
        wrapper.weight(lora,x.device)
        replacement=copy.deepcopy(net.state_dict())
        for key in replacement:replacement[key].sub_(.06)
        net.load_state_dict(replacement,assign=True)
        compare()
        wrapper.multiplier=.35
        compare()
        styles=[(network(204),.8),(network(205),1.1)]
        wrapper.styles(styles)
        compare()
        wrapper.multiplier=0.
        wrapper.styles([]);styles=[]
        compare()
    finally:
        wrapper.detach()
        assert wrapper.base_unchanged()
        assert all(host.forward==old_forwards[name] for name,host in net.hosts.items())


def test_parameter_mutation_before_backward_is_rejected():
    net=network();wrapper=CachedMergedForward(net).attach()
    try:
        value=net.hosts['projection-0'](torch.ones(2,12,dtype=torch.bfloat16))
        with torch.no_grad():net.unet_loras[0].lora_up.weight.add_(.01)
        with pytest.raises(RuntimeError,match='modified by an inplace operation'):
            value.float().sum().backward()
    finally:wrapper.detach()


def test_actual_small_music3_blocks_match_two_optimizer_steps():
    from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    from conceptmod.textsliders.reward_game.ff_network import make,check_frozen_attention
    torch.set_num_threads(2);torch.manual_seed(79)
    original=MiniMaxMusic3Transformer1DModel(in_channels=2,condition_dim=8,num_layers=36,
        num_attention_heads=2,attention_head_dim=4,ff_inner_dim=16,rotary_dim=4,fourier_embedding_dim=4).to(torch.bfloat16).eval().requires_grad_(False)
    reference=[]
    for implementation in (MergedForward,CachedMergedForward):
        tf=copy.deepcopy(original);tf.enable_gradient_checkpointing()
        torch.manual_seed(741);net=make(tf,'cpu');wrapper=implementation(net).attach()
        before={k:v.detach().clone() for k,v in net.state_dict().items()}
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0.)
        results=[]
        try:
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs=[]
                for call in range(2):
                    torch.manual_seed(500+call)
                    args=dict(hidden_states=torch.randn(1,2,6,dtype=torch.bfloat16),
                        encoder_hidden_states=torch.randn(1,6,8,dtype=torch.bfloat16),
                        timestep=torch.ones(1,dtype=torch.bfloat16),return_dict=False)
                    output=tf(**args)[0]
                    outputs.append(output.detach().clone())
                    output.float().square().mean().backward()
                grads={name:p.grad.clone() for name,p in net.named_parameters() if p.grad is not None}
                optimizer.step()
                assert check_frozen_attention(net,before)
                results.append((outputs,grads,{k:v.detach().clone() for k,v in net.state_dict().items()}))
            reference.append(results)
        finally:wrapper.detach();assert wrapper.base_unchanged()
    for old,new in zip(*reference):
        assert all(torch.equal(a,b) for a,b in zip(old[0],new[0]))
        for group in (1,2):
            assert old[group].keys()==new[group].keys()
            for key in old[group]:assert torch.equal(old[group][key],new[group][key]),key
