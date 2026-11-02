"""Native full-block export with only feed-forward factors trainable."""
from .core import IntegrityError


def make(transformer, device='cuda:0'):
    from app.lora_runtime import LoRANetwork
    network=LoRANetwork(transformer,rank=8,alpha=8.,multiplier=1.,
        target_replace=['MiniMaxMusic3TransformerBlock'],prefix='lora_unet',
        delimiter='-',train_method='full',attach=False).to(device)
    network.requires_grad_(False)
    selected=[]
    for module in network.unet_loras:
        if module.lora_name.endswith(('-ff_in','-ff_out')):
            module.lora_down.requires_grad_(True)
            module.lora_up.requires_grad_(True)
            selected.append(module.lora_name)
    if len(network.unet_loras)!=216 or len(selected)!=72:
        raise IntegrityError('Unexpected native feed-forward block topology')
    return network


def check_frozen_attention(network, before=None):
    import torch
    for module in network.unet_loras:
        if '-attn-' not in module.lora_name:
            continue
        if bool(module.lora_up.weight.count_nonzero()):
            raise IntegrityError('Attention delta changed in feed-forward-only experiment')
        for part in ('lora_up','lora_down'):
            tensor=getattr(module,part).weight
            if tensor.requires_grad or tensor.grad is not None:
                raise IntegrityError('Frozen attention factor received a gradient')
            key=module.lora_name+'.'+part+'.weight'
            if before is not None and not torch.equal(tensor.detach().cpu(),before[key]):
                raise IntegrityError('Frozen attention factor bytes changed')
    return True
