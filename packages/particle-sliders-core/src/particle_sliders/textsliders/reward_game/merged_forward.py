"""Differentiable ordinary merger: pristine + summed FP32 deltas, one cast.

Factor contractions and accumulation deliberately run on CPU, as deployment
does. Autograd follows the casts to FP32 factors. As with ordinary mixed dtype
training, gradients through the BF16 cast use PyTorch's cast derivative.
No base parameter is modified or included in an optimizer.
"""
import torch
import torch.nn.functional as F


def merged_weight(pristine, style_sum, up, down, multiplier=1., scale=1.):
    delta=(up.float().cpu() @ down.float().cpu()) * (multiplier*scale)
    total=delta if style_sum is None else style_sum + delta
    return (pristine.float()+total).to(dtype=pristine.dtype)


class MergedForward:
    def __init__(self, network):
        self.network=network;self.original={};self.pristine={};self.style_sum={};self.multiplier=1.
        for lora in network.unet_loras:
            host=network.hosts[lora.lora_name]
            if host.weight.requires_grad:raise ValueError('Base must be frozen before merged training')
            self.pristine[lora.lora_name]=host.weight.detach().cpu().clone()
        self.attached=False

    def styles(self, networks):
        """Ordered (ordinary style network, multiplier) pairs, already audited."""
        self.style_sum={}
        with torch.no_grad():
            for network,multiplier in networks:
                if multiplier==0:continue
                for lora in network.unet_loras:
                    name=lora.lora_name
                    delta=(lora.lora_up.weight.detach().float().cpu() @ lora.lora_down.weight.detach().float().cpu())*(multiplier*float(lora.scale))
                    self.style_sum[name]=delta if name not in self.style_sum else self.style_sum[name]+delta

    def attach(self):
        if self.attached:raise RuntimeError('Merged forward is already attached')
        for lora in self.network.unet_loras:
            name=lora.lora_name;host=self.network.hosts[name];self.original[name]=host.forward
            def forward(x,lora=lora,host=host,name=name):
                weight=merged_weight(self.pristine[name],self.style_sum.get(name),lora.lora_up.weight,
                                     lora.lora_down.weight,self.multiplier,float(lora.scale)).to(x.device)
                return F.linear(x,weight,host.bias)
            host.forward=forward
        self.attached=True
        return self

    def detach(self):
        if self.attached:
            for name,original in self.original.items():self.network.hosts[name].forward=original
            self.attached=False

    def base_unchanged(self):
        return all(torch.equal(self.network.hosts[name].weight.detach().cpu(),value) for name,value in self.pristine.items())
