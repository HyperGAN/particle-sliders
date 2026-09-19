"""Unselected speed prototype: cache ordinary weights, retain per-call backward.

The active experiments use the separately frozen MergedForward implementation.
This prototype requires an actual-host numerical/gradient audit before use.
Only first-order derivatives are supported.
"""
import torch
import torch.nn.functional as F
from .merged_forward import merged_weight


class _CachedWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx,up,down,value,up_cpu,down_cpu,scale):
        ctx.save_for_backward(up,down,up_cpu,down_cpu)
        ctx.scale=scale
        return value.view_as(value)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient):
        up,down,up_cpu,down_cpu=ctx.saved_tensors
        # Preserve the original CPU FP32 contraction and BF16 cast derivative.
        # Each projection call owns its own gradient contraction, avoiding a
        # shared BF16 weight-gradient sum across different CFG/model calls.
        scaled=gradient.cpu().float()*ctx.scale
        grad_up=(scaled@down_cpu.T).to(up) if ctx.needs_input_grad[0] else None
        grad_down=(up_cpu.T@scaled).to(down) if ctx.needs_input_grad[1] else None
        return grad_up,grad_down,None,None,None,None


class CachedMergedForward:
    def __init__(self,network):
        self.network=network;self.original={};self.pristine={};self.style_sum={}
        self.multiplier=1.;self.style_version=0;self.cache={};self.attached=False
        for lora in network.unet_loras:
            host=network.hosts[lora.lora_name]
            if host.weight.requires_grad:raise ValueError('Base must be frozen before merged training')
            self.pristine[lora.lora_name]=host.weight.detach().cpu().clone()

    def styles(self,networks):
        self.style_sum={};self.style_version+=1;self.cache={}
        with torch.no_grad():
            for network,multiplier in networks:
                if multiplier==0:continue
                for lora in network.unet_loras:
                    name=lora.lora_name
                    delta=(lora.lora_up.weight.detach().float().cpu()@lora.lora_down.weight.detach().float().cpu())*(multiplier*float(lora.scale))
                    self.style_sum[name]=delta if name not in self.style_sum else self.style_sum[name]+delta

    def weight(self,lora,device):
        name=lora.lora_name;up=lora.lora_up.weight;down=lora.lora_down.weight
        scale=self.multiplier*float(lora.scale)
        key=(id(up),up._version,id(down),down._version,str(device),scale,self.style_version)
        entry=self.cache.get(name)
        if entry is None or entry['key']!=key:
            with torch.no_grad():
                up_cpu=up.detach().float().cpu().clone();down_cpu=down.detach().float().cpu().clone()
                value=merged_weight(self.pristine[name],self.style_sum.get(name),up_cpu,down_cpu,
                    multiplier=self.multiplier,scale=float(lora.scale)).to(device)
            entry=dict(key=key,value=value,up_cpu=up_cpu,down_cpu=down_cpu)
            self.cache[name]=entry
        return _CachedWeight.apply(up,down,entry['value'],entry['up_cpu'],entry['down_cpu'],scale)

    def attach(self):
        if self.attached:raise RuntimeError('Merged forward is already attached')
        for lora in self.network.unet_loras:
            name=lora.lora_name;host=self.network.hosts[name];self.original[name]=host.forward
            def forward(x,lora=lora,host=host):
                return F.linear(x,self.weight(lora,x.device),host.bias)
            host.forward=forward
        self.attached=True
        return self

    def detach(self):
        if self.attached:
            for name,original in self.original.items():self.network.hosts[name].forward=original
            self.attached=False
        self.cache={}

    def base_unchanged(self):
        return all(torch.equal(self.network.hosts[name].weight.detach().cpu(),value) for name,value in self.pristine.items())
