"""YuE2 routed AR architecture; no trainer, model download or audio renderer.

Adapted from HyperGAN/yue2-particle-sliders b33794d81b92c125891a011531f55554d65e2082.
Copyright (c) 2026 HyperGAN. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn

from .reference import RoutedMLP


def attention_targets(model):
    if getattr(model.config, 'model_type', None) != 'yue2':
        raise ValueError('genuine YuE2 AR configuration required')
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('nonfinite YuE2 base parameter before attachment')
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None or len(layers) != model.config.num_hidden_layers:
        raise ValueError('YuE2 decoder layout differs from configuration')
    targets = {}
    for index, layer in enumerate(layers):
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            module = getattr(getattr(layer, 'self_attn', None), name, None)
            if not isinstance(module, nn.Linear) or not all(torch.isfinite(p).all() for p in module.parameters()):
                raise ValueError('missing or nonfinite YuE2 attention projection')
            if isinstance(getattr(module.forward, '__self__', None), RoutedProjection):
                raise ValueError('a routed YuE2 adapter is already attached')
            targets[f'model.layers.{index}.self_attn.{name}'] = module
    if not targets:
        raise ValueError('nonempty YuE2 decoder required')
    return targets


class RoutedProjection(nn.Module):
    def __init__(self, module, particles):
        super().__init__()
        self.lora_down = nn.Linear(module.in_features, 8, bias=False)
        self.lora_up = nn.Linear(8, module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        self.bridge = RoutedMLP(8, 8)
        self.multiplier = 0.0
        object.__setattr__(self, '_particles', particles)
        self.org_forward = module.forward

    def forward(self, x):
        base = self.org_forward(x)
        if self.multiplier == 0:
            return base
        features = self.lora_down(x.to(device=self.lora_down.weight.device, dtype=torch.float32))
        delta = self.lora_up(self.bridge(features, self._particles)).to(device=x.device, dtype=x.dtype)
        return base + delta * self.multiplier  # fixed alpha/rank = 8/8


class ParticleSlider(nn.Module):
    """One real shared 128x4 bank, attached after complete target validation."""
    def __init__(self, model):
        super().__init__()
        targets = attention_targets(model)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.adapters = nn.ModuleDict({name.replace('.', '-'): RoutedProjection(module, self.particles)
                                       for name, module in targets.items()})
        self.to(device=device, dtype=torch.float32)
        model.requires_grad_(False)
        for name, module in targets.items():
            module.forward = self.adapters[name.replace('.', '-')].forward

    @contextmanager
    def scaled(self, scale):
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError('finite adapter scale required')
        previous = [module.multiplier for module in self.adapters.values()]
        try:
            for module in self.adapters.values():
                module.multiplier = scale
            yield self
        finally:
            for module, value in zip(self.adapters.values(), previous):
                module.multiplier = value


class YuE2ModelBackend:
    """Wrap an explicitly supplied native host; never construct/load weights."""
    def __init__(self, model):
        self.model = model

    def hidden(self, ids, *, checkpointing=False):
        if checkpointing:
            raise ValueError('shared adapter owns captured-scale checkpointing')
        device = next(self.model.parameters()).device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        positions = torch.arange(len(ids), device=device)[None]
        return self.model.model(input_ids=tokens, position_ids=positions, use_cache=False)[0]
