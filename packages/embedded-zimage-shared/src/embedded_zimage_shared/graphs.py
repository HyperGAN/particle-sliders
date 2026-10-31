"""The historical native attention LoRA delta, executed by HyperGAN/HNDL.

This is ordinary rank16/alpha16 adapter GAN, not a routed particle bank.
All leading token/spatial axes are independent native Linear coordinates.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import nn
from particle_sliders.fibo_hndl import verified_builder
from particle_sliders.lifecycle import _restore_all
from particle_sliders.native_lora import lora_target_plan

GRAPH = 'x = linear(value, $rank, bias=False, name="down")\nlinear(x, $output_width, bias=False, name="up")\n'
GRAPH_SHA256 = hashlib.sha256(GRAPH.encode()).hexdigest()


def targets(model):
    plan = lora_target_plan(model, ['Attention'], prefix='lora_unet', delimiter='-', train_method='full')
    if not plan or any(type(m) is not nn.Linear for _, m in plan):
        raise ValueError('Z-Image requires the exact native Attention Linear target plan')
    expected = ('-to_q', '-to_k', '-to_v', '-to_out-0')
    if any(not name.endswith(expected) for name, _ in plan):
        raise ValueError('unsupported native Z-Image attention projection')
    return plan


class Projection(nn.Module):
    def __init__(self, host, builder, identity):
        super().__init__()
        self.input_width, self.output_width = host.in_features, host.out_features
        self.graph = builder(GRAPH, input_shape={'value': ['B', self.input_width]},
            output_shape=['B', self.output_width], parameters={'rank': 16, 'output_width': self.output_width})
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=1)
        nn.init.zeros_(self.graph['up'].weight)
        self._identity = dict(deepcopy(identity), graph_sha256=GRAPH_SHA256, rank=16, alpha=16,
            input_shape=['B', self.input_width], output_shape=['B', self.output_width],
            native_axes='preserve all leading axes; final axis is the native Linear feature',
            plan_sha256=hashlib.sha256(str(self.graph.plan).encode()).hexdigest(),
            build_receipt=dict(self.graph.build_receipt))

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, value):
        dtype = self.graph['down'].weight.dtype
        x = value.to(device=self.graph['down'].weight.device, dtype=dtype)
        return self.graph(value=x.reshape(-1, self.input_width)).reshape(*value.shape[:-1], self.output_width).to(value)


class ZImageLoRA(nn.Module):
    """Own only new HNDL adapter leaves and reversible public forward hooks."""
    def __init__(self, model):
        super().__init__()
        builder, identity = verified_builder()  # Always precedes native mutation.
        plan = targets(model)
        self.names = tuple(name for name, _ in plan)
        self.projections = nn.ModuleDict({name: Projection(host, builder, identity) for name, host in plan})
        self.scale = 0.
        self._handles = []
        try:
            for name, host in plan:
                projection = self.projections[name]
                def apply(module, args, output, projection=projection):
                    if self.scale == 0:
                        return output
                    return output + projection(args[0]) * self.scale
                self._handles.append(host.register_forward_hook(apply))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('finite adapter strength required')
        previous = self.scale
        self.scale = value
        try:
            yield self
        finally:
            self.scale = previous

    def close(self):
        handles, self._handles = self._handles, []
        _restore_all([handle.remove for handle in handles])
