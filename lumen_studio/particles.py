"""Independent routed rank-8 branches, summed on the same linear input."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .contracts import canonical, digest, file_hash, normalized_strengths
from .vendor.reference import RoutedMLP

FORMAT = "anima-turbo-routed-particles-v1"
ARCHITECTURE = dict(rank=8, particles=128, particle_dim=4, router_width=16, branch_width=48,
                    targets=["to_q", "to_k", "to_v", "to_out.0"])


class Branch(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.down = nn.Linear(inputs, 8, bias=False)
        self.routed = RoutedMLP(8, 8, z_dim=4, width=48, router_width=16)
        self.up = nn.Linear(8, outputs, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** .5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, particles):
        return self.up(self.routed(self.down(x.float()), particles))


class ParticleAdapter(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.names = [name for name, mod in transformer.named_modules()
                      if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in ARCHITECTURE["targets"])]
        if not self.names:
            raise ValueError("No Anima attention targets found")
        modules = dict(transformer.named_modules())
        self.branches = nn.ModuleList([Branch(modules[n].in_features, modules[n].out_features) for n in self.names])

    def export(self, path, *, model_identity, normalization, manifest_hash, step, ema=None):
        if not model_identity or not normalization or len(manifest_hash) != 64:
            raise ValueError("Native exports require model, normalization and manifest provenance")
        metadata = dict(format=FORMAT, architecture=ARCHITECTURE, targets=self.names,
                        model_identity=model_identity, normalization=normalization,
                        manifest_sha256=manifest_hash, step=step, weights="ema" if ema is not None else "live")
        state = self.state_dict() if ema is None else ema
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Exports are immutable; use a new checkpoint path")
        tmp = path.with_suffix(".tmp")
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(tmp),
                  metadata={"anima": canonical(metadata)})
        tmp.replace(path)
        return file_hash(path)

    def load_export(self, path, *, model_identity):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = json.loads(f.metadata()["anima"])
        if meta["format"] != FORMAT or meta["architecture"] != ARCHITECTURE or meta["targets"] != self.names:
            raise ValueError("Incompatible particle architecture")
        if meta["model_identity"] != model_identity:
            raise ValueError("Checkpoint belongs to a different model runtime")
        self.load_state_dict(load_file(str(path)), strict=True)
        return meta


class Mixer:
    def __init__(self, transformer):
        self.adapters = {}
        self.strengths = {}
        self.handles = []
        self.transformer = transformer

    def add(self, name, adapter):
        if name in self.adapters:
            raise ValueError(f"Duplicate adapter {name}")
        modules = dict(self.transformer.named_modules())
        if self.adapters and adapter.names != next(iter(self.adapters.values())).names:
            raise ValueError("Adapter targets do not agree")
        self.adapters[name] = adapter
        self.strengths[name] = 0.
        if not self.handles:
            for index, target in enumerate(adapter.names):
                def hook(module, args, output, index=index):
                    active = [(name, self.strengths.get(name, 0.)) for name in sorted(self.adapters)
                              if self.strengths.get(name, 0.) != 0.]
                    if not active:
                        return output  # Exact base, including NaNs and signed zero.
                    delta = None
                    for key, scale in active:
                        a = self.adapters[key]
                        part = a.branches[index](args[0], a.particles) * scale
                        delta = part if delta is None else delta + part
                    return output + delta.to(output.dtype)
                self.handles.append(modules[target].register_forward_hook(hook))

    def set_mix(self, mix, energy):
        strengths = normalized_strengths(mix, energy)
        if any(value and name not in self.adapters for name, value in strengths.items()):
            raise ValueError("Select a checkpoint for every active slider")
        self.strengths = strengths
        return strengths

    @contextmanager
    def scales(self, values):
        before = self.strengths.copy()
        self.strengths = values.copy()
        try:
            yield
        finally:
            self.strengths = before

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.adapters.clear()


def init_ema(adapter):
    return {k: v.detach().clone() for k, v in adapter.state_dict().items()}


@torch.no_grad()
def update_ema(ema, adapter):
    for k, v in adapter.state_dict().items():
        ema[k].lerp_(v, .005)
