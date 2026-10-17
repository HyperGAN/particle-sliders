"""Native routed TinyQwen adapter, extracted from reviewed source fe54d05.

The frozen host and native adapter are independent of historical training games.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .reference import RoutedMLP
from .tiny_llm_backend import ATTN_CLASS_NAMES, LORA_LINEAR_NAMES, ArchitectureMismatch

FORMAT = "conceptmod-tiny-llm-particle-v1"


def _adapter_key(dotted: str) -> str:
    return f"lora_tiny-{dotted}".replace(".", "-")


class TinyParticleProjection(nn.Module):
    """One routed low-rank branch on a frozen attention Linear.

    Mirrors ``ParticleProjection``: ``down`` is kaiming, ``up`` starts at
    zero (base preserved at init), one shared 128x4 cloud, per-projection
    router + MLP, multiplier 0 bypasses the host exactly.
    """

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        particles: nn.Parameter,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.lora_name = name
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 0.0
        host_kwargs: dict = {}
        if hasattr(module, "weight"):
            host_kwargs["device"] = module.weight.device
            host_kwargs["dtype"] = module.weight.dtype
        self.lora_down = nn.Linear(module.in_features, rank, bias=False, **host_kwargs)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False, **host_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        # The cloud belongs to the root exactly once, not to every projection.
        object.__setattr__(self, "_particles", particles)
        self.bridge = RoutedMLP(rank, rank)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0:
            return self.org_forward(x)
        features = self.lora_down(x.to(device=self.lora_down.weight.device,
                                       dtype=self.lora_down.weight.dtype))
        routed = self.bridge(features, self._particles)
        delta = self.lora_up(routed).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class TinyParticleSlider(nn.Module):
    """Wrap attention ``q/k/v/o`` Linears only, sharing one 128x4 cloud."""

    def __init__(self, model: nn.Module, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if int(rank) != 8 or float(alpha) != 8.0:
            raise ValueError("The tiny particle format pins rank/alpha 8")
        targets = _attention_targets(model)
        if any(
            isinstance(getattr(m.forward, "__self__", None), TinyParticleProjection)
            for m in targets.values()
        ):
            raise ValueError("A tiny particle slider is already attached")
        self.rank, self.alpha = int(rank), float(alpha)
        self.target_names = list(targets)
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.adapters = nn.ModuleDict()
        for name, module in targets.items():
            key = _adapter_key(name)
            adapter = TinyParticleProjection(key, module, self.particles,
                                             self.rank, self.alpha)
            self.adapters[key] = adapter
        host_dtype = _module_param_dtype(model)
        if host_dtype is not None:
            self.to(device=device, dtype=host_dtype)

    @contextmanager
    def scaled(self, scale: float):
        previous = [a.multiplier for a in self.adapters.values()]
        for adapter in self.adapters.values():
            adapter.multiplier = float(scale)
        try:
            yield
        finally:
            for adapter, value in zip(self.adapters.values(), previous):
                adapter.multiplier = value

    def save(self, path, metadata):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            targets=self.target_names,
            particles=128, particle_dim=4, bridge_width=48, router_width=16,
        )
        state = self.state_dict()
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid tiny particle-slider export")
        save_file(
            {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
            str(path), metadata={"conceptmod": json.dumps(record, sort_keys=True)},
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, model, path, *, adapter_dtype=None):
        """Validate an export before attaching, optionally retaining its FP32 leaves.

        The legacy default follows the host dtype. PG0.9 callers explicitly use
        ``adapter_dtype=torch.float32`` so a BF16 host cannot round the saved
        FP32 weights before the adapter is promoted back to FP32.
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        if adapter_dtype not in (None, torch.float32):
            raise ValueError("adapter_dtype must be None or torch.float32")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        required = dict(format=FORMAT, rank=8, alpha=8.0, particles=128,
                        particle_dim=4, bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()):
            raise ValueError("Incompatible tiny routed-particle checkpoint")
        if record.get("targets") != list(_attention_targets(model)):
            raise ValueError("Tiny particle-slider targets differ")
        state = load_file(str(path), device="cpu")
        expected = {"particles": (128, 4)}
        for name, module in _attention_targets(model).items():
            prefix = "adapters." + _adapter_key(name)
            expected.update({
                prefix + ".lora_down.weight": (8, module.in_features),
                prefix + ".lora_up.weight": (module.out_features, 8),
            })
            for branch, inputs, outputs, width in [("router", 8, 4, 16),
                                                   ("net", 12, 8, 48)]:
                sizes = [inputs, width, width, width, outputs]
                for j, (a, b) in enumerate(zip(sizes, sizes[1:])):
                    expected[f"{prefix}.bridge.{branch}.{j * 2}.weight"] = (b, a)
                    expected[f"{prefix}.bridge.{branch}.{j * 2}.bias"] = (b,)
        if set(state) != set(expected) or any(
            tuple(state[k].shape) != shape for k, shape in expected.items()
        ):
            raise ValueError("Incomplete tiny routed-particle tensors")
        if any(not t.is_floating_point() or
               (adapter_dtype is not None and t.dtype != adapter_dtype)
               for t in state.values()):
            raise ValueError("Invalid tiny routed-particle tensor dtype")
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite tiny routed-particle tensors")
        network = cls(model)
        if adapter_dtype is not None:
            network.to(dtype=adapter_dtype)
        network.load_state_dict(state, strict=True)
        return network, record


def _attention_targets(model: nn.Module) -> dict[str, nn.Linear]:
    targets: dict[str, nn.Linear] = {}
    seen: set[int] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ not in ATTN_CLASS_NAMES:
            continue
        for child_name, child in module.named_modules():
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in LORA_LINEAR_NAMES:
                continue
            if id(child) in seen:
                continue
            seen.add(id(child))
            targets[f"{name}.{child_name}" if name else child_name] = child
    if not targets:
        raise ArchitectureMismatch(
            "no attention q/k/v/o Linears found; expected Qwen3-shaped "
            f"{ATTN_CLASS_NAMES} hosts"
        )
    return targets


def _module_param_dtype(module: nn.Module):
    for param in module.parameters():
        return param.dtype
    return None
