"""Routed low-rank projection shared by image and historical tiny hosts."""
import torch
from torch import nn
from .reference import RoutedMLP


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
