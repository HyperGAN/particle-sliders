"""Native routed residual head for fixed Bonsai readouts.

No GGUF loader, server, historical game or model weight access is included.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .reference import RoutedMLP

HIDDEN = 5120
RANK = 8
ALPHA = 8.0
FORMAT = "conceptmod-bonsai-gguf-particle-v1"


class BonsaiParticleHead(nn.Module):
    """Routed low-rank residual on the frozen 5120-dim readout.

    ``up`` starts at zero so a fresh head is the base behavior;
    ``multiplier`` 0 bypasses it exactly. The 128x4 cloud belongs to
    this head exactly once.
    """

    def __init__(self, hidden: int = HIDDEN, rank: int = RANK,
                 alpha: float = ALPHA) -> None:
        super().__init__()
        if int(rank) != RANK or float(alpha) != ALPHA:
            raise ValueError("The Bonsai particle format pins rank/alpha 8")
        if int(hidden) != HIDDEN:
            raise ValueError(
                f"Bonsai readout width is {HIDDEN}, got {hidden}"
            )
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 0.0
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.bridge = RoutedMLP(rank, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0:
            return x
        host_device, host_dtype = x.device, x.dtype
        features = self.down(x.to(device=self.down.weight.device,
                                   dtype=self.down.weight.dtype))
        routed = self.bridge(features, self.particles)
        delta = self.up(routed).to(device=host_device, dtype=host_dtype)
        return x + delta * (self.multiplier * self.scale)

    @contextmanager
    def scaled(self, scale: float):
        previous = self.multiplier
        self.multiplier = float(scale)
        try:
            yield
        finally:
            self.multiplier = previous

    def save(self, path, metadata):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            hidden=HIDDEN, particles=128, particle_dim=4,
            bridge_width=48, router_width=16,
        )
        state = self.state_dict()
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid Bonsai particle-head export")
        save_file(
            {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
            str(path), metadata={"conceptmod": json.dumps(record, sort_keys=True)},
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        required = dict(format=FORMAT, rank=8, alpha=8.0, hidden=HIDDEN,
                        particles=128, particle_dim=4,
                        bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()):
            raise ValueError("Incompatible Bonsai routed-particle checkpoint")
        state = load_file(str(path), device="cpu")
        expected = {"particles": (128, 4), "down.weight": (RANK, HIDDEN),
                    "up.weight": (HIDDEN, RANK)}
        for branch, inputs, outputs, width in (("router", RANK, 4, 16),
                                               ("net", RANK + 4, RANK, 48)):
            sizes = [inputs, width, width, width, outputs]
            for index, (a, b) in enumerate(zip(sizes, sizes[1:])):
                expected[f"bridge.{branch}.{index * 2}.weight"] = (b, a)
                expected[f"bridge.{branch}.{index * 2}.bias"] = (b,)
        if set(state) != set(expected) or any(
            tuple(state[k].shape) != shape for k, shape in expected.items()
        ):
            raise ValueError("Incomplete Bonsai routed-particle tensors")
        if any(t.dtype != torch.float32 for t in state.values()):
            raise ValueError("Bonsai routed-particle tensors must be float32")
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite Bonsai routed-particle tensors")
        with torch.random.fork_rng(devices=[]):
            network = cls().float()
            network.load_state_dict(state, strict=True)
        return network, record
