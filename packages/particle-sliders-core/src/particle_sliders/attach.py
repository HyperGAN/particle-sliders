"""Bind ``winning_formulation()`` onto an ``nn.Conv1d``.

``RoutedMLP`` stays rank-in, rank-out. This module does not build a game and
does not train a checkpoint. ``FormulationGame.step`` still takes features
shaped ``[batch, adapter_rank]``. Pass ``attachment.bridge`` (the existing
``stamp.bridge()``) into that game.
"""
from __future__ import annotations

import torch
from torch import nn


class _ParticleRef:
    """Share one particle Parameter across bindings without registering it twice."""

    def __init__(self, param: nn.Parameter):
        self.param = param


def _host_convs(module: nn.Module) -> list[nn.Conv1d]:
    if isinstance(module, nn.Conv1d):
        _require_bindable(module, module.__class__.__name__)
        return [module]
    found: list[nn.Conv1d] = []
    seen: set[int] = set()
    for name, child in module.named_modules():
        if not isinstance(child, nn.Conv1d):
            continue
        if getattr(child, "_stamp_conv_internal", False):
            continue
        if id(child) in seen:
            continue
        seen.add(id(child))
        label = name or child.__class__.__name__
        _require_bindable(child, label)
        found.append(child)
    if not found:
        present = sorted({item.__class__.__name__ for item in module.modules()})
        raise RuntimeError(
            f"stamp.attach found no nn.Conv1d to bind. Classes present: {present}."
        )
    return found


def _require_bindable(conv: nn.Conv1d, label: str) -> None:
    if getattr(conv, "_stamp_conv_internal", False):
        raise ValueError("stamp.attach does not bind its own down or up convolution")
    if getattr(conv, "_stamp_conv_binding", None) is not None:
        raise RuntimeError(f"{label} is already bound to the stamp")
    if int(conv.groups) != 1:
        raise ValueError(
            f"{label} has groups={conv.groups}; stamp.attach supports nn.Conv1d with groups=1"
        )


def _down_conv(conv: nn.Conv1d, rank: int) -> nn.Conv1d:
    return nn.Conv1d(
        int(conv.in_channels),
        int(rank),
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=False,
        padding_mode=conv.padding_mode,
    )


class Conv1dBinding(nn.Module):
    """One host ``nn.Conv1d`` plus the stamp's ``RoutedMLP``.

    The down convolution copies the host kernel, stride, padding, and dilation
    and emits ``adapter_rank`` channels. Mean over the length axis is
    ``[batch, adapter_rank]``. That vector is the input to the existing
    ``RoutedMLP``. A zero-initialized pointwise convolution writes the routed
    residual back onto the host output. Scale 0 leaves the host convolution
    unchanged.
    """

    def __init__(self, stamp, conv: nn.Conv1d, particles: _ParticleRef):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        if rank < 1:
            raise ValueError("adapter_rank must be positive")
        self.rank = rank
        self.rank_scale = float(stamp.spec["adapter_alpha"]) / float(rank)
        self.scale = 1.0
        self.down = _down_conv(conv, rank)
        self.up = nn.Conv1d(rank, int(conv.out_channels), kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.down._stamp_conv_internal = True
        self.up._stamp_conv_internal = True
        self.bridge = stamp.bridge()
        object.__setattr__(self, "_particles", particles)
        self._org_forward = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled down features, ``[batch, adapter_rank]``, before the bridge."""
        if x.ndim != 3:
            raise ValueError(
                f"Conv1d attach expects input [batch, channels, length], got {tuple(x.shape)}"
            )
        if int(x.shape[1]) != int(self.down.in_channels):
            raise ValueError(
                f"Conv1d attach expected {self.down.in_channels} channels, got {int(x.shape[1])}"
            )
        weight = self.down.weight
        activation = self.down(x.to(device=weight.device, dtype=weight.dtype))
        if activation.shape[-1] < 1:
            raise ValueError("Conv1d attach produced an empty length")
        pooled = activation.mean(dim=-1)
        if pooled.shape != (x.shape[0], self.rank):
            raise RuntimeError(
                f"pooled Conv1d features are {tuple(pooled.shape)}; expected [batch, {self.rank}]"
            )
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._org_forward is None:
            raise RuntimeError("Conv1d binding was not installed")
        base = self._org_forward(x)
        if self.scale == 0.0:
            return base
        routed = self.bridge(self.features(x), self._particles.param)
        if routed.ndim != 2 or int(routed.shape[1]) != self.rank or routed.shape[0] != base.shape[0]:
            raise RuntimeError(
                f"RoutedMLP returned {tuple(routed.shape)}; expected [{base.shape[0]}, {self.rank}]"
            )
        spatial = routed.unsqueeze(-1).expand(-1, -1, base.shape[-1])
        delta = self.up(spatial)
        if delta.shape != base.shape:
            raise RuntimeError(
                f"Conv1d residual shape {tuple(delta.shape)} != host {tuple(base.shape)}"
            )
        return base + delta.to(device=base.device, dtype=base.dtype) * (self.scale * self.rank_scale)

    def install(self, conv: nn.Conv1d) -> None:
        object.__setattr__(self, "_conv", conv)
        self._org_forward = conv.forward
        conv.forward = self.forward
        # A Module assignment would reparent this binding out of the attachment.
        object.__setattr__(conv, "_stamp_conv_binding", self)


class ConvAttachment(nn.Module):
    """Shared particles and one ``RoutedMLP`` per bound ``nn.Conv1d``.

    ``bridge`` is the first binding's ``stamp.bridge()``. ``features`` is
    defined when exactly one convolution was bound. ``FormulationGame`` keeps
    taking that ``[batch, adapter_rank]`` tensor and this ``RoutedMLP``.
    """

    def __init__(self, stamp, convs: list[nn.Conv1d]):
        super().__init__()
        if not convs:
            raise ValueError("stamp.attach needs at least one nn.Conv1d")
        devices = {conv.weight.device for conv in convs}
        dtypes = {conv.weight.dtype for conv in convs}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("stamp.attach requires every Conv1d on the same device and dtype")
        parts = int(stamp.spec["parts"])
        dim = int(stamp.spec["particle_dim"])
        self.particles = nn.Parameter(torch.randn(parts, dim) * 0.02)
        shared = _ParticleRef(self.particles)
        self.bindings = nn.ModuleList(Conv1dBinding(stamp, conv, shared) for conv in convs)
        object.__setattr__(self, "_convs", list(convs))

    @property
    def bridge(self) -> nn.Module:
        """Existing ``RoutedMLP`` for ``FormulationGame(..., bridge=)``."""
        return self.bindings[0].bridge

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, adapter_rank]`` features of the single bound convolution."""
        if len(self.bindings) != 1:
            raise ValueError(
                f"features() is for a single bound Conv1d; this attachment bound {len(self.bindings)}. "
                "Call features() on one binding with that convolution's input."
            )
        return self.bindings[0].features(x)

    def extra_generator(self) -> list[nn.Parameter]:
        """Down/up and any other bridges. The game already owns ``bridge`` and particles."""
        owned = {id(param) for param in self.bridge.parameters()}
        owned.add(id(self.particles))
        return [param for param in self.parameters() if id(param) not in owned]

    def set_scale(self, scale: float) -> None:
        value = float(scale)
        for binding in self.bindings:
            binding.scale = value

    def install(self) -> None:
        for binding, conv in zip(self.bindings, self._convs):
            binding.install(conv)


def bind_conv1d(stamp, module: nn.Module) -> ConvAttachment:
    """Attach ``stamp.bridge()`` to every host ``nn.Conv1d`` and return the binding."""
    if not isinstance(module, nn.Module):
        raise TypeError("stamp.attach expects an nn.Module")
    if getattr(module, "_stamp_conv_attachment", None) is not None and not isinstance(module, nn.Conv1d):
        raise RuntimeError("this module already has a stamp Conv1d attachment")
    convs = _host_convs(module)
    attachment = ConvAttachment(stamp, convs)
    # Stay in the attachment's dtype (float32 unless the caller moves it).
    # A narrower host is cast on the way in and the residual is cast back.
    attachment.to(device=convs[0].weight.device)
    module.add_module("_stamp_conv_attachment", attachment)
    attachment.install()
    return attachment
