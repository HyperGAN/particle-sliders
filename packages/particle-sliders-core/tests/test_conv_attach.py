"""Bind stamp.attach to a synthetic nn.Conv1d and step FormulationGame.

No checkpoint and no weight download. One step shows the convolution's
pooled features reach the existing RoutedMLP inside FormulationGame.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from particle_sliders import (
    ConvAttachment,
    FormulationGame,
    RoutedMLP,
    winning_formulation,
)


class SyntheticConv(nn.Module):
    """One convolution. No nn.Linear and no loaded weights."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(6, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.proj(x)


def _declared(stamp):
    declared = {**stamp.as_dict(), "g_lr": 2e-5, "adv_batch": 2}
    stamp.require(declared)
    return declared


def test_conv1d_features_reach_formulation_game_through_routed_mlp():
    torch.manual_seed(7)
    stamp = winning_formulation()
    rank = int(stamp.spec["adapter_rank"])
    host = SyntheticConv()
    assert not any(isinstance(module, nn.Linear) for module in host.modules())
    x = torch.randn(2, 6, 5)
    with torch.no_grad():
        unbound = host(x).clone()
    host_weight = host.proj.weight.detach().clone()

    attachment = stamp.attach(host)
    assert isinstance(attachment, ConvAttachment)
    assert len(attachment.bindings) == 1
    binding = attachment.bindings[0]
    assert binding.down.kernel_size == (3,)
    assert binding.down.in_channels == 6
    assert binding.up.kernel_size == (1,)
    assert binding.up.out_channels == 4
    assert isinstance(attachment.bridge, RoutedMLP)
    assert attachment.bridge.router[0].in_features == rank
    assert attachment.bridge.net[-1].out_features == rank
    assert type(stamp.bridge()) is RoutedMLP
    for name, module in host.named_modules():
        if isinstance(module, nn.Linear):
            assert name.startswith("_stamp_conv_attachment")

    attachment.set_scale(0.0)
    with torch.no_grad():
        assert torch.equal(host(x), unbound)
    attachment.set_scale(1.0)
    with torch.no_grad():
        assert torch.equal(host(x), unbound)

    features = attachment.features(x)
    assert features.shape == (2, rank)
    assert features.requires_grad
    assert torch.isfinite(features).all()

    seen = []
    real_forward = attachment.bridge.forward

    def spy(value, particles):
        seen.append(value.detach().clone())
        return real_forward(value, particles)

    attachment.bridge.forward = spy
    with torch.no_grad():
        host(x)
    assert seen[-1].shape == (2, rank)
    assert torch.allclose(seen[-1], features.detach())

    extra = attachment.extra_generator()
    extra_ids = {id(param) for param in extra}
    assert id(attachment.particles) not in extra_ids
    assert id(host.proj.weight) not in extra_ids
    assert all(id(param) not in extra_ids for param in attachment.bridge.parameters())
    assert id(binding.down.weight) in extra_ids

    game = FormulationGame(
        stamp,
        _declared(stamp),
        seed=7,
        device="cpu",
        bridge=attachment.bridge,
        particles=attachment.particles,
        extra_generator=extra,
    )
    assert game.bridge is attachment.bridge
    seen.clear()
    stats = game.step(1, features)
    assert seen[-1].shape == (2, rank)
    assert torch.allclose(seen[-1], features.detach())
    assert torch.isfinite(torch.tensor([stats["d_loss"], stats["g_loss"], stats["vic"]])).all()
    assert binding.down.weight.grad is not None
    assert torch.isfinite(binding.down.weight.grad).all()
    assert binding.down.weight.grad.abs().sum() > 0
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in attachment.bridge.parameters()
    )
    assert torch.equal(host.proj.weight, host_weight)


def test_several_convs_share_particles_and_keep_their_own_routed_mlp():
    torch.manual_seed(3)
    stamp = winning_formulation()
    rank = int(stamp.spec["adapter_rank"])

    class TwoConvs(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Conv1d(3, 5, kernel_size=3, padding=1)
            self.right = nn.Conv1d(5, 2, kernel_size=1)
            self.skip = nn.Linear(5, 5)

        def forward(self, x):
            hidden = self.left(x)
            return self.right(hidden), self.skip(hidden.transpose(1, 2)).transpose(1, 2)

    host = TwoConvs()
    linear_forward = host.skip.forward.__func__
    attachment = stamp.attach(host)
    assert len(attachment.bindings) == 2
    assert attachment.bindings[0].bridge is not attachment.bindings[1].bridge
    assert isinstance(attachment.bindings[0].bridge, RoutedMLP)
    assert isinstance(attachment.bindings[1].bridge, RoutedMLP)
    assert attachment.bindings[0]._particles.param is attachment.particles
    assert attachment.bindings[1]._particles.param is attachment.particles
    assert host.skip.forward.__func__ is linear_forward
    x = torch.randn(2, 3, 6)
    left = attachment.bindings[0].features(x)
    assert left.shape == (2, rank)
    with torch.no_grad():
        hidden = host.left(x)
    right = attachment.bindings[1].features(hidden)
    assert right.shape == (2, rank)
    with pytest.raises(ValueError, match="single bound Conv1d"):
        attachment.features(x)


def test_attach_rejects_non_conv_and_a_second_bind():
    stamp = winning_formulation()
    with pytest.raises(RuntimeError, match="no nn.Conv1d"):
        stamp.attach(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="groups"):
        stamp.attach(nn.Conv1d(4, 4, kernel_size=1, groups=4))

    host = SyntheticConv()
    stamp.attach(host)
    with pytest.raises(RuntimeError, match="already"):
        stamp.attach(host)
    with pytest.raises(RuntimeError, match="already"):
        stamp.attach(host.proj)
