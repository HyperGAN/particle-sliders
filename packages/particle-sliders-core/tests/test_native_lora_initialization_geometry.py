"""Relate scoped down-projection direction to the native LoRA first gradient."""

import torch
from torch import nn

from particle_sliders import ComponentInitialization, initialize_fresh_components
from particle_sliders.native_lora import LoRANetwork


def _fresh_gradient(seed):
    with torch.random.fork_rng(devices=[]):
        host = nn.Linear(4, 1, bias=False).requires_grad_(False)
        with torch.no_grad():
            host.weight.fill_(.25)
        network = LoRANetwork(host, rank=1, alpha=1, multiplier=0,
                              target_plan=[("edit", host)])
    adapter = network.unet_loras[0]
    before = host.weight.detach().clone()
    receipt = initialize_fresh_components(
        {"edit.down": ComponentInitialization(adapter.lora_down, "deterministic_orthogonal",
                                               {"weight": seed})},
        cohort_id="native-lora-initialization-fixture",
        source_identity={"host": "linear-4-to-1"},
        completed_steps=0, restored_from=None, optimizers=(),
    )
    token = torch.tensor([[0., 0., 0., 1.]])
    off = host(token)
    with network:
        on = host(token)
        assert torch.equal(on, off)
        (.5 * (on - (off + 1.)).square().sum()).backward()

    down = adapter.lora_down.weight.detach()
    assert torch.equal(host.weight, before)
    assert host.weight.grad is None
    assert torch.count_nonzero(adapter.lora_up.weight) == 0
    assert torch.count_nonzero(adapter.lora_down.weight.grad) == 0
    torch.testing.assert_close(adapter.lora_up.weight.grad, -down[:, 3:4])
    return down, adapter.lora_up.weight.grad.abs().item(), receipt.sha256


def test_fresh_projection_direction_changes_the_first_gradient_at_equal_host_scale():
    weak, weak_gradient, weak_receipt = _fresh_gradient(3)
    strong, strong_gradient, strong_receipt = _fresh_gradient(17)
    torch.testing.assert_close(weak.norm(), strong.norm())
    assert weak_receipt != strong_receipt
    assert strong_gradient > 2. * weak_gradient > 0.
