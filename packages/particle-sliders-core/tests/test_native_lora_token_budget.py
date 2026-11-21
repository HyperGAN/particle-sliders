"""Exercise the native LoRA prefix envelope at a matched edit budget."""

import pytest
import torch
from torch import nn

from particle_sliders.native_lora import LoRANetwork


@pytest.mark.parametrize("batched", [False, True])
def test_prefix_envelope_concentrates_a_matched_budget_at_the_declared_position(batched):
    with torch.random.fork_rng(devices=[]):
        host = nn.Linear(2, 2, bias=False).requires_grad_(False)
        network = LoRANetwork(host, rank=1, alpha=1, multiplier=0,
                              target_plan=[("edit", host)])
    layer = network.unet_loras[0]
    tokens = torch.tensor([[1., 2.], [1., 4.], [1., 6.], [1., 8.], [1., 10.]])
    if batched:
        tokens = torch.stack((tokens, tokens + torch.tensor([0., 1.])))

    with torch.no_grad():
        host.weight.copy_(torch.eye(2))
        layer.lora_down.weight.copy_(torch.tensor([[1., 0.]]))
        layer.lora_up.weight.copy_(torch.tensor([[1.], [0.]]))
        off = host(tokens)

        network.set_lora_slider(.25)
        with network:
            uniform = host(tokens[..., :4, :]) - off[..., :4, :]

        network.set_lora_slider(1.)
        network.set_seq_gain([0., 1., 0., 0.], mode="prefix")
        with network:
            gated = host(tokens[..., :4, :])
            assert torch.equal(host(tokens[..., :2, :]), gated[..., :2, :])
            assert torch.equal(host(tokens)[..., 4:, :], off[..., 4:, :])
        assert torch.equal(host(tokens), off)

    delta = gated - off[..., :4, :]
    torch.testing.assert_close(delta.norm(dim=-1).sum(dim=-1),
                               uniform.norm(dim=-1).sum(dim=-1))
    assert torch.all(delta[..., 1, 0] > uniform[..., 1, 0])
    assert torch.equal(gated[..., [0, 2, 3], :], off[..., [0, 2, 3], :])
