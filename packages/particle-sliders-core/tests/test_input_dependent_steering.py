"""Per-input destination transport through the packaged particle projection."""
import torch
from torch import nn

from particle_sliders.routed_projection import TinyParticleProjection


def set_positive_axis_map(network, input_column):
    """Configure one path through the installed LeakyReLU MLP for positive values."""
    for layer in (module for module in network if isinstance(module, nn.Linear)):
        layer.weight.zero_()
        layer.bias.zero_()
        layer.weight[0, input_column] = 1.
        input_column = 0


def test_routed_projection_distinguishes_varying_destinations_from_a_shared_destination():
    with torch.random.fork_rng(devices=[]):
        host = nn.Linear(2, 2, bias=False).requires_grad_(False)
        bank = nn.Parameter(torch.tensor([[0., 0., 0., 0.], [2., 0., 0., 0.]]))
        branch = TinyParticleProjection('attention.to_q', host, bank, rank=1, alpha=1)
    tokens = torch.tensor([[[.25, 4.], [1., 4.], [2., 4.]]])
    with torch.no_grad():
        host.weight.copy_(torch.eye(2))
        branch.lora_down.weight.copy_(torch.tensor([[1., 0.]]))
        branch.lora_up.weight.copy_(torch.tensor([[0.], [1.]]))
        set_positive_axis_map(branch.bridge.router, input_column=0)
        set_positive_axis_map(branch.bridge.net, input_column=1)

        off = host(tokens)
        torch.testing.assert_close(off, tokens, rtol=0, atol=0)
        branch.multiplier = 1.
        on = host(tokens)
        # Two bank entries yield logits [0, x] in this fixture. Their first
        # coordinates are 0 and 2, so the routed edit is 2 sigmoid(x).
        expected = off.clone()
        expected[..., 1] += 2 * tokens[..., 0].sigmoid()
        torch.testing.assert_close(on, expected)
        assert on[..., 1].unique().numel() == tokens.shape[1]

        bank.copy_(bank.flip(0))
        torch.testing.assert_close(host(tokens), expected)

        # With a shared destination, routing has no per-token effect. The
        # ablation uses the same host, adapter, inputs and slider strength.
        bank.copy_(bank[0].clone().expand_as(bank))
        constant = off.clone()
        constant[..., 1] += 2.
        torch.testing.assert_close(host(tokens), constant, rtol=0, atol=0)
