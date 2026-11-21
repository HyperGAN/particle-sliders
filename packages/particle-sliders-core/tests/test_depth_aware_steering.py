"""Apply a position envelope to explicit LoRA targets across model depth."""
import torch
from torch import nn

from particle_sliders.native_lora import LoRANetwork


def test_distributed_lora_targets_own_envelope_and_restore_nested_scale():
    with torch.random.fork_rng(devices=[]):
        host = nn.Sequential(*(nn.Linear(3, 3, bias=False) for _ in range(3)))
        network = LoRANetwork(
            host, rank=1, alpha=1., multiplier=0.,
            target_plan=[('early', host[0]), ('late', host[2])],
        )
    tokens = torch.tensor([[[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]]])
    with torch.no_grad():
        for layer in host:
            layer.weight.copy_(torch.eye(3))
        for branch, strength in zip(network.unet_loras, (2., 4.)):
            branch.lora_down.weight.copy_(torch.tensor([[1., 0., 0.]]))
            branch.lora_up.weight.copy_(torch.tensor([[0.], [0.], [strength]]))
        assert torch.equal(host(tokens), tokens)

        envelope = torch.tensor([0., .5, 1.])
        network.set_seq_gain(envelope, mode='prefix')
        envelope.fill_(float('nan'))
        network.set_lora_slider(.5)
        with network:
            early_expected = tokens.clone()
            early_expected[..., 2] += torch.tensor([0., 1., 3.])
            assert torch.equal(host[0](tokens), early_expected)
            assert torch.equal(host[1](early_expected), early_expected)
            distributed = tokens.clone()
            distributed[..., 2] += torch.tensor([0., 3., 9.])
            assert torch.equal(host(tokens), distributed)

            network.set_lora_slider(-.25)
            with network:
                reverse = tokens.clone()
                reverse[..., 2] -= torch.tensor([0., 1.5, 4.5])
                assert torch.equal(host(tokens), reverse)
            assert torch.equal(host(tokens), distributed)
        assert torch.equal(host(tokens), tokens)
