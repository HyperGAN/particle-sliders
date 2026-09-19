"""Transported targets through a real small acoustic host and checkpointing."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch
from test_reward_game_recomputed_flow import capture_fixture
from conceptmod.textsliders.reward_game.latent_transport import schedule, transported, velocity_loss
from conceptmod.textsliders.reward_game.merged_forward import MergedForward


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_transport_updates_on_real_acoustic_host_with_frozen_base(dtype):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.lora_runtime import LoRANetwork
    tf, capture = capture_fixture(dtype)
    initial_base = deepcopy(tf.state_dict())
    tf.enable_gradient_checkpointing()
    torch.manual_seed(9131)
    network = LoRANetwork(tf, rank=8, alpha=8., multiplier=1., target_replace=['MiniMaxMusic3Attention'],
        prefix='lora_unet', delimiter='-', train_method='full', attach=False).requires_grad_(True)
    initial = deepcopy(network.state_dict())
    wrapper = MergedForward(network).attach()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0003, weight_decay=0.)
    sigmas = schedule(capture).sigmas
    assert float(sigmas[0]) == 0 and float(sigmas[-1]) == 1
    delta = capture['chunks'][0]['latent'].float()*.01
    try:
        for iteration, index in enumerate((0, 15)):
            optimizer.zero_grad(set_to_none=True)
            for branch in (0, 1):
                source = capture['chunks'][0]['steps'][index]['branches'][branch]
                kwargs, target = transported(source, delta, sigmas[index], sigmas[0], sigmas[-1], 'cpu')
                output = tf(**kwargs)[0]
                if iteration == 0:
                    assert torch.equal(output.detach(), source['velocity'])
                loss = .5*velocity_loss(output, target, source['velocity'])
                loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
            assert any(p.grad.count_nonzero() for p in network.parameters())
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        assert any(not torch.equal(initial[k], value) for k, value in network.state_dict().items())
        assert wrapper.base_unchanged()
        assert all(p.grad is None for p in tf.parameters())
    finally:
        wrapper.detach()
    assert all(torch.equal(initial_base[k], value) for k, value in tf.state_dict().items())
