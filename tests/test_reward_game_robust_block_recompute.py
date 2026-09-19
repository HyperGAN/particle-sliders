"""Recomputed joint block gradients under the earlier established tolerance.

The original bit-exact test and its failed result remain in the run audit.
"""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch
from test_reward_game_recomputed_flow import capture_fixture
from conceptmod.textsliders.reward_game.acoustic_tail import replay_latents
from conceptmod.textsliders.reward_game.merged_forward import MergedForward
from conceptmod.textsliders.reward_game.recomputed_flow_forward import RecomputedFlowForward


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_recomputed_joint_groups_and_partial_batch_match_legacy(dtype):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.lora_runtime import LoRANetwork
    tf, capture = capture_fixture(dtype)
    capture['retained_steps'] = 2
    pristine = deepcopy(tf.state_dict()); tf.enable_gradient_checkpointing()
    torch.manual_seed(9131)
    network = LoRANetwork(tf, rank=8, alpha=8., multiplier=1., target_replace=['MiniMaxMusic3TransformerBlock'],
        prefix='lora_unet', delimiter='-', train_method='full', attach=False).requires_grad_(True)
    with torch.no_grad():
        for module in network.unet_loras:
            module.lora_up.weight.normal_(0., .002)
    parent = deepcopy(network.state_dict()); wrapper = MergedForward(network).attach()
    recomputed = RecomputedFlowForward(tf)
    results = []
    try:
        for resume in (False, True):
            if resume:
                recomputed.attach()
            network.load_state_dict(parent)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=0.)
            optimizer.zero_grad(set_to_none=True)
            for index in range(4):
                outputs, _ = replay_latents(tf, capture, device='cpu', chunk_indices=(index % 2,))
                with torch.no_grad():
                    ordinary, _ = replay_latents(tf, capture, device='cpu', chunk_indices=(index % 2,))
                assert torch.equal(outputs[0].detach(), ordinary[0])
                # Toy differentiable reward isolates balanced accumulation and
                # joint support; actual frozen waveform CE is audited on host.
                ce = 7.+outputs[0].float().mean()
                baseline = 7.+index*.03
                loss = .1*torch.nn.functional.softplus((.1-(ce-baseline))/.1)/4
                loss.backward()
                if resume and index == 1:
                    state = deepcopy(dict(network=network.state_dict(), optimizer=optimizer.state_dict(),
                        gradients={k: p.grad for k, p in network.named_parameters()}))
                    optimizer.zero_grad(set_to_none=True)
                    network.load_state_dict(state['network']); optimizer.load_state_dict(state['optimizer'])
                    for key, parameter in network.named_parameters():
                        parameter.grad = state['gradients'][key]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
            for kind in ('-attn-', '-ff_'):
                assert sum(float(p.grad.abs().sum()) for k, p in network.named_parameters() if kind in k) > 0
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            results.append(deepcopy(dict(network=network.state_dict(), optimizer=optimizer.state_dict(),
                gradients={k: p.grad for k, p in network.named_parameters()})))
            recomputed.detach()
        for key in results[0]['network']:
            torch.testing.assert_close(results[0]['network'][key], results[1]['network'][key], rtol=2e-5, atol=1e-7)
        for key in results[0]['gradients']:
            torch.testing.assert_close(results[0]['gradients'][key], results[1]['gradients'][key], rtol=2e-5, atol=1e-7)
        for ident, state in results[0]['optimizer']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, results[1]['optimizer']['state'][ident][key], rtol=2e-5, atol=1e-7)
        assert wrapper.base_unchanged() and all(p.grad is None for p in tf.parameters())
    finally:
        recomputed.detach(); wrapper.detach()
    assert all(torch.equal(value, pristine[key]) for key, value in tf.state_dict().items())
