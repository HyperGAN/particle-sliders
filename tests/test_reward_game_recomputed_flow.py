"""Real small Music 3 flow: gradients and values through all thirty steps."""
from copy import deepcopy
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch


def capture_fixture(dtype):
    from diffusers import MiniMaxMusic3Transformer1DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines.minimax_music3.denoise import (
        MiniMaxMusic3ChunkDenoiseInner, MiniMaxMusic3ChunkUpdateStep,
        MiniMaxMusic3ChunkSetTimestepsStep,
    )
    from conceptmod.textsliders.reward_game.full_flow import FullFlowCapture
    torch.set_num_threads(2); torch.manual_seed(23)
    tf = MiniMaxMusic3Transformer1DModel(in_channels=4, condition_dim=8, num_layers=2,
        num_attention_heads=2, attention_head_dim=8, ff_inner_dim=32, rotary_dim=8,
    ).to(dtype).eval().requires_grad_(False)
    pipe = SimpleNamespace(transformer=tf, condition_encoder=torch.nn.Identity(),
        vocoder=torch.nn.Identity(), scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1, invert_sigmas=True),
        guider=ClassifierFreeGuidance(guidance_scale=1.7), _execution_device='cpu',
        latent_hop_length=512, sampling_rate=44100)
    state = SimpleNamespace(previous_latent=None, previous_condition=None, latent_chunks=[],
        num_inference_steps=30, progress_bar=SimpleNamespace(update=lambda: None))
    with FullFlowCapture(pipe) as capture:
        for chunk in range(2):
            state.condition = pipe.condition_encoder(torch.randn(1, 200, 8, dtype=dtype))
            state.overlap = 0
            if state.previous_latent is not None:
                state.overlap = min(state.previous_latent.shape[-1], 200)
                state.condition[:, :state.overlap] = state.previous_condition[:, :state.overlap]
            state.latents = torch.randn(1, 4, 200, dtype=dtype)
            state.noise_prompt = state.latents[..., :state.overlap].clone() if state.overlap else None
            MiniMaxMusic3ChunkSetTimestepsStep()(pipe, state, chunk)
            MiniMaxMusic3ChunkDenoiseInner()(pipe, state, chunk)
            MiniMaxMusic3ChunkUpdateStep()(pipe, state, chunk)
        for latent in state.latent_chunks:
            pipe.vocoder(latent)
    return tf, capture.result()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_complete_flow_recomputed_factors_and_optimizer_match(dtype):
    from conceptmod.textsliders.reward_game.full_flow import replay
    from conceptmod.textsliders.reward_game.merged_forward import MergedForward
    from conceptmod.textsliders.reward_game.recomputed_flow_forward import RecomputedFlowForward
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.lora_runtime import LoRANetwork
    tf, capture = capture_fixture(dtype)
    pristine = deepcopy(tf.state_dict())
    tf.enable_gradient_checkpointing()
    torch.manual_seed(9131)
    network = LoRANetwork(tf, rank=8, alpha=8., multiplier=1.,
        target_replace=['MiniMaxMusic3Attention'], prefix='lora_unet', delimiter='-',
        train_method='full', attach=False).requires_grad_(True)
    merger = MergedForward(network).attach()
    recompute = RecomputedFlowForward(tf)
    initial = deepcopy(network.state_dict())
    results = []
    try:
        for enabled in (False, True):
            network.load_state_dict(initial)
            opt = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=0.)
            if enabled:
                recompute.attach()
            rows = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                outputs, _ = replay(tf, capture, device='cpu')
                loss = sum(x.float().square().mean() for x in outputs)
                loss.backward()
                gradients = {k:p.grad.clone() for k,p in network.named_parameters()}
                assert all(torch.isfinite(g).all() for g in gradients.values())
                assert any(g.count_nonzero() for g in gradients.values())
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                rows.append(dict(outputs=[x.detach().clone() for x in outputs], grads=gradients,
                                 state=deepcopy(network.state_dict()), optimizer=deepcopy(opt.state_dict())))
            results.append(rows)
            recompute.detach()
        for ordinary, rebuilt in zip(*results):
            assert all(torch.equal(a,b) for a,b in zip(ordinary['outputs'],rebuilt['outputs']))
            for key in ordinary['grads']:
                torch.testing.assert_close(ordinary['grads'][key], rebuilt['grads'][key], rtol=2e-5, atol=1e-7)
            for key in ordinary['state']:
                torch.testing.assert_close(ordinary['state'][key], rebuilt['state'][key], rtol=2e-5, atol=1e-7)
            assert ordinary['optimizer']['param_groups'] == rebuilt['optimizer']['param_groups']
            for ident, values in ordinary['optimizer']['state'].items():
                for key, value in values.items():
                    torch.testing.assert_close(value, rebuilt['optimizer']['state'][ident][key], rtol=2e-5, atol=1e-7)
        assert recompute.metrics['recomputed_calls'] == 240
        assert recompute.metrics['no_grad_calls'] == 240
        assert merger.base_unchanged()
        assert all(p.grad is None for p in tf.parameters())
    finally:
        recompute.detach(); merger.detach()
    assert all(torch.equal(v, pristine[k]) for k,v in tf.state_dict().items())


def test_memory_guard_rejects_before_model_call():
    from conceptmod.textsliders.reward_game.recomputed_flow_forward import RecomputedFlowForward
    tf = SimpleNamespace(forward=lambda **kwargs: (_ for _ in ()).throw(AssertionError('model called')))
    wrapper = RecomputedFlowForward(tf, max_rss_gib=0).attach()
    try:
        with pytest.raises(MemoryError, match='memory budget'):
            tf.forward(torch.ones(1), torch.ones(1), torch.ones(1), False)
    finally:
        wrapper.detach()
