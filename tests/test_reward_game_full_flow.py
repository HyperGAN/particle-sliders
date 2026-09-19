"""Small real Music 3 blocks; full-size waveform/GPU audits remain required."""
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch


def test_thirty_step_replay_and_early_credit_with_fixed_overlap():
    from diffusers import MiniMaxMusic3Transformer1DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines.minimax_music3.denoise import (
        MiniMaxMusic3ChunkDenoiseInner, MiniMaxMusic3ChunkUpdateStep,
        MiniMaxMusic3ChunkSetTimestepsStep,
    )
    from conceptmod.textsliders.reward_game.full_flow import FullFlowCapture, replay, validate
    from conceptmod.textsliders.reward_game.core import IntegrityError

    torch.set_num_threads(2)
    torch.manual_seed(15)
    tf = MiniMaxMusic3Transformer1DModel(
        in_channels=4, condition_dim=8, num_layers=2, num_attention_heads=2,
        attention_head_dim=8, ff_inner_dim=32, rotary_dim=8,
    ).eval().requires_grad_(False)
    pipe = SimpleNamespace(
        transformer=tf, condition_encoder=torch.nn.Identity(), vocoder=torch.nn.Identity(),
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, invert_sigmas=True),
        guider=ClassifierFreeGuidance(guidance_scale=1.7), _execution_device='cpu',
        latent_hop_length=512, sampling_rate=44100,
    )
    state = SimpleNamespace(
        previous_latent=None, previous_condition=None, latent_chunks=[],
        num_inference_steps=30, progress_bar=SimpleNamespace(update=lambda: None),
    )
    with FullFlowCapture(pipe) as capture:
        for chunk in range(2):
            state.condition = pipe.condition_encoder(torch.randn(1, 200, 8))
            state.overlap = 0
            if state.previous_latent is not None:
                state.overlap = min(state.previous_latent.shape[-1], 200)
                state.condition[:, :state.overlap] = state.previous_condition[:, :state.overlap]
            state.latents = torch.randn(1, 4, 200)
            state.noise_prompt = state.latents[..., :state.overlap].clone() if state.overlap else None
            MiniMaxMusic3ChunkSetTimestepsStep()(pipe, state, chunk)
            MiniMaxMusic3ChunkDenoiseInner()(pipe, state, chunk)
            MiniMaxMusic3ChunkUpdateStep()(pipe, state, chunk)
        for latent in state.latent_chunks:
            pipe.vocoder(latent)
    result = capture.result()
    assert state.previous_latent.shape[-1] == 28
    replayed, checks = replay(tf, result, device='cpu', record_checks=True)
    assert all(c['exact'] for c in checks)
    assert all(torch.equal(a, b) for a, b in zip(replayed, state.latent_chunks))

    # A loss on the second chunk reaches its first flow step, while the
    # reference overlap deliberately prevents credit into the first chunk.
    tf.requires_grad_(True)
    outputs = []
    def retain(module, args, output):
        output[0].retain_grad()
        outputs.append(output[0])
    handle = tf.register_forward_hook(retain)
    try:
        differentiable, _ = replay(tf, result, device='cpu')
        assert all(torch.equal(a, b) for a, b in zip(differentiable, replayed))
        differentiable[1].square().mean().backward()
    finally:
        handle.remove()
    assert len(outputs) == 120
    assert all(o.grad is None for o in outputs[:60])
    assert all(o.grad is not None and torch.isfinite(o.grad).all() and o.grad.count_nonzero() for o in outputs[60:])

    truncated = deepcopy(result)
    truncated.update(version=1, retained_steps=2)
    with pytest.raises(IntegrityError, match='thirty-step'):
        validate(truncated)
    incomplete = deepcopy(result)
    del incomplete['chunks'][0]['steps'][0]
    with pytest.raises(IntegrityError, match='Incomplete'):
        validate(incomplete)
