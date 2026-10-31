"""Source/CPU software fixtures; never a pretrained H3 or audiovisual run.

Requires the installed exact core0.12 candidate, not this checkout's core0.11.
Run with ``-o pythonpath=.`` in the separately qualified shared environment.
"""
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("particle_sliders")
if version("particle-sliders-core") != "0.12.0":
    pytest.skip("requires the separately installed reviewed core0.12 cohort", allow_module_level=True)

from conceptmod.textsliders import pika_h3_flex_session as h3
from conceptmod.textsliders.pika_h3_flex import PIKA_ADAPTER
from conceptmod.textsliders.minimax_h3_backend import PackedLayout
from particle_sliders.lifecycle import SessionConfig, Binding, _open_registered, CheckpointError
from particle_sliders.response import ResponseBackend, ResponseCapabilities, TRAINING, INFERENCE
from particle_sliders.pg09 import _hash, PG_VERSION, PG_SOURCE_PROFILE


def fixture_runtime(seed=7):
    from diffusers import MiniMaxH3Transformer3DModel, PikaH3FlexModularPipeline
    from peft import LoraConfig
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=8, hidden_size=16,
        num_layers=1, num_refiner_layers=1, ffn_dim=32, in_channels=1, audio_in_channels=2,
        patch_size=(1, 1, 1), text_dim=8, freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=1)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=list(h3.RELEASE["fl2va"]["target_modules"])), adapter_name=PIKA_ADAPTER)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.fill_(0.03)
    model.requires_grad_(False).eval()
    identity = {"qualification": "random miniature native H3 architecture, not released H3 geometry or weights",
                "frozen_base_sha256": _hash(model.state_dict())}
    pipe = PikaH3FlexModularPipeline(workflow="t2va")
    pipe.update_components(transformer=model)
    runtime = h3.HNDLH3FlexBackend.__new__(h3.HNDLH3FlexBackend)
    runtime.pipe, runtime.transformer, runtime.device = pipe, model, torch.device("cpu")
    runtime._attach_slider(rank=8, alpha=8)
    pairs = []
    for index in range(2):
        generator = torch.Generator().manual_seed(seed + index + 100)
        layout = PackedLayout(hidden_states=torch.randn(1, 2, 1, generator=generator),
            audio_hidden_states=torch.randn(1, 2, 2, generator=generator),
            encoder_hidden_states=torch.randn(1, 2, 8, generator=generator),
            timestep=torch.tensor([0.2, 0.4]), timestep_indices=torch.tensor([0, 0, 0, 0, 1, 1]),
            token_tags=torch.tensor([1, 1, 0, 0, 2, 2]), position_ids=torch.zeros(6, 3),
            text_indices=torch.tensor([0, 1]), video_indices=torch.tensor([2, 3]), audio_indices=torch.tensor([4, 5]))
        positive = deepcopy(layout)
        positive.encoder_hidden_states = positive.encoder_hidden_states + 0.5
        pairs.append(h3.H3Pair(layout, positive))
    return runtime, pairs, identity


def fixture_prepare(options):
    runtime, pairs, identity = fixture_runtime(options.get("seed", 7))
    return h3.bind(runtime, pairs, options=options, model_identity=identity, shape=(1, 6))


FIXTURE = ResponseBackend("h3-random-architecture-fixture",
    ResponseCapabilities("adapter_gan", "video_audio_fixture", "random_packed_velocity", "continuous_vector", (1, 6),
        PG_VERSION, PG_SOURCE_PROFILE, qualification="six-coordinate random architecture fixture; not H3 release qualification"),
    fixture_prepare, lambda options: None)


def open_fixture(config, **kwargs):
    return _open_registered(config, resume=kwargs.get("resume"), artifact=kwargs.get("artifact"),
        operation=kwargs.get("operation", "train"), request={}, resolve=lambda name: FIXTURE,
        training_schema=TRAINING, inference_schema=INFERENCE)


def test_native_preflight_uses_actual_shape_and_refuses_before_loader(monkeypatch):
    called = []
    monkeypatch.setattr(h3, "HNDLH3FlexBackend", lambda *_a, **_k: called.append(True))
    options = dict(snapshot="/never-loaded", rows=[{"neutral": "a fox", "positive": "a red fox"},
                                                  {"neutral": "a room", "positive": "a bright room"}])
    assert h3.native_response_shape(options) == (1, 1398528)
    config = SessionConfig(h3.NAME, "unqualified-native", **options)
    with pytest.raises(ValueError, match="released PG09 cannot admit flat continuous vectors with width >=170"):
        h3.open_session(config)
    assert called == []
    for override in ("width", "response_shape", "dummy", "penalty_input_domain", "pg_source_profile"):
        with pytest.raises(ValueError, match="overrides"):
            h3.validate(dict(options, **{override: 6}))


def test_hndl_projection_matches_native_lora_values_and_gradients():
    runtime, pairs, _ = fixture_runtime()
    branch = runtime.network.loras[0]
    value = torch.randn(1, 2, branch.lora_down.in_features, requires_grad=True)
    with torch.no_grad():
        branch.lora_up.weight.fill_(0.02)
    branch.multiplier = 0.5
    actual = branch(value)
    reference = branch.org_forward(value) + 0.5 * torch.nn.functional.linear(
        torch.nn.functional.linear(value, branch.lora_down.weight), branch.lora_up.weight)
    assert torch.equal(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), value, retain_graph=True)[0],
                               torch.autograd.grad(reference.sum(), value)[0], rtol=0, atol=0)
    assert type(branch.graph).__module__.startswith("hndl")
    assert branch._identity["hypergan_commit"] == "5f368a3770cf5bdae0448995c9f76e29af8049f7"
    runtime.network.close()


def test_bfloat16_native_host_keeps_hndl_leaves_float32():
    runtime, _, _ = fixture_runtime()
    runtime.transformer.to(dtype=torch.bfloat16)
    branch = runtime.network.loras[0]
    with torch.no_grad():
        branch.lora_up.weight.fill_(0.02)
    value = torch.randn(1, 2, branch.lora_down.in_features, dtype=torch.bfloat16, requires_grad=True)
    with runtime.scaled(0.5):
        output = branch(value)
        grads = torch.autograd.grad(output.float().square().mean(), tuple(branch.parameters()))
    assert output.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in branch.parameters())
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    runtime._check_fixed_pika()
    runtime.network.close()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_shared_game_updates_hndl_leaves_only_and_preserves_pika(checkpointing):
    config = SessionConfig(FIXTURE.name, "random-software-v1", seed=7, steps=2, batch=2, checkpointing=checkpointing, ema_decay=0.5)
    with open_fixture(config) as session:
        runtime = session.game.backend.runtime
        base = {name: value.detach().clone() for name, value in runtime.transformer.named_parameters()}
        before = [p.detach().clone() for p in session.game.network.parameters()]
        critic = session.game.critic
        assert isinstance(critic, h3.HNDLCritic)
        assert type(critic.graph).__module__.startswith("hndl")
        session.train()
        assert session.game.completed_steps == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, session.game.network.parameters()))
        assert all(torch.equal(base[name], value) for name, value in runtime.transformer.named_parameters())
        for strength in (0, 0.5, -1, 1):
            with runtime.scaled(strength):
                assert all(layer.scaling[PIKA_ADAPTER] == 1 for layer in runtime._release_layers.values())
        runtime._check_fixed_pika()
        assert session.manifest["game_signature"]["pg_version"] == "0.9.4"
        assert session.manifest["capabilities"]["generator_arm"] == "adapter_gan"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_shared_checkpoint_resume_export_and_observation_are_exact(tmp_path, checkpointing):
    config = SessionConfig(FIXTURE.name, "random-exact-resume-v1", seed=7, steps=4, batch=2, ema_decay=0.5, checkpointing=checkpointing)
    checkpoint = tmp_path / "training.pt"
    artifact = tmp_path / "adapter.pt"
    with open_fixture(config) as uninterrupted:
        uninterrupted.train()
        expected = deepcopy(uninterrupted.game.network.state_dict())
    with open_fixture(config) as first:
        first.train(stop_after=2)
        state = _hash(first.game.state_dict())
        first.evaluate(0, seed=19, strength=0.5)
        assert _hash(first.game.state_dict()) == state
        first.checkpoint(checkpoint)
    with open_fixture(config, resume=checkpoint) as second:
        second.train()
        assert all(torch.equal(expected[name], value) for name, value in second.game.network.state_dict().items())
        second.export(artifact, selection="raw")
        result = second.evaluate(0, seed=19)
    with open_fixture(config, artifact=artifact, operation="response") as inference:
        torch.testing.assert_close(inference.evaluate(0, seed=19), result, rtol=0, atol=0)
    wrong = SessionConfig(FIXTURE.name, "different-cohort", seed=7, steps=4, batch=2, ema_decay=0.5, checkpointing=checkpointing)
    with pytest.raises(CheckpointError, match="cohort"):
        open_fixture(wrong, resume=checkpoint)
