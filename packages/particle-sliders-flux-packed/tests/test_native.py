"""Installed native prerequisites only; no synthetic claim of shared training."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from diffusers import FluxTransformer2DModel
from particle_sliders.pg09_backends import ResponseRow

from ps_flux_packed_preview import COHORT
from ps_flux_packed_preview.native import (
    FluxLoRA, HNDLLoRA, PackedFluxResponse, PackedTrainingUnavailable, preflight_training,
)
from ps_flux_packed_preview.provenance import ROOT, checked_builder


@pytest.fixture(autouse=True)
def cpu_rng():
    with torch.random.fork_rng():
        torch.manual_seed(94)
        yield


def model(guided=False):
    return FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
        num_single_layers=1, attention_head_dim=8, num_attention_heads=2,
        joint_attention_dim=12, pooled_projection_dim=8, guidance_embeds=guided,
        axes_dims_rope=(2, 2, 4)).requires_grad_(False)


def condition(tokens=3, image_tokens=4, guided=False):
    return dict(embeds=torch.randn(1, tokens, 12), pooled=torch.randn(1, 8),
        text_ids=torch.zeros(tokens, 3),
        image_ids=torch.stack([torch.zeros(image_tokens), torch.arange(image_tokens), torch.zeros(image_tokens)], -1),
        guidance=torch.tensor([3.5]) if guided else None)


def response(m, bank):
    return PackedFluxResponse(m, bank, model_identity={"kind": "random native Flux CPU fixture", "seed": 94})


def historical_class():
    path = Path(__file__).resolve().parents[3] / "conceptmod/textsliders/lora.py"
    spec = importlib.util.spec_from_file_location("historical_flux_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_identity_and_actual_hndl_graph():
    from importlib.metadata import version
    import ps_flux_packed_preview
    assert "site-packages" in ps_flux_packed_preview.__file__
    assert version("hndl") == "0.10.0"
    builder, identity = checked_builder()
    assert identity["hypergan_commit"] == "5f368a3770cf5bdae0448995c9f76e29af8049f7"
    branch = HNDLLoRA(torch.nn.Linear(12, 16), 4, 1., builder, identity, tokens=5)
    assert type(branch.graph).__module__.startswith("hndl.")
    assert isinstance(branch.graph["down"], torch.nn.Linear)
    assert isinstance(branch.graph["up"], torch.nn.Linear)
    assert branch.graph["down"].weight.shape == (4, 12)


@pytest.mark.parametrize("shape", [(1, 1, 64), (1, 4, 64), (1, 1024, 64)])
def test_shared_packed_refusal_does_not_consume_rng_or_mutate_native(shape):
    m = model()
    modes = [v.training for v in m.modules()]
    grads = [v.requires_grad for v in m.parameters()]
    before = torch.get_rng_state().clone()
    with pytest.raises(PackedTrainingUnavailable, match="no native packed-token"):
        preflight_training(shape)
    assert torch.equal(before, torch.get_rng_state())
    assert [v.training for v in m.modules()] == modes
    assert [v.requires_grad for v in m.parameters()] == grads
    assert all(not v._forward_hooks for v in m.modules())


def test_shared_response_descriptor_cannot_advertise_packed_geometry():
    from particle_sliders.response import ResponseCapabilities
    for domain in ("continuous_vector", "continuous_field", "continuous_packed_tokens"):
        with pytest.raises(ValueError, match="native continuous response geometry"):
            ResponseCapabilities("adapter_gan", "image", "packed_velocity", domain,
                (1, 4, 64), "0.9.4", "release-0.9.4")


@pytest.mark.parametrize("shape", [(1, 64), (1, 16, 8, 8), (2, 4, 64), (1, 4, 32)])
def test_preflight_rejects_recast_or_malformed_axes(shape):
    with pytest.raises(ValueError, match="must be"):
        preflight_training(shape)


def test_no_silent_source_or_cohort_fallback():
    with pytest.raises(ValueError, match="cohort"):
        preflight_training((1, 4, 64), cohort="historical")
    original = Path.read_bytes
    def changed(path):
        data = original(path)
        return data + b"\n# changed\n" if path.name == "pg09.py" else data
    with patch.object(Path, "read_bytes", changed), pytest.raises(ValueError, match="core source closure"):
        FluxLoRA(model(), image_tokens=4, text_tokens=3)


def test_builder_tamper_refused_before_any_hook():
    original = Path.read_bytes
    def changed(path):
        data = original(path)
        return data + b"\n# changed\n" if path.name == "hndl_networks.py" else data
    m = model()
    with patch.object(Path, "read_bytes", changed), pytest.raises(ValueError, match="builder closure"):
        FluxLoRA(m, image_tokens=4, text_tokens=3)
    assert all(not v._forward_hooks for v in m.modules())


def test_native_source_and_local_graph_tamper_refused():
    original = Path.read_bytes
    for name, message in [("transformer_flux.py", "native Flux"), ("lora.hndl", "LoRA graph")]:
        def changed(path, name=name):
            data = original(path)
            return data + b"\n# changed\n" if path.name == name else data
        m = model()
        with patch.object(Path, "read_bytes", changed), pytest.raises(ValueError, match=message):
            FluxLoRA(m, image_tokens=4, text_tokens=3)
        assert all(not v._forward_hooks for v in m.modules())


def test_fused_projections_refused_instead_of_attaching_unused_lora_leaves():
    m = model()
    m.fuse_qkv_projections()
    with pytest.raises(ValueError, match="unfused"):
        FluxLoRA(m, image_tokens=4, text_tokens=3)
    assert all(not v._forward_hooks for v in m.modules())


@pytest.mark.parametrize("scale", [0., .3, 1., -1.])
def test_matched_historical_branch_outputs_input_and_all_parameter_vjps(scale):
    historical = historical_class()
    host = torch.nn.Linear(12, 16).requires_grad_(False)
    reference_host = deepcopy(host)
    old = historical.LoRAModule("reference", reference_host, lora_dim=4, alpha=1.)
    old.apply_to()
    builder, identity = checked_builder()
    branch = HNDLLoRA(host, 4, 1., builder, identity, tokens=5)
    with torch.no_grad():
        old.lora_up.weight.normal_(std=.1)
        branch.graph["down"].weight.copy_(old.lora_down.weight)
        branch.graph["up"].weight.copy_(old.lora_up.weight)
    x = torch.randn(2, 5, 12, requires_grad=True)
    old.multiplier = scale
    native = host(x) + branch(x) * scale
    expected = reference_host(x)
    torch.testing.assert_close(native, expected, atol=1e-7, rtol=1e-6)
    cotangent = torch.randn_like(native)
    actual_vjp = torch.autograd.grad(native, (x, *branch.parameters()), cotangent, retain_graph=True)
    expected_vjp = torch.autograd.grad(expected, (x, *old.parameters()), cotangent)
    for actual, wanted in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("guided", [False, True])
def test_actual_flux_packed_forward_and_target_geometry(guided):
    m = model(guided)
    c = condition(guided=guided)
    latent, time = torch.randn(1, 4, 64), torch.tensor(777.)
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    adapter = response(m, bank)
    with bank.borrowed():
        expected = m(hidden_states=latent, timestep=time.reshape(1) / 1000,
            encoder_hidden_states=c["embeds"], pooled_projections=c["pooled"],
            txt_ids=c["text_ids"], img_ids=c["image_ids"], guidance=c["guidance"], return_dict=False)[0]
        for strength in [0., .5, 1.]:
            torch.testing.assert_close(adapter.velocity(c, latent, time, scale=strength), expected, rtol=0, atol=0)
        row = ResponseRow(c, condition(guided=guided), c, latent, timestep=time)
        prepared = adapter.prepare(row)
        assert prepared.target.shape == prepared.neutral.shape == (1, 4, 64)
        assert not prepared.target.requires_grad
        assert adapter.metadata()["shared_training_admitted"] is False


@pytest.mark.parametrize("strength", [.3, 1., -1.])
def test_full_native_flux_matches_historical_lora_and_bank_vjps(strength):
    native, reference = model(), model()
    reference.load_state_dict(native.state_dict())
    historical = historical_class().LoRANetwork(reference, rank=4, alpha=1.,
        target_replace=["FluxAttention"], train_method="full")
    bank = FluxLoRA(native, image_tokens=4, text_tokens=3)
    assert tuple("lora_unet_" + n.replace(".", "_") for n in bank.target_names) == tuple(
        layer.lora_name for layer in historical.unet_loras)
    assert len(historical.unet_loras) == len(bank.branches) == 7
    with torch.no_grad():
        for old, branch in zip(historical.unet_loras, bank.branches):
            old.lora_up.weight.normal_(std=.03)
            branch.graph["down"].weight.copy_(old.lora_down.weight)
            branch.graph["up"].weight.copy_(old.lora_up.weight)
    c = condition()
    x = torch.randn(1, 4, 64, requires_grad=True)
    t = torch.tensor(400.)
    with bank.borrowed(), bank.scaled(strength):
        actual = response(native, bank)._velocity(c, x, t)
        for old in historical.unet_loras:
            old.multiplier = strength
        expected = reference(hidden_states=x, timestep=t.reshape(1) / 1000,
            encoder_hidden_states=c["embeds"], pooled_projections=c["pooled"],
            txt_ids=c["text_ids"], img_ids=c["image_ids"], guidance=None, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        cotangent = torch.randn_like(actual)
        actual_vjps = torch.autograd.grad(actual, (x, *bank.parameters()), cotangent, retain_graph=True)
        expected_vjps = torch.autograd.grad(expected, (x, *historical.parameters()), cotangent)
        for a, b in zip(actual_vjps, expected_vjps):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_strength_nesting_failure_cleanup_and_borrowed_state():
    m = model()
    list(m.modules())[3].eval()
    modes = [v.training for v in m.modules()]
    flags = [v.requires_grad for v in m.parameters()]
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    for p in bank.parameters():
        p.grad = torch.full_like(p, .25)
    before_grads = [p.grad.clone() for p in bank.parameters()]
    with pytest.raises(RuntimeError, match="native forward failed"):
        with bank.borrowed(), bank.scaled(.25):
            with bank.scaled(0.):
                assert bank.strength == 0.
            assert bank.strength == .25
            raise RuntimeError("native forward failed")
    assert bank.strength == 0. and not bank.attached
    assert [v.training for v in m.modules()] == modes
    assert [v.requires_grad for v in m.parameters()] == flags
    for p, wanted in zip(bank.parameters(), before_grads):
        assert torch.equal(p.grad, wanted)
    assert all(not v._forward_hooks for v in m.modules())
    bank.close()


def test_partial_hook_failure_rolls_back_all_prior_hooks_and_modes():
    m = model()
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    original_modes = [v.training for v in m.modules()]
    second = bank._targets[1][1]
    with patch.object(second, "register_forward_hook", side_effect=RuntimeError("attach failure")):
        with pytest.raises(RuntimeError, match="attach failure"):
            bank.attach()
    assert not bank.attached
    assert [v.training for v in m.modules()] == original_modes
    assert all(not v._forward_hooks for v in m.modules())


def test_every_cleanup_attempted_and_original_error_preserved():
    m = model()
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    calls = []
    original = torch.utils.hooks.RemovableHandle.remove
    def noisy(handle):
        original(handle)
        calls.append(handle.id)
        raise RuntimeError("cleanup failure after removal")
    with pytest.raises(RuntimeError, match="original failure") as error:
        with patch.object(torch.utils.hooks.RemovableHandle, "remove", noisy):
            with bank.borrowed():
                raise RuntimeError("original failure")
    assert len(calls) == len(bank.branches)
    assert any("cleanup also failed" in note for note in error.value.__notes__)
    assert all(not v._forward_hooks for v in m.modules())
    assert all(v.training for v in m.modules())
    # An uncertain failed removal remains retryable; close is idempotent after
    # the injected failure is removed, even if the original removal succeeded.
    assert bank.attached
    bank.close()
    assert not bank.attached


def test_existing_host_hook_is_preserved_and_attachment_is_refused():
    m = model()
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    existing = bank._targets[0][1].register_forward_hook(lambda *args: None)
    modes = [v.training for v in m.modules()]
    with pytest.raises(ValueError, match="already carries"):
        bank.attach()
    assert [v.training for v in m.modules()] == modes
    assert not bank.attached
    assert len(bank._targets[0][1]._forward_hooks) == 1
    existing.remove()


@pytest.mark.parametrize("change", ["grad", "checkpointing", "config", "parameters"])
def test_staged_mutation_refused_before_new_hooks(change):
    m = model()
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    if change == "grad":
        next(m.parameters()).requires_grad_(True)
    elif change == "checkpointing":
        m.enable_gradient_checkpointing()
    elif change == "config":
        m.register_to_config(guidance_embeds=True)
    else:
        m.x_embedder.weight = torch.nn.Parameter(m.x_embedder.weight.detach().clone(), requires_grad=False)
    with pytest.raises(ValueError, match="changed before"):
        bank.attach()
    assert all(not v._forward_hooks for v in m.modules())


@pytest.mark.parametrize("change", ["text_ids", "image_ids", "guidance", "time", "latent", "config"])
def test_malformed_native_contract_refused_before_forward(change):
    m = model()
    bank = FluxLoRA(m, image_tokens=4, text_tokens=3)
    adapter = response(m, bank)
    c, x, time = condition(), torch.randn(1, 4, 64), torch.tensor(500.)
    if change in ("text_ids", "image_ids"):
        c[change] = c[change].unsqueeze(0)
    elif change == "guidance":
        c[change] = torch.tensor([3.5])
    elif change == "time":
        time = torch.tensor(1001.)
    elif change == "latent":
        x = x.reshape(1, 64, 2, 2)
    with bank.borrowed(), patch.object(m, "forward", side_effect=AssertionError("forward ran")):
        if change == "config":
            m.register_to_config(guidance_embeds=True)
        with pytest.raises(ValueError):
            adapter.velocity(c, x, time)
