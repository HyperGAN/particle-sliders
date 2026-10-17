"""Independent CPU regressions from native LTX package review.

Actual PEFT adapters wrap tiny random modules; no pretrained loader, Hub,
accelerator allocation or audiovisual generation is admitted by these tests.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders import ltx25_backend as native


def components():
    return SimpleNamespace(
        tokenizer=native.DummyTokenizer(),
        text_encoder=native.DummyEncoder(),
        connectors=native.DummyLTX2Connectors(),
        transformer=native.DummyLTX2Transformer(),
        vae=None,
        audio_vae=None,
        vocoder=None,
    )


@pytest.mark.parametrize("recipe", ["embed", "velocity"])
def test_actual_peft_preserves_fractional_alpha_and_float32_adapter_leaves(recipe):
    pytest.importorskip("peft")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        backend = native.LTX25Backend(
            components=components(), device="cpu", encoder_device="cpu",
            recipe=recipe, te_last_n=1, lora_rank=2, lora_alpha=.5,
        )
    roots = (backend.encoder, backend.connectors) if recipe == "embed" else (backend.transformer,)
    layers = [module for root in roots for module in root.modules()
              if hasattr(module, "lora_A") and hasattr(module, "scaling")]
    assert layers, "the native fixture must exercise actual PEFT layers"
    backend.network.set_lora_slider(1.)
    assert all(module.scaling["default"] == pytest.approx(.25) for module in layers)
    assert all(module.lora_alpha["default"] == .5 for module in layers)
    assert backend.trainable_parameters()
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
               for parameter in backend.trainable_parameters())
    backend.network.set_lora_slider(0.)
    assert all(module.scaling["default"] == 0. for module in layers)
    backend.network.set_lora_slider(.5)
    assert all(module.scaling["default"] == pytest.approx(.125) for module in layers)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("recipe", ["embed", "velocity"])
def test_partial_peft_attachment_error_propagates_without_custom_fallback(recipe, monkeypatch):
    peft = pytest.importorskip("peft")
    wrap = peft.get_peft_model
    calls = []

    def interrupted(model, config):
        calls.append(model)
        if recipe == "embed" and len(calls) == 1:
            return wrap(model, config)  # Encoder mutation has already happened.
        if recipe == "velocity":
            wrap(model, config)  # Fault after the transformer was mutated.
        raise RuntimeError("injected native PEFT attachment failure")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("a partially modified native model cannot use custom fallback")

    monkeypatch.setattr(peft, "get_peft_model", interrupted)
    monkeypatch.setattr(native, "ModuleLoRANetwork", forbidden_fallback)
    monkeypatch.setattr(native, "AttnLoRANetwork", forbidden_fallback)
    with pytest.raises(RuntimeError, match="injected native PEFT attachment failure"):
        native.LTX25Backend(
            components=components(), device="cpu", encoder_device="cpu",
            recipe=recipe, te_last_n=1, lora_rank=2, lora_alpha=.5,
        )
    assert len(calls) == (2 if recipe == "embed" else 1)
    assert not torch.cuda.is_initialized()


def test_unqualified_constructor_rejects_before_loader_or_device_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified constructor touched a loader or device")

    monkeypatch.setattr(native, "_load_ltx25_pipeline", forbidden)
    monkeypatch.setattr(torch, "device", forbidden)
    with pytest.raises(native.ArchitectureMismatch, match="production runtime is not qualified"):
        native.LTX25Backend(dummy=False, device="cuda:0")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("operation", ["save", "load", "render"])
def test_unqualified_native_exports_and_render_reject_before_access(operation, tmp_path):
    backend = object.__new__(native.LTX25Backend)
    backend.dummy = False
    actions = {
        "save": lambda: backend.save_trained(str(tmp_path / "must-not-exist")),
        "load": lambda: backend.load_trained("missing.safetensors"),
        "render": lambda: backend.generate_t2v("neutral caption"),
    }
    with pytest.raises(native.ArchitectureMismatch, match="not qualified"):
        actions[operation]()
    assert not list(tmp_path.iterdir())
    assert not torch.cuda.is_initialized()


def test_components_injection_rejects_non_cpu_modules_before_attachment(monkeypatch):
    fixture = components()
    fixture.connectors = nn.Linear(2, 2, device="meta")

    def forbidden(*args, **kwargs):
        raise AssertionError("non-CPU components reached attachment")

    monkeypatch.setattr(native.LTX25Backend, "_init_live", forbidden)
    with pytest.raises(native.ArchitectureMismatch, match="CPU native-module fixtures"):
        native.LTX25Backend(components=fixture, device="cpu", encoder_device="cpu")
    assert not torch.cuda.is_initialized()
