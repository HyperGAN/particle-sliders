"""Weights-free gates for the installed Music3 native architecture surface."""
import ast
from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders import music3_native as native
from particle_sliders import native_lora
from particle_sliders.reference import RoutedMLP


def test_native_imports_do_not_require_trainers_or_optional_model_runtimes(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(Path(native.__file__).resolve().parents[1]), environment.get("PYTHONPATH", ""),
    ))
    script = """
import importlib.abc
import sys
class NoOptionalOrHistoricalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'conceptmod', 'analysis', 'transformers', 'diffusers', 'safetensors'}:
            raise AssertionError('native architecture import leaked: ' + fullname)
sys.meta_path.insert(0, NoOptionalOrHistoricalImports())
from particle_sliders.music3_native import Music3Backend, ParticleNetwork
from particle_sliders.native_lora import LoRAModule, LoRANetwork
assert Music3Backend.__module__ == 'particle_sliders.music3_native'
assert LoRANetwork.__module__ == 'particle_sliders.native_lora'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_historical_imports_alias_one_native_implementation_and_token_helpers_match():
    if importlib.util.find_spec("conceptmod") is None:
        pytest.skip("historical source parity requires the repository checkout")
    root = Path(__file__).resolve().parents[3]
    trainer_path = root / "conceptmod/textsliders/train_lm_slider_music3.py"
    if not trainer_path.is_file():
        pytest.skip("historical token source is not included in the installed package")
    old_lora = importlib.import_module("conceptmod.textsliders.lora")
    old_music3 = importlib.import_module("conceptmod.textsliders.music3_particle_bridge")
    assert old_lora.LoRAModule is native_lora.LoRAModule
    assert old_lora.LoRANetwork is native_lora.LoRANetwork
    assert old_music3.Music3Backend is native.Music3Backend
    assert old_music3.ParticleNetwork is native.ParticleNetwork
    assert old_music3.Qwen3Attention is native.Qwen3Attention
    trainer = trainer_path.read_text()

    def functions(text):
        return {node.name: ast.dump(node) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef)}

    before, after = functions(trainer), functions(Path(native.__file__).read_text())
    for name in ("_assemble", "_tokenize", "_last_real_index", "_gather_last_hidden",
                 "_audio_start_token_id", "_assert_last_token_is_audio_start",
                 "_encode_static", "_encode_train"):
        assert after[name] == before[name], name


def test_native_prompt_template_tokenizer_flags_and_last_audio_start(monkeypatch):
    encoders = ModuleType("diffusers.modular_pipelines.minimax_music3.encoders")
    encoders._clean_caption = lambda text: "caption:" + text.strip()
    encoders._normalize_lyrics = lambda text: "lyrics:" + text.strip()
    monkeypatch.setitem(sys.modules, encoders.__name__, encoders)
    text = native.assemble_text(" soft piano ", " morning light ")
    assert text == ("<|im_start|><|caption_start|>caption:soft piano<|caption_end|>"
                    "<|lyrics_start|>lyrics:morning light<|lyrics_end|><|im_end|><|audio_start|>")

    class Tokenizer:
        def __call__(self, value, **kwargs):
            assert value == text
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": torch.tensor([[5, 31, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0]])}

        def convert_tokens_to_ids(self, value):
            assert value == "<|audio_start|>"
            return 31

    tokenizer = Tokenizer()
    ids, mask = native.tokenize_text(tokenizer, text, torch.device("cpu"))
    native.assert_audio_start(ids, mask, tokenizer, where="native test")
    with pytest.raises(RuntimeError, match="last real token is not"):
        native.assert_audio_start(torch.tensor([[31, 5, 0]]), mask, tokenizer, where="native test")


class NativeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=1)
        self.embed = nn.Embedding(32, 16)
        self.attention = native.Qwen3Attention(16)

    def model(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.attention(self.embed(input_ids)))


def test_model_loader_is_local_frozen_and_last_hidden_train_path_keeps_gradients(tmp_path, monkeypatch):
    model = NativeLM()
    calls = []
    tokenizer = object()
    transformers = ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("model", path, kwargs))
            return model

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    backend = native.Music3Backend(tmp_path, device="cpu")
    assert calls == [
        ("tokenizer", str(tmp_path / "tokenizer"), {"local_files_only": True}),
        ("model", str(tmp_path / "language_model"),
         {"torch_dtype": torch.bfloat16, "local_files_only": True}),
    ]
    assert backend.tokenizer is tokenizer
    assert backend.model is model and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    ids, mask = torch.tensor([[1, 7, 0], [2, 3, 4]]), torch.tensor([[1, 1, 0], [1, 1, 1]])
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    expected = torch.stack((hidden[0, 1], hidden[1, 2])).float()
    assert torch.equal(backend.hidden(ids, mask), expected)

    adapter = native.ParticleNetwork(model)
    for layer in adapter.network.unet_loras:
        nn.init.normal_(layer.lora_up.weight, std=.1)
    with adapter.scaled(1.):
        assert not backend.hidden(ids, mask).requires_grad
        value = backend.hidden(ids, mask, train=True)
        assert value.requires_grad
        value.square().mean().backward()
    assert adapter.particles.grad is not None and adapter.particles.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_routed_weights_zero_scale_nested_restore_and_existing_storage_format(tmp_path):
    torch.manual_seed(17)
    model = native._TinyLM(32).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 31]])
    baseline = model(ids).detach().clone()
    model_weights = deepcopy(model.state_dict())
    adapter = native.ParticleNetwork(model)
    assert all(isinstance(bridge, RoutedMLP) for bridge in adapter.bridges.values())
    for layer in adapter.network.unet_loras:
        nn.init.normal_(layer.lora_up.weight, std=.1)
    assert torch.equal(model(ids), baseline)
    with adapter.scaled(1.):
        expected = model(ids).detach().clone()
        assert not torch.equal(expected, baseline)
        with adapter.scaled(0.):
            assert torch.equal(model(ids), baseline)
        assert torch.equal(model(ids), expected)
    assert torch.equal(model(ids), baseline)
    path = tmp_path / "music3.safetensors"
    adapter.save(path, {"weights_kind": "live"})
    restored_model = native._TinyLM(32).eval().requires_grad_(False)
    restored_model.load_state_dict(model_weights)
    restored, metadata = native.ParticleNetwork.load(restored_model, path)
    assert metadata["format"] == "conceptmod-music3-routed-particle-lm-v1"
    assert metadata["weights_kind"] == "live"
    assert restored.state_dict().keys() == adapter.state_dict().keys()
    assert all(torch.equal(v, restored.state_dict()[name]) for name, v in adapter.state_dict().items())
    with restored.scaled(1.):
        assert torch.equal(restored_model(ids), expected)
    assert torch.equal(restored_model(ids), baseline)
