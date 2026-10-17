"""Continuous PEFT-style delta scales retain their original alpha/r factors."""
import ast
from pathlib import Path

import torch
from torch import nn

from particle_sliders import native_lora


class DeltaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.lora_B = nn.Linear(2, 2, bias=False)
        self.scaling = {"default": 2., "secondary": .5}
        self.lora_scale = 1.
        self.multiplier = 1.

    def forward(self, value):
        return value + self.lora_B(self.lora_A(value)) * self.scaling["default"]


def test_continuous_scale_restores_native_adapter_factors_without_compounding():
    layer = DeltaLayer()
    # A non-LoRA child may have fields named scaling or multiplier. It must
    # not be changed merely because the surrounding model owns adapters.
    unrelated = nn.Linear(2, 2)
    unrelated.scaling, unrelated.multiplier = {"default": 7.}, 9.
    model = nn.ModuleList([layer, unrelated])
    value = torch.tensor([[1., 2.]])
    baseline, full = value.clone(), layer(value).detach().clone()
    assert native_lora.is_lora_delta_layer is native_lora._is_lora_delta_layer
    assert native_lora.is_lora_delta_layer(layer)
    assert not native_lora.is_lora_delta_layer(unrelated)
    for scale in (.25, .5, 0., 1., .25, 1.):
        assert native_lora.apply_continuous_lora_scale(model, scale) == 3
        assert layer.scaling == {"default": 2. * scale, "secondary": .5 * scale}
        assert layer.lora_scale == layer.multiplier == scale
        torch.testing.assert_close(layer(value), baseline + (full - baseline) * scale)
        assert unrelated.scaling == {"default": 7.}
        assert unrelated.multiplier == 9.
    assert layer._krea_base_scaling == {"default": 2., "secondary": .5}


def test_embedding_delta_layers_and_local_network_scale_api():
    embedding = nn.Module()
    embedding.lora_embedding_A = nn.Parameter(torch.ones(1))
    embedding.scaling = {"embedding": 4.}
    assert native_lora.apply_continuous_lora_scale(embedding, .5) == 1
    assert embedding.scaling["embedding"] == 2.
    assert native_lora.apply_continuous_lora_scale(embedding, 1.) == 1
    assert embedding.scaling["embedding"] == 4.

    # Native down/up LoRA uses its explicit network scale/context API. The
    # extracted PEFT predicate must retain its original field convention.
    host = nn.Linear(2, 2, bias=False)
    local = native_lora.LoRAModule("local", host, lora_dim=2, alpha=4.)
    local.apply_to()
    nn.init.ones_(local.lora_up.weight)
    value = torch.ones(1, 2)
    base = local.org_forward(value)
    local.multiplier = .25
    quarter = host(value)
    assert native_lora.apply_continuous_lora_scale(local, .5) == 0
    assert local.multiplier == .25
    local.multiplier = 1.
    torch.testing.assert_close(quarter, base + (host(value) - base) * .25)


def test_extracted_scale_helpers_match_historical_source_when_available():
    historical = Path(__file__).resolve().parents[3] / "conceptmod/textsliders/slider_targets.py"
    if not historical.is_file():
        import pytest
        pytest.skip("historical scale source is not included in the installed package")

    def functions(text):
        return {node.name: ast.dump(node) for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef)}

    before = functions(historical.read_text())
    after = functions(Path(native_lora.__file__).read_text())
    for name in ("_is_lora_delta_layer", "apply_continuous_lora_scale"):
        assert before[name] == after[name]
