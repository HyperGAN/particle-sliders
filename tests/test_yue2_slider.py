"""Exercise real tiny upstream layers; no model downloads or GPU required."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("yue2", reason="Install the isolated YuE2 runtime to test this backend")
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.pipeline import SemanticResult, SymbolicPlan

from conceptmod.textsliders.yue2_backend import (
    YuE2Backend, YuE2Slider, aligned_suffix, attention_targets, normalized_mse,
)
from conceptmod.textsliders.train_lora_yue2 import load_rows, parse_args, prepare_pairs, train
from conceptmod.textsliders.infer_yue2 import parse_args as infer_args, render


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny():
    torch.manual_seed(4)
    return YuE2ForCausalLM(YuE2Config(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=48, vocab_size=128, max_latent_frames=32)).eval()


def backend_for(model):
    backend = YuE2Backend.__new__(YuE2Backend)
    backend.model = model
    return backend


def test_target_selection_and_freezing():
    model = tiny()
    targets = attention_targets(model)
    assert len(targets) == 8
    assert all(".self_attn." in k and "nar_" not in k for k in targets)
    network = YuE2Slider(model, rank=2, alpha=2)
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in network.parameters())
    with pytest.raises(ValueError, match="already attached"):
        YuE2Slider(model)


def test_scale_zero_exact_and_scope_restored():
    model = tiny()
    backend = backend_for(model)
    ids = [1, 2, 3, 4]
    baseline = backend.hidden(ids).detach()
    network = YuE2Slider(model, rank=2, alpha=2)
    for a in network.adapters.values():
        torch.nn.init.normal_(a.lora_up.weight, std=0.1)
    with network.scaled(0):
        assert torch.equal(baseline, backend.hidden(ids))
    with network.scaled(1):
        assert not torch.equal(baseline, backend.hidden(ids))
        with pytest.raises(RuntimeError), network.scaled(-0.5):
            raise RuntimeError("failed render")
        assert all(a.multiplier == 1 for a in network.adapters.values())
    assert torch.equal(baseline, backend.hidden(ids))
    with pytest.raises(ValueError, match="finite"):
        with network.scaled(float("nan")):
            pass


def test_native_hidden_matches_causal_lm_and_checkpoint_gradients():
    model = tiny()
    backend = backend_for(model)
    ids = [5, 6, 7, 8, 9]
    with torch.no_grad():
        assert torch.equal(model.lm_head(backend.hidden(ids)),
                           model(torch.tensor([ids]), use_cache=False).logits)
    network = YuE2Slider(model, rank=2, alpha=2)
    target = backend.hidden([3, 4, 7, 8, 9]).detach()
    grads = []
    for checkpointing in (False, True):
        network.zero_grad(set_to_none=True)
        with network.scaled(1):
            loss = normalized_mse(backend.hidden(ids, checkpointing=checkpointing), target)
            loss.backward()
        grads.append([p.grad.clone() for p in network.parameters()])
    assert any(g.abs().sum() > 0 for g in grads[0])
    assert all(torch.allclose(a, b) for a, b in zip(*grads))
    assert all(p.grad is None for p in model.parameters())


def test_checkpoint_round_trip_and_foreign_rejection(tmp_path):
    model = tiny()
    original = copy.deepcopy(model)
    network = YuE2Slider(model, rank=2, alpha=3)
    for a in network.adapters.values():
        torch.nn.init.normal_(a.lora_up.weight, std=0.01)
    path = tmp_path / "slider.safetensors"
    network.save(path, {"dummy": False, "model_identity": {"test": True}})
    loaded, record = YuE2Slider.load(original, path)
    assert record["rank"] == 2
    with network.scaled(0.75), loaded.scaled(0.75):
        assert torch.equal(backend_for(model).hidden([1, 2, 3]), backend_for(original).hidden([1, 2, 3]))
    foreign = tmp_path / "foreign.safetensors"
    save_file({"weight": torch.ones(1)}, str(foreign))
    with pytest.raises(ValueError, match="Not a native"):
        YuE2Slider.load(tiny(), foreign)
    record["targets"].append("model.layers.0.nar_self_attn.q_proj")
    save_file(load_file(str(path)), str(foreign), metadata={"conceptmod": json.dumps(record)})
    with pytest.raises(ValueError, match="target list"):
        YuE2Slider.load(tiny(), foreign)


@pytest.mark.parametrize("damage", ["missing", "nan", "alpha", "architecture", "dummy"])
def test_checkpoint_validation_before_attachment(tmp_path, damage):
    network = YuE2Slider(tiny(), rank=2, alpha=2)
    path = tmp_path / "slider.safetensors"
    network.save(path, {"dummy": False})
    state = load_file(str(path))
    record = json.loads(path.with_suffix(".json").read_text())
    if damage == "missing":
        state.pop(next(iter(state)))
    elif damage == "nan":
        key = next(k for k in state if k.endswith("lora_up.weight"))
        state[key].fill_(float("nan"))
    elif damage == "alpha":
        record["alpha"] = 3
    elif damage == "architecture":
        record["architecture"]["hidden_size"] = 64
    else:
        record["dummy"] = True
    save_file(state, str(path), metadata={"conceptmod": json.dumps(record)})
    model = tiny()
    original_forward = model.model.layers[0].self_attn.q_proj.forward
    with pytest.raises(ValueError):
        YuE2Slider.load(model, path)
    assert model.model.layers[0].self_attn.q_proj.forward == original_forward


def test_unequal_prefix_alignment_and_native_dummy(tmp_path):
    backend = YuE2Backend(dummy=True)
    rows = [{"neutral": "Dry clear voice", "positive": "Soft close breathy voice with audible air",
             "lyrics": "[Verse]\nThe rain falls on the step\n"}]
    pairs = prepare_pairs(backend, rows, 3, 7, 512)
    assert pairs[0]["teacher"].shape[1] == 4
    assert pairs[0]["hold"].shape[1] > 0
    assert not pairs[0]["teacher"].requires_grad
    assert aligned_suffix([1, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8]) == 4
    with pytest.raises(ValueError, match="max_seq_len"):
        prepare_pairs(backend, rows, 3, 7, 20)


def test_train_smoke_and_dummy_export_rejected(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("rows:\n  - neutral: clear vocal\n    positive: breathy vocal\n    lyrics: '[Verse] rain on the step'\n")
    args = parse_args(["--recipe", "hidden", "--dummy", "--steps", "3", "--train_tokens", "2", "--rank", "2",
                       "--alpha", "2", "--prompts_file", str(path), "--save_dir", str(tmp_path)])
    train(args)
    weights = tmp_path / "breath-yue2-ar_last.safetensors"
    state = load_file(str(weights))
    assert any(t.abs().sum() > 0 for k, t in state.items() if k.endswith("lora_up.weight"))
    log = [json.loads(line) for line in (tmp_path / "breath-yue2-ar_train.jsonl").read_text().splitlines()]
    assert len(log) == 3 and all(np.isfinite(r["loss"]) for r in log)
    with pytest.raises(ValueError, match="Dummy"):
        YuE2Slider.load(tiny(), weights)


def test_prompt_and_config_validation(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("rank: 4\nsteps: 20\n")
    args = parse_args(["--config_file", str(config), "--steps", "3"])
    assert (args.rank, args.steps) == (4, 3)
    assert args.save_dir.parent.name == "models"
    prompts = Path(__file__).resolve().parents[1] / "conceptmod/textsliders/data/prompts-yue2.yaml"
    rows = load_rows(prompts)
    assert len(rows) == 4 and rows[0]["neutral"].startswith("Low male register.")
    bad = tmp_path / "bad.yaml"
    bad.write_text("rows:\n  - positive: same\n    neutral: same\n    lyrics: rain\n")
    with pytest.raises(ValueError, match="must differ"):
        load_rows(bad)
    bad.write_text("rows:\n  - positive: 'Example — meaning a dry kit'\n    neutral: voice\n    lyrics: rain\n")
    with pytest.raises(ValueError, match="named references"):
        load_rows(bad)
    with pytest.raises(SystemExit):
        parse_args(["--hold_weight", "nan"])


def test_render_stages_and_adapter_identity():
    events = []
    class Network:
        scale = 0
        @contextmanager
        def scaled(self, scale):
            old, self.scale = self.scale, scale
            try:
                yield
            finally:
                self.scale = old
    network = Network()
    class Pipeline:
        backend = "torch-eager"
        quantization = "none"
        offload_ar = False
        weights = {"mot": "test"}
        def plan(self, request):
            events.append(("plan", network.scale))
            return SymbolicPlan(request, None, [], [1])
        def generate_semantic(self, plan, sampling=None):
            events.append(("semantic", network.scale))
            return SemanticResult(plan, [1, 2], {}, True)
        def synthesize(self, semantic):
            events.append(("synthesize", network.scale))
            return np.zeros((2, 64))
        def decode(self, latents):
            events.append(("decode", network.scale))
            return np.zeros((480, 2))
        def effective_config(self, request, **kwargs):
            return {"cot": request.cot}
    first = render(Pipeline(), network, style="dry vocal", lyrics="[Verse]\nRain", scale=0.5,
                   seed=7, adapter_identity="abc")
    assert events == [("plan", .5), ("semantic", .5), ("synthesize", 0), ("decode", 0)]
    assert first.truncated["semantic"] and first.sample_rate == 48000
    assert first.config["conceptmod"]["adapter"] == "abc"
    second = render(Pipeline(), network, style="dry vocal", lyrics="[Verse]\nRain", scale=1,
                    seed=7, adapter_identity="abc")
    assert first.request_identity != second.request_identity


def test_render_cli_validation():
    required = ["--weights", "a.safetensors", "--style", "dry voice", "--lyrics_file", "lyrics.txt",
                "--output_dir", "/tmp/out"]
    assert infer_args(required).scales == [0, .5, 1]
    for extra in (["--scales=nan"], ["--scales=0,0"], ["--abc_file", "score.abc"]):
        with pytest.raises(SystemExit):
            infer_args(required + extra)
