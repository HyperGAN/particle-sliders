"""Tiny-LLM UNI slider: CPU stand-in only. No Hub, no GPU, no Qwen weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders.tiny_llm_backend import (
    ATTN_CLASS_NAMES,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_MODEL,
    FORMAT,
    LORA_LINEAR_NAMES,
    MODEL_PARAMS_TOTAL,
    ArchitectureMismatch,
    DummyTinyCausalLM,
    TinyLLMBackend,
    TinySlider,
    hidden_delta_metrics,
    resolve_tiny_lora_path,
    tiny_uni_loss,
)
from conceptmod.textsliders.train_lora_tiny_llm import (
    DEFAULT_SAMPLE_SCALES,
    load_slider_rows,
    main as train_main,
    parse_args,
    parse_report_scales,
    train,
)


def test_pinned_model_is_small_recent_public_lm():
    assert DEFAULT_MODEL == "Qwen/Qwen3-0.6B-Base"
    assert MODEL_PARAMS_TOTAL == 596049920
    assert MODEL_PARAMS_TOTAL < 1_100_000_000
    assert FORMAT == "conceptmod-tiny-llm-uni-v1"
    assert DEFAULT_LORA_UP_INIT_STD == 0.02
    args = parse_args(["--dummy"])
    assert args.model_id == "Qwen/Qwen3-0.6B-Base"
    assert args.rank == 8
    assert args.alpha == 8.0
    assert args.steps == 20
    assert args.device == "cpu"
    assert args.dummy is True
    assert args.load_tiny_lora is None
    assert args.no_report is False
    assert parse_report_scales(args.report_scales) == list(DEFAULT_SAMPLE_SCALES)


def test_lora_attaches_only_to_attention_qkvo():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    names = backend.lora_module_names()
    assert names
    assert len(names) == 2 * len(LORA_LINEAR_NAMES)  # dummy has 2 layers
    assert LORA_LINEAR_NAMES == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert set(ATTN_CLASS_NAMES) >= {"TinyAttention", "Qwen3Attention"}
    joined = " ".join(names)
    for piece in LORA_LINEAR_NAMES:
        assert piece in joined, (piece, names)
    for banned in ("mlp", "gate", "up_proj", "down_proj", "embed", "norm", "lm_head"):
        assert banned not in joined.lower(), (banned, names)
    params = backend.trainable_parameters()
    assert params
    trainable_ids = {id(p) for p in params}
    host_params = [p for p in backend.model.parameters() if p.requires_grad]
    assert host_params == [], "host model must stay frozen"
    mlp_ids = {id(p) for m in backend.model.model.layers for p in m.mlp.parameters()}
    assert not mlp_ids.intersection(trainable_ids)
    assert id(backend.model.lm_head.weight) not in trainable_ids


def test_scale_zero_is_exact_base():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    ids = backend.encode("a blooming rose garden at dawn").ids
    teacher = backend.teacher_hidden(ids)
    student = backend.hidden(ids, scale=0.0)
    assert torch.equal(teacher, student)
    moved = backend.hidden(ids, scale=1.0)
    assert not torch.equal(moved, teacher)


def test_plus_and_neu_hiddens_differ():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    plus = backend.teacher_hidden(backend.encode("a blooming rose garden").ids)
    neu = backend.teacher_hidden(backend.encode("a garden").ids)
    common = min(plus.shape[1], neu.shape[1])
    assert not torch.allclose(plus[:, :common], neu[:, :common])
    loss = tiny_uni_loss(plus[:, :common], plus[:, :common], neu, neu)
    assert float(loss.item()) == pytest.approx(0.0)
    assert float(tiny_uni_loss(plus[:, :common], neu, neu, neu).item()) > 0


def _uni_gap(backend: TinyLLMBackend) -> float:
    ids = backend.encode("a blooming rose garden at dawn").ids
    pred = backend.hidden(ids, scale=1.0)
    tgt = backend.teacher_hidden(ids)
    return float(torch.mean((pred - tgt) ** 2).item())


def test_zero_init_lora_up_is_uni_identity():
    model = DummyTinyCausalLM()
    slider = TinySlider(model, rank=4, alpha=4.0, up_init_std=0.0)
    for adapter in slider.adapters.values():
        assert torch.count_nonzero(adapter.lora_up.weight).item() == 0
    backend = TinyLLMBackend(device="cpu", dummy=True, lora_up_init_std=0.0)
    assert _uni_gap(backend) < 1e-10


def test_noisy_lora_up_gives_nonzero_uni_gap():
    torch.manual_seed(0)
    model = DummyTinyCausalLM()
    slider = TinySlider(model, rank=4, alpha=4.0, up_init_std=DEFAULT_LORA_UP_INIT_STD)
    assert any(
        float(a.lora_up.weight.detach().abs().max()) > 0
        for a in slider.adapters.values()
    )
    torch.manual_seed(0)
    backend = TinyLLMBackend(device="cpu", dummy=True)
    assert _uni_gap(backend) > 1e-8


def test_dummy_train_drops_uni_loss_and_writes_sidecar(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy",
        "--steps", "20",
        "--name", "tiny-llm-dummy",
        "--save_dir", str(tmp_path),
        "--prompts_file", str(prompts),
        "--lr", "1e-3",
        "--no_report",
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert sidecar["resolved_model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert sidecar["model_params_total"] == 596049920
    assert sidecar["backend"] == "tiny_llm"
    assert sidecar["format"] == "conceptmod-tiny-llm-uni-v1"
    assert sidecar["stack"] == "causal_lm_hidden"
    assert sidecar["recipe"] == "tiny_llm_uni_hidden"
    assert sidecar["plus_neu"] is True
    assert sidecar["minus_teacher"] is False
    assert sidecar["lora_only"] is True
    assert sidecar["lora_linears"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert sidecar["train_mlp"] is False
    assert sidecar["train_lm_head"] is False
    assert sidecar["lora_up_init_std"] == 0.02
    assert sidecar["first_loss"] > sidecar["last_loss"]
    data = json.loads((tmp_path / "tiny-llm-dummy_last.json").read_text())
    assert data["backend"] == "tiny_llm"


def test_dummy_train_writes_hidden_delta_report(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy",
        "--steps", "4",
        "--name", "tiny-llm-report",
        "--save_dir", str(tmp_path),
        "--prompts_file", str(prompts),
        "--report_scales", "0,1",
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["report"]["scales"] == [0.0, 1.0]
    report = json.loads((tmp_path / "report" / "hidden_delta.json").read_text())
    assert len(report) == 2
    by_scale = {r["scale"]: r for r in report}
    assert by_scale[0.0]["delta_l2"] == pytest.approx(0.0, abs=1e-5)
    assert by_scale[0.0]["delta_cos"] == pytest.approx(1.0, abs=1e-4)
    assert by_scale[1.0]["delta_l2"] >= 0.0
    for rec in report:
        assert rec["teacher_delta_norm"] > 0


def test_load_tiny_lora_roundtrip_and_steps_zero_report(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    train_dir = tmp_path / "trained"
    train(parse_args([
        "--dummy",
        "--steps", "4",
        "--name", "tiny-llm-load",
        "--save_dir", str(train_dir),
        "--prompts_file", str(prompts),
        "--no_report",
        "--seed", "1",
    ]))
    lora_path = resolve_tiny_lora_path(str(train_dir))
    assert lora_path.name.endswith(".safetensors")
    src = TinyLLMBackend(device="cpu", dummy=True)
    src.load_trained(str(train_dir))
    dst = TinyLLMBackend(device="cpu", dummy=True)
    dst.load_trained(str(lora_path))
    for a, b in zip(src.slider.parameters(), dst.slider.parameters()):
        assert torch.allclose(a, b)

    reload_dir = tmp_path / "reload"
    sidecar = train(parse_args([
        "--dummy",
        "--steps", "0",
        "--name", "tiny-llm-reload",
        "--save_dir", str(reload_dir),
        "--prompts_file", str(prompts),
        "--load_tiny_lora", str(train_dir),
        "--report_scales", "0,1",
    ]))
    assert sidecar["steps"] == 0
    assert sidecar["first_loss"] is None
    assert sidecar["load_tiny_lora"]
    assert (reload_dir / "report" / "hidden_delta.json").is_file()


def test_load_tiny_lora_rejects_non_tiny_keys(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "peft.safetensors"
    save_file({"base_model.model.foo.weight": torch.zeros(2, 2)}, str(path))
    backend = TinyLLMBackend(device="cpu", dummy=True)
    with pytest.raises(ValueError, match="lora_tiny"):
        backend.load_trained(str(path))


def test_slider_rejects_non_attention_host():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(ArchitectureMismatch, match="Qwen3-shaped"):
        TinySlider(model, rank=4, alpha=4.0)


def test_live_load_is_not_imported_on_dummy():
    import sys

    import conceptmod.textsliders.tiny_llm_backend as tiny

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live loader must not run in dummy mode")

    orig = tiny._load_live_model
    tiny._load_live_model = boom
    try:
        backend = TinyLLMBackend(device="cpu", dummy=True)
        _ = backend.encode("a garden")
        _ = backend.hidden(backend.encode("a garden").ids, scale=1.0)
    finally:
        tiny._load_live_model = orig
    assert called["n"] == 0
    assert "transformers" not in sys.modules


def test_yaml_rows_and_config_card():
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-tiny-llm.yaml")
    assert len(rows) == 3
    for row in rows:
        assert row["positive"] != row["neutral"]
        assert row["positive"] and row["neutral"]
    cfg = yaml.safe_load(
        Path("conceptmod/textsliders/data/config-tiny-llm.yaml").read_text()
    )
    assert cfg["pretrained_model"]["name_or_path"] == "Qwen/Qwen3-0.6B-Base"
    assert cfg["pretrained_model"]["params_total"] == 596049920
    assert cfg["train"]["recipe"] == "tiny_llm_uni_hidden"
    assert cfg["train"]["iterations"] == 20
    assert cfg["network"]["target"] == "Qwen3Attention"
    assert cfg["network"]["linears"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg["network"]["train_mlp"] is False
    assert cfg["network"]["train_lm_head"] is False


def test_production_trainer_defaults_unchanged():
    tf_src = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert 'parser.add_argument("--steps", type=int, default=500)' in tf_src
    assert 'parser.add_argument("--rank", type=int, default=8)' in tf_src
    assert 'parser.add_argument("--lr", type=float, default=2e-3' in tf_src
    lm_src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"--lm_target"' in lm_src and 'default="v9"' in lm_src
    assert '"--pole_mode"' in lm_src and 'default="hidden"' in lm_src
    music3_yaml = Path("conceptmod/textsliders/data/prompts-music3.yaml").read_text()
    assert "tiny-llm" not in music3_yaml
    assert "Qwen3-0.6B" not in music3_yaml
    for name in (
        "train_lora_yue2.py",
        "train_lora_yue2_arm_b.py",
        "yue2_backend.py",
    ):
        src = Path("conceptmod/textsliders") / name
        if src.is_file():
            assert "tiny-llm" not in src.read_text(), name


def test_hidden_delta_metrics_identity():
    h = torch.ones(1, 3, 8)
    z = torch.zeros(1, 3, 8)
    m = hidden_delta_metrics(h, z, h, z)
    assert m["delta_cos"] == pytest.approx(1.0)
    assert m["delta_l2"] == pytest.approx(0.0)
    assert m["teacher_delta_norm"] > 0
    gap = hidden_delta_metrics(h, z, z, z)
    assert gap["delta_l2"] > 0


def test_config_file_flows_into_train(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "pretrained_model:\n  name_or_path: Qwen/Qwen3-0.6B-Base\n"
        "network:\n  rank: 4\n  alpha: 4.0\n"
        "train:\n  iterations: 2\n  lr: 0.01\n"
        f"prompts_file: {prompts}\n"
        "save:\n  name: tiny-llm-cfg\n"
    )
    sidecar = train_main([
        "--dummy",
        "--config_file", str(cfg),
        "--save_dir", str(tmp_path / "out"),
        "--no_report",
        "--seed", "0",
    ])
    assert sidecar["rank"] == 4
    assert sidecar["name"] == "tiny-llm-cfg"
    assert sidecar["steps"] == 2
