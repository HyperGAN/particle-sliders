"""Tiny-LLM particle-bridge slider: CPU stand-in only. No Hub, no GPU, no Qwen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders import particle_bridge_gan as yue2_game
from conceptmod.textsliders import tiny_llm_particle as tiny_game
from conceptmod.textsliders.tiny_llm_backend import (
    ATTN_CLASS_NAMES,
    DEFAULT_MODEL,
    LORA_LINEAR_NAMES,
    MODEL_PARAMS_TOTAL,
    ArchitectureMismatch,
    TinyLLMBackend,
    hidden_delta_metrics,
)
from conceptmod.textsliders.tiny_llm_particle import (
    FORMAT,
    RECIPE,
    TinyParticleSlider,
    _attention_targets,
    build_game,
    prepare_rows,
    resolve_particle_path,
    update,
)
from conceptmod.textsliders.train_lora_tiny_llm import (
    DEFAULT_SAMPLE_SCALES,
    load_slider_rows,
    main as train_main,
    parse_args,
    parse_report_scales,
    train,
)


def test_pinned_model_and_recipe_names():
    assert DEFAULT_MODEL == "Qwen/Qwen3-0.6B-Base"
    assert MODEL_PARAMS_TOTAL == 596049920
    assert MODEL_PARAMS_TOTAL < 1_100_000_000
    assert FORMAT == "conceptmod-tiny-llm-particle-v1"
    assert RECIPE["name"] == "anneal-routed-particle-error-tiny-llm-v1"
    assert RECIPE["source_recipe"] == "anneal-routed-particle-error-yue2-v1"
    assert RECIPE["propose_only"] is True
    args = parse_args(["--dummy"])
    assert args.recipe == "particle_bridge"
    assert args.model_id == "Qwen/Qwen3-0.6B-Base"
    assert args.seed == 7
    assert args.device == "cpu"
    assert args.dummy is True
    assert parse_report_scales(args.report_scales) == list(DEFAULT_SAMPLE_SCALES)
    with pytest.raises(SystemExit):
        parse_args(["--dummy", "--recipe", "unipolar_gan"])


def test_game_is_the_shared_yue2_module_not_a_copy():
    assert tiny_game.shared is yue2_game
    ref = yue2_game.REFERENCE
    assert RECIPE["g_lr"] == ref["g_lr"] == 0.0006
    assert RECIPE["d_lr"] == ref["d_lr"] == 0.0009
    assert RECIPE["particle_lr"] == ref["particle_lr"] == 0.006
    assert tuple(RECIPE["optimizer_betas"]) == tuple(ref["betas"]) == (0.0, 0.999)
    assert RECIPE["ema"] == ref["ema"] == 0.995
    assert RECIPE["vicreg_weight"] == ref["vic_coeff"] == 1.0
    assert RECIPE["penalty_lazy_k"] == ref["cap_every"] == 4
    assert RECIPE["noise_floor"] == ref["noise_floor"] == 0.03
    assert RECIPE["noise_decay_steps"] == ref["noise_decay_steps"] == 8000
    assert RECIPE["parts"] == ref["particles"] == 128
    assert RECIPE["particle_dim"] == ref["z_dim"] == 4
    assert RECIPE["adapter_width"] == ref["width"] == 48
    assert RECIPE["router_width"] == ref["router_width"] == 16
    assert RECIPE["adv_batch"] == ref["batch_size"] == 64
    assert yue2_game.noise_std(0) == pytest.approx(1.0)
    assert yue2_game.noise_std(8000) == pytest.approx(0.03)
    assert yue2_game.noise_std(20000) == pytest.approx(0.03)


def test_build_game_uses_reference_lrs_and_betas():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-tiny-llm.yaml")
    fixed = prepare_rows(backend, rows)
    network = TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    critic, g, d = build_game(backend, network, fixed)
    assert critic.target_mean.shape == (backend.model.config["hidden_size"],)
    roles = {group.get("role", "generator"): group for group in g.param_groups}
    ref = yue2_game.REFERENCE
    assert roles["generator"]["lr"] == ref["g_lr"]
    assert roles["particles"]["lr"] == ref["particle_lr"]
    assert list(roles["generator"]["betas"]) == list(ref["betas"])
    assert d.param_groups[0]["lr"] == ref["d_lr"]
    assert list(d.param_groups[0]["betas"]) == list(ref["betas"])


def test_particle_branches_attach_only_to_attention_qkvo():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    network = TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    assert network.particles.shape == (128, 4)
    names = list(network.adapters.keys())
    assert len(names) == 2 * len(LORA_LINEAR_NAMES)  # dummy has 2 layers
    assert LORA_LINEAR_NAMES == ("q_proj", "k_proj", "v_proj", "o_proj")
    joined = " ".join(names)
    for piece in LORA_LINEAR_NAMES:
        assert piece in joined, (piece, names)
    for banned in ("mlp", "gate", "embed", "norm", "lm_head"):
        assert banned not in joined.lower(), (banned, names)
    assert all("lora_tiny-" in name for name in names)
    # Host stays frozen; only adapters + the one shared cloud train.
    assert [p for p in backend.model.parameters() if p.requires_grad] == []
    trainable = {id(p) for p in network.parameters() if p.requires_grad}
    mlp_ids = {id(p) for m in backend.model.model.layers for p in m.mlp.parameters()}
    assert not mlp_ids.intersection(trainable)
    assert id(backend.model.lm_head.weight) not in trainable
    assert id(network.particles) in trainable
    # up starts at zero: freshly attached adapter is the base behavior.
    for adapter in network.adapters.values():
        assert torch.count_nonzero(adapter.lora_up.weight).item() == 0


def test_rank_alpha_pin_and_double_attach_guard():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    with pytest.raises(ValueError, match="pins rank/alpha 8"):
        TinyParticleSlider(backend.model, rank=4, alpha=4.0)
    TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    with pytest.raises(ValueError, match="already attached"):
        TinyParticleSlider(backend.model, rank=8, alpha=8.0)


def test_slider_rejects_non_attention_host():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(ArchitectureMismatch, match="Qwen3-shaped"):
        TinyParticleSlider(model, rank=8, alpha=8.0)


def test_scale_zero_is_exact_base_before_and_after_train(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy", "--steps", "4", "--name", "tiny-zero",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["steps"] == 4
    # Post-train live weights still bypass exactly at scale 0.
    backend = TinyLLMBackend(device="cpu", dummy=True)
    network, _ = TinyParticleSlider.load(
        backend.model, resolve_particle_path(str(tmp_path)))
    ids = backend.encode("a blooming rose garden at dawn").ids
    teacher = backend.teacher_hidden(ids)
    with network.scaled(0.0):
        assert torch.equal(backend.hidden(ids), teacher)
    with network.scaled(1.0):
        assert not torch.equal(backend.hidden(ids), teacher)


def test_shared_build_game_rejects_wrong_cloud_shape():
    backend = TinyLLMBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-tiny-llm.yaml")
    fixed = prepare_rows(backend, rows)
    network = TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    network.particles = torch.nn.Parameter(torch.randn(16, 4))
    with pytest.raises(ValueError, match="128x4"):
        build_game(backend, network, fixed)


def test_dummy_train_runs_shared_game_with_no_mse(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy", "--steps", "6", "--name", "tiny-game",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--report_scales", "0,1", "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["recipe"] == "anneal-routed-particle-error-tiny-llm-v1"
    assert sidecar["game_module"] == "conceptmod.textsliders.particle_bridge_gan"
    assert sidecar["output_mse"] is False
    assert sidecar["propose_only"] is True
    assert sidecar["unipolar"] is True
    ref = yue2_game.REFERENCE
    for key in ("g_lr", "d_lr", "particle_lr", "ema", "vic_coeff",
                "cap_coeff", "cap_kappa", "cap_every",
                "noise_floor", "noise_decay_steps"):
        assert sidecar[key] == ref[key], key
    assert sidecar["optimizer_betas"] == list(ref["betas"])
    assert sidecar["particles"] == [128, 4]
    history = sidecar["history"]
    assert len(history) == 6
    for rec in history:
        assert set(rec) >= {
            "step", "loss", "g_adv", "particle_vic", "d_loss", "d_adv",
            "d_pen", "grad_norm", "particle_grad_norm",
            "particle_gan_grad_norm", "noise_std", "cos_pos",
        }
        assert "mse" not in json.dumps(rec).lower()
        for num in ("loss", "g_adv", "particle_vic", "d_loss", "cos_pos"):
            assert abs(rec[num]) != float("inf") and rec[num] == rec[num]
        assert rec["noise_std"] == pytest.approx(
            yue2_game.noise_std(rec["step"]))
    assert history[0]["noise_std"] == pytest.approx(yue2_game.noise_std(1))
    # EMA + live exports exist; report pins scale-0 exactness post-train.
    assert (tmp_path / "tiny-game_last.safetensors").is_file()
    assert (tmp_path / "tiny-game_live_last.safetensors").is_file()
    report = json.loads((tmp_path / "report" / "hidden_delta.json").read_text())
    assert len(report) == 4
    by_scale = {}
    for rec in report:
        by_scale.setdefault(rec["scale"], []).append(rec)
    for rec in by_scale[0.0]:
        assert rec["delta_l2"] == pytest.approx(0.0, abs=1e-5)
        assert rec["delta_cos"] == pytest.approx(1.0, abs=1e-4)


def test_save_load_roundtrip_and_rejections(tmp_path):
    backend = TinyLLMBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-tiny-llm.yaml")
    fixed = prepare_rows(backend, rows)
    network = TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    critic, g, d = build_game(backend, network, fixed)
    sampler = yue2_game.BridgeSampler(len(rows), 0)
    update(backend, network, critic, g, d, fixed, sampler=sampler, step=1)
    path = tmp_path / "tiny-particle.safetensors"
    network.save(path, {"weights_kind": "live"})
    fresh = TinyLLMBackend(device="cpu", dummy=True)
    reloaded, record = TinyParticleSlider.load(fresh.model, path)
    assert record["format"] == FORMAT
    assert record["particles"] == 128 and record["particle_dim"] == 4
    for a, b in zip(network.parameters(), reloaded.parameters()):
        assert torch.allclose(a, b)

    from safetensors.torch import save_file
    peft = tmp_path / "peft.safetensors"
    save_file({"base_model.model.foo.weight": torch.zeros(2, 2)}, str(peft))
    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        TinyParticleSlider.load(fresh.model, peft)
    bad = dict(network.state_dict())
    bad["adapters.unrelated.lora_down.weight"] = torch.zeros(1, 1)
    bad_path = tmp_path / "bad.safetensors"
    meta = {"conceptmod": json.dumps({
        "format": FORMAT, "rank": 8, "alpha": 8.0, "particles": 128,
        "particle_dim": 4, "bridge_width": 48, "router_width": 16,
        "targets": record["targets"],
    })}
    save_file({k: v.cpu() for k, v in bad.items()}, str(bad_path), metadata=meta)
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        TinyParticleSlider.load(fresh.model, bad_path)


def test_steps_zero_load_reports_without_training(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    train_dir = tmp_path / "trained"
    train(parse_args([
        "--dummy", "--steps", "2", "--name", "tiny-load",
        "--save_dir", str(train_dir), "--prompts_file", str(prompts),
        "--no_report", "--seed", "1",
    ]))
    reload_dir = tmp_path / "reload"
    sidecar = train(parse_args([
        "--dummy", "--steps", "0", "--name", "tiny-reload",
        "--save_dir", str(reload_dir), "--prompts_file", str(prompts),
        "--load_tiny_lora", str(train_dir), "--report_scales", "0,1",
    ]))
    assert sidecar["steps"] == 0
    assert sidecar["history"] == []
    assert sidecar["load_tiny_lora"].endswith("_last.safetensors")
    assert (reload_dir / "report" / "hidden_delta.json").is_file()


def test_needs_two_rows(tmp_path):
    one = tmp_path / "one.yaml"
    one.write_text(
        "- target: garden caption\n  positive: a blooming rose garden\n"
        "  neutral: a garden\n  unconditional: ''\n"
    )
    with pytest.raises(ValueError, match="at least two"):
        load_slider_rows(str(one))


def test_live_load_is_not_imported_on_dummy():
    import subprocess
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
        _ = backend.hidden(backend.encode("a garden").ids)
    finally:
        tiny._load_live_model = orig
    assert called["n"] == 0
    # Other tests may already have imported transformers in this process.
    probe = (
        "import sys\n"
        "from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend\n"
        "backend = TinyLLMBackend(device='cpu', dummy=True)\n"
        "backend.hidden(backend.encode('a garden').ids)\n"
        "assert 'transformers' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_yaml_rows_and_config_card_match_shared_reference():
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-tiny-llm.yaml")
    assert len(rows) == 3
    for row in rows:
        assert row["positive"] != row["neutral"]
    cfg = yaml.safe_load(
        Path("conceptmod/textsliders/data/config-tiny-llm.yaml").read_text()
    )
    assert cfg["pretrained_model"]["name_or_path"] == "Qwen/Qwen3-0.6B-Base"
    assert cfg["pretrained_model"]["params_total"] == 596049920
    assert cfg["train"]["recipe"] == "particle_bridge"
    assert cfg["game"]["module"] == "conceptmod.textsliders.particle_bridge_gan"
    ref = yue2_game.REFERENCE
    for key in ("g_lr", "d_lr", "particle_lr", "ema", "vic_coeff",
                "cap_coeff", "cap_kappa", "cap_every",
                "noise_floor", "noise_decay_steps"):
        assert cfg["game"][key] == ref[key], key
    assert cfg["game"]["particles"] == ref["particles"] == 128
    assert cfg["game"]["particle_dim"] == ref["z_dim"] == 4
    assert cfg["network"]["target"] == "Qwen3Attention"
    assert cfg["network"]["linears"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg["network"]["train_mlp"] is False
    assert cfg["network"]["train_lm_head"] is False


def test_production_files_unchanged():
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
        "yue2_particle_bridge.py",
        "particle_bridge_gan.py",
        "yue2_arm_b.py",
    ):
        src = Path("conceptmod/textsliders") / name
        if src.is_file():
            assert "tiny-llm" not in src.read_text(), name
            assert "tiny_llm" not in src.read_text(), name
    adv_src = Path("analysis/slider2d/adv.py").read_text()
    assert "tiny" not in adv_src


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
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "pretrained_model:\n  name_or_path: Qwen/Qwen3-0.6B-Base\n"
        "train:\n  iterations: 2\n"
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
    assert sidecar["name"] == "tiny-llm-cfg"
    assert sidecar["steps"] == 2
    assert sidecar["recipe"] == "anneal-routed-particle-error-tiny-llm-v1"
