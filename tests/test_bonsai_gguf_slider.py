"""Bonsai-GGUF particle slider: pins, guards, dummy smoke. No Hub, no GPU."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders import particle_bridge_gan as yue2_game
from conceptmod.textsliders import bonsai_gguf_particle as bonsai_game
from conceptmod.textsliders.bonsai_gguf_backend import (
    ARCH,
    BASE_MODEL,
    CONTEXT,
    DEMO_URL,
    EMBED_POOLING,
    F16_FILENAME,
    FILE_BYTES,
    FILENAME,
    FORK_RELEASE_PIN,
    FORK_URL,
    HIDDEN,
    LAYERS,
    MMPROJ_BF16_FILENAME,
    MMPROJ_Q8_FILENAME,
    N_TENSORS,
    PQ2_0_FILENAME,
    Q2_0_DEV_FILENAME,
    REPO,
    TERNARY_TYPE_IDS,
    VOCAB,
    BonsaiGGUFBackend,
    BonsaiGGUFError,
    StockLlamaRejected,
    build_server_cmd,
    check_fork_log,
    check_weights_size,
    download_weights,
    probe_gguf,
    probe_gguf_head,
    require_fork_binary,
    resolve_weights,
)
from conceptmod.textsliders.bonsai_gguf_particle import (
    FORMAT,
    RECIPE,
    BonsaiParticleHead,
    build_game,
    prepare_rows,
    resolve_head_path,
    update,
)
from conceptmod.textsliders.train_lora_bonsai_gguf import (
    DEFAULT_SAMPLE_SCALES,
    load_slider_rows,
    main as train_main,
    parse_args,
    parse_report_scales,
    train,
)


def test_pinned_model_file_and_readout():
    assert REPO == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert FILENAME == "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    assert FILE_BYTES == 5946648928
    assert BASE_MODEL == "Qwen/Qwen3.8-27B"
    assert ARCH == "qwen35"
    assert HIDDEN == 5120
    assert LAYERS == 64
    assert CONTEXT == 262144
    assert VOCAB == 248320
    assert N_TENSORS == 851
    assert TERNARY_TYPE_IDS == (142, 143)
    assert EMBED_POOLING == "last"
    assert FORK_URL == "https://github.com/PrismML-Eng/llama.cpp"
    assert DEMO_URL == "https://github.com/PrismML-Eng/Bonsai-demo"
    assert FORK_RELEASE_PIN == "prism-b10685-7dffb15"
    assert FORMAT == "conceptmod-bonsai-gguf-particle-v1"
    assert RECIPE["name"] == "anneal-routed-particle-error-bonsai-gguf-v1"
    assert RECIPE["source_recipe"] == "anneal-routed-particle-error-yue2-v1"
    assert RECIPE["propose_only"] is True
    assert RECIPE["adapter_placement"] == "readout_residual_zero_init"
    assert RECIPE["readout"] == "fork_server_last_token_embedding"
    args = parse_args(["--dummy"])
    assert args.recipe == "particle_bridge"
    assert args.seed == 7
    assert args.device == "cpu"
    assert args.dummy is True
    assert parse_report_scales(args.report_scales) == list(DEFAULT_SAMPLE_SCALES)
    with pytest.raises(SystemExit):
        parse_args(["--dummy", "--recipe", "unipolar_gan"])


def test_game_is_the_shared_yue2_module_not_a_copy():
    assert bonsai_game.shared is yue2_game
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


def test_pinned_file_resolves_and_everything_else_is_refused(tmp_path):
    good = tmp_path / FILENAME
    good.write_bytes(b"x")
    assert resolve_weights(str(good)) == str(good)
    assert resolve_weights(tmp_path / "some" / FILENAME) == str(tmp_path / "some" / FILENAME)
    cases = [
        (F16_FILENAME, "F16"),
        (MMPROJ_Q8_FILENAME, "vision"),
        (MMPROJ_BF16_FILENAME, "vision"),
        (PQ2_0_FILENAME, "pins"),
        (Q2_0_DEV_FILENAME, "garbage"),
        ("random-weights.gguf", "pinned file"),
    ]
    for name, needle in cases:
        with pytest.raises(BonsaiGGUFError, match=needle):
            resolve_weights(str(tmp_path / name))
    with pytest.raises(BonsaiGGUFError, match="truncated"):
        check_weights_size(str(good))
    exact = tmp_path / "exact.gguf"
    exact.write_bytes(b"\0" * 64)
    # Size gate works on any path; the name gate is separate.
    with pytest.raises(BonsaiGGUFError, match="expected 5946648928"):
        check_weights_size(str(exact))


def _pack_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf_bytes(*, arch="qwen35", hidden=5120, layers=64, ctx=262144,
                hadamard=True, tensor_types=(143,), magic=b"GGUF") -> bytes:
    kv: list[bytes] = []

    def add_str(k, v):
        kv.append(_pack_str(k) + struct.pack("<I", 8) + _pack_str(v))

    def add_u32(k, v):
        kv.append(_pack_str(k) + struct.pack("<I", 4) + struct.pack("<I", v))

    add_str("general.architecture", arch)
    add_u32(f"{arch}.embedding_length", hidden)
    add_u32(f"{arch}.block_count", layers)
    add_u32(f"{arch}.context_length", ctx)
    if hadamard:
        add_u32("prism.hadamard.version", 1)
    infos = b""
    for i, ty in enumerate(tensor_types):
        infos += _pack_str(f"blk.{i}.attn_qkv.weight")
        infos += struct.pack("<I", 2) + struct.pack("<QQ", 8, 8)
        infos += struct.pack("<i", ty) + struct.pack("<Q", 0)
    return (magic + struct.pack("<IQQ", 3, len(tensor_types), len(kv))
            + b"".join(kv) + infos)


def test_probe_accepts_ternary_hadamard_gguf():
    verdict = probe_gguf_head(_gguf_bytes())
    assert verdict["arch"] == "qwen35"
    assert verdict["hidden"] == 5120
    assert verdict["has_ternary"] is True
    assert verdict["has_hadamard"] is True
    assert 143 in verdict["tensor_types"]


def test_probe_rejects_non_gguf_wrong_shape_and_missing_markers():
    with pytest.raises(BonsaiGGUFError, match="not a GGUF"):
        probe_gguf_head(b"definitely not a model file" * 4)
    with pytest.raises(BonsaiGGUFError, match="expected hidden=5120"):
        probe_gguf_head(_gguf_bytes(hidden=1024))
    with pytest.raises(BonsaiGGUFError, match="GGUF arch"):
        probe_gguf_head(_gguf_bytes(arch="llama"))
    # No Hadamard markers: not a rotated ternary pack (or stock-stripped).
    with pytest.raises(StockLlamaRejected, match="hadamard"):
        probe_gguf_head(_gguf_bytes(hadamard=False))
    # F32-only tensors: an F16/F32 file can never serve as ternary readout.
    with pytest.raises(StockLlamaRejected, match="no ternary tensor types"):
        probe_gguf_head(_gguf_bytes(tensor_types=(0,)))
    with pytest.raises(BonsaiGGUFError, match="too short"):
        probe_gguf_head(_gguf_bytes()[:64])


def test_probe_real_ptq1_0_when_present():
    weights = os.environ.get("BONSAI_PTQ1_0")
    if not weights or not Path(weights).is_file():
        pytest.skip("BONSAI_PTQ1_0 not set; live header probe is opt-in")
    verdict = probe_gguf(weights)
    assert verdict["arch"] == "qwen35"
    assert verdict["hidden"] == HIDDEN
    assert verdict["n_tensors"] == N_TENSORS
    assert set(verdict["tensor_types"]) >= {0, 143}


def test_fork_log_scanner_rejects_stock_and_ambiguous():
    stock = (
        "llama_model_load_from_file_impl: loading model from "
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf\n"
        "error loading model: unknown model type PTQ1_0 (ggml type 143 "
        "exceeds GGML_TYPE_COUNT)\n"
    )
    with pytest.raises(StockLlamaRejected, match="stock llama.cpp"):
        check_fork_log(stock)
    garbage_q2 = "main: error: Q2_0 weights produced garbage output, no warning"
    # Ambiguous logs still fail closed (never silently OK).
    with pytest.raises(BonsaiGGUFError, match="could not confirm"):
        check_fork_log(garbage_q2)
    fork_ok = (
        "llama_model_load_from_file_impl: loading model from "
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf\n"
        "llama_model_load_from_file_impl: model loaded\n"
        "main: server is listening on 127.0.0.1:8080\n"
    )
    assert check_fork_log(fork_ok) is None


def test_require_fork_binary_fails_closed_without_a_binary(tmp_path):
    with pytest.raises(BonsaiGGUFError, match="not found"):
        require_fork_binary(str(tmp_path / "llama-server"))
    with pytest.raises(BonsaiGGUFError, match="not found"):
        require_fork_binary(str(tmp_path / "missing"))


def test_require_fork_binary_accepts_a_versioned_executable(tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text(
        "#!/bin/sh\necho 'version: 0.2.0-dev (build 10685, commit 7dffb158d)'\n"
    )
    fake.chmod(0o755)
    out = require_fork_binary(str(fake))
    assert "10685" in out
    cmd = build_server_cmd(fake, tmp_path / FILENAME, port=18081)
    assert "--embedding" in cmd and "--pooling" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "last"
    assert "-ngl" in cmd and cmd[cmd.index("-ngl") + 1] == "0"
    assert not any("mmproj" in c for c in cmd), "text-only: never --mmproj"
    with pytest.raises(BonsaiGGUFError, match="F16"):
        build_server_cmd(fake, tmp_path / F16_FILENAME, port=18081)


def test_download_refuses_without_explicit_opt_in(tmp_path):
    with pytest.raises(BonsaiGGUFError, match="--allow_download"):
        download_weights(tmp_path, allow_download=False)


def test_dummy_backend_readout_shape_and_determinism():
    backend = BonsaiGGUFBackend(device="cpu", dummy=True)
    enc_a = backend.encode("a blooming rose garden at dawn")
    enc_b = backend.encode("a sea under soft light")
    ha = backend.teacher_hidden(enc_a)
    assert tuple(ha.shape) == (1, HIDDEN)
    assert torch.isfinite(ha).all()
    assert torch.equal(ha, backend.hidden(enc_a))
    assert torch.equal(ha, backend.teacher_hidden(backend.encode(enc_a.text)))
    assert not torch.equal(ha, backend.teacher_hidden(enc_b))


def test_live_backend_needs_weights_and_server():
    with pytest.raises(BonsaiGGUFError, match="--server_url"):
        BonsaiGGUFBackend(device="cpu", dummy=False)


def test_head_pins_and_zero_init():
    head = BonsaiParticleHead()
    assert head.particles.shape == (128, 4)
    assert tuple(head.down.weight.shape) == (8, HIDDEN)
    assert tuple(head.up.weight.shape) == (HIDDEN, 8)
    assert torch.count_nonzero(head.up.weight).item() == 0
    x = torch.randn(2, HIDDEN)
    assert torch.equal(head(x), x)
    # Fresh head is the base behavior at every scale (up starts at zero);
    # steering appears only after training (covered post-train below).
    with head.scaled(1.0):
        assert torch.equal(head(x), x)
    assert torch.equal(head(x), x)
    with pytest.raises(ValueError, match="pins rank/alpha 8"):
        BonsaiParticleHead(rank=4, alpha=4.0)
    with pytest.raises(ValueError, match="readout width"):
        BonsaiParticleHead(hidden=1024)


def test_build_game_uses_reference_lrs_and_betas():
    backend = BonsaiGGUFBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-bonsai-gguf.yaml")
    fixed = prepare_rows(backend, rows)
    network = BonsaiParticleHead()
    critic, g, d = build_game(backend, network, fixed)
    assert critic.target_mean.shape == (HIDDEN,)
    roles = {group.get("role", "generator"): group for group in g.param_groups}
    ref = yue2_game.REFERENCE
    assert roles["generator"]["lr"] == ref["g_lr"]
    assert roles["particles"]["lr"] == ref["particle_lr"]
    assert list(roles["generator"]["betas"]) == list(ref["betas"])
    assert d.param_groups[0]["lr"] == ref["d_lr"]
    assert list(d.param_groups[0]["betas"]) == list(ref["betas"])


def test_shared_build_game_rejects_wrong_cloud_shape():
    backend = BonsaiGGUFBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-bonsai-gguf.yaml")
    fixed = prepare_rows(backend, rows)
    network = BonsaiParticleHead()
    network.particles = torch.nn.Parameter(torch.randn(16, 4))
    with pytest.raises(ValueError, match="128x4"):
        build_game(backend, network, fixed)


def test_scale_zero_is_exact_base_before_and_after_train(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy", "--steps", "2", "--name", "bonsai-zero",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["steps"] == 2
    backend = BonsaiGGUFBackend(device="cpu", dummy=True)
    network, _ = BonsaiParticleHead.load(resolve_head_path(str(tmp_path)))
    enc = backend.encode("a blooming rose garden at dawn")
    teacher = backend.teacher_hidden(enc)
    with network.scaled(0.0):
        assert torch.equal(network(backend.hidden(enc)), teacher)
    with network.scaled(1.0):
        assert not torch.equal(network(backend.hidden(enc)), teacher)


def test_dummy_train_runs_shared_game_with_no_mse(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: garden caption\n  positive: a blooming rose garden at dawn\n"
        "  neutral: a garden at dawn\n  unconditional: ''\n"
        "- target: sea caption\n  positive: a calm turquoise sea\n"
        "  neutral: a sea\n  unconditional: ''\n"
    )
    args = parse_args([
        "--dummy", "--steps", "4", "--name", "bonsai-game",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--report_scales", "0,1", "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["recipe"] == "anneal-routed-particle-error-bonsai-gguf-v1"
    assert sidecar["game_module"] == "conceptmod.textsliders.particle_bridge_gan"
    assert sidecar["backend"] == "bonsai_gguf"
    assert sidecar["model_repo"] == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert sidecar["model_file"] == "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    assert sidecar["model_file_bytes"] == 5946648928
    assert sidecar["model_hidden"] == 5120
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
    assert len(history) == 4
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
    assert (tmp_path / "bonsai-game_last.safetensors").is_file()
    assert (tmp_path / "bonsai-game_live_last.safetensors").is_file()
    report = json.loads((tmp_path / "report" / "readout_delta.json").read_text())
    assert len(report) == 4
    by_scale = {}
    for rec in report:
        by_scale.setdefault(rec["scale"], []).append(rec)
    for rec in by_scale[0.0]:
        assert rec["delta_l2"] == pytest.approx(0.0, abs=1e-5)
        assert rec["delta_cos"] == pytest.approx(1.0, abs=1e-4)


def test_save_load_roundtrip_and_rejections(tmp_path):
    backend = BonsaiGGUFBackend(device="cpu", dummy=True)
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-bonsai-gguf.yaml")
    fixed = prepare_rows(backend, rows)
    network = BonsaiParticleHead()
    critic, g, d = build_game(backend, network, fixed)
    sampler = yue2_game.BridgeSampler(len(rows), 0)
    update(backend, network, critic, g, d, fixed, sampler=sampler, step=1)
    path = tmp_path / "bonsai-head.safetensors"
    network.save(path, {"weights_kind": "live"})
    reloaded, record = BonsaiParticleHead.load(path)
    assert record["format"] == FORMAT
    assert record["hidden"] == HIDDEN
    assert record["particles"] == 128 and record["particle_dim"] == 4
    for a, b in zip(network.parameters(), reloaded.parameters()):
        assert torch.allclose(a, b)

    from safetensors.torch import save_file
    peft = tmp_path / "peft.safetensors"
    save_file({"base_model.model.foo.weight": torch.zeros(2, 2)}, str(peft))
    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        BonsaiParticleHead.load(peft)
    bad = dict(network.state_dict())
    bad["unrelated.weight"] = torch.zeros(1, 1)
    bad_path = tmp_path / "bad.safetensors"
    meta = {"conceptmod": json.dumps({
        "format": FORMAT, "rank": 8, "alpha": 8.0, "hidden": HIDDEN,
        "particles": 128, "particle_dim": 4,
        "bridge_width": 48, "router_width": 16,
    })}
    save_file({k: v.cpu() for k, v in bad.items()}, str(bad_path), metadata=meta)
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        BonsaiParticleHead.load(bad_path)


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
        "--dummy", "--steps", "2", "--name", "bonsai-load",
        "--save_dir", str(train_dir), "--prompts_file", str(prompts),
        "--no_report", "--seed", "1",
    ]))
    reload_dir = tmp_path / "reload"
    sidecar = train(parse_args([
        "--dummy", "--steps", "0", "--name", "bonsai-reload",
        "--save_dir", str(reload_dir), "--prompts_file", str(prompts),
        "--load_head", str(train_dir), "--report_scales", "0,1",
    ]))
    assert sidecar["steps"] == 0
    assert sidecar["history"] == []
    assert sidecar["load_head"].endswith("_last.safetensors")
    assert (reload_dir / "report" / "readout_delta.json").is_file()


def test_needs_two_rows(tmp_path):
    one = tmp_path / "one.yaml"
    one.write_text(
        "- target: garden caption\n  positive: a blooming rose garden\n"
        "  neutral: a garden\n  unconditional: ''\n"
    )
    with pytest.raises(ValueError, match="at least two"):
        load_slider_rows(str(one))


def test_yaml_rows_and_config_card_match_shared_reference():
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-bonsai-gguf.yaml")
    assert len(rows) == 3
    for row in rows:
        assert row["positive"] != row["neutral"]
    cfg = yaml.safe_load(
        Path("conceptmod/textsliders/data/config-bonsai-gguf.yaml").read_text()
    )
    assert cfg["pretrained_model"]["repo"] == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert cfg["pretrained_model"]["file"] == "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    assert cfg["pretrained_model"]["file_bytes"] == 5946648928
    assert "F16" not in yaml.safe_dump(cfg)
    assert "mmproj" not in yaml.safe_dump(cfg).lower()
    assert cfg["train"]["recipe"] == "particle_bridge"
    assert cfg["game"]["module"] == "conceptmod.textsliders.particle_bridge_gan"
    ref = yue2_game.REFERENCE
    for key in ("g_lr", "d_lr", "particle_lr", "ema", "vic_coeff",
                "cap_coeff", "cap_kappa", "cap_every",
                "noise_floor", "noise_decay_steps"):
        assert cfg["game"][key] == ref[key], key
    assert cfg["game"]["particles"] == ref["particles"] == 128
    assert cfg["game"]["particle_dim"] == ref["z_dim"] == 4
    assert cfg["network"]["placement"] == "readout_residual_zero_init"
    assert cfg["network"]["hidden"] == 5120


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
        "pretrained_model:\n  repo: prism-ml/Ternary-Bonsai-2-27B-gguf\n"
        "  file: Ternary-Bonsai-2-27B-PTQ1_0.gguf\n"
        "train:\n  iterations: 2\n"
        f"prompts_file: {prompts}\n"
        "save:\n  name: bonsai-gguf-cfg\n"
    )
    sidecar = train_main([
        "--dummy",
        "--config_file", str(cfg),
        "--save_dir", str(tmp_path / "out"),
        "--no_report",
        "--seed", "0",
    ])
    assert sidecar["name"] == "bonsai-gguf-cfg"
    assert sidecar["steps"] == 2
    assert sidecar["recipe"] == "anneal-routed-particle-error-bonsai-gguf-v1"


def test_live_server_readout_when_present():
    server_url = os.environ.get("BONSAI_SERVER_URL")
    if not server_url:
        pytest.skip("BONSAI_SERVER_URL not set; live readout is opt-in")
    backend = BonsaiGGUFBackend(
        device="cpu", weights=os.environ["BONSAI_PTQ1_0"],
        server_url=server_url, dummy=False,
    )
    enc_a = backend.encode("a blooming rose garden at dawn")
    enc_b = backend.encode("a sea under soft light")
    ha, hb = backend.teacher_hidden(enc_a), backend.teacher_hidden(enc_b)
    assert tuple(ha.shape) == (1, HIDDEN)
    assert torch.isfinite(ha).all() and torch.isfinite(hb).all()
    assert not torch.equal(ha, hb)


def test_production_files_unchanged():
    tf_src = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert 'parser.add_argument("--steps", type=int, default=500)' in tf_src
    assert 'parser.add_argument("--rank", type=int, default=8)' in tf_src
    assert 'parser.add_argument("--lr", type=float, default=2e-3' in tf_src
    lm_src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"--lm_target"' in lm_src and 'default="v9"' in lm_src
    assert '"--pole_mode"' in lm_src and 'default="hidden"' in lm_src
    music3_yaml = Path("conceptmod/textsliders/data/prompts-music3.yaml").read_text()
    assert "bonsai" not in music3_yaml.lower()
    assert "Bonsai-2-27B" not in music3_yaml
    for name in (
        "train_lora_yue2.py",
        "train_lora_yue2_arm_b.py",
        "yue2_backend.py",
        "yue2_particle_bridge.py",
        "particle_bridge_gan.py",
        "yue2_arm_b.py",
        "tiny_llm_backend.py",
        "tiny_llm_particle.py",
        "train_lora_tiny_llm.py",
    ):
        src = Path("conceptmod/textsliders") / name
        if src.is_file():
            assert "bonsai" not in src.read_text().lower(), name
    adv_src = Path("analysis/slider2d/adv.py").read_text()
    assert "bonsai" not in adv_src.lower()
