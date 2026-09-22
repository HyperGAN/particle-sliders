"""Supra2-IMG slider: load contract, LoRA attach, dummy train step. CPU only."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders.supra_fake import FakeSupraBackend
from conceptmod.textsliders.supra_model import (
    CKPT_FILENAME,
    D_CTX,
    D_MODEL,
    DEPTH,
    HEAD_DIM,
    HF_REPO,
    IMG_SIZE,
    LATENT_CH,
    LATENT_SIZE,
    MAX_CTX_LEN,
    MLP_RATIO,
    N_HEADS,
    NUM_TOKENS,
    PATCH,
    T5_NAME,
    VAE_NAME,
    SUPRA_PARAM_COUNT,
    VAE_SCALE,
    SupraDiT,
    attach_supra_lora,
    count_parameters,
    load_supra_weights,
)
from conceptmod.textsliders.supra_slider import (
    CROSS_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_CONCEPT_WORDS,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_LM_TARGET,
    DEFAULT_LORA_TARGETS,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_TRAJ_STEPS,
    DIT_LORA_TARGETS,
    MAN_NEU,
    MAN_PLUS,
    WOMAN_NEU,
    WOMAN_PLUS,
    architecture_card,
    concept_tokens,
    infer_sample_prompts,
    live_train_card,
    live_train_command,
    load_supra_prompts,
    resolve_supra_lm_target,
    resolve_supra_lora_targets,
    row_token_plan,
    supra_cfg_delta,
    supra_euler_sample_latents,
    supra_euler_step,
    supra_euler_times,
    supra_sample_cfg,
    supra_short_trajectory,
    supra_uni_loss,
    supra_uni_teachers,
    unused_vocab,
)
from conceptmod.textsliders.train_lora_krea import assert_krea_only
from conceptmod.textsliders.train_lora_sana import assert_sana_only
from conceptmod.textsliders.train_lora_supra import (
    _sample_zt,
    LiveSupraBackend,
    assert_supra_only,
    load_live_backend,
    parse_args,
    train,
    train_dummy,
)

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "conceptmod/textsliders/data/prompts-supra.yaml"
CANARY = REPO / "conceptmod/textsliders/data/prompts-supra-canary.yaml"
CONFIG = REPO / "conceptmod/textsliders/data/config-supra.yaml"


def test_music3_defaults_stay_v9_hidden():
    from conceptmod.textsliders.train_lm_slider_music3 import parse_args as lm_parse

    args = lm_parse(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_cli_defaults_match_hub_card():
    args = parse_args([])
    assert args.model_id == DEFAULT_MODEL_ID == HF_REPO == "SupraLabs/Supra2-IMG"
    assert args.rank == DEFAULT_RANK == 16
    assert args.resolution == DEFAULT_RESOLUTION == IMG_SIZE == 256
    assert args.sample_steps == DEFAULT_SAMPLE_STEPS == 50
    assert args.cfg == DEFAULT_CFG == 3.0
    assert args.lr == DEFAULT_LR == 1e-4
    assert args.lm_target == DEFAULT_LM_TARGET == "trajectory"
    assert args.traj_steps == DEFAULT_TRAJ_STEPS == 4
    assert args.lora_targets == DEFAULT_LORA_TARGETS == "cross"
    assert args.sample_every == DEFAULT_SAMPLE_EVERY == 100
    assert args.sample_seed == DEFAULT_SAMPLE_SEED == 42
    assert args.sample_mode == "train_faithful"
    assert args.control_prompt == DEFAULT_CONTROL_PROMPT
    assert args.dummy is False
    assert args.allow_hub is False
    card = live_train_card()
    arch = card["arch"]
    assert arch["model_id"] == "SupraLabs/Supra2-IMG"
    assert arch["d_model"] == D_MODEL == 576
    assert arch["depth"] == DEPTH == 14
    assert arch["n_heads"] == N_HEADS == 9
    assert arch["head_dim"] == HEAD_DIM == 64
    assert D_MODEL == N_HEADS * HEAD_DIM
    assert arch["mlp_ratio"] == MLP_RATIO == 4.0
    assert arch["d_ctx"] == D_CTX == 768
    assert arch["encoder"] == T5_NAME == "google/flan-t5-base"
    assert arch["ctx_len"] == MAX_CTX_LEN == 128
    assert arch["vae"] == VAE_NAME == "stabilityai/sd-vae-ft-mse"
    assert arch["vae_scale"] == pytest.approx(VAE_SCALE)
    assert VAE_SCALE == pytest.approx(0.18215)
    assert arch["latent_size"] == LATENT_SIZE == 32
    assert arch["latent_ch"] == LATENT_CH == 4
    assert arch["patch"] == PATCH == 2
    assert arch["num_tokens"] == NUM_TOKENS == 256
    assert (LATENT_SIZE // PATCH) ** 2 == NUM_TOKENS
    assert arch["ckpt"] == CKPT_FILENAME == "model_final_ema.pt"
    assert arch["sample_steps"] == 50
    assert arch["sample_cfg"] == 3.0
    assert "v_uncond + cfg" in arch["sample_cfg_formula"]
    assert card["lora"]["lora_targets"] == "cross"
    assert card["lora"]["train_cross"] is True
    assert card["lora"]["train_dit"] is False
    assert card["lora"]["train_text_encoder"] is False
    assert card["lora"]["targets"] == list(CROSS_LORA_TARGETS)
    assert card["lm_target"] == "trajectory"
    assert "Hub Euler" in card["traj_loop"]
    assert card["music3_default_untouched"] == {"lm_target": "v9", "pole_mode": "hidden"}
    assert "Comfy" in " ".join(card["non_goals"])
    assert "product repo" in " ".join(card["non_goals"])
    cmd = live_train_command()
    assert "SupraLabs/Supra2-IMG" in cmd
    assert "--lora_targets cross" in cmd
    assert "--resolution 256" in cmd
    assert "--sample_steps 50" in cmd
    assert "--cfg 3" in cmd
    assert "--lm_target trajectory" in cmd
    assert "HF_HUB_OFFLINE=1" in cmd
    assert architecture_card()["params"] == SUPRA_PARAM_COUNT
    assert architecture_card()["params_millions"] == pytest.approx(104.1)


def test_config_yaml_matches_cli_card():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["pretrained_model"]["name_or_path"] == DEFAULT_MODEL_ID
    assert raw["pretrained_model"]["checkpoint"] == CKPT_FILENAME
    assert raw["pretrained_model"]["text_encoder"] == T5_NAME
    assert raw["pretrained_model"]["text_encoder_frozen"] is True
    assert raw["pretrained_model"]["vae"] == VAE_NAME
    assert raw["pretrained_model"]["vae_scale"] == pytest.approx(VAE_SCALE)
    assert raw["network"]["lora_targets"] == "cross"
    assert raw["network"]["rank"] == 16
    assert raw["network"]["train_text_encoder"] is False
    assert raw["arch"]["d_model"] == 576
    assert raw["arch"]["depth"] == 14
    assert raw["arch"]["n_heads"] == 9
    assert raw["arch"]["head_dim"] == 64
    assert raw["arch"]["resolution"] == 256
    assert raw["arch"]["latent_size"] == 32
    assert raw["arch"]["patch"] == 2
    assert raw["train"]["cfg"] == 3.0
    assert raw["train"]["sample_steps"] == 50
    assert raw["train"]["lm_target"] == "trajectory"
    assert raw["train"]["lr"] == pytest.approx(1e-4)


def test_yaml_is_lighting_not_age_and_does_not_prefix():
    rows, meta = load_supra_prompts(PROMPTS)
    text = PROMPTS.read_text(encoding="utf-8").lower()
    assert "elderly" not in text
    assert "years old" not in text
    assert meta.plus_label == "sunlit"
    assert meta.concept_words == DEFAULT_CONCEPT_WORDS
    assert len(rows) == 2
    assert rows[0].infer_prompt == rows[0].neutral == WOMAN_NEU
    assert rows[1].infer_prompt == rows[1].neutral == MAN_NEU
    assert rows[0].positive == WOMAN_PLUS
    assert rows[1].positive == MAN_PLUS
    assert all(not row.has_minus_canary for row in rows)
    assert all(not row.positive.startswith("indoor") for row in rows)
    assert all(not row.neutral.startswith("portrait") for row in rows)
    assert rows[0].attributes == ["indoor", "portrait"]
    plan = row_token_plan(rows[0])
    for word in ("warm", "golden", "sunlit", "glow"):
        assert word in plan["concept"]
        assert word not in plan["unused"]
    assert "woman" in plan["unused"]
    assert "window" in plan["unused"]
    assert "indoor" in plan["unused"]
    prompts = infer_sample_prompts(rows)
    assert prompts[0] == WOMAN_NEU
    assert prompts[1] == MAN_NEU
    assert prompts[-1] == DEFAULT_CONTROL_PROMPT
    assert all("warm" not in p for p in prompts)
    unused = unused_vocab(WOMAN_NEU, WOMAN_NEU, attributes=["indoor"], concept_words=DEFAULT_CONCEPT_WORDS)
    assert "warm" in concept_tokens(WOMAN_PLUS, unused)
    assert "warm" not in unused


def test_cfg_delta_and_hub_sample_cfg():
    v_c = torch.tensor([1.5, 0.25])
    v_u = torch.tensor([0.5, 0.05])
    assert torch.allclose(supra_cfg_delta(v_c, v_u), v_c - v_u)
    assert torch.allclose(supra_sample_cfg(v_c, v_u, 3.0), v_u + 3.0 * (v_c - v_u))
    assert torch.allclose(supra_sample_cfg(v_c, v_u, 1.0), v_c)
    teachers = supra_uni_teachers(v_c, v_u, torch.zeros_like(v_c), v_neg=-v_c)
    assert teachers["minus"] is None
    base = supra_uni_loss(v_c, v_c, v_u, v_u)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved = supra_uni_loss(
        v_c,
        v_c,
        v_u,
        v_u,
        student_minus=torch.tensor([9.0, 9.0]),
        teacher_minus=torch.tensor([-4.0, 4.0]),
    )
    assert float(moved) == pytest.approx(0.0, abs=1e-8)


def test_hub_euler_step_is_z_plus_dt_v():
    sample = torch.tensor([1.0, -2.0])
    vel = torch.tensor([0.5, 0.25])
    out = supra_euler_step(sample, vel, 0.02)
    assert torch.allclose(out, sample + 0.02 * vel)
    times, dt = supra_euler_times(50)
    assert times.numel() == 50
    assert dt == pytest.approx(1.0 / 50)
    assert float(times[0]) == pytest.approx(0.0)
    assert float(times[1]) == pytest.approx(dt)
    assert float(times[-1]) == pytest.approx(49 / 50)
    with pytest.raises(ValueError, match="lm_target"):
        resolve_supra_lm_target("v9")
    with pytest.raises(ValueError, match="lm_target"):
        resolve_supra_lm_target("embed_struct")
    with pytest.raises(ValueError, match="lm_target"):
        resolve_supra_lm_target("same_crop")


def test_lora_targets_resolver():
    spec = resolve_supra_lora_targets()
    assert spec.label == "cross"
    assert spec.train_cross is True
    assert spec.train_dit is False
    assert spec.active_attn_targets == list(CROSS_LORA_TARGETS)
    assert "text_encoder" in spec.frozen_modules
    assert "vae" in spec.frozen_modules
    assert "self_attn" in spec.frozen_modules
    dit = resolve_supra_lora_targets("dit")
    assert dit.train_dit is True
    assert dit.train_cross is False
    assert "cross_attn" in dit.frozen_modules
    joint = resolve_supra_lora_targets("self+cross")
    assert joint.label == "dit+cross"
    assert joint.train_dit and joint.train_cross
    with pytest.raises(ValueError, match="text_encoder|Flan-T5|lora_targets"):
        resolve_supra_lora_targets("text_encoder")
    args = parse_args(["--lora_targets", "dit"])
    assert args.lora_targets == "dit"


def test_lora_attach_on_tiny_dit_matches_hub_names():
    model = SupraDiT(
        latent_ch=4,
        d_model=32,
        depth=2,
        n_heads=4,
        ctx_dim=8,
        mlp_ratio=2.0,
        num_tokens=16,
        patch=2,
    )
    keys = set(model.state_dict())
    assert "ctx_proj.weight" in keys
    assert "blocks.0.cross_attn.q.weight" in keys
    assert "blocks.0.cross_attn.kv.weight" in keys
    assert "blocks.0.cross_attn.proj.weight" in keys
    assert "blocks.1.self_attn.qkv.weight" in keys
    assert "blocks.1.self_attn.proj.weight" in keys
    assert "x_embed.weight" in keys
    assert "pos_embed" in keys
    assert not any(k.endswith("cross_attn.qkv.weight") for k in keys)
    assert not any(k.endswith("self_attn.q.weight") for k in keys)
    wrapped = attach_supra_lora(model, rank=4, alpha=4.0, targets=CROSS_LORA_TARGETS)
    assert wrapped
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names
    assert any("ctx_proj.up.weight" in n or n.endswith("ctx_proj.up.weight") for n in names)
    assert any("cross_attn.q.up.weight" in n for n in names)
    assert any("cross_attn.kv.down.weight" in n for n in names)
    assert any("cross_attn.proj.up.weight" in n for n in names)
    assert not any("self_attn" in n for n in names)
    assert all(not p.requires_grad for n, p in model.named_parameters() if "self_attn" in n)
    with pytest.raises(ValueError, match="proj"):
        attach_supra_lora(model, rank=2, alpha=2.0, targets=("proj",))


def test_checkpoint_ema_wrapper_roundtrip():
    model = SupraDiT(
        latent_ch=4,
        d_model=32,
        depth=1,
        n_heads=4,
        ctx_dim=8,
        mlp_ratio=2.0,
        num_tokens=16,
        patch=2,
    )
    blob = {"ema": {k: v.clone() for k, v in model.state_dict().items()}, "config": {"patch": 2}}
    other = SupraDiT(
        latent_ch=4,
        d_model=32,
        depth=1,
        n_heads=4,
        ctx_dim=8,
        mlp_ratio=2.0,
        num_tokens=16,
        patch=2,
    )
    load_supra_weights(other, blob)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, other.state_dict()[key])
    with pytest.raises(ValueError, match="PATCH"):
        load_supra_weights(other, {"ema": blob["ema"], "config": {"patch": 4}})


def test_default_dit_param_count_is_about_104_1m():
    model = SupraDiT()
    n = count_parameters(model)
    assert n == SUPRA_PARAM_COUNT
    assert abs(n / 1e6 - 104.1) < 0.05
    assert model.blocks.__len__() == DEPTH
    assert model.ctx_proj.in_features == D_CTX
    assert model.ctx_proj.out_features == D_MODEL


def test_tiny_forward_and_dummy_lora_grad():
    backend = FakeSupraBackend(device="cpu", rank=4, seed=0, lora_targets="cross")
    z = torch.randn(1, *backend.latent_shape)
    t = torch.tensor([0.5])
    v = backend.predict_v(WOMAN_PLUS, z, t, scale=1.0)
    assert tuple(v.shape) == (1, *backend.latent_shape)
    loss = v.float().square().mean()
    loss.backward()
    grads = [p.grad for p in backend.trainable_parameters() if p.grad is not None]
    assert grads
    assert any(float(g.abs().sum()) > 0 for g in grads)
    names = backend.named_trainable()
    assert any("cross_attn" in n or "ctx_proj" in n for n in names)
    assert not any("self_attn" in n for n in names)
    with backend.disable_adapter():
        assert all(lora.multiplier == 0.0 for lora in backend.loras)


def test_dummy_euler_sample_keeps_latent_shape():
    backend = FakeSupraBackend(device="cpu", rank=4, seed=1, lora_targets="dit")
    assert any("self_attn" in n for n in backend.named_trainable())
    assert not any("cross_attn" in n or "ctx_proj" in n for n in backend.named_trainable())
    z0 = torch.randn(1, *backend.latent_shape)
    out = supra_euler_sample_latents(backend, WOMAN_NEU, num_steps=2, cfg=3.0, z=z0)
    assert tuple(out.shape) == tuple(z0.shape)
    assert not torch.allclose(out, z0)


def test_short_trajectory_matches_manual_euler():
    backend = FakeSupraBackend(device="cpu", rank=2, seed=0, lora_targets="cross")
    z = torch.zeros(1, *backend.latent_shape)
    with torch.no_grad():
        got = supra_short_trajectory(backend, WOMAN_NEU, z, num_steps=2, frozen=True)
        x = z.clone()
        times, dt = supra_euler_times(2)
        for i in range(2):
            v = backend.predict_v(WOMAN_NEU, x, times[i].reshape(1), frozen=True)
            x = supra_euler_step(x, v, dt)
    assert torch.allclose(got, x)


def test_dummy_train_step_cross(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--name",
            "supra-unit",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--traj_steps",
            "2",
            "--resolution",
            "32",
            "--sample_every",
            "0",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["dummy"] is True
    assert sidecar["model_id"] == "SupraLabs/Supra2-IMG"
    assert sidecar["lora_targets"] == "cross"
    assert sidecar["train_cross"] is True
    assert sidecar["train_dit"] is False
    assert sidecar["train_text_encoder"] is False
    assert sidecar["cross_lora_targets"] == list(CROSS_LORA_TARGETS)
    assert sidecar["dit_lora_targets"] == []
    assert sidecar["frozen_modules"][0] == "text_encoder"
    assert "vae" in sidecar["frozen_modules"]
    assert sidecar["lm_target"] == "trajectory"
    assert sidecar["minus_canary"] is False
    assert math.isfinite(sidecar["loss_last"])
    assert sidecar["lora_b_norm"] > 0.0
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"
    assert WOMAN_NEU in sidecar["train_infer_prompts"]
    assert MAN_NEU in sidecar["train_infer_prompts"]
    assert sidecar["sample_grid"]["scales"] == list(DEFAULT_SAMPLE_SCALES)
    assert (tmp_path / "supra-unit_dummy_last.json").is_file()
    assert list((tmp_path / "samples").glob("*.png"))


def test_dummy_train_dit_and_canary(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--name",
            "supra-canary",
            "--prompts_file",
            str(CANARY),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lora_targets",
            "dit",
            "--lm_target",
            "cfg_delta",
            "--traj_steps",
            "2",
            "--sample_every",
            "0",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lora_targets"] == "dit"
    assert sidecar["train_dit"] is True
    assert sidecar["train_cross"] is False
    assert sidecar["dit_lora_targets"] == list(DIT_LORA_TARGETS)
    assert sidecar["lm_target"] == "cfg_delta"
    assert sidecar["minus_canary"] is True
    assert sidecar["canary_cos_last"] is not None
    assert math.isfinite(sidecar["loss_last"])


def test_print_card_does_not_train(capsys):
    out = train(parse_args(["--print_card"]))
    assert out["model_id"] == DEFAULT_MODEL_ID
    assert out["lora"]["rank"] == 16
    captured = capsys.readouterr().out
    assert "SupraLabs/Supra2-IMG" in captured
    assert "model_final_ema" not in captured or "ckpt" in captured


def test_live_loader_stays_offline(monkeypatch):
    args = parse_args(["--device", "cpu"])
    assert args.allow_hub is False

    def _boom(*_a, **_k):
        raise AssertionError("live loader must not download in this test")

    monkeypatch.setattr(
        "conceptmod.textsliders.train_lora_supra._download_checkpoint",
        _boom,
    )
    with pytest.raises(RuntimeError, match="dummy"):
        load_live_backend(args, torch.device("cpu"))


def test_refuses_foreign_backends():
    with pytest.raises(ValueError, match="Supra-only"):
        assert_supra_only("circlestone-labs/Anima-Base-v1.0-Diffusers")
    with pytest.raises(ValueError, match="Supra-only"):
        assert_supra_only("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="Supra-only"):
        assert_supra_only("Efficient-Large-Model/Sana_600M_512px_diffusers")
    with pytest.raises(ValueError, match="Supra-only"):
        assert_supra_only("Tongyi-MAI/Z-Image-Turbo")
    assert_supra_only("SupraLabs/Supra2-IMG")
    with pytest.raises(ValueError, match="Supra-only"):
        train(parse_args(["--dummy", "--model_id", "krea/Krea-2-Raw", "--steps", "1"]))
    with pytest.raises(ValueError, match="Sana-only"):
        assert_sana_only("SupraLabs/Supra2-IMG")
    with pytest.raises(ValueError, match="Krea-only"):
        assert_krea_only("SupraLabs/Supra2-IMG")


def test_live_backend_predicts_when_context_is_injected():
    model = SupraDiT(
        latent_ch=4,
        d_model=32,
        depth=1,
        n_heads=4,
        ctx_dim=8,
        mlp_ratio=2.0,
        num_tokens=16,
        patch=2,
    )
    loras = attach_supra_lora(model, rank=4, alpha=4.0, targets=CROSS_LORA_TARGETS)
    backend = LiveSupraBackend(
        model, loras, torch.device("cpu"), resolve_supra_lora_targets("cross")
    )
    assert backend.latent_shape == (4, 8, 8)
    z = torch.randn(1, 4, 8, 8)
    with pytest.raises(RuntimeError, match="Flan-T5"):
        backend.predict_v("warm glow", z, torch.tensor([0.2]))

    def encode(prompt: str):
        n = max(1, len(prompt.split()))
        return torch.zeros(1, n, 8), prompt.split()

    backend.encode_fn = encode
    velocity = backend.predict_v("warm golden glow", z, torch.tensor([0.2]), scale=1.0)
    assert tuple(velocity.shape) == (1, 4, 8, 8)
    velocity.square().mean().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in backend.trainable_parameters()
    )


def test_live_train_refuses_cpu_without_dummy():
    with pytest.raises(RuntimeError, match="dummy"):
        train(parse_args(["--device", "cpu", "--steps", "1"]))


def test_sample_zt_is_unit_interval_not_anima_thousand():
    backend = FakeSupraBackend(device="cpu", rank=2, seed=0)
    z, t = _sample_zt(backend, seed=7, step=3)
    assert tuple(z.shape) == (1, *backend.latent_shape)
    assert z.device == backend.device
    assert float(t) >= 0.0
    assert float(t) < 1.0
