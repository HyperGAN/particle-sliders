"""Krea2 turbo-bbox slider: distilled card, load plan, dummy train. CPU only.

No Hub download and no GPU. Stock Raw tests stay in test_krea_slider.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders.krea2_bbox import (
    assert_krea2_only,
    assert_krea2_skeleton,
    live_train_card,
    live_train_command,
    resolve_krea2_bbox_card,
    resolve_krea2_bbox_source,
)
from conceptmod.textsliders.slider_targets import (
    KREA2_BBOX_CFG,
    KREA2_BBOX_COMFY_FILE,
    KREA2_BBOX_MODEL,
    KREA2_BBOX_MU,
    KREA2_BBOX_SKELETON,
    KREA2_BBOX_STEPS,
    KREA2_BBOX_SUBFOLDER,
    KREA_CONTROL_PROMPT,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_DISTILLED_MU,
    KREA_HOLD_WEIGHT,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_SMILE_HOLD_WEIGHT,
    is_krea2_turbo_bbox_id,
    krea_looks_turbo,
    krea_plus_neu_teachers,
    krea_sample_card,
    krea_scheduler_mu,
)
from conceptmod.textsliders.train_lora_anima import (
    assert_anima_only,
    parse_args as parse_anima,
    train as train_anima,
)
from conceptmod.textsliders.train_lora_krea import (
    assert_krea_only,
    parse_args as parse_krea,
    resolve_krea_card,
    train as train_krea,
)
from conceptmod.textsliders.train_lora_krea2 import (
    parse_args,
    train,
)
from conceptmod.textsliders.train_lora_krea import load_prompts, unused_words_for
from conceptmod.textsliders.train_lora_sana import assert_sana_only, parse_args as parse_sana, train as train_sana
from conceptmod.textsliders.train_lora_supra import assert_supra_only, parse_args as parse_supra, train as train_supra


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "conceptmod/textsliders/data/prompts-krea2-bbox.yaml"
CONFIG = ROOT / "conceptmod/textsliders/data/config-krea2-bbox.yaml"
LIVE = ROOT / "conceptmod/textsliders/krea2_bbox_live.py"
BBOX_ID = "jimmycarter/krea2-turbo-bbox"


def test_music3_lm_default_is_still_v9_hidden():
    from conceptmod.textsliders.train_lm_slider_music3 import parse_args as parse_lm

    args = parse_lm(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_cli_defaults_are_distilled_not_raw():
    args = parse_args([])
    assert args.model_id == KREA2_BBOX_MODEL == BBOX_ID
    assert args.transformer_subfolder == KREA2_BBOX_SUBFOLDER
    assert args.transformer_subfolder == "epoch-14-step-73184/transformer"
    assert args.skeleton_model == KREA2_BBOX_SKELETON == "krea/Krea-2-Raw"
    assert args.rank == KREA_DEFAULT_RANK == 16
    assert args.resolution == KREA_DEFAULT_RESOLUTION == 512
    assert args.sample_steps == KREA2_BBOX_STEPS == 8
    assert args.sample_guidance == KREA2_BBOX_CFG == 0.0
    assert args.mu == KREA2_BBOX_MU == KREA_DISTILLED_MU == 1.15
    assert args.sample_steps != KREA_RAW_STEPS
    assert args.sample_guidance != KREA_RAW_CFG
    assert args.dummy is False
    assert args.allow_hub is False
    assert args.lora_targets == "dit"
    assert args.lm_target == "v"
    assert args.hold_weight == KREA_SMILE_HOLD_WEIGHT == 0.1
    assert args.hold_weight != KREA_HOLD_WEIGHT
    card = resolve_krea2_bbox_card(None, None, None)
    assert card["variant"] == "turbo_bbox"
    assert card["sample_steps"] == 8
    assert card["sample_guidance"] == 0.0
    assert card["mu"] == 1.15
    assert card["is_distilled"] is True


def test_print_card_command_is_the_product_cli():
    args = parse_args(["--print_card"])
    card = train(args)
    assert card["backend"] == "krea2_turbo_bbox"
    assert card["model_id"] == BBOX_ID
    assert card["hold_weight"] == pytest.approx(0.1)
    assert card["sample_steps"] == 8
    assert card["sample_guidance"] == 0.0
    assert card["mu"] == 1.15
    assert card["not_raw_card"]["raw_cfg"] == 4.5
    assert card["not_raw_card"]["raw_steps"] == 28
    assert card["music3_default_untouched"] == {"lm_target": "v9", "pole_mode": "hidden"}
    assert "product repo" in card["non_goals"]
    cmd = live_train_command()
    assert "train_lora_krea2.py" in cmd
    assert BBOX_ID in cmd
    assert "--transformer_subfolder epoch-14-step-73184/transformer" in cmd
    assert "--skeleton_model krea/Krea-2-Raw" in cmd
    assert "--sample_steps 8" in cmd
    assert "--sample_guidance 0" in cmd
    assert "--mu 1.15" in cmd
    assert "--allow_hub" in cmd
    assert "--sample_guidance 4.5" not in cmd
    assert "--sample_steps 28" not in cmd
    documented = live_train_card()
    assert documented["comfy_filename"] == KREA2_BBOX_COMFY_FILE


def test_config_yaml_matches_distilled_card():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["pretrained_model"]["name_or_path"] == BBOX_ID
    assert raw["pretrained_model"]["transformer_subfolder"] == KREA2_BBOX_SUBFOLDER
    assert raw["pretrained_model"]["skeleton"] == "krea/Krea-2-Raw"
    assert raw["pretrained_model"]["comfy_filename"] == KREA2_BBOX_COMFY_FILE
    assert raw["pretrained_model"]["is_distilled"] is True
    assert raw["network"]["rank"] == 16
    assert raw["network"]["lora_targets"] == "dit"
    assert raw["train"]["resolution"] == 512
    assert raw["train"]["hold_weight"] == pytest.approx(KREA_SMILE_HOLD_WEIGHT)
    assert raw["sample"]["steps"] == 8
    assert raw["sample"]["guidance"] == 0.0
    assert raw["sample"]["mu"] == pytest.approx(1.15)
    assert raw["sample"]["raw_steps"] == 28
    assert raw["sample"]["raw_guidance"] == pytest.approx(4.5)


def test_yaml_is_bare_smile_and_keeps_bbox_dsl():
    rows, meta = load_prompts(PROMPTS)
    text = PROMPTS.read_text(encoding="utf-8")
    assert meta.bare_captions is True
    assert meta.plus_label == "Happy"
    assert "teeth" in meta.concept_words
    assert meta.control_prompt == KREA_CONTROL_PROMPT
    assert len(rows) == 3
    assert rows[0].positive.startswith("a person, big smile")
    assert not rows[0].positive.startswith("male ")
    assert "male" in rows[0].attributes
    assert rows[0].guidance_scale == 0.0
    grounded = rows[2]
    assert "pe[200,80,800,980]" in grounded.positive
    assert "pe[200,80,800,980]" in grounded.neutral
    assert "[x0,y0,x1,y1]" not in grounded.positive
    assert "p[0,0,1000,1000]" in grounded.target
    unused = unused_words_for(grounded)
    assert "male" in unused
    assert "female" in unused
    assert "smile" not in unused
    assert "teeth" not in unused
    assert "PROMPTING.md" in text


def test_cfg0_teacher_is_not_the_raw_4_5_teacher():
    v_pos = torch.tensor([2.0, 0.0])
    v_neu = torch.tensor([0.0, 1.0])
    v_uncond = torch.tensor([0.5, 0.5])
    plus, zero = krea_plus_neu_teachers(
        v_pos, v_neu, v_uncond, guidance=KREA2_BBOX_CFG
    )
    assert torch.allclose(plus, v_pos)
    assert torch.allclose(zero, v_neu)
    raw_plus, _raw_zero = krea_plus_neu_teachers(
        v_pos, v_neu, v_uncond, guidance=KREA_RAW_CFG
    )
    assert not torch.allclose(plus, raw_plus)
    assert krea_scheduler_mu(is_distilled=True, mu=None) == pytest.approx(1.15)
    assert krea_scheduler_mu(is_distilled=False, mu=1.15) == pytest.approx(1.15)
    assert krea_scheduler_mu(is_distilled=False, mu=None) is None
    assert krea_scheduler_mu(is_distilled=True, mu=1.15) == pytest.approx(KREA2_BBOX_MU)


def test_stock_krea_refuses_bbox_id_and_keeps_raw_turbo_cards():
    assert is_krea2_turbo_bbox_id(BBOX_ID) is True
    assert is_krea2_turbo_bbox_id(KREA2_BBOX_COMFY_FILE) is True
    assert is_krea2_turbo_bbox_id(KREA_RAW_MODEL) is False
    assert is_krea2_turbo_bbox_id("/comfy/Krea-2-Turbo.safetensors") is False
    assert krea_looks_turbo(BBOX_ID) is False
    assert krea_looks_turbo("/comfy/Krea-2-Turbo.safetensors") is True
    assert krea_looks_turbo(KREA_RAW_MODEL) is False
    with pytest.raises(ValueError, match="krea2-turbo-bbox"):
        krea_sample_card(BBOX_ID)
    with pytest.raises(ValueError, match="krea2-turbo-bbox"):
        resolve_krea_card(BBOX_ID, None, None)
    raw = krea_sample_card(KREA_RAW_MODEL)
    assert raw["variant"] == "raw"
    assert raw["sample_steps"] == 28
    assert raw["sample_guidance"] == 4.5
    turbo = krea_sample_card("/comfy/Krea-2-Turbo.safetensors")
    assert turbo["variant"] == "turbo"
    assert turbo["sample_steps"] == 8
    assert turbo["sample_guidance"] == 0.0
    with pytest.raises(ValueError, match="krea2-turbo-bbox"):
        assert_krea_only(BBOX_ID)
    with pytest.raises(ValueError, match="krea2-turbo-bbox"):
        assert_krea_only(KREA2_BBOX_COMFY_FILE)
    assert_krea_only(KREA_RAW_MODEL)
    assert_krea_only("/comfy/Krea-2-Turbo.safetensors")
    with pytest.raises(ValueError, match="krea2-turbo-bbox"):
        train_krea(
            parse_krea(
                ["--dummy", "--model_id", BBOX_ID, "--steps", "1", "--save_dir", "/tmp/nope"]
            )
        )


def test_foreign_trainers_refuse_bbox_id():
    with pytest.raises(ValueError, match="Supra-only"):
        assert_supra_only(BBOX_ID)
    with pytest.raises(ValueError, match="Supra-only"):
        train_supra(parse_supra(["--dummy", "--model_id", BBOX_ID, "--steps", "1"]))
    with pytest.raises(ValueError, match="Sana-only"):
        assert_sana_only(BBOX_ID)
    with pytest.raises(ValueError, match="Sana-only"):
        train_sana(parse_sana(["--dummy", "--model_id", BBOX_ID, "--steps", "1"]))
    with pytest.raises(ValueError, match="Anima-only"):
        assert_anima_only(BBOX_ID)
    with pytest.raises(ValueError, match="Anima-only"):
        train_anima(parse_anima(["--dummy", "--model_id", BBOX_ID, "--steps", "1"]))
    assert_anima_only("circlestone-labs/Anima-Base-v1.0-Diffusers")


def test_krea2_refuses_stock_raw_and_foreign_ids():
    with pytest.raises(ValueError, match="Krea2-turbo-bbox-only"):
        assert_krea2_only("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="Krea2-turbo-bbox-only"):
        assert_krea2_only("/comfy/Krea-2-Turbo.safetensors")
    with pytest.raises(ValueError, match="anima"):
        assert_krea2_only("circlestone-labs/Anima-Base-v1.0-Diffusers")
    with pytest.raises(ValueError, match="supra"):
        assert_krea2_only("SupraLabs/Supra2-IMG")
    with pytest.raises(ValueError, match="sana"):
        assert_krea2_only("Efficient-Large-Model/Sana_600M_512px_diffusers")
    assert_krea2_only(BBOX_ID)
    assert_krea2_only("/weights/" + KREA2_BBOX_COMFY_FILE)
    assert_krea2_skeleton("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="skeleton"):
        assert_krea2_skeleton("krea/Krea-2-Turbo")
    with pytest.raises(ValueError, match="anima"):
        assert_krea2_skeleton("circlestone-labs/Anima-Base-v1.0-Diffusers")
    with pytest.raises(ValueError, match="Krea2-turbo-bbox-only"):
        train(parse_args(["--dummy", "--model_id", "krea/Krea-2-Raw", "--steps", "1"]))


def test_load_plan_hub_local_and_comfy_without_download(tmp_path: Path):
    hub = resolve_krea2_bbox_source(BBOX_ID)
    assert hub["source"] == "hub_subfolder"
    assert hub["repo_id"] == BBOX_ID
    assert hub["subfolder"] == "epoch-14-step-73184/transformer"
    assert hub["skeleton"] == "krea/Krea-2-Raw"
    assert hub["mu"] == pytest.approx(1.15)
    assert hub["sample_steps"] == 8
    assert hub["sample_guidance"] == 0.0
    assert hub["is_distilled"] is True
    assert hub["transformer_path"] is None

    comfy = tmp_path / KREA2_BBOX_COMFY_FILE
    comfy.write_bytes(b"not-a-real-weight")
    plan = resolve_krea2_bbox_source(BBOX_ID, transformer=str(comfy))
    assert plan["source"] == "comfy_safetensors"
    assert plan["transformer_path"] == str(comfy.resolve())
    assert plan["subfolder"] is None

    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    local = resolve_krea2_bbox_source(str(root))
    assert local["source"] == "local_transformer"
    assert local["subfolder"] == ""

    clone = tmp_path / "clone"
    nested = clone / "epoch-14-step-73184" / "transformer"
    nested.mkdir(parents=True)
    (clone / "model_index.json").write_text("{}", encoding="utf-8")
    (clone / "config.json").write_text("{}", encoding="utf-8")
    (nested / "config.json").write_text("{}", encoding="utf-8")
    cloned = resolve_krea2_bbox_source(str(clone))
    assert cloned["source"] == "local_transformer"
    assert cloned["subfolder"] == "epoch-14-step-73184/transformer"

    with pytest.raises(ValueError, match="not a local"):
        resolve_krea2_bbox_source(BBOX_ID, transformer=str(tmp_path / "missing.safetensors"))


def test_dummy_train_uses_distilled_card_not_row_guidance(tmp_path: Path):
    raw_guidance = tmp_path / "prompts.yaml"
    raw_guidance.write_text(
        """
plus_label: Happy
minus_label: Sad
concept_words: "smiling, smile, teeth"
control_prompt: "a bowl of fruit on a table"
bare_captions: true
rows:
  - target: "a person, neutral expression, closed mouth"
    positive: "a person, big smile showing teeth, happy joyful expression"
    neutral: "a person, neutral expression, closed mouth"
    negative: "a sad person"
    attributes: ["male", "female"]
    guidance_scale: 4.5
""",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-krea2-dummy",
            "--prompts_file",
            str(raw_guidance),
            "--save_dir",
            str(tmp_path / "out"),
            "--steps",
            "8",
            "--hold_weight",
            "0.1",
            "--seed",
            "7",
        ]
    )
    sidecar_path = train(args)
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "krea2_turbo_bbox"
    assert payload["variant"] == "turbo_bbox"
    assert payload["model_id"] == BBOX_ID
    assert payload["sample_steps"] == 8
    assert payload["sample_guidance"] == 0.0
    assert payload["train_guidance"] == 0.0
    assert payload["mu"] == pytest.approx(1.15)
    assert payload["is_distilled"] is True
    assert payload["skeleton"] == "krea/Krea-2-Raw"
    assert payload["weights"]["source"] == "hub_subfolder"
    assert payload["not_raw_card"]["raw_cfg"] == 4.5
    assert payload["not_raw_card"]["raw_steps"] == 28
    assert payload["minus_teacher"] is False
    assert payload["minus_canary"] is True
    assert payload["token_hold"] == "unused_to_neu"
    assert payload["lyric_hold"] is False
    assert payload["bare_captions"] is True
    assert payload["control_prompt"] == KREA_CONTROL_PROMPT
    assert payload["dummy"] is True
    assert payload["allow_hub"] is False
    assert payload["hold_weight"] == pytest.approx(0.1)
    assert payload["lora_targets"] == "dit"
    assert "4.5" not in payload["official"]
    assert "mu=1.15" in payload["official"]
    lines = [
        json.loads(line)
        for line in (tmp_path / "out" / "smile-krea2-dummy_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(lines) == 2
    assert lines[-1]["minus_teacher"] == 0.0
    meta = json.loads((tmp_path / "out" / "samples" / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["gate"] == "smile-first"
    assert "a bowl of fruit on a table" in meta["prompts"]
    assert all(shot["cfg"] == 0.0 for shot in meta["samples"])
    assert all(shot["sample_steps"] == 2 for shot in meta["samples"])


def test_dummy_default_yaml_writes_grid_without_hub(tmp_path: Path):
    sys.modules.pop("conceptmod.textsliders.krea2_bbox_live", None)
    sys.modules.pop("conceptmod.textsliders.krea_live", None)
    sys.modules.pop("conceptmod.textsliders.krea_weights", None)
    args = parse_args(
        [
            "--dummy",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--steps",
            "4",
            "--seed",
            "3",
        ]
    )
    sidecar = json.loads(Path(train(args)).read_text(encoding="utf-8"))
    assert sidecar["transformer_subfolder"] == KREA2_BBOX_SUBFOLDER
    assert sidecar["bare_captions"] is True
    pngs = list((tmp_path / "samples").glob("*.png"))
    # 3 neu captions + fruit-bowl control, 4 scales
    assert len(pngs) == 16
    assert "conceptmod.textsliders.krea2_bbox_live" not in sys.modules
    assert "conceptmod.textsliders.krea_live" not in sys.modules
    assert "conceptmod.textsliders.krea_weights" not in sys.modules


def test_explicit_guidance_override_is_not_silent(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--sample_guidance",
            "1.5",
            "--sample_steps",
            "6",
            "--mu",
            "0.8",
            "--save_dir",
            str(tmp_path),
            "--steps",
            "1",
        ]
    )
    payload = json.loads(Path(train(args)).read_text(encoding="utf-8"))
    assert payload["sample_guidance"] == pytest.approx(1.5)
    assert payload["sample_steps"] == 6
    assert payload["mu"] == pytest.approx(0.8)
    assert payload["train_guidance"] == pytest.approx(1.5)


def test_embed_dummy_stays_on_distilled_card(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--lm_target",
            "embed",
            "--lora_targets",
            "te",
            "--save_dir",
            str(tmp_path),
            "--steps",
            "1",
            "--hold_weight",
            "0.1",
        ]
    )
    payload = json.loads(Path(train(args)).read_text(encoding="utf-8"))
    assert payload["lm_target"] == "embed"
    assert payload["lora_targets"] == "te"
    assert payload["sample_guidance"] == 0.0
    assert payload["sample_steps"] == 8
    assert payload["mu"] == pytest.approx(1.15)
    assert payload["dit_velocity_supervised"] is False


def test_live_module_pins_distilled_load_and_is_not_the_raw_trainer():
    src = LIVE.read_text(encoding="utf-8")
    assert "Krea2Transformer2DModel" in src
    assert "Krea2Pipeline" in src
    assert "subfolder" in src
    assert "is_distilled" in src
    assert "load_comfy_krea_transformer" in src
    assert "krea/Krea-2-Raw" in src
    assert "epoch-14-step-73184/transformer" in src
    trainer = (ROOT / "conceptmod/textsliders/train_lora_krea2.py").read_text(encoding="utf-8")
    assert "train_lora_krea.py" in trainer
    assert "CFG 4.5" in trainer or "4.5" in trainer
    stock = (ROOT / "conceptmod/textsliders/train_lora_krea.py").read_text(encoding="utf-8")
    assert "train_lora_krea2.py" in stock
