"""Krea2 turbo-bbox load plan and trainer card.

Pure helpers: no Hub download and no ``krea_live`` import. ``--dummy``
and pytest stay on this module. Live weights go through
``krea2_bbox_live.py``.

Defaults are the distilled card (8 steps, CFG 0, mu=1.15). They are
not the Raw train card (CFG 4.5 / 28).
"""

from __future__ import annotations

from pathlib import Path

from conceptmod.textsliders.slider_targets import (
    KREA2_BBOX_CFG,
    KREA2_BBOX_COMFY_FILE,
    KREA2_BBOX_EPOCH,
    KREA2_BBOX_MODEL,
    KREA2_BBOX_MU,
    KREA2_BBOX_SKELETON,
    KREA2_BBOX_STEPS,
    KREA2_BBOX_SUBFOLDER,
    KREA_CONTROL_PROMPT,
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_HOLD_WEIGHT,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_SMILE_HOLD_WEIGHT,
    is_krea2_turbo_bbox_id,
)

# Substring bans, same idea as Sana / Supra / stock Krea. "krea" itself
# is not banned: this trainer's model id and Raw skeleton contain it.
_FOREIGN_BACKENDS = (
    "anima",
    "sana",
    "supra",
    "z-image",
    "zimage",
    "zit",
    "minimax",
    "h3",
)

# Stock full pipelines. ``krea2-turbo-bbox`` does not contain these.
_STOCK_PIPELINE_MARKERS = ("krea-2-raw", "krea-2-turbo")

DEFAULT_PROMPTS = "conceptmod/textsliders/data/prompts-krea2-bbox.yaml"


def _refuse_krea2_candidate(text: str, *, role: str) -> None:
    lowered = text.lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                "this trainer is Krea2-turbo-bbox-only; "
                f"refused foreign backend {name!r} in {role} {text!r}"
            )
    if is_krea2_turbo_bbox_id(lowered):
        return
    for marker in _STOCK_PIPELINE_MARKERS:
        if marker in lowered:
            raise ValueError(
                "this trainer is Krea2-turbo-bbox-only; refused stock Krea "
                f"pipeline id {text!r} ({role}). Raw CFG "
                f"{KREA_RAW_CFG:g} / {KREA_RAW_STEPS} steps stay on "
                "train_lora_krea.py. This backend trains and samples at "
                f"CFG {KREA2_BBOX_CFG:g}, {KREA2_BBOX_STEPS} steps, "
                f"mu={KREA2_BBOX_MU:g}."
            )


def assert_krea2_only(model_id: str, transformer: str | None = None) -> None:
    """Fail closed on foreign backends and on stock Raw / Turbo pipeline ids."""
    _refuse_krea2_candidate(str(model_id), role="model_id")
    if transformer:
        _refuse_krea2_candidate(str(transformer), role="transformer")


def assert_krea2_skeleton(skeleton: str) -> None:
    """VAE + Qwen3-VL come from Raw, not from the transformer-only repo."""
    text = str(skeleton)
    lowered = text.lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                "Krea2-turbo-bbox skeleton is krea/Krea-2-Raw "
                f"(VAE + Qwen3-VL); refused foreign backend {name!r} "
                f"in skeleton {text!r}"
            )
    if Path(text).exists():
        return
    if "krea-2-raw" in lowered:
        return
    raise ValueError(
        "Krea2-turbo-bbox skeleton must be krea/Krea-2-Raw "
        "(VAE, text encoder, tokenizer, scheduler). "
        f"Refused {text!r}."
    )


def resolve_krea2_bbox_card(
    sample_steps: int | None,
    sample_guidance: float | None,
    mu: float | None = None,
) -> dict[str, float | int | str | bool]:
    """Distilled defaults. Explicit CLI values win; Raw 4.5 / 28 never do."""
    return {
        "variant": "turbo_bbox",
        "sample_steps": int(KREA2_BBOX_STEPS if sample_steps is None else sample_steps),
        "sample_guidance": float(
            KREA2_BBOX_CFG if sample_guidance is None else sample_guidance
        ),
        "mu": float(KREA2_BBOX_MU if mu is None else mu),
        "is_distilled": True,
        "skeleton": KREA2_BBOX_SKELETON,
    }


def _local_safetensors(path: Path) -> Path | None:
    if path.suffix == ".safetensors" and path.is_file():
        return path.resolve()
    return None


def _local_transformer_subfolder(path: Path, subfolder: str) -> str | None:
    """Subfolder for ``from_pretrained``, ``""`` if ``path`` is the transformer root.

    None when no ``config.json`` is present. A pipeline ``model_index.json``
    at the root is not a transformer folder.
    """
    if (path / "config.json").is_file() and not (path / "model_index.json").is_file():
        return ""
    nested = path / subfolder
    if subfolder and nested.is_dir() and (nested / "config.json").is_file():
        return subfolder
    if (path / "transformer" / "config.json").is_file():
        return "transformer"
    return None


def _plan(**fields: object) -> dict[str, object]:
    fields.update(
        {
            "mu": float(KREA2_BBOX_MU),
            "sample_steps": int(KREA2_BBOX_STEPS),
            "sample_guidance": float(KREA2_BBOX_CFG),
            "is_distilled": True,
            "comfy_filename": KREA2_BBOX_COMFY_FILE,
            "epoch": KREA2_BBOX_EPOCH,
        }
    )
    return fields


def resolve_krea2_bbox_source(
    model_id: str,
    *,
    subfolder: str | None = None,
    transformer: str | None = None,
    skeleton: str | None = None,
) -> dict[str, object]:
    """Decide hub subfolder vs local diffusers dir vs Comfy safetensors.

    Does not download. A missing local path is a Hub id.
    """
    sub = str(subfolder or KREA2_BBOX_SUBFOLDER).strip()
    skel = str(skeleton or KREA2_BBOX_SKELETON)
    chosen = str(transformer).strip() if transformer else str(model_id)
    path = Path(chosen)
    weights = _local_safetensors(path)
    if weights is not None:
        return _plan(
            source="comfy_safetensors",
            model_id=str(model_id),
            transformer_path=str(weights),
            subfolder=None,
            repo_id=None,
            skeleton=skel,
        )
    if path.is_dir():
        local_sub = _local_transformer_subfolder(path, sub)
        if local_sub is None:
            raise ValueError(
                f"local path {path} has no Krea2 transformer config.json "
                f"(looked at the directory, {sub!r}, and transformer/)"
            )
        return _plan(
            source="local_transformer",
            model_id=str(model_id),
            transformer_path=str(path.resolve()),
            subfolder=local_sub,
            repo_id=None,
            skeleton=skel,
        )
    if transformer:
        raise ValueError(
            f"--transformer {transformer!r} is not a local .safetensors file "
            "or diffusers directory"
        )
    if not sub:
        raise ValueError("hub krea2-turbo-bbox load needs a transformer subfolder")
    return _plan(
        source="hub_subfolder",
        model_id=str(model_id),
        transformer_path=None,
        subfolder=sub,
        repo_id=str(model_id),
        skeleton=skel,
    )


def live_train_card(
    *,
    name: str = "smile-krea2-bbox",
    prompts_file: str = DEFAULT_PROMPTS,
    model_id: str = KREA2_BBOX_MODEL,
    subfolder: str = KREA2_BBOX_SUBFOLDER,
    skeleton: str = KREA2_BBOX_SKELETON,
    rank: int = KREA_DEFAULT_RANK,
    resolution: int = KREA_DEFAULT_RESOLUTION,
    sample_steps: int = KREA2_BBOX_STEPS,
    sample_guidance: float = KREA2_BBOX_CFG,
    mu: float = KREA2_BBOX_MU,
    hold_weight: float = KREA_SMILE_HOLD_WEIGHT,
    lora_targets: str = KREA_DEFAULT_LORA_TARGETS,
) -> dict[str, object]:
    """Card a product repo can wrap. Distilled defaults, not Raw."""
    return {
        "backend": "krea2_turbo_bbox",
        "name": name,
        "prompts_file": prompts_file,
        "model_id": model_id,
        "transformer_subfolder": subfolder,
        "skeleton": skeleton,
        "comfy_filename": KREA2_BBOX_COMFY_FILE,
        "epoch": KREA2_BBOX_EPOCH,
        "load": (
            "Krea2Transformer2DModel.from_pretrained(model_id, subfolder=...) "
            "into Krea2Pipeline.from_pretrained(skeleton, transformer=tf)"
        ),
        "rank": int(rank),
        "resolution": int(resolution),
        "lora_targets": lora_targets,
        "sample_steps": int(sample_steps),
        "sample_guidance": float(sample_guidance),
        "mu": float(mu),
        "is_distilled": True,
        "hold_weight": float(hold_weight),
        "control_prompt": KREA_CONTROL_PROMPT,
        "uni": {
            "plus": "v(pos) at CFG 0 (not Raw CFG 4.5)",
            "zero": "v(neu)",
            "minus": "canary only",
            "hold": "unused tokens to encode(neu); concept words are not held",
            "captions": "bare (attributes are not prefixed)",
            "control_prompt": "verify only, never a teacher",
        },
        "not_raw_card": {
            "raw_cfg": KREA_RAW_CFG,
            "raw_steps": KREA_RAW_STEPS,
            "raw_model": KREA_RAW_MODEL,
        },
        "comfy_cfg_note": (
            "Comfy's Krea-2 Turbo template says CFG 1.0, which is guidance "
            "off in that UI. This trainer's diffusers convention is "
            "guidance_scale 0 (v = v(cond)). Do not pass 4.5."
        ),
        "age_hold_default": KREA_HOLD_WEIGHT,
        "smile_hold": KREA_SMILE_HOLD_WEIGHT,
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
        "non_goals": (
            "product repo",
            "vendored weights",
            "Music 3 default changes",
            "GPU train in CI",
        ),
    }


def live_train_command(
    *,
    name: str = "smile-krea2-bbox",
    prompts_file: str = DEFAULT_PROMPTS,
    model_id: str = KREA2_BBOX_MODEL,
    subfolder: str = KREA2_BBOX_SUBFOLDER,
    skeleton: str = KREA2_BBOX_SKELETON,
    rank: int = KREA_DEFAULT_RANK,
    resolution: int = KREA_DEFAULT_RESOLUTION,
    sample_steps: int = KREA2_BBOX_STEPS,
    sample_guidance: float = KREA2_BBOX_CFG,
    mu: float = KREA2_BBOX_MU,
    hold_weight: float = KREA_SMILE_HOLD_WEIGHT,
    lora_targets: str = KREA_DEFAULT_LORA_TARGETS,
    save_dir: str | None = None,
) -> str:
    """CLI a product repo can shell out to. Hub download stays opt-in."""
    dest = save_dir or f"models/{name}"
    guidance = f"{float(sample_guidance):g}"
    return (
        "CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea2.py \\\n"
        f"  --name {name} \\\n"
        f"  --prompts_file {prompts_file} \\\n"
        f"  --model_id {model_id} --allow_hub \\\n"
        f"  --transformer_subfolder {subfolder} \\\n"
        f"  --skeleton_model {skeleton} \\\n"
        f"  --lora_targets {lora_targets} --rank {int(rank)} "
        f"--resolution {int(resolution)} \\\n"
        f"  --sample_steps {int(sample_steps)} --sample_guidance {guidance} "
        f"--mu {float(mu):g} \\\n"
        f"  --hold_weight {float(hold_weight):g} --steps 800 --lr 1e-4 "
        "--seed 7 --device 0 \\\n"
        f"  --save_dir {dest}"
    )
