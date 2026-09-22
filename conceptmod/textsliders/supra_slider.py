"""Supra2-IMG slider geometry: UNI + unused-token hold.

Image analog of the Anima trainer contract (not Music 3 lyric-hold):

- student +1 fits the neutral / infer caption
- the + caption is the teacher only
- scale 0 stays on the neutral caption
- unused prompt tokens / pinned attributes are held to encode(neu)
- declared ``concept_words`` are not held
- attributes are pins only — never prefixed onto captions
- minus is a canary, never a teacher

Sampling matches the Hub script, not Anima's FlowMatch σ schedule:

    t_i = i / K          # i = 0 .. K-1, t in [0, 1)
    z  <- z + (1/K) * v

Train-time CFG direction is conceptmod's ``v(z, t, c) − v(z, t, '')``.
Hub sample CFG is ``v_u + cfg * (v_c − v_u)`` at cfg 3.0, 50 Euler steps.

CPU-pure. No Hub, no GPU, no ``model_final_ema.pt``. Does not change the
Music 3 trainer default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from conceptmod.textsliders.supra_model import (
    CKPT_FILENAME,
    SUPRA_PARAM_COUNT,
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
    VAE_SCALE,
)

DEFAULT_MODEL_ID = HF_REPO
DEFAULT_RANK = 16
DEFAULT_ALPHA = 16.0
DEFAULT_RESOLUTION = IMG_SIZE
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_CFG = 3.0
DEFAULT_HOLD_WEIGHT = 1.0
DEFAULT_LR = 1e-4
DEFAULT_CONTROL_PROMPT = "a bowl of fruit on a table"
DEFAULT_SAMPLE_SCALES = (0.0, 0.25, 0.5, 1.0)
DEFAULT_SAMPLE_SEED = 42
DEFAULT_SAMPLE_EVERY = 100
# No ModularPipeline. Sample is the same Euler ``predict_v`` loop as the loss.
SUPRA_SAMPLE_MODES = ("train_faithful",)
DEFAULT_SAMPLE_MODE = "train_faithful"
SUPRA_LM_TARGETS = ("trajectory", "direct", "cfg_delta")
DEFAULT_LM_TARGET = "trajectory"
DEFAULT_TRAJ_STEPS = 4
DEFAULT_TRAJ_IDENTITY_WEIGHT = 0.25
DEFAULT_TEACHER_GAP_BOOST = 1.0
# Cross-attn is the text path (frozen Flan-T5 has no AnimaTextConditioner).
# ``dit`` is self-attn only. ``proj`` collides, so targets are path suffixes.
CROSS_LORA_TARGETS = (
    "ctx_proj",
    "cross_attn.q",
    "cross_attn.kv",
    "cross_attn.proj",
)
DIT_LORA_TARGETS = (
    "self_attn.qkv",
    "self_attn.proj",
)
SUPRA_LORA_TARGET_CHOICES = ("cross", "dit", "dit+cross")
DEFAULT_LORA_TARGETS = "cross"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NOISE_MEAN_LO = 110.0
_NOISE_MEAN_HI = 145.0
_NOISE_STD_LO = 68.0
_NOISE_STD_HI = 82.0
_NOISE_CORR_MAX = 0.15

# Lighting, not age. Neutral daylight vs warm sun. + is the teacher only.
WOMAN_NEU = "a woman sitting by a window, neutral daylight, even illumination"
WOMAN_PLUS = "a woman sitting by a window, warm golden sunlit glow"
MAN_NEU = "a man reading at a table, neutral daylight, even illumination"
MAN_PLUS = "a man reading at a table, warm golden sunlit glow"
DEFAULT_CONCEPT_WORDS = "warm, golden, sunlit, glow"
PRODUCT_HANDOFF = (
    "Backend only. A future HyperGAN/supra-concept-sliders product repo "
    "(Comfy, studio UI) should consume train_lora_supra.py the way "
    "anima-concept-sliders consumes the Anima backend. This PR does not "
    "ship that repo, a Comfy plugin, or Hub weights."
)


@dataclass
class SupraSliderRow:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = DEFAULT_CFG
    resolution: int = DEFAULT_RESOLUTION
    batch_size: int = 1
    pins: list[str] = field(default_factory=list)
    concept_words: str = ""

    @property
    def has_minus_canary(self) -> bool:
        return bool(str(self.negative).strip())

    @property
    def infer_prompt(self) -> str:
        return self.target.strip() or self.neutral


@dataclass
class SupraPromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    concept_words: str = ""


def word_tokens(text: str) -> list[str]:
    """Whitespace / alnum tokenizer. Images have no lyric special tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def parse_concept_words(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    return word_tokens(" ".join(parts))


def unused_vocab(
    target: str,
    neutral: str,
    attributes: Iterable[str] | None = None,
    extra_pins: Iterable[str] | None = None,
    concept_words: str | Iterable[str] | None = None,
) -> set[str]:
    """Subject, composition, and pinned attributes — never concept words."""
    vocab = set(word_tokens(target)) | set(word_tokens(neutral))
    for item in list(attributes or []) + list(extra_pins or []):
        vocab.update(word_tokens(str(item)))
    for tok in parse_concept_words(concept_words):
        vocab.discard(tok)
    return vocab


def concept_tokens(positive: str, unused: set[str]) -> list[str]:
    return [tok for tok in word_tokens(positive) if tok not in unused]


def unused_token_mask(tokens: Sequence[str], unused: set[str]) -> list[bool]:
    return [tok in unused for tok in tokens]


def align_unused_positions(
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused: set[str],
) -> list[tuple[int, int]]:
    """Pair unused + tokens with the next matching unused neu token."""
    pairs: list[tuple[int, int]] = []
    neu_idx = 0
    for pos_i, tok in enumerate(pos_tokens):
        if tok not in unused:
            continue
        found = None
        for j in range(neu_idx, len(neu_tokens)):
            if neu_tokens[j] == tok:
                found = j
                break
        if found is None:
            for j, neu_tok in enumerate(neu_tokens):
                if neu_tok == tok:
                    found = j
                    break
        if found is None:
            continue
        pairs.append((pos_i, found))
        neu_idx = found + 1
    return pairs


def splice_unused_embeds(
    pos_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    held = pos_embeds.clone()
    for pos_i, neu_i in pairs:
        held[..., pos_i, :] = neu_embeds[..., neu_i, :]
    return held


def supra_cfg_delta(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """Train-time conceptmod CFG direction: ``v(z,t,c) − v(z,t,'')``."""
    return v_cond - v_uncond


def supra_sample_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    cfg: float,
) -> torch.Tensor:
    """Hub Euler CFG: ``v_u + cfg * (v_c − v_u)``. ``cfg <= 1`` is conditional only."""
    scale = float(cfg)
    if scale <= 1.0 + 1e-8:
        return v_cond
    return v_uncond + scale * (v_cond - v_uncond)


def supra_uni_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    v_uncond: torch.Tensor,
    v_neg: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    del v_neg
    return {
        "plus": supra_cfg_delta(v_pos, v_uncond),
        "zero": supra_cfg_delta(v_neu, v_uncond),
        "minus": None,
    }


def supra_uni_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
    student_minus: torch.Tensor | None = None,
    teacher_minus: torch.Tensor | None = None,
) -> torch.Tensor:
    """``MSE(+) + MSE(0)``. Minus tensors are accepted as a canary and ignored."""
    del student_minus, teacher_minus
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def supra_direct_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    return {"plus": v_pos, "zero": v_neu, "minus": None}


def supra_direct_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def supra_boost_teacher(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """``v_neu + boost * (v_pos − v_neu)`` for direct / cfg_delta. ``boost <= 1`` is off."""
    scale = float(boost)
    if scale <= 1.0 + 1e-8:
        return v_pos
    return v_neu + scale * (v_pos - v_neu)


def supra_euler_times(
    num_steps: int,
    device=None,
    dtype=None,
) -> tuple[torch.Tensor, float]:
    """Hub Euler grid: ``t_i = i/K`` for ``i = 0 .. K-1``, ``dt = 1/K``."""
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"traj_steps must be >= 1, got {num_steps!r}")
    dt = 1.0 / steps
    times = torch.arange(steps, device=device, dtype=dtype) * dt
    return times, dt


def supra_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Hub update ``z <- z + dt * v`` (t runs 0 → 1, not Anima σ 1 → 0)."""
    return sample + dt * velocity


def supra_short_trajectory(
    backend,
    prompt: str,
    z_t: torch.Tensor,
    *,
    num_steps: int,
    frozen: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """K-step Hub Euler over ``predict_v``. ``z_t`` is the t=0 noise latent."""
    steps = int(num_steps)
    times, dt = supra_euler_times(steps, device=z_t.device, dtype=torch.float32)
    x = z_t
    for i in range(steps):
        t = times[i].reshape(1).to(device=z_t.device)
        velocity = backend.predict_v(prompt, x, t, frozen=frozen, scale=scale)
        x = supra_euler_step(x, velocity, dt)
    return x


def supra_euler_sample_latents(
    backend,
    prompt: str,
    *,
    num_steps: int,
    cfg: float = DEFAULT_CFG,
    z: torch.Tensor | None = None,
    uncond_prompt: str = "",
) -> torch.Tensor:
    """Hub Euler sample in latent space. No VAE decode.

    ``cfg > 1`` uses ``v_u + cfg * (v_c − v_u)`` with a frozen pass, matching
    ``inference.py``. ``z`` defaults to ``N(0, I)`` at ``backend.latent_shape``.
    """
    if z is None:
        z = torch.randn((1, *backend.latent_shape), device=getattr(backend, "device", "cpu"))
    x = z
    times, dt = supra_euler_times(int(num_steps), device=x.device, dtype=torch.float32)
    use_cfg = float(cfg) > 1.0 + 1e-8
    for i in range(int(num_steps)):
        t = times[i].reshape(1).to(device=x.device)
        v_cond = backend.predict_v(prompt, x, t, frozen=True)
        if use_cfg:
            v_uncond = backend.predict_v(uncond_prompt, x, t, frozen=True)
            velocity = supra_sample_cfg(v_cond, v_uncond, cfg)
        else:
            velocity = v_cond
        x = supra_euler_step(x, velocity, dt)
    return x


def supra_trajectory_loss(
    x_student: torch.Tensor,
    x_plus: torch.Tensor,
    x_zero: torch.Tensor | None = None,
    x_neu: torch.Tensor | None = None,
    identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
) -> torch.Tensor:
    """``MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)``."""
    loss = F.mse_loss(x_student, x_plus)
    weight = float(identity_weight)
    if weight > 0.0 and x_zero is not None and x_neu is not None:
        loss = loss + weight * F.mse_loss(x_zero, x_neu)
    return loss


def supra_unused_hold_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pairs: Sequence[tuple[int, int]] | None = None,
    pred_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE of unused positions to encode(neu). Concept words skipped."""
    if pairs is not None:
        if not pairs:
            return pred.reshape(-1)[:1].new_zeros(())
        pred_idx = [p for p, _ in pairs]
        tgt_idx = [n for _, n in pairs]
        return F.mse_loss(pred[..., pred_idx, :], tgt[..., tgt_idx, :])
    if pred_mask is None:
        raise ValueError("supra_unused_hold_loss needs pairs or pred_mask")
    mask = pred_mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - tgt).pow(2) * mask).sum() / denom


def minus_canary_cosine(
    student_plus: torch.Tensor,
    v_neg: torch.Tensor,
    v_uncond: torch.Tensor,
) -> torch.Tensor:
    minus = supra_cfg_delta(v_neg, v_uncond).flatten().unsqueeze(0)
    plus = student_plus.flatten().unsqueeze(0)
    return F.cosine_similarity(plus, minus, dim=1, eps=1e-6).mean()


def resolve_supra_lm_target(lm_target: str | None = None) -> str:
    recipe = str(lm_target or DEFAULT_LM_TARGET).strip().lower()
    if recipe not in SUPRA_LM_TARGETS:
        raise ValueError(
            f"supra lm_target must be one of {SUPRA_LM_TARGETS}, got {lm_target!r}. "
            "embed_struct / same_crop stay on the Anima trainer; Music 3 stays v9."
        )
    return recipe


@dataclass(frozen=True)
class SupraLoraSpec:
    """Which SupraDiT linears receive LoRA.

    Flan-T5 and the SD VAE stay frozen. There is no separate text
    conditioner: caption signal enters through ``ctx_proj`` + cross-attn.
    """

    label: str
    train_dit: bool
    train_cross: bool

    @property
    def frozen_modules(self) -> tuple[str, ...]:
        frozen = ["text_encoder", "vae"]
        if not self.train_dit:
            frozen.append("self_attn")
        if not self.train_cross:
            frozen.extend(["cross_attn", "ctx_proj"])
        return tuple(frozen)

    @property
    def active_attn_targets(self) -> list[str]:
        names: list[str] = []
        if self.train_dit:
            names.extend(DIT_LORA_TARGETS)
        if self.train_cross:
            names.extend(CROSS_LORA_TARGETS)
        return names

    @property
    def adapted_module_names(self) -> list[str]:
        names: list[str] = []
        if self.train_cross:
            names.append("cross_attn")
        if self.train_dit:
            names.append("self_attn")
        return names


def resolve_supra_lora_targets(lora_targets: str | None = None) -> SupraLoraSpec:
    raw = str(lora_targets if lora_targets is not None else DEFAULT_LORA_TARGETS)
    label = raw.strip().lower().replace(" ", "")
    aliases = {
        "cross_attn": "cross",
        "ctx": "cross",
        "self": "dit",
        "self_attn": "dit",
        "transformer": "dit",
        "cross+dit": "dit+cross",
        "dit+self": "dit",
        "self+cross": "dit+cross",
    }
    label = aliases.get(label, label)
    if label not in SUPRA_LORA_TARGET_CHOICES:
        raise ValueError(
            f"supra lora_targets must be one of {SUPRA_LORA_TARGET_CHOICES}, "
            f"got {lora_targets!r}. Flan-T5 text_encoder and the SD VAE are not adapted."
        )
    parts = set(label.split("+"))
    return SupraLoraSpec(
        label=label,
        train_dit="dit" in parts,
        train_cross="cross" in parts,
    )


def expand_attributes_supra(row: dict) -> list[dict]:
    """Pin attributes for unused-token hold. Do not prefix captions."""
    item = dict(row)
    attributes = [
        str(a).strip() for a in (row.get("attributes") or []) if str(a).strip()
    ]
    pins = [str(p).strip() for p in (row.get("pins") or []) if str(p).strip()]
    for attr in attributes:
        if attr not in pins:
            pins.append(attr)
    item["pins"] = pins
    item["attributes"] = attributes
    return [item]


def _as_row(item: dict) -> SupraSliderRow:
    if "target" not in item and "positive" not in item:
        raise ValueError(f"supra prompt row needs target or positive: {item!r}")
    target = str(item.get("target") or item.get("neutral") or "")
    positive = str(item.get("positive") or target)
    neutral = str(item.get("neutral") or target)
    attributes = [str(a).strip() for a in (item.get("attributes") or []) if str(a).strip()]
    pins = [str(p).strip() for p in (item.get("pins") or []) if str(p).strip()]
    for attr in attributes:
        if attr not in pins:
            pins.append(attr)
    return SupraSliderRow(
        target=target,
        positive=positive,
        neutral=neutral,
        negative=str(item.get("negative") or ""),
        attributes=attributes,
        action=str(item.get("action") or "enhance"),
        guidance_scale=float(item.get("guidance_scale", DEFAULT_CFG)),
        resolution=int(item.get("resolution", DEFAULT_RESOLUTION)),
        batch_size=int(item.get("batch_size", 1)),
        pins=pins,
        concept_words=str(item.get("concept_words") or ""),
    )


def load_supra_prompts(path: Path | str) -> tuple[list[SupraSliderRow], SupraPromptsMeta]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    meta = SupraPromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        meta.concept_words = str(raw.get("concept_words") or "")
        raw = raw.get("rows", raw.get("prompts"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"supra prompts file is empty: {path}")
    rows: list[SupraSliderRow] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"each supra prompt must be a mapping: {item!r}")
        row_concept = str(item.get("concept_words") or meta.concept_words)
        for expanded in expand_attributes_supra(item):
            expanded.setdefault("concept_words", row_concept)
            rows.append(_as_row(expanded))
    return rows, meta


def row_token_plan(row: SupraSliderRow) -> dict[str, Any]:
    unused = unused_vocab(
        row.target,
        row.neutral,
        row.attributes,
        row.pins,
        concept_words=row.concept_words,
    )
    pos_tokens = word_tokens(row.positive)
    neu_tokens = word_tokens(row.neutral)
    return {
        "unused": unused,
        "concept": concept_tokens(row.positive, unused),
        "pos_tokens": pos_tokens,
        "neu_tokens": neu_tokens,
        "pos_hold_mask": unused_token_mask(pos_tokens, unused),
        "pairs": align_unused_positions(pos_tokens, neu_tokens, unused),
    }


class SupraSampleGateError(RuntimeError):
    """In-process sample failed the RGB-noise gate."""


def infer_sample_prompts(
    rows: Sequence[SupraSliderRow],
    control_prompt: str = DEFAULT_CONTROL_PROMPT,
) -> list[str]:
    seen: list[str] = []
    for row in rows:
        prompt = (row.infer_prompt or row.neutral or "").strip()
        if prompt and prompt not in seen:
            seen.append(prompt)
    control = str(control_prompt or "").strip()
    if control and control not in seen:
        seen.append(control)
    return seen


def image_mean_std(arr: np.ndarray) -> tuple[float, float]:
    pixels = np.asarray(arr, dtype=np.float64)
    if pixels.size == 0:
        raise ValueError("empty image for mean/std")
    return float(pixels.mean()), float(pixels.std())


def looks_like_rgb_noise(arr: np.ndarray) -> bool:
    """True for TV-static RGB (~122/75): high std, mid mean, no spatial corr."""
    pixels = np.asarray(arr)
    if pixels.ndim == 3 and pixels.shape[0] in (1, 3) and pixels.shape[-1] not in (1, 3):
        pixels = np.transpose(pixels, (1, 2, 0))
    mean, std = image_mean_std(pixels)
    if not (_NOISE_MEAN_LO <= mean <= _NOISE_MEAN_HI and _NOISE_STD_LO <= std <= _NOISE_STD_HI):
        return False
    gray = pixels.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=-1)
    if gray.shape[1] < 2:
        return True
    left = gray[:, :-1].ravel()
    right = gray[:, 1:].ravel()
    left = left - left.mean()
    right = right - right.mean()
    denom = float(left.std() * right.std())
    if denom < 1e-6:
        return True
    corr = float((left * right).mean() / denom)
    return abs(corr) < _NOISE_CORR_MAX


def assert_sample_gate(records: Sequence[dict[str, Any]]) -> None:
    if not records:
        raise SupraSampleGateError("in-process sample grid is empty")

    def _is_scale(row: dict[str, Any], target: float) -> bool:
        return abs(float(row.get("scale", 1e9)) - target) < 1e-6

    scale0 = [row for row in records if _is_scale(row, 0.0)]
    scale025 = [row for row in records if _is_scale(row, 0.25)]
    if not scale0:
        raise SupraSampleGateError("in-process sample grid missing scale 0.0")
    if any(bool(row.get("looks_like_noise")) for row in scale0):
        bad = next(row for row in scale0 if row.get("looks_like_noise"))
        raise SupraSampleGateError(
            "scale-0 sample looks like RGB noise "
            f"(mean={bad.get('mean')}/std={bad.get('std')}); base path is broken"
        )
    if scale025 and any(bool(row.get("looks_like_noise")) for row in scale025):
        bad = next(row for row in scale025 if row.get("looks_like_noise"))
        raise SupraSampleGateError(
            "scale 0.25 sample looks like RGB noise while scale 0 is fine "
            f"(mean={bad.get('mean')}/std={bad.get('std')}); adapter is broken"
        )


def resolve_supra_recipe_label(lm_target: str | None = None) -> str:
    recipe = resolve_supra_lm_target(lm_target)
    if recipe == "trajectory":
        return (
            "trajectory Hub Euler (t=i/K, z<-z+dt*v) + unused_token_hold"
        )
    if recipe == "direct":
        return "direct velocity UNI + unused_token_hold"
    return "cfg_delta UNI + unused_token_hold"


def architecture_card() -> dict[str, Any]:
    """Hub card constants. CI checks these without downloading weights."""
    return {
        "model_id": DEFAULT_MODEL_ID,
        "ckpt": CKPT_FILENAME,
        "arch": "SupraDiT",
        "params": SUPRA_PARAM_COUNT,
        "params_millions": 104.1,
        "encoder": T5_NAME,
        "encoder_frozen": True,
        "ctx_len": MAX_CTX_LEN,
        "vae": VAE_NAME,
        "vae_scale": VAE_SCALE,
        "resolution": IMG_SIZE,
        "latent_size": LATENT_SIZE,
        "latent_ch": LATENT_CH,
        "patch": PATCH,
        "num_tokens": NUM_TOKENS,
        "d_model": D_MODEL,
        "depth": DEPTH,
        "n_heads": N_HEADS,
        "head_dim": HEAD_DIM,
        "mlp_ratio": MLP_RATIO,
        "d_ctx": D_CTX,
        "sampler": "euler_flow",
        "sample_cfg": DEFAULT_CFG,
        "sample_steps": DEFAULT_SAMPLE_STEPS,
        "sample_cfg_formula": "v_uncond + cfg * (v_cond - v_uncond)",
        "train_cfg_delta": "v_cond - v_uncond",
        "euler": "t_i = i/K; z <- z + (1/K) * v",
    }


def live_train_card(
    *,
    name: str = "lighting-supra",
    prompts_file: str = "conceptmod/textsliders/data/prompts-supra.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
    lr: float = DEFAULT_LR,
    control_prompt: str = DEFAULT_CONTROL_PROMPT,
    lm_target: str = DEFAULT_LM_TARGET,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    traj_identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
    teacher_gap_boost: float = DEFAULT_TEACHER_GAP_BOOST,
    lora_targets: str = DEFAULT_LORA_TARGETS,
) -> dict[str, Any]:
    recipe = resolve_supra_lm_target(lm_target)
    spec = resolve_supra_lora_targets(lora_targets)
    return {
        "name": name,
        "model_id": model_id,
        "arch": architecture_card(),
        "lora": {
            "lora_targets": spec.label,
            "rank": rank,
            "alpha": float(rank),
            "targets": spec.active_attn_targets,
            "dit_targets": list(DIT_LORA_TARGETS),
            "cross_targets": list(CROSS_LORA_TARGETS),
            "train_dit": spec.train_dit,
            "train_cross": spec.train_cross,
            "train_text_encoder": False,
            "adapted_modules": spec.adapted_module_names,
            "frozen_modules": list(spec.frozen_modules),
            "note": (
                "Flan-T5-Base stays frozen. Cross-attn + ctx_proj is the "
                "text path (no AnimaTextConditioner). Self-attn is --lora_targets dit."
            ),
        },
        "resolution": resolution,
        "sample_steps": sample_steps,
        "cfg": cfg,
        "lr": lr,
        "device": device,
        "prompts_file": prompts_file,
        "control_prompt": control_prompt,
        "sample_scales": list(DEFAULT_SAMPLE_SCALES),
        "sample_seed": DEFAULT_SAMPLE_SEED,
        "sample_every": int(sample_every),
        "sample_mode": DEFAULT_SAMPLE_MODE,
        "sample_modes": list(SUPRA_SAMPLE_MODES),
        "lm_target": recipe,
        "lm_targets": list(SUPRA_LM_TARGETS),
        "traj_steps": int(traj_steps),
        "traj_identity_weight": float(traj_identity_weight),
        "teacher_gap_boost": float(teacher_gap_boost),
        "recipe": resolve_supra_recipe_label(recipe),
        "traj_loop": "Hub Euler over predict_v: t=i/K, z<-z+(1/K)*v",
        "traj_loss": "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu)",
        "concept": "lighting (warm sun vs neutral daylight); not age",
        "concept_words": DEFAULT_CONCEPT_WORDS,
        "product_handoff": PRODUCT_HANDOFF,
        "non_goals": [
            "Comfy plugin",
            "Hub weight vendoring",
            "supra-concept-sliders product repo",
            "VAE image decode in CI",
        ],
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }


def live_train_command(
    *,
    name: str = "lighting-supra",
    prompts_file: str = "conceptmod/textsliders/data/prompts-supra.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
    save_dir: str = "models/lighting-supra",
    lr: float = DEFAULT_LR,
    lm_target: str = DEFAULT_LM_TARGET,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    lora_targets: str = DEFAULT_LORA_TARGETS,
) -> str:
    recipe = resolve_supra_lm_target(lm_target)
    spec = resolve_supra_lora_targets(lora_targets)
    return (
        "HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_supra.py \\\n"
        f"  --name {name} \\\n"
        f"  --prompts_file {prompts_file} \\\n"
        f"  --model_id {model_id} \\\n"
        f"  --lora_targets {spec.label} --rank {rank} "
        f"--resolution {resolution} "
        f"--sample_steps {sample_steps} --cfg {cfg:g} \\\n"
        f"  --lr {lr} --lm_target {recipe} --traj_steps {int(traj_steps)} \\\n"
        f"  --sample_every {int(sample_every)} \\\n"
        f"  --device {device} --save_dir {save_dir}"
    )
