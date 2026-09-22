#!/usr/bin/env python3
"""Opt-in Supra2-IMG slider trainer (UNI + unused-token hold).

Live card:

    model  SupraLabs/Supra2-IMG
    arch   ~104.1M SupraDiT, frozen Flan-T5-Base, SD-VAE-FT-MSE
    res    256²   latent 32²   patch 2
    lora   --lora_targets cross (default): ctx_proj + cross_attn q/kv/proj
           dit = self_attn qkv/proj. Flan-T5 and the VAE stay frozen.
    sample 50 Euler steps   CFG 3.0   lr 1e-4
    lm     --lm_target trajectory (Hub Euler t=i/K, z<-z+dt*v).
           direct / cfg_delta kept. Not Anima embed_struct / same_crop.
           Not Music 3 v9.

CI / this repo: pass ``--dummy``. Dummy never downloads Supra weights.
Live load is local-checkpoint / ``HF_HUB_OFFLINE=1`` unless ``--allow_hub``.

This file does not vendor ``model_final_ema.pt`` and does not ship a
Comfy plugin or a supra-concept-sliders product repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.supra_fake import FakeSupraBackend
from conceptmod.textsliders.supra_model import (
    CKPT_FILENAME,
    HF_REPO,
    SupraDiT,
    attach_supra_lora,
    load_supra_weights,
)
from conceptmod.textsliders.supra_slider import (
    CROSS_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_HOLD_WEIGHT,
    DEFAULT_LM_TARGET,
    DEFAULT_LORA_TARGETS,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_MODE,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_TEACHER_GAP_BOOST,
    DEFAULT_TRAJ_IDENTITY_WEIGHT,
    DEFAULT_TRAJ_STEPS,
    DIT_LORA_TARGETS,
    SUPRA_LM_TARGETS,
    SUPRA_LORA_TARGET_CHOICES,
    SUPRA_SAMPLE_MODES,
    assert_sample_gate,
    image_mean_std,
    infer_sample_prompts,
    live_train_card,
    live_train_command,
    load_supra_prompts,
    looks_like_rgb_noise,
    minus_canary_cosine,
    resolve_supra_lm_target,
    resolve_supra_lora_targets,
    resolve_supra_recipe_label,
    row_token_plan,
    supra_boost_teacher,
    supra_cfg_delta,
    supra_direct_loss,
    supra_direct_teachers,
    supra_short_trajectory,
    supra_trajectory_loss,
    supra_uni_loss,
    supra_uni_teachers,
    supra_unused_hold_loss,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-supra.yaml"
DEFAULT_SAVE_DIR = Path("models/supra-slider")

# This trainer is Supra-only. Sibling cards stay on their own files.
_FOREIGN_BACKENDS = (
    "anima",
    "krea",
    "sana",
    "z-image",
    "zimage",
    "zit",
    "minimax",
)


def assert_supra_only(model_id: str) -> None:
    lowered = str(model_id).lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                f"this trainer is Supra-only; refused foreign backend {name!r}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="lighting-supra")
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--lora_targets",
        type=str,
        choices=SUPRA_LORA_TARGET_CHOICES,
        default=DEFAULT_LORA_TARGETS,
        help=(
            "cross (default): ctx_proj + cross_attn.q/kv/proj. "
            "dit: self_attn.qkv/proj. dit+cross: both. "
            "Flan-T5 and the SD VAE are not adapted."
        ),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--sample_steps", type=int, default=DEFAULT_SAMPLE_STEPS)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--hold_weight", type=float, default=DEFAULT_HOLD_WEIGHT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--control_prompt", type=str, default=DEFAULT_CONTROL_PROMPT)
    parser.add_argument(
        "--lm_target",
        type=str,
        choices=SUPRA_LM_TARGETS,
        default=DEFAULT_LM_TARGET,
        help=(
            "trajectory (default): K-step Hub Euler MSE(x_student, x_plus). "
            "direct / cfg_delta are 1-step. Anima embed_struct / same_crop "
            "and Music 3 v9 are not accepted here."
        ),
    )
    parser.add_argument("--traj_steps", type=int, default=DEFAULT_TRAJ_STEPS)
    parser.add_argument(
        "--traj_identity_weight",
        type=float,
        default=DEFAULT_TRAJ_IDENTITY_WEIGHT,
    )
    parser.add_argument(
        "--teacher_gap_boost",
        type=float,
        default=DEFAULT_TEACHER_GAP_BOOST,
        help="direct / cfg_delta only. Default 1 (off).",
    )
    parser.add_argument("--sample_every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--sample_first_n", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=SUPRA_SAMPLE_MODES,
        default=DEFAULT_SAMPLE_MODE,
        help="train_faithful: backend.encode_text + Euler (same path as the loss).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"local {CKPT_FILENAME}. CI must not rely on a Hub download.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU SupraDiT, never loads Supra2-IMG / never hits the Hub",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="permit a Hub download of the DiT checkpoint (off; CI must not set this)",
    )
    parser.add_argument(
        "--print_card",
        action="store_true",
        help="print the live train card and exit",
    )
    return parser.parse_args(argv)


def _device(arg: str, dummy: bool) -> torch.device:
    if dummy:
        if arg in ("cpu", "dummy"):
            return torch.device("cpu")
        if arg.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
    if arg.isdigit():
        return torch.device(f"cuda:{arg}" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _sample_zt(backend, seed: int, step: int):
    """Draw noise on CPU, then ``.to(device)``. Timestep is Hub ``t in [0, 1]``."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1009 + step)
    z = torch.randn((1, *backend.latent_shape), generator=g, device="cpu")
    z = z.to(device=backend.device)
    # Spread 1-step probes across the Euler interval. Not Anima's t in [0, 1000].
    t_value = ((step * 37) % 100) / 100.0
    t = torch.tensor([t_value], device=backend.device)
    return z, t


def _cycle_row(rows, plans, step: int):
    idx = int(step) % len(rows)
    return rows[idx], plans[idx]


def _sidecar_lora_fields(spec) -> dict[str, Any]:
    return {
        "lora_targets": spec.label,
        "dit_lora_targets": list(DIT_LORA_TARGETS) if spec.train_dit else [],
        "cross_lora_targets": list(CROSS_LORA_TARGETS) if spec.train_cross else [],
        "train_cross": spec.train_cross,
        "train_dit": spec.train_dit,
        "train_text_encoder": False,
        "adapted_modules": spec.adapted_module_names,
        "frozen_modules": list(spec.frozen_modules),
    }


def _assert_lora_train_state(backend, spec) -> None:
    names = backend.named_trainable()
    if not names:
        raise RuntimeError("Supra LoRA enabled but no trainable parameters")
    cross_hit = any("cross_attn" in n or n.startswith("ctx_proj") or ".ctx_proj." in n for n in names)
    dit_hit = any("self_attn" in n for n in names)
    if spec.train_cross and not cross_hit:
        raise RuntimeError("cross LoRA requested but ctx_proj/cross_attn are not trainable")
    if not spec.train_cross and cross_hit:
        raise RuntimeError("cross LoRA attached while --lora_targets excludes cross")
    if spec.train_dit and not dit_hit:
        raise RuntimeError("dit LoRA requested but self_attn is not trainable")
    if not spec.train_dit and dit_hit:
        raise RuntimeError("self_attn LoRA attached while --lora_targets excludes dit")


def _hold_and_canary(backend, row, plan, hold_weight: float, canary_student, v_neg, v_null):
    feat_s, tokens_s = backend.text_features(row.positive, frozen=False, scale=1.0)
    feat_n, tokens_n = backend.text_features(row.neutral, frozen=True)
    extra = None
    if tokens_s and tokens_n and plan["pairs"]:
        extra = hold_weight * supra_unused_hold_loss(feat_s, feat_n, pairs=plan["pairs"])
    canary = None
    if v_neg is not None:
        canary = float(minus_canary_cosine(canary_student, v_neg, v_null))
    return extra, canary


def _trajectory_step(
    backend,
    row,
    z,
    plan,
    hold_weight: float,
    traj_steps: int,
    identity_weight: float,
):
    infer = row.infer_prompt
    with torch.no_grad():
        x_plus = supra_short_trajectory(
            backend, row.positive, z, num_steps=traj_steps, frozen=True
        )
        x_neu = supra_short_trajectory(
            backend, row.neutral, z, num_steps=traj_steps, frozen=True
        )
        t_probe = torch.tensor([0.0], device=z.device)
        v_null = backend.predict_v("", z, t_probe, frozen=True)
        v_neg = (
            backend.predict_v(row.negative, z, t_probe, frozen=True)
            if row.has_minus_canary
            else None
        )
    x_student = supra_short_trajectory(
        backend, infer, z, num_steps=traj_steps, frozen=False, scale=1.0
    )
    x_zero = None
    if float(identity_weight) > 0.0:
        with torch.no_grad():
            x_zero = supra_short_trajectory(
                backend, infer, z, num_steps=traj_steps, frozen=False, scale=0.0
            )
    loss = supra_trajectory_loss(
        x_student, x_plus, x_zero, x_neu, identity_weight=identity_weight
    )
    extra, canary = _hold_and_canary(
        backend, row, plan, hold_weight, x_student.detach(), v_neg, v_null
    )
    if extra is not None:
        loss = loss + extra
    return loss, canary


def _train_step(
    backend,
    row,
    plan,
    z,
    t,
    hold_weight: float,
    lm_target: str,
    *,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    traj_identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
    teacher_gap_boost: float = DEFAULT_TEACHER_GAP_BOOST,
):
    """Student +1 stays on infer/neu. + caption is teacher only."""
    infer = row.infer_prompt
    recipe = resolve_supra_lm_target(lm_target)
    if recipe == "trajectory":
        return _trajectory_step(
            backend,
            row,
            z,
            plan,
            hold_weight,
            traj_steps=int(traj_steps),
            identity_weight=float(traj_identity_weight),
        )
    with torch.no_grad():
        v_pos = backend.predict_v(row.positive, z, t, frozen=True)
        v_neu = backend.predict_v(row.neutral, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
        v_neg = (
            backend.predict_v(row.negative, z, t, frozen=True)
            if row.has_minus_canary
            else None
        )
        v_pos = supra_boost_teacher(v_pos, v_neu, teacher_gap_boost)
    v_student_plus = backend.predict_v(infer, z, t, frozen=False, scale=1.0)
    v_student_zero = backend.predict_v(infer, z, t, frozen=False, scale=0.0)
    if recipe == "direct":
        teachers = supra_direct_teachers(v_pos, v_neu)
        loss = supra_direct_loss(
            v_student_plus, teachers["plus"], v_student_zero, teachers["zero"]
        )
        canary_student = supra_cfg_delta(v_student_plus.detach(), v_null)
    else:
        teachers = supra_uni_teachers(v_pos, v_neu, v_null, v_neg)
        s_plus = supra_cfg_delta(
            v_student_plus,
            backend.predict_v("", z, t, frozen=False, scale=1.0),
        )
        s_zero = supra_cfg_delta(
            v_student_zero,
            backend.predict_v("", z, t, frozen=False, scale=0.0),
        )
        loss = supra_uni_loss(s_plus, teachers["plus"], s_zero, teachers["zero"])
        canary_student = s_plus.detach()
    extra, canary = _hold_and_canary(
        backend, row, plan, hold_weight, canary_student, v_neg, v_null
    )
    if extra is not None:
        loss = loss + extra
    return loss, canary


def _should_sample(step: int, args: argparse.Namespace, *, last: bool) -> bool:
    if last:
        return True
    every = int(getattr(args, "sample_every", 0) or 0)
    first_n = int(getattr(args, "sample_first_n", 0) or 0)
    idx = int(step) + 1
    if first_n and idx <= first_n:
        return True
    if every > 0 and idx % every == 0:
        return True
    return False


def _image_from_pipe(image) -> np.ndarray:
    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def emit_inprocess_samples(backend, args, save_dir: Path, *, step: int, rows, dummy: bool):
    """Scale grid through the dummy pipe (structured RGB, not a VAE decode)."""
    del dummy
    prompts = infer_sample_prompts(rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT))
    scales = list(DEFAULT_SAMPLE_SCALES)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seed = int(getattr(args, "sample_seed", DEFAULT_SAMPLE_SEED))
    height = int(getattr(args, "resolution", DEFAULT_RESOLUTION))
    # Structured stand-in, not a 256² VAE decode. Card resolution stays on the sidecar.
    side = min(height, 64)
    for prompt in prompts:
        for scale in scales:
            if hasattr(backend, "set_adapter_scale"):
                backend.set_adapter_scale(scale)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            out = backend.pipe(
                prompt=prompt,
                height=side,
                width=side,
                num_inference_steps=min(int(args.sample_steps), 4),
                cfg=float(args.cfg),
                generator=generator,
            )
            arr = _image_from_pipe(out.images[0])
            mean, std = image_mean_std(arr)
            noise = bool(looks_like_rgb_noise(arr))
            slug = f"step{int(step):04d}_s{scale:.2f}_{len(records):02d}.png"
            from PIL import Image

            Image.fromarray(arr, mode="RGB").save(out_dir / slug)
            records.append(
                {
                    "prompt": prompt,
                    "scale": float(scale),
                    "mean": mean,
                    "std": std,
                    "looks_like_noise": noise,
                    "file": slug,
                    "method": "train_faithful_structured",
                    "sample_mode": str(getattr(args, "sample_mode", DEFAULT_SAMPLE_MODE)),
                    "cfg": float(args.cfg),
                }
            )
    if hasattr(backend, "set_adapter_scale"):
        backend.set_adapter_scale(1.0)
    meta = {
        "step": int(step),
        "seed": seed,
        "scales": scales,
        "prompts": prompts,
        "sample_mode": str(getattr(args, "sample_mode", DEFAULT_SAMPLE_MODE)),
        "samples": records,
    }
    (out_dir / f"step{int(step):04d}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    assert_sample_gate(records)
    return records


def _base_sidecar(args, spec, lm_target, meta, rows, device, history, canary, sample_records, trained_infer, backend):
    return {
        "name": args.name,
        "model_id": args.model_id,
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "sample_steps": int(args.sample_steps),
        "cfg": float(args.cfg),
        "lr": float(args.lr),
        "lm_target": lm_target,
        "sample_every": int(args.sample_every),
        "device": str(device),
        **_sidecar_lora_fields(spec),
        "recipe": resolve_supra_recipe_label(lm_target),
        "traj_steps": int(args.traj_steps),
        "traj_identity_weight": float(args.traj_identity_weight),
        "teacher_gap_boost": float(args.teacher_gap_boost),
        "traj_loop": "Hub Euler over predict_v (t=i/K, z<-z+(1/K)*v)",
        "traj_loss": "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu)",
        "plus_label": meta.plus_label,
        "minus_canary": any(r.has_minus_canary for r in rows),
        "train_infer_prompts": trained_infer,
        "sample_infer_prompts": infer_sample_prompts(
            rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT)
        ),
        "canary_cos_last": canary[-1] if canary else None,
        "loss_last": history[-1] if history else None,
        "lora_b_norm": float(backend.lora_B_norm()) if hasattr(backend, "lora_B_norm") else None,
        "steps": int(args.steps),
        "prompts_file": str(args.prompts_file),
        "control_prompt": str(args.control_prompt),
        "sample_grid": {
            "n": len(sample_records),
            "scales": list(DEFAULT_SAMPLE_SCALES),
            "seed": int(args.sample_seed),
            "method": "train_faithful_structured",
            "sample_mode": str(getattr(args, "sample_mode", DEFAULT_SAMPLE_MODE)),
        },
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }


def train_dummy(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=True)
    rows, meta = load_supra_prompts(args.prompts_file)
    if not rows:
        raise ValueError("no supra prompt rows")
    rank = int(args.rank)
    spec = resolve_supra_lora_targets(getattr(args, "lora_targets", DEFAULT_LORA_TARGETS))
    backend = FakeSupraBackend(
        device=str(device),
        rank=rank,
        seed=int(args.seed),
        lora_targets=spec.label,
    )
    _assert_lora_train_state(backend, spec)
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    history: list[float] = []
    canary: list[float] = []
    lm_target = resolve_supra_lm_target(getattr(args, "lm_target", DEFAULT_LM_TARGET))
    plans = [row_token_plan(row) for row in rows]
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    trained_infer: list[str] = []
    total = int(args.steps)
    pbar = tqdm(range(total), disable=total < 3)
    for step in pbar:
        row, plan = _cycle_row(rows, plans, step)
        if row.infer_prompt not in trained_infer:
            trained_infer.append(row.infer_prompt)
        z, t = _sample_zt(backend, int(args.seed), step)
        loss, canary_v = _train_step(
            backend,
            row,
            plan,
            z,
            t,
            float(args.hold_weight),
            lm_target,
            traj_steps=int(args.traj_steps),
            traj_identity_weight=float(args.traj_identity_weight),
            teacher_gap_boost=float(args.teacher_gap_boost),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if canary_v is not None:
            canary.append(canary_v)
        pbar.set_postfix(loss=f"{history[-1]:.4f}")
        if _should_sample(step, args, last=False):
            sample_records = emit_inprocess_samples(
                backend, args, save_dir, step=step + 1, rows=rows, dummy=True
            )
    sample_records = emit_inprocess_samples(
        backend, args, save_dir, step=total, rows=rows, dummy=True
    )
    sidecar = _base_sidecar(
        args, spec, lm_target, meta, rows, device, history, canary, sample_records, trained_infer, backend
    )
    sidecar["dummy"] = True
    if sidecar["loss_last"] is None or not math.isfinite(float(sidecar["loss_last"])):
        raise RuntimeError(f"dummy Supra step produced a non-finite loss: {sidecar['loss_last']}")
    side_path = save_dir / f"{args.name}_dummy_last.json"
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))
    return sidecar


def _resolve_local_checkpoint(args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "checkpoint", None)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise RuntimeError(
                f"Supra checkpoint not found at {path}. "
                "CI must use --dummy. Do not vendor model_final_ema.pt."
            )
        return path
    local = Path(CKPT_FILENAME)
    if local.is_file():
        return local
    return None


def _download_checkpoint(args: argparse.Namespace) -> Path:
    """Hub fetch. Only called when ``--allow_hub`` is set."""
    if not args.allow_hub:
        raise RuntimeError("refusing Hub download without --allow_hub; use --dummy")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "live Supra2-IMG download needs huggingface_hub. Use --dummy for CPU tests."
        ) from exc
    downloaded = hf_hub_download(repo_id=args.model_id or HF_REPO, filename=CKPT_FILENAME)
    return Path(downloaded)


def load_live_backend(args: argparse.Namespace, device: torch.device):
    """Build ``SupraDiT`` and attach LoRA. Never downloads unless ``--allow_hub``."""
    if not args.allow_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
    ckpt = _resolve_local_checkpoint(args)
    if ckpt is None:
        if not args.allow_hub:
            raise RuntimeError(
                f"live Supra2-IMG weights not available for {args.model_id!r}. "
                "CI must use --dummy. Pass --checkpoint model_final_ema.pt or --allow_hub. "
                "This repo does not vendor the checkpoint."
            )
        ckpt = _download_checkpoint(args)
    spec = resolve_supra_lora_targets(getattr(args, "lora_targets", DEFAULT_LORA_TARGETS))
    model = SupraDiT()
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    load_supra_weights(model, state, strict=True)
    alpha = float(args.alpha if args.alpha is not None else args.rank)
    targets: list[str] = []
    if spec.train_cross:
        targets.extend(CROSS_LORA_TARGETS)
    if spec.train_dit:
        targets.extend(DIT_LORA_TARGETS)
    loras = attach_supra_lora(model, rank=int(args.rank), alpha=alpha, targets=tuple(targets))
    model.to(device)
    backend = LiveSupraBackend(model, loras, device, spec)
    _assert_lora_train_state(backend, spec)
    return backend


class LiveSupraBackend:
    """Checkpoint DiT plus LoRA. Flan-T5 is injected, not downloaded.

    Set ``encode_fn(prompt) -> embeds`` or ``(embeds, mask)`` to a local
    frozen ``google/flan-t5-base`` before ``predict_v``. This PR does not
    fetch that encoder or the SD VAE. ``--dummy`` is the CPU contract.
    """

    def __init__(self, model: SupraDiT, loras, device: torch.device, lora_spec):
        self.transformer = model
        self.loras = list(loras)
        self.device = device
        self.lora_spec = lora_spec
        grid = int(round(model.num_tokens**0.5))
        side = grid * int(model.patch)
        self.latent_shape = (model.latent_ch, side, side)
        self.encode_fn = None

    def set_lora_scale(self, scale: float) -> None:
        for lora in self.loras:
            lora.multiplier = float(scale)

    def set_adapter_scale(self, scale: float) -> None:
        self.set_lora_scale(scale)

    def named_trainable(self) -> list[str]:
        names = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("down.weight" in name or "up.weight" in name):
                names.append(name)
        return names

    def trainable_parameters(self):
        params = []
        for lora in self.loras:
            params.extend([lora.down.weight, lora.up.weight])
        if not params:
            raise RuntimeError("Supra LoRA returned no trainable params")
        return params

    def lora_B_norm(self) -> float:
        total = 0.0
        for lora in self.loras:
            total += float(lora.up.weight.detach().float().pow(2).sum().item())
        return total**0.5

    def encode_text(self, prompt: str):
        if self.encode_fn is None:
            raise RuntimeError(
                "live Supra encode needs a local frozen Flan-T5 (google/flan-t5-base). "
                "This PR does not download it. Use --dummy for the CPU contract, "
                "or set backend.encode_fn to a local encoder."
            )
        return self.encode_fn(prompt)

    def _context(self, prompt: str):
        encoded = self.encode_text(prompt)
        mask = None
        if isinstance(encoded, tuple):
            embeds = encoded[0]
            if len(encoded) >= 2 and torch.is_tensor(encoded[1]) and encoded[1].dtype != torch.long:
                mask = encoded[1]
        else:
            embeds = encoded
        if not torch.is_tensor(embeds):
            raise RuntimeError("encode_fn must return a context tensor [B, T, D_CTX]")
        return embeds, mask

    def predict_v(self, prompt, z, timestep, frozen=False, scale=None):
        """Hub forward: ``t`` in ``[0, 1]``, context from ``encode_fn``."""
        scale_value = 0.0 if frozen or scale == 0.0 else (1.0 if scale is None else float(scale))
        prev = [lora.multiplier for lora in self.loras]
        self.set_lora_scale(scale_value)
        try:
            embeds, mask = self._context(prompt)
            hidden = z.to(self.device)
            if not torch.is_tensor(timestep):
                timestep = torch.tensor(timestep, device=self.device)
            t = timestep.reshape(-1).to(device=self.device, dtype=torch.float32)
            if t.numel() == 1:
                t = t.expand(hidden.shape[0])
            ctx = embeds.to(device=self.device, dtype=hidden.dtype)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            ctx_mask = None if mask is None else mask.to(self.device)
            if frozen or scale == 0.0:
                with torch.no_grad():
                    return self.transformer(hidden, t, ctx, ctx_mask).float()
            return self.transformer(hidden, t, ctx, ctx_mask).float()
        finally:
            for lora, value in zip(self.loras, prev):
                lora.multiplier = value


def train_live(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=False)
    if device.type != "cuda":
        raise RuntimeError(
            "live Supra2-IMG needs CUDA. Pass --dummy for the CPU fake, or "
            "--device cuda:0 with a local checkpoint."
        )
    backend = load_live_backend(args, device)
    raise RuntimeError(
        f"loaded {args.model_id} and attached {len(backend.loras)} LoRA linears, "
        "but frozen Flan-T5 encode is not auto-loaded in this PR. "
        "Set a local encoder or use --dummy. VAE decode is a product-repo step."
    )


def train(args: argparse.Namespace) -> dict:
    assert_supra_only(str(args.model_id))
    if args.print_card:
        card = live_train_card(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            cfg=float(args.cfg),
            device=str(args.device),
            lr=float(args.lr),
            control_prompt=str(args.control_prompt),
            lm_target=str(args.lm_target),
            sample_every=int(args.sample_every),
            traj_steps=int(args.traj_steps),
            traj_identity_weight=float(args.traj_identity_weight),
            teacher_gap_boost=float(args.teacher_gap_boost),
            lora_targets=str(args.lora_targets),
        )
        print(json.dumps(card, indent=2))
        print()
        print(
            live_train_command(
                name=args.name,
                prompts_file=args.prompts_file,
                model_id=args.model_id,
                rank=int(args.rank),
                resolution=int(args.resolution),
                sample_steps=int(args.sample_steps),
                cfg=float(args.cfg),
                device=str(args.device),
                lr=float(args.lr),
                lm_target=str(args.lm_target),
                sample_every=int(args.sample_every),
                traj_steps=int(args.traj_steps),
                lora_targets=str(args.lora_targets),
            )
        )
        return card
    if args.dummy:
        return train_dummy(args)
    return train_live(args)


def main(argv: list[str] | None = None) -> dict:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
