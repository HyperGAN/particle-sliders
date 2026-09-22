"""Live loader for ``jimmycarter/krea2-turbo-bbox``.

Lazy-imported from ``train_lora_krea2._load_live_backend`` only.
``--dummy`` must never import this module (no Hub, no 12B, no peft).

The Hub repo is transformer-only. VAE, Qwen3-VL, tokenizer, and
scheduler still come from ``krea/Krea-2-Raw``:

    tf = Krea2Transformer2DModel.from_pretrained(
        "jimmycarter/krea2-turbo-bbox",
        subfolder="epoch-14-step-73184/transformer",
    )
    pipe = Krea2Pipeline.from_pretrained(
        "krea/Krea-2-Raw", transformer=tf,
    )

``is_distilled`` is a pipeline flag and is not stored on the
transformer upload, so this loader sets it. Sampling uses the
distilled shift ``mu=1.15``, 8 steps, guidance 0. A local Comfy
``.safetensors`` (``krea2-bbox-turbo-comfy-latest.safetensors``)
replaces only the DiT, same as stock Turbo files, but the sample
card stays the bbox card rather than Raw.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from conceptmod.textsliders.krea2_bbox import resolve_krea2_bbox_source
from conceptmod.textsliders.krea_live import LiveKreaBackend, _from_pretrained
from conceptmod.textsliders.slider_targets import (
    KREA2_BBOX_CFG,
    KREA2_BBOX_MODEL,
    KREA2_BBOX_MU,
    KREA2_BBOX_SKELETON,
    KREA2_BBOX_STEPS,
    KREA2_BBOX_SUBFOLDER,
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
)


def _mark_distilled(pipe) -> None:
    """Pipeline-level distilled flag. Not a transformer config field."""
    pipe.register_to_config(is_distilled=True)
    print("krea2-turbo-bbox: is_distilled=True (mu=1.15, CFG 0, 8 steps)")


def load_krea2_bbox_pipeline(
    *,
    model_id: str,
    subfolder: str,
    skeleton: str,
    transformer: str | None,
    allow_hub: bool,
):
    """Build a Raw pipeline whose DiT is the bbox turbo transformer."""
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel

    plan = resolve_krea2_bbox_source(
        model_id,
        subfolder=subfolder,
        transformer=transformer,
        skeleton=skeleton,
    )
    local_files_only = not bool(allow_hub)
    source = str(plan["source"])
    print(
        "krea2-turbo-bbox load "
        f"source={source} skeleton={plan['skeleton']} "
        f"subfolder={plan['subfolder']!r} allow_hub={bool(allow_hub)}"
    )
    if source == "comfy_safetensors":
        from conceptmod.textsliders.krea_weights import load_comfy_krea_transformer

        tf = load_comfy_krea_transformer(
            str(plan["transformer_path"]),
            skeleton=str(plan["skeleton"]),
        )
    else:
        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "local_files_only": True if source == "local_transformer" else local_files_only,
        }
        sub = plan["subfolder"]
        if sub:
            kwargs["subfolder"] = str(sub)
        repo = (
            str(plan["transformer_path"])
            if source == "local_transformer"
            else str(plan["repo_id"])
        )
        tf = _from_pretrained(Krea2Transformer2DModel, repo, **kwargs)
    pipe = _from_pretrained(
        Krea2Pipeline,
        str(plan["skeleton"]),
        transformer=tf,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    _mark_distilled(pipe)
    return pipe


class LiveKrea2BboxBackend(LiveKreaBackend):
    """Stock Krea UNI surface on the bbox turbo transformer."""

    def __init__(
        self,
        device: torch.device,
        model_id: str = KREA2_BBOX_MODEL,
        resolution: int = KREA_DEFAULT_RESOLUTION,
        rank: int = KREA_DEFAULT_RANK,
        sample_steps: int | None = None,
        sample_guidance: float | None = None,
        allow_hub: bool = False,
        lora_targets: str = KREA_DEFAULT_LORA_TARGETS,
        subfolder: str = KREA2_BBOX_SUBFOLDER,
        skeleton: str = KREA2_BBOX_SKELETON,
        transformer: str | None = None,
        mu: float = KREA2_BBOX_MU,
    ):
        self.bbox_subfolder = str(subfolder or KREA2_BBOX_SUBFOLDER)
        self.bbox_skeleton = str(skeleton or KREA2_BBOX_SKELETON)
        self.bbox_transformer = (
            None if transformer in (None, "") else str(transformer)
        )
        self.mu = float(mu)
        steps = int(KREA2_BBOX_STEPS if sample_steps is None else sample_steps)
        guidance = float(
            KREA2_BBOX_CFG if sample_guidance is None else sample_guidance
        )
        super().__init__(
            device=device,
            model_id=str(model_id),
            resolution=int(resolution),
            rank=int(rank),
            sample_steps=steps,
            sample_guidance=guidance,
            allow_hub=bool(allow_hub),
            lora_targets=str(lora_targets),
        )
        # Parent may read pipe.config before register, and looks_turbo()
        # is intentionally false for this id. Pin the distilled card.
        self.is_distilled = True
        self.mu = float(mu)
        self.generate_steps = steps
        self.generate_guidance = guidance

    def build_pipeline(self):
        return load_krea2_bbox_pipeline(
            model_id=self.model_id,
            subfolder=self.bbox_subfolder,
            skeleton=self.bbox_skeleton,
            transformer=self.bbox_transformer,
            allow_hub=self.allow_hub,
        )


def load_live_krea2_backend(args: Any, device: torch.device) -> LiveKrea2BboxBackend:
    """Hub / local loader. Never called from ``--dummy``."""
    allow_hub = bool(getattr(args, "allow_hub", False))
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_hub else "1"
    try:
        from diffusers import Krea2Pipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "live Krea2-turbo-bbox needs diffusers + peft. Use --dummy for CPU tests."
        ) from exc
    model_id = str(getattr(args, "model_id", KREA2_BBOX_MODEL))
    try:
        backend = LiveKrea2BboxBackend(
            device=device,
            model_id=model_id,
            resolution=int(args.resolution),
            rank=int(args.rank),
            sample_steps=getattr(args, "sample_steps", None),
            sample_guidance=getattr(args, "sample_guidance", None),
            allow_hub=allow_hub,
            lora_targets=str(getattr(args, "lora_targets", KREA_DEFAULT_LORA_TARGETS)),
            subfolder=str(getattr(args, "transformer_subfolder", KREA2_BBOX_SUBFOLDER)),
            skeleton=str(getattr(args, "skeleton_model", KREA2_BBOX_SKELETON)),
            transformer=getattr(args, "transformer", None),
            mu=float(getattr(args, "mu", KREA2_BBOX_MU)),
        )
    except Exception as exc:
        raise RuntimeError(
            f"live Krea2-turbo-bbox weights not available for {model_id!r} "
            f"(local_files_only={not allow_hub}). "
            "Pass --allow_hub to download the transformer subfolder and the "
            "gated krea/Krea-2-Raw skeleton (~48GB GPU). CI uses --dummy."
        ) from exc
    return backend
