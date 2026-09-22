"""CPU fake Supra2-IMG: tiny ``SupraDiT`` with the live LoRA contract.

No Hub, no GPU, no ``model_final_ema.pt``. The module is the real DiT
class at a toy width so LoRA path suffixes (``cross_attn.q``,
``self_attn.qkv``, ``ctx_proj``) match the checkpoint. A fixed token
table stands in for frozen Flan-T5.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from conceptmod.textsliders.supra_model import (
    LATENT_CH,
    PATCH,
    LoRALinear,
    SupraDiT,
    attach_supra_lora,
)
from conceptmod.textsliders.supra_slider import (
    CROSS_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_RANK,
    DIT_LORA_TARGETS,
    resolve_supra_lora_targets,
    supra_cfg_delta,
    word_tokens,
)

TEXT_DIM = 8
LATENT_HW = 8
MAX_TOKENS = 24
_CONCEPTISH = {
    "warm",
    "golden",
    "sunlit",
    "glow",
    "smiling",
    "smile",
    "happy",
    "joyful",
    "teeth",
}


def tiny_supra_dit(seed: int = 0) -> SupraDiT:
    """One-block DiT. Spatial 8² / patch 2 → 16 tokens. Not the 104.1M card."""
    torch.manual_seed(seed)
    num_tokens = (LATENT_HW // PATCH) ** 2
    return SupraDiT(
        latent_ch=LATENT_CH,
        d_model=32,
        depth=1,
        n_heads=4,
        ctx_dim=TEXT_DIM,
        mlp_ratio=2.0,
        num_tokens=num_tokens,
        patch=PATCH,
    )


class FakeSupraBackend:
    """In-repo Supra stand-in. ``--dummy`` on the trainer."""

    def __init__(
        self,
        device: str = "cpu",
        rank: int = DEFAULT_RANK,
        seed: int = 0,
        resolution: int = 64,
        lora_targets: str = "cross",
    ):
        del resolution
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.rank = int(rank)
        self.lora_spec = resolve_supra_lora_targets(lora_targets)
        self.latent_shape = (LATENT_CH, LATENT_HW, LATENT_HW)
        torch.manual_seed(seed)
        self.transformer = tiny_supra_dit(seed=seed).to(self.device)
        table = torch.zeros(64, TEXT_DIM)
        table[1, 0] = 1.0
        table[2, 1] = 1.0
        self.register_table = table.to(self.device)
        self._token_ids: dict[str, int] = {}
        targets: list[str] = []
        if self.lora_spec.train_cross:
            targets.extend(CROSS_LORA_TARGETS)
        if self.lora_spec.train_dit:
            targets.extend(DIT_LORA_TARGETS)
        self.loras = attach_supra_lora(
            self.transformer,
            rank=self.rank,
            alpha=float(self.rank),
            targets=tuple(targets),
        )
        self.transformer.to(self.device)
        self.set_lora_scale(1.0)
        self.pipe = FakeSupraPipe(self)

    def token_id(self, word: str) -> int:
        key = word.lower()
        if key in self._token_ids:
            return self._token_ids[key]
        if key in ("",):
            idx = 0
        elif key in _CONCEPTISH:
            idx = 2
        else:
            idx = 1
        self._token_ids[key] = idx
        return idx

    def encode_tokens(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        tokens = word_tokens(prompt)[:MAX_TOKENS] or [""]
        ids = torch.tensor([self.token_id(tok) for tok in tokens], dtype=torch.long)
        embeds = self.register_table[ids].unsqueeze(0)
        return embeds, tokens

    def set_lora_scale(self, scale: float) -> None:
        for lora in self.loras:
            lora.multiplier = float(scale)

    def set_adapter_scale(self, scale: float) -> None:
        self.set_lora_scale(scale)

    def disable_adapter(self):
        return _scale_ctx(self, 0.0)

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        embeds, tokens = self.encode_tokens(prompt)
        return embeds.to(self.device), tokens

    def _timestep(self, timestep: torch.Tensor, batch: int) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device)
        t = timestep.reshape(-1).to(device=self.device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.expand(batch)
        return t

    def _forward(self, z: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor):
        hidden = z.to(self.device)
        t = self._timestep(timestep, hidden.shape[0])
        ctx = embeds.to(device=self.device, dtype=hidden.dtype)
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0)
        return self.transformer(hidden, t, ctx).float()

    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        timestep: torch.Tensor,
        frozen: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        if frozen or scale == 0.0:
            with torch.no_grad(), _scale_ctx(self, 0.0):
                embeds, _tokens = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
        if scale is not None:
            with _scale_ctx(self, float(scale)):
                embeds, _tokens = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
        embeds, _tokens = self.encode_text(prompt)
        return self._forward(z, timestep, embeds)

    def text_features(
        self, prompt: str, frozen: bool = False, scale: float | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        """Context after ``ctx_proj`` (the LoRA text path, stand-in for T5)."""

        def _run(embeds: torch.Tensor) -> torch.Tensor:
            ctx = embeds.to(self.device)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            return self.transformer.ctx_proj(ctx)

        if frozen or scale == 0.0:
            with torch.no_grad(), _scale_ctx(self, 0.0):
                embeds, tokens = self.encode_text(prompt)
                return _run(embeds), tokens
        if scale is not None:
            with _scale_ctx(self, float(scale)):
                embeds, tokens = self.encode_text(prompt)
                return _run(embeds), tokens
        embeds, tokens = self.encode_text(prompt)
        return _run(embeds), tokens

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = []
        for lora in self.loras:
            params.extend([lora.down.weight, lora.up.weight])
        return params

    def lora_B_norm(self) -> float:
        total = 0.0
        for lora in self.loras:
            total += lora.up.weight.detach().float().pow(2).sum().item()
        return total**0.5

    def named_trainable(self) -> list[str]:
        names = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("down.weight" in name or "up.weight" in name):
                names.append(name)
        return names


class _scale_ctx:
    def __init__(self, backend: FakeSupraBackend, scale: float):
        self.backend = backend
        self.scale = float(scale)
        self._prev: list[float] = []

    def __enter__(self):
        self._prev = [lora.multiplier for lora in self.backend.loras]
        self.backend.set_lora_scale(self.scale)
        return self

    def __exit__(self, *_exc):
        for lora, value in zip(self.backend.loras, self._prev):
            lora.multiplier = value
        return False


class FakeSupraPipe:
    """Dummy infer surface. Images are structured ramps, not VAE decodes."""

    def __init__(self, backend: FakeSupraBackend):
        self.backend = backend
        self.transformer = backend.transformer
        self.last_prompt = ""
        self.prompts_seen: list[str] = []
        self.last_cfg = float(DEFAULT_CFG)

    def __call__(
        self,
        prompt: str = "",
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 4,
        cfg: float = DEFAULT_CFG,
        generator=None,
        output_type: str = "pil",
        **_kwargs,
    ):
        del num_inference_steps
        self.last_prompt = str(prompt)
        self.prompts_seen.append(str(prompt))
        self.last_cfg = float(cfg)
        scale = 1.0
        if self.backend.loras:
            scale = float(self.backend.loras[0].multiplier)
        seed = 0
        if generator is not None and hasattr(generator, "initial_seed"):
            try:
                seed = int(generator.initial_seed())
            except Exception:
                seed = 0
        arr = _structured_dummy_image(
            prompt, scale=scale, height=max(8, int(height)), width=max(8, int(width)), seed=seed
        )
        if output_type == "np":
            images = [arr.astype(np.float32) / 255.0]
        elif output_type == "pt":
            images = [torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0]
        else:
            from PIL import Image

            images = [Image.fromarray(arr, mode="RGB")]
        return SimpleNamespace(images=images)


def _structured_dummy_image(
    prompt: str, *, scale: float, height: int, width: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(abs(int(seed)) % (2**31) + (hash(prompt) % 997))
    ys = np.linspace(48.0, 176.0, height, dtype=np.float64)
    xs = np.linspace(36.0, 168.0, width, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    tint = 12.0 * float(scale)
    blob = 18.0 * np.sin(yy / 18.0) * np.cos(xx / 22.0)
    noise = 4.0 * rng.standard_normal((height, width))
    img = np.stack(
        [yy + tint + blob, xx + 0.4 * tint, 0.45 * yy + 0.45 * xx + 16.0 + 0.6 * tint + noise],
        axis=-1,
    )
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def write_plus_alignment(
    backend: FakeSupraBackend,
    neu: str,
    pos: str,
    seed: int = 0,
) -> float:
    """Cosine of student(+1, neu) CFG vs frozen(pos) CFG."""
    g = torch.Generator(device="cpu").manual_seed(seed + 3)
    z = torch.randn((1, *backend.latent_shape), generator=g, device=backend.device)
    t = torch.tensor([0.5], device=backend.device)
    with torch.no_grad():
        v_pos = backend.predict_v(pos, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
        v_s = backend.predict_v(neu, z, t, frozen=False, scale=1.0)
        v_s_null = backend.predict_v("", z, t, frozen=False, scale=1.0)
    d_t = supra_cfg_delta(v_pos, v_null).flatten().unsqueeze(0)
    d_s = supra_cfg_delta(v_s, v_s_null).flatten().unsqueeze(0)
    return torch.nn.functional.cosine_similarity(d_s, d_t, dim=1, eps=1e-6).item()
