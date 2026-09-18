"""Tiny recent causal-LM concept-slider test target (opt-in, not a default).

Pinned model: ``Qwen/Qwen3-0.6B-Base`` — 596,049,920 params (Hub safetensors
total, BF16), Apache-2.0, ungated, Qwen3 release April 2025. It is the
smallest 2025+ public causal LM that is actually downloadable without a
gated-access token: ``google/gemma-3-270m(-it)`` is smaller (268M) but
gated (manual approval), and ``HuggingFaceTB/SmolLM2-135M`` is smaller
still (134M) but a November-2024 release, so both fail the pick criteria.
See ``docs/tiny-llm-slider.md`` for the full comparison.

Shape of the stack (mirrors ``minimax_h3_backend.py`` + ``yue2_backend.py``):

- Live path loads ``transformers.AutoModelForCausalLM`` and trains a rank-8
  LoRA on attention ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` only
  (the live ``Qwen3Attention`` names). MLP, embeddings, norms and ``lm_head``
  stay frozen and are never wrapped.
- Dummy path is a tiny randomly-initialized causal LM with the same
  contract (``model.layers[i].self_attn.{q,k,v,o}_proj``, ``embed_tokens``,
  ``norm``, ``lm_head`` unused by the slider). No ``transformers`` import,
  no Hub, no GPU. This is the CI / CPU path.
- UNI polarity (no minus teacher): student scale +1 fits the + caption
  hidden states, student scale 0 fits the neutral caption hidden states
  (``faithful_plus_neu`` analog on full-sequence last-hidden MSE).
- Like MiniMax-H3 UNI, LoRA-up defaults to ``N(0, 0.02)``: zero-init is the
  UNI identity (scale-1 vs scale-0 gap is 0, gradients vanish, loss sits at
  0). Pass ``lora_up_init_std=0`` to restore zeros for ablations.

This module never touches Music 3, YuE2, Music Arm B, ``locked_shared`` or
any live ``--lm_target`` default.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pinned Hub id. Live runs need transformers>=4.51 (Qwen3 support); the dummy
# path below needs nothing beyond torch.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
# Hub safetensors.total for the pinned revision (BF16). Rejects > ~1B params.
MODEL_PARAMS_TOTAL = 596049920
MODEL_RELEASE = "2025-04 (Qwen3)"
MODEL_LICENSE = "Apache-2.0"
MODEL_CONTEXT = 40960
# Live Qwen3 attention class + projection names; dummy mirrors the names.
ATTN_CLASS_NAMES = ("TinyAttention", "Qwen3Attention")
LORA_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
# Same UNI lesson as MiniMax-H3: noisy LoRA-up breaks the zero-adapter
# identity so a +1/0 UNI loss has a gradient at step 0.
DEFAULT_LORA_UP_INIT_STD = 0.02
FORMAT = "conceptmod-tiny-llm-uni-v1"
DEFAULT_RANK = 8
DEFAULT_ALPHA = 8.0


class ArchitectureMismatch(RuntimeError):
    """Raised when the host model is not a Qwen3-shaped causal LM."""


class DummyTokenizer:
    """Whitespace tokenizer with a stable word vocab. Not the Qwen tokenizer."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for word in str(text).split():
            key = word.lower()
            if key not in self._vocab:
                # Reserve 0 for padding; ids start at 1.
                self._vocab[key] = len(self._vocab) + 1
            ids.append(self._vocab[key])
        return ids or [1]


class TinyAttention(nn.Module):
    """Stand-in whose Linear names match live ``Qwen3Attention`` projections.

    Tiny GQA: ``num_heads`` query heads over ``num_kv_heads`` key/value
    heads. MLP lives beside this class, never inside it.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
        scale = self.head_dim ** -0.5
        attn = torch.softmax(
            (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) * scale, dim=-1
        )
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class TinyMLP(nn.Module):
    """SwiGLU-ish MLP. Named so the LoRA walk never wraps it."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = TinyMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class DummyTinyCausalLM(nn.Module):
    """Randomly-initialized stand-in with the Qwen3-0.6B-Base config shape.

    Live reference (fetched from the Hub config): hidden 1024, 28 layers,
    16 query / 8 kv heads, head_dim 128, intermediate 3072, vocab 151936
    tied, context 40960, RoPE theta 1e6. Dummy shrinks every axis but keeps
    the module nesting (``model.layers[i].self_attn.{q,k,v,o}_proj``,
    ``model.embed_tokens``, ``model.norm``) so the LoRA walk, save format
    and trainer run unchanged against the real checkpoint.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        intermediate_size: int = 64,
        vocab_size: int = 256,
        max_position_embeddings: int = 128,
    ) -> None:
        super().__init__()
        self.config = {
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": hidden_size // num_attention_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_position_embeddings,
            "tie_word_embeddings": True,
        }
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        self.model.layers = nn.ModuleList([
            TinyDecoderLayer(
                hidden_size, num_attention_heads, num_key_value_heads,
                intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ])
        self.model.norm = nn.LayerNorm(hidden_size)
        # Frozen output head: the slider trains on hidden states, never
        # logits (same reason YuE2 skips its 184k-vocab logits).
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be [batch, seq] with seq >= 1")
        seq = input_ids.shape[1]
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.model.embed_tokens(input_ids) + self.model.pos_embed(pos)
        for layer in self.model.layers:
            x = layer(x)
        return self.model.norm(x)


class _AttnLoRA(nn.Module):
    """LoRA on one attention Linear. Zero multiplier == exact base."""

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        rank: int,
        alpha: float,
        up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
    ) -> None:
        super().__init__()
        self.lora_name = name
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 0.0
        host_kwargs: dict[str, Any] = {}
        if hasattr(module, "weight"):
            host_kwargs["device"] = module.weight.device
            host_kwargs["dtype"] = module.weight.dtype
        self.lora_down = nn.Linear(module.in_features, rank, bias=False, **host_kwargs)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False, **host_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        if float(up_init_std) > 0:
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=float(up_init_std))
        else:
            nn.init.zeros_(self.lora_up.weight)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0:
            return self.org_forward(x)
        weight = self.lora_down.weight
        x_lora = x.to(device=weight.device, dtype=weight.dtype)
        delta = self.lora_up(self.lora_down(x_lora)).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class TinySlider(nn.Module):
    """Wrap attention ``q/k/v/o`` Linears only. Skip MLP, norms, embeds."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
    ) -> None:
        super().__init__()
        if int(rank) < 1 or not float(alpha) > 0:
            raise ValueError("rank and alpha must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.up_init_std = float(up_init_std)
        self.adapters = nn.ModuleDict()
        seen: set[int] = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ not in ATTN_CLASS_NAMES:
                continue
            for child_name, child in module.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                if child_name not in LORA_LINEAR_NAMES:
                    continue
                if id(child) in seen:
                    continue
                seen.add(id(child))
                key = f"lora_tiny-{name}-{child_name}".replace(".", "-")
                adapter = _AttnLoRA(
                    key, child, self.rank, self.alpha, up_init_std=self.up_init_std
                )
                self.adapters[key] = adapter
        if len(self.adapters) == 0:
            raise ArchitectureMismatch(
                "no attention q/k/v/o Linears found; expected Qwen3-shaped "
                f"{ATTN_CLASS_NAMES} hosts"
            )

    @contextmanager
    def scaled(self, scale: float):
        previous = [a.multiplier for a in self.adapters.values()]
        for adapter in self.adapters.values():
            adapter.multiplier = float(scale)
        try:
            yield
        finally:
            for adapter, value in zip(self.adapters.values(), previous):
                adapter.multiplier = value

    def save_weights(self, file: str, dtype=None) -> None:
        from safetensors.torch import save_file

        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        if dtype is not None:
            state = {k: v.to(dtype) for k, v in state.items()}
        save_file(state, file)

    def load_weights(self, file: str) -> None:
        from safetensors.torch import load_file

        state = load_file(file)
        tiny_keys = [k for k in state if "lora_tiny-" in str(k)]
        if not tiny_keys:
            raise ValueError(
                f"{file} has no lora_tiny-* keys; tiny-llm sliders save a "
                "custom TinySlider, not PEFT adapter_model.safetensors"
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_lora = [k for k in missing if "lora_tiny-" in str(k)]
        if missing_lora:
            raise ValueError(f"{file} is missing LoRA keys: {missing_lora[:8]}")
        _ = unexpected


@dataclass
class EncodedText:
    ids: list[int]


class TinyLLMBackend:
    """Opt-in tiny-LM slider host. Dummy is the CI / CPU path."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        lora_up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
        allow_hub: bool = False,
        dummy: bool = False,
    ) -> None:
        self.device = torch.device(device if not dummy else "cpu")
        self.model_id = model_id
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_up_init_std = float(lora_up_init_std)
        self.allow_hub = bool(allow_hub)
        self.dummy = bool(dummy)
        self.tokenizer: Any
        self.model: nn.Module
        self.slider: TinySlider
        if self.dummy:
            self._init_dummy()
        else:
            self._init_live()

    def _init_dummy(self) -> None:
        self.tokenizer = DummyTokenizer()
        self.model = DummyTinyCausalLM()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.slider = TinySlider(
            self.model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            up_init_std=self.lora_up_init_std,
        )
        self.slider.to(self.device)

    def _init_live(self) -> None:
        self.tokenizer, model = _load_live_model(
            self.model_id, self.device, allow_hub=self.allow_hub
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        _assert_qwen3_shape(model, self.model_id)
        self.model = model
        self.slider = TinySlider(
            self.model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            up_init_std=self.lora_up_init_std,
        )
        self.slider.to(self.device)
        host_dtype = _module_param_dtype(self.model)
        if host_dtype is not None:
            self.slider.to(dtype=host_dtype)

    def lora_module_names(self) -> list[str]:
        return list(self.slider.adapters.keys())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.slider.parameters() if p.requires_grad]

    def encode(self, text: str) -> EncodedText:
        if self.dummy:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [int(x) for x in ids] or [0]
        limit = int(getattr(self.model.config, "max_position_embeddings", MODEL_CONTEXT))
        if len(ids) > limit:
            raise ValueError(f"prompt exceeds context ({len(ids)} > {limit})")
        return EncodedText(ids=ids)

    @torch.no_grad()
    def teacher_hidden(self, ids: list[int]) -> torch.Tensor:
        """Frozen base hidden states (LoRA scale 0)."""
        return self.hidden(ids, scale=0.0)

    def hidden(self, ids: list[int], *, scale: float) -> torch.Tensor:
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.slider.scaled(float(scale)):
            if self.dummy:
                return self.model.forward_hidden(tokens)
            return self._live_hidden(tokens)

    def _live_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model.model(
            input_ids=tokens, use_cache=False, return_dict=True
        )
        return out.last_hidden_state

    def save_trained(self, path: str) -> None:
        self.slider.save_weights(path + ".safetensors", dtype=torch.float32)

    def load_trained(self, path: str) -> str:
        resolved = resolve_tiny_lora_path(path)
        state_dict = _read_safetensors(str(resolved))
        _validate_tiny_state(self.model, self.slider, state_dict, str(resolved))
        self.slider.load_weights(str(resolved))
        self.slider.to(self.device)
        return str(resolved)


def tiny_uni_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
) -> torch.Tensor:
    """UNI hidden MSE: +1 fits raw ``h+``, 0 fits ``h0``. No minus teacher."""
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def hidden_delta_metrics(
    student_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_plus: torch.Tensor,
    teacher_zero: torch.Tensor,
) -> dict[str, float]:
    """How far the LoRA delta moved along the teacher concept delta.

    Last-token readout (the live Music3 LM convention): plus/neutral
    captions have different lengths, so the delta is ``h_last(+1) -
    h_last(0)`` vs ``h+_last - h0_last``, all ``[H]`` regardless of ``T``.
    """
    with torch.no_grad():
        s_delta = (student_plus[:, -1] - student_zero[:, -1]).float().reshape(-1)
        t_delta = (teacher_plus[:, -1] - teacher_zero[:, -1]).float().reshape(-1)
        cos = F.cosine_similarity(s_delta, t_delta, dim=0).item()
        l2 = float(torch.linalg.vector_norm(s_delta - t_delta).item())
        t_norm = float(torch.linalg.vector_norm(t_delta).item())
    return {
        "delta_cos": cos,
        "delta_l2": l2,
        "teacher_delta_norm": t_norm,
        "rel_l2": l2 / max(t_norm, 1e-8),
    }


def _load_live_model(model_id: str, device: torch.device, *, allow_hub: bool):
    """Live path. Dummy never calls this. Do not download Qwen weights in CI."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=not allow_hub,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=not allow_hub,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def _assert_qwen3_shape(model: nn.Module, model_id: str) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ArchitectureMismatch(
            f"{model_id} is not a Qwen3 causal LM "
            f"(model_type={getattr(config, 'model_type', None)!r})"
        )
    total = sum(p.numel() for p in model.parameters())
    if total > 1_100_000_000:
        raise ArchitectureMismatch(
            f"{model_id} has {total} params; tiny-llm test target rejects > ~1B"
        )


def _module_param_dtype(module: nn.Module):
    for param in module.parameters():
        return param.dtype
    return None


def resolve_tiny_lora_path(path: str):
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        return p
    candidates = []
    if p.suffix == ".safetensors":
        candidates.append(p)
    candidates.append(Path(str(p) + ".safetensors"))
    if p.is_dir():
        candidates.extend(sorted(p.glob("*_lora.safetensors")))
        candidates.extend(sorted(p.glob("*.safetensors")))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no tiny-llm LoRA safetensors under {path} "
        "(expected {name}_lora.safetensors from save_trained)"
    )


def _read_safetensors(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def _validate_tiny_state(
    model: nn.Module, slider: TinySlider, state: dict[str, torch.Tensor], path: str
) -> None:
    expected: dict[str, tuple[int, ...]] = {
        key: tuple(tensor.shape)
        for key, tensor in slider.state_dict().items()
        if "lora_tiny-" in str(key)
    }
    if not expected:
        raise ValueError(f"{path}: host slider has no lora_tiny-* parameters")
    missing = [k for k in expected if k not in state]
    extra = [k for k in state if "lora_tiny-" in str(k) and k not in expected]
    bad_shape = [
        k for k, shape in expected.items()
        if k in state and tuple(state[k].shape) != shape
    ]
    if missing or extra or bad_shape:
        raise ValueError(
            f"{path} does not match this tiny-llm host "
            f"(missing={missing[:4]} extra={extra[:4]} bad_shape={bad_shape[:4]})"
        )
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError(f"{path} has non-finite LoRA weights")
