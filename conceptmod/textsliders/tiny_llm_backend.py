"""Tiny recent causal-LM concept-slider test target (opt-in, not a default).

Pinned model: ``Qwen/Qwen3-0.6B-Base`` — 596,049,920 params (Hub safetensors
total, BF16), Apache-2.0, ungated, Qwen3 release April 2025. It is the
smallest 2025+ public causal LM that is actually downloadable without a
gated-access token: ``google/gemma-3-270m(-it)`` is smaller (268M) but
gated (manual approval), and ``HuggingFaceTB/SmolLM2-135M`` is smaller
still (134M) but a November-2024 release, so both fail the pick criteria.
See ``docs/tiny-llm-slider.md`` for the full comparison.

Working game (transferred from YuE2, not invented here): the routed
particle bridge ``anneal-routed-particle-error`` — Rp paired-error GAN on
``e = T(student) - T(positive)`` plus particle VIC, implemented in
``conceptmod/textsliders/tiny_llm_particle.py`` on top of the shared
``particle_bridge_gan`` module. This backend only hosts the frozen model:
live ``transformers.AutoModelForCausalLM`` or a tiny randomly-initialized
dummy of the same config shape (no ``transformers`` import, no Hub, no
GPU — the CI / CPU path). The particle slider attaches its own
attention ``q/k/v/o`` branches itself, like ``ParticleSlider`` does.

This module never touches Music 3, YuE2, Music Arm B, ``locked_shared`` or
any live ``--lm_target`` default.
"""

from __future__ import annotations

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
# The particle slider (tiny_llm_particle.py) wraps exactly these Linears.
ATTN_CLASS_NAMES = ("TinyAttention", "Qwen3Attention")
LORA_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


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
    """SwiGLU-ish MLP. Named so the particle walk never wraps it."""

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
    ``model.embed_tokens``, ``model.norm``) so the particle walk, save
    format and trainer run unchanged against the real checkpoint.
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


@dataclass
class EncodedText:
    ids: list[int]


class TinyLLMBackend:
    """Frozen tiny-LM host. The particle slider attaches its own branches."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL,
        allow_hub: bool = False,
        dummy: bool = False,
    ) -> None:
        self.device = torch.device(device if not dummy else "cpu")
        self.model_id = model_id
        self.allow_hub = bool(allow_hub)
        self.dummy = bool(dummy)
        self.tokenizer: Any
        self.model: nn.Module
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

    def _init_live(self) -> None:
        self.tokenizer, model = _load_live_model(
            self.model_id, self.device, allow_hub=self.allow_hub
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        _assert_qwen3_shape(model, self.model_id)
        self.model = model

    def encode(self, text: str) -> EncodedText:
        ids = [int(x) for x in self.tokenizer.encode(text, add_special_tokens=False)]
        ids = ids or [0]
        limit = int(getattr(self.model.config, "max_position_embeddings", MODEL_CONTEXT))
        if len(ids) > limit:
            raise ValueError(f"prompt exceeds context ({len(ids)} > {limit})")
        return EncodedText(ids=ids)

    @torch.no_grad()
    def teacher_hidden(self, ids: list[int]) -> torch.Tensor:
        """Frozen base hidden states (no adapter attached at rest)."""
        return self.hidden(ids)

    def hidden(self, ids: list[int]) -> torch.Tensor:
        """Full-sequence last-hidden states; particle scale comes from the
        network's ``scaled()`` context, never from this backend."""
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        if self.dummy:
            return self.model.forward_hidden(tokens)
        return self._live_hidden(tokens)

    def _live_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model.model(
            input_ids=tokens, use_cache=False, return_dict=True
        )
        return out.last_hidden_state


def hidden_delta_metrics(
    student_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_plus: torch.Tensor,
    teacher_zero: torch.Tensor,
) -> dict[str, float]:
    """Report-only diagnostic: LoRA delta vs teacher concept delta.

    Last-token readout (the live Music3 LM convention): plus/neutral
    captions have different lengths, so the delta is ``h_last(+1) -
    h_last(0)`` vs ``h+_last - h0_last``, all ``[H]`` regardless of ``T``.
    Not part of the game (the game has no output MSE).
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
