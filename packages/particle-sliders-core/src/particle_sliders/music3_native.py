"""Native Music 3 prompt-state loader and routed adapter architecture.

Model dependencies load only when requested. This module owns no training
objective, optimizer, prompt dataset, AR token sampler or audio renderer.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re

import torch
from torch import nn

from .native_lora import LoRAModule, LoRANetwork, lora_target_plan
from .reference import RoutedMLP

DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
TARGET_REPLACE = ["Qwen3Attention"]
FORMAT = "conceptmod-music3-routed-particle-lm-v1"
_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"


def sound_only(text: str) -> str:
    """Preserve the native adapter export's descriptive metadata contract."""
    if re.search(r"\b(?:in the style of|inspired by|sounds like)\b|\S\s+[—–]\s+meaning\b", text, re.I):
        raise ValueError("Describe instruments, voice, room and timing; remove named references")
    return text


def _assemble(prompt: str, lyrics: str) -> str:
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _clean_caption,
        _normalize_lyrics,
    )

    return (
        f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
        f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )


def _tokenize(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return out["input_ids"].to(device), out["attention_mask"].to(device)


@torch.no_grad()
def _last_real_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real token per row (attention_mask sum − 1)."""
    return attention_mask.to(dtype=torch.long).sum(dim=1) - 1


def _gather_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden at the last real token. Respects padding; do not use shape-1."""
    lengths = _last_real_index(attention_mask)
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather.clamp(min=0)).squeeze(1)


def _audio_start_token_id(tokenizer):
    """Tokenizer id for ``<|audio_start|>``, or None if the tokenizer cannot say."""
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(_AUDIO_START)
    if tid is None:
        return None
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    if tid < 0:
        return None
    return tid


def _assert_last_token_is_audio_start(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, *, where: str
) -> None:
    """Fail closed if last-hidden MSE would pin the wrong continue-from token."""
    audio_id = _audio_start_token_id(tokenizer)
    if audio_id is None:
        return
    last_idx = _last_real_index(attention_mask).clamp(min=0)
    batch = torch.arange(input_ids.size(0), device=input_ids.device)
    last_ids = input_ids[batch, last_idx]
    expected = torch.full_like(last_ids, audio_id)
    if not torch.equal(last_ids, expected):
        raise RuntimeError(
            f"{where}: last real token is not {_AUDIO_START} "
            f"(got ids={last_ids.tolist()}, expected={audio_id}). "
            "Last-hidden MSE would pin the wrong continue-from token."
        )


def _encode_static(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Last real token (the audio-start token) is what AR continues from.
    return _gather_last_hidden(hidden, attention_mask).float()


def _encode_train(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return _gather_last_hidden(hidden, attention_mask).float()


class Music3Backend:
    """Frozen Music 3 LM + tokenizer for prompt-state last-hidden reads."""

    def __init__(self, model_dir=DEFAULT_MODEL, device="cuda:0", dummy=False):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if not dummy else "cpu")
        self.dummy = bool(dummy)
        if dummy:
            self.tokenizer = None
            self.model = _TinyLM(32).to(self.device).eval().requires_grad_(False)
            self.identity = {"dummy": True, "hidden_size": 32, "fixture": "byte-projection-v1"}
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir / "tokenizer"), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir / "language_model"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.identity = {
            "model_dir": str(self.model_dir.resolve()),
            "hidden_size": int(self.model.config.hidden_size),
            "num_layers": int(self.model.config.num_hidden_layers),
        }

    def encode_text(self, style: str, lyrics: str):
        text = _assemble(style, lyrics)
        if self.dummy:
            ids = torch.tensor([[b % 31 for b in text.encode("utf-8")] + [31]], device=self.device)
            return ids, torch.ones_like(ids)
        ids, mask = _tokenize(self.tokenizer, text, self.device)
        _assert_last_token_is_audio_start(ids, mask, self.tokenizer, where="particle bridge")
        return ids, mask

    def hidden(self, ids, mask, *, train=False):
        if self.dummy:
            return self.model(ids)
        if train:
            return _encode_train(self.model, ids, mask)
        with torch.no_grad():
            return _encode_static(self.model, ids, mask)


class Qwen3Attention(nn.Module):
    """CPU dummy projection with the production adapter target class name.

    Used only by _TinyLM; it is not a model of Qwen attention or audio quality.
    """
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class _TinyLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "num_hidden_layers": 1})()
        self.embed = nn.Embedding(32, hidden)
        self.attention = Qwen3Attention(hidden)

    def forward(self, ids):
        return self.attention(self.embed(ids.clamp(0, 31).long())).mean(1, keepdim=False)


class ParticleNetwork(nn.Module):
    """Music 3 LM LoRA with a shared routed particle cloud between down and up."""

    def __init__(self, lm, rank=8, alpha=8.0):
        super().__init__()
        if rank != 8 or float(alpha) != 8.0:
            raise ValueError("Particle format pins rank/alpha 8")
        self.rank = rank
        self.alpha = float(alpha)
        self.network = LoRANetwork(
            lm,
            rank=rank,
            alpha=alpha,
            multiplier=0.0,
            delimiter="-",
            target_replace=TARGET_REPLACE,
            prefix="lora_te",
            train_method="full",
        )
        if not self.network.unet_loras:
            raise RuntimeError("LoRA wrapped 0 Qwen3Attention linears")
        device = next(lm.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.bridges = nn.ModuleDict()
        for lora in self.network.unet_loras:
            bridge = RoutedMLP(rank, rank)
            key = lora.lora_name.replace(".", "-")
            self.bridges[key] = bridge
        # Move weights first, then install routed forwards that look up
        # self.particles at call time (not a Parameter captured before .to()).
        self.to(device=device, dtype=torch.float32)
        for lora in self.network.unet_loras:
            key = lora.lora_name.replace(".", "-")
            self._install(lora, self.bridges[key])
        self.set_scale(0.0)

    def _install(self, lora, bridge):
        network = self

        def forward(x, lora=lora, bridge=bridge, network=network):
            if lora.multiplier == 0:
                return lora.org_forward(x)
            weight = lora.lora_down.weight
            x_lora = x.to(device=weight.device, dtype=weight.dtype)
            features = lora.lora_down(x_lora).float()
            routed = bridge(features, network.particles)
            delta = lora.lora_up(routed.to(dtype=weight.dtype)).to(device=x.device, dtype=x.dtype)
            return lora.org_forward(x) + delta * (lora.multiplier * lora.scale)

        # LoRANetwork.apply_to() binds Linear.forward to LoRAModule.forward as a
        # MethodType. Assigning lora.forward alone does not change that binding,
        # so the LM never entered the particle bridge and GAN grads stayed 0.
        lora.forward = forward
        host = getattr(lora.org_forward, "__self__", None)
        if host is None:
            raise RuntimeError(f"Cannot rebind host forward for {lora.lora_name}")
        host.forward = forward

    def set_scale(self, scale: float):
        self.network.set_lora_slider(float(scale))
        for lora in self.network.unet_loras:
            lora.multiplier = float(scale)

    @contextmanager
    def scaled(self, scale: float):
        previous = float(self.network.lora_scale)
        self.set_scale(scale)
        try:
            yield
        finally:
            self.set_scale(previous)

    def save(self, path, metadata, *, state=None):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            metadata,
            format=FORMAT,
            rank=self.rank,
            alpha=self.alpha,
            particles=128,
            particle_dim=4,
            bridge_width=48,
            router_width=16,
            target_replace=TARGET_REPLACE,
            prefix="lora_te",
        )
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        sound_only(payload)
        state = self.state_dict() if state is None else state
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid particle-slider export")
        save_file(
            {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
            str(path),
            metadata={"conceptmod": payload},
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @staticmethod
    def _checkpoint_schema(targets):
        """Derive exact native adapter shapes without attaching to the host.

        Only metadata-sized modules are constructed on the meta device. The
        same LoRAModule and RoutedMLP builders define the actual state keys;
        no pretrained weights are copied or random streams consumed here.
        """
        schema = {"particles": (128, 4)}
        with torch.device("meta"):
            bridge = RoutedMLP(8, 8)
            for name, host in targets:
                kind = host.__class__.__name__
                if "Linear" in kind:
                    proxy = nn.Linear(host.in_features, host.out_features, bias=False)
                elif kind == "Conv1d":
                    proxy = nn.Conv1d(host.in_channels, host.out_channels,
                                      host.kernel_size, host.stride, host.padding, bias=False)
                else:
                    proxy = nn.Conv2d(host.in_channels, host.out_channels,
                                      host.kernel_size, host.stride, host.padding, bias=False)
                adapter = LoRAModule(name, proxy, multiplier=0., lora_dim=8, alpha=8.)
                adapter.apply_to()
                schema.update({f"network.{name}.{key}": tuple(value.shape)
                               for key, value in adapter.state_dict().items()})
                key = name.replace(".", "-")
                schema.update({f"bridges.{key}.{field}": tuple(value.shape)
                               for field, value in bridge.state_dict().items()})
        return schema

    @classmethod
    def load(cls, lm, path):
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        sound_only(json.dumps(record, ensure_ascii=False))
        required = dict(
            format=FORMAT, rank=8, alpha=8.0, particles=128, particle_dim=4,
            bridge_width=48, router_width=16, target_replace=TARGET_REPLACE, prefix="lora_te",
        )
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in required.items()):
            raise ValueError("Incompatible Music 3 routed-particle checkpoint")
        targets = lora_target_plan(lm, TARGET_REPLACE, prefix="lora_te", delimiter="-",
                                   train_method="full")
        if not targets:
            raise ValueError("Music 3 checkpoint host has no Qwen3Attention targets")
        state = load_file(str(path), device="cpu")
        schema = cls._checkpoint_schema(targets)
        if state.keys() != schema.keys():
            raise ValueError("Music 3 checkpoint tensor keys differ from native adapter")
        for name, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != schema[name] or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid Music 3 checkpoint tensor: " + name)
            if name.endswith(".alpha") and tensor.item() != 8.:
                raise ValueError("Music 3 checkpoint alpha differs from native adapter")

        # A qualified load constructs the native adapter exactly once. Preserve
        # caller RNG, and restore the exact original forward attribute bindings
        # if construction or the public state loader unexpectedly rejects it.
        forwards = [(host, "forward" in host.__dict__, host.__dict__.get("forward"))
                    for _, host in targets]
        device = next(lm.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            try:
                network = cls(lm)
                network.load_state_dict(state, strict=True)
            except Exception:
                for host, owned, forward in forwards:
                    if owned:
                        host.forward = forward
                    else:
                        host.__dict__.pop("forward", None)
                raise
        return network, record


# Public spellings for loader integrations; historical trainer spellings remain
# importable for compatibility without importing any trainer module.
assemble_text = _assemble
tokenize_text = _tokenize
assert_audio_start = _assert_last_token_is_audio_start
encode_last_hidden = _encode_train

__all__ = ["DEFAULT_MODEL", "TARGET_REPLACE", "FORMAT", "Music3Backend",
           "ParticleNetwork", "assemble_text", "tokenize_text", "assert_audio_start",
           "encode_last_hidden"]
