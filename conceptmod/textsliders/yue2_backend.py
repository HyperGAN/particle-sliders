"""YuE2 composition sliders on the native AR attention path.

The acoustic NAR projections, MLPs, embeddings and VAE are frozen. Importing
this module needs no YuE2 installation; live loading uses the official runtime.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from conceptmod.textsliders.lora import LoRAModule

DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"
UPSTREAM_REVISION = "ef1936f2ee39fe8de486a0f47a481c95f8d4da87"
FORMAT = "conceptmod-yue2-ar-v1"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def require_yue2():
    try:
        from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install the isolated YuE2 runtime; see docs/yue2-slider.md") from exc
    return YuE2Config, YuE2ForCausalLM


def sound_only(text: str) -> str:
    """Reject named-reference syntax; prompts must describe sound directly."""
    if re.search(r"\b(?:in the style of|inspired by|sounds like)\b|\S\s+[—–]\s+meaning\b", text, re.I):
        raise ValueError("Describe instruments, voice, room and timing; remove named references")
    return text


def architecture(model) -> dict:
    config = model.config
    if getattr(config, "model_type", None) != "yue2":
        raise ValueError("Expected the official YuE2 AR–NAR model")
    return {key: getattr(config, key) for key in (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
    )}


def attention_targets(model) -> dict[str, nn.Linear]:
    architecture(model)
    targets = {}
    for i, layer in enumerate(model.model.layers):
        for name in PROJECTIONS:
            module = getattr(layer.self_attn, name, None)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Unsupported YuE2 projection: layer {i} {name}")
            targets[f"model.layers.{i}.self_attn.{name}"] = module
    if len(targets) != model.config.num_hidden_layers * len(PROJECTIONS):
        raise ValueError("YuE2 attention layout does not match its configuration")
    return targets


class _Adapter(LoRAModule):
    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        return super().forward(x)


class YuE2Slider(nn.Module):
    def __init__(self, model, rank=8, alpha=8.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        targets = attention_targets(model)
        if any(isinstance(getattr(m.forward, "__self__", None), _Adapter) for m in targets.values()):
            raise ValueError("A YuE2 slider is already attached")
        self.rank, self.alpha = rank, float(alpha)
        self.architecture = architecture(model)
        self.target_names = list(targets)
        model.requires_grad_(False)
        self.adapters = nn.ModuleDict()
        for path, module in targets.items():
            key = path.replace(".", "-")
            adapter = _Adapter(key, module, multiplier=0.0, lora_dim=rank, alpha=alpha)
            adapter.apply_to()
            self.adapters[key] = adapter
        self.to(device=next(model.parameters()).device, dtype=torch.float32)

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale):
            raise ValueError("Slider scale must be finite")
        previous = [a.multiplier for a in self.adapters.values()]
        for adapter in self.adapters.values():
            adapter.multiplier = float(scale)
        try:
            yield
        finally:
            for adapter, value in zip(self.adapters.values(), previous):
                adapter.multiplier = value

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
                      architecture=self.architecture, targets=self.target_names,
                      upstream_revision=UPSTREAM_REVISION)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        sound_only(payload)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(path), metadata={"conceptmod": payload})
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, model, path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        if record.get("format") == 'conceptmod-yue2-routed-particle-ar-v1':
            from conceptmod.textsliders.yue2_particle_bridge import ParticleSlider
            return ParticleSlider.load(model, path)
        if record.get("format") != FORMAT:
            raise ValueError("Not a native YuE2 composition-slider checkpoint")
        sound_only(json.dumps(record, ensure_ascii=False))
        if record.get("dummy"):
            raise ValueError("Dummy checkpoints cannot be used for song generation")
        if record.get("architecture") != architecture(model):
            raise ValueError("Slider architecture does not match this YuE2 model")
        if record.get("targets") != list(attention_targets(model)):
            raise ValueError("Slider target list does not match YuE2 AR attention")
        state = load_file(str(path), device="cpu")
        # Check every key/shape before attaching hooks to a live model.
        rank, alpha = record["rank"], record["alpha"]
        expected = {}
        for name, module in attention_targets(model).items():
            prefix = "adapters." + name.replace(".", "-")
            expected[prefix + ".lora_down.weight"] = (rank, module.in_features)
            expected[prefix + ".lora_up.weight"] = (module.out_features, rank)
            expected[prefix + ".alpha"] = ()
        if set(state) != set(expected) or any(tuple(state[k].shape) != shape for k, shape in expected.items()):
            raise ValueError("Incomplete or incompatible YuE2 slider tensors")
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite YuE2 slider weights")
        if any(float(state[k]) != alpha for k in state if k.endswith(".alpha")):
            raise ValueError("Slider alpha metadata disagrees with its tensors")
        network = cls(model, rank=rank, alpha=alpha)
        network.load_state_dict(state, strict=True)
        return network, record


class _ByteTokenizer:
    """Only used with the tiny random architecture in --dummy smoke runs."""
    def encode(self, text):
        return list(text.encode("utf-8"))


class YuE2Backend:
    def __init__(self, model_id=DEFAULT_MODEL, *, device="cuda:0", allow_hub=False,
                 revision=None, cache_dir=None, dummy=False):
        config_cls, model_cls = require_yue2()
        self.dummy = dummy
        if dummy:
            self.model = model_cls(config_cls(hidden_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                intermediate_size=64, max_latent_frames=64)).eval()
            self.tokenizer = _ByteTokenizer()
            self.identity = {"dummy": True}
        else:
            from yue2.storage import resolve_model, model_identity
            from yue2.tokenization_yue2 import YuE2TextTokenizer
            path = resolve_model(model_id, revision=revision, cache_dir=cache_dir,
                                 local_files_only=not allow_hub)
            self.identity = model_identity(path)
            self.tokenizer = YuE2TextTokenizer(path / "qwen.tiktoken")
            self.model = model_cls.from_pretrained(path, local_files_only=True,
                torch_dtype=torch.float32 if str(device) == "cpu" else torch.bfloat16).eval()
        self.model.to("cpu" if dummy else device).requires_grad_(False)

    def prefix(self, style, lyrics, cot="off") -> list[int]:
        from yue2.protocol import SongRequest, token_prefixes
        sound_only(style)
        sound_only(lyrics)
        return token_prefixes(SongRequest(style=style, lyrics=lyrics, cot=cot), self.tokenizer)

    def continuation(self, prefix, count, seed) -> list[int]:
        from yue2.protocol import CODEC_OFFSET, Sampling
        if self.dummy:
            rng = torch.Generator().manual_seed(seed)
            return (torch.randint(0, 64, (count,), generator=rng) + CODEC_OFFSET).tolist()
        from yue2.sampling import generate_tokens
        tokens, _, _ = generate_tokens(self.model, prefix,
            Sampling(min_tokens=count, max_tokens=count), seed, "semantic",
            use_cuda_graph=False, legacy_off=True)
        return tokens

    def hidden(self, ids, *, checkpointing=False):
        """Unpadded causal AR forward without the 184k-vocabulary logits."""
        model = self.model.model
        if not ids or len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("YuE2 training sequence is empty or exceeds model context")
        device = next(model.parameters()).device
        tokens = torch.tensor([ids], device=device, dtype=torch.long)
        positions = torch.arange(len(ids), device=device)[None]
        if not checkpointing:
            return model(input_ids=tokens, position_ids=positions, use_cache=False)[0]
        x = model.embed_tokens(tokens)
        cos, sin = model.rotary_emb(positions)
        for layer in model.layers:
            x = checkpoint(layer, x, cos, sin, use_reentrant=False)
        return model.norm(x)


def aligned_suffix(neutral, positive):
    """Align exact token positions from the end, never crop unequal prompts."""
    count = 0
    for a, b in zip(reversed(neutral), reversed(positive)):
        if a != b:
            break
        count += 1
    if count < 4:
        raise ValueError("Prompt pair needs shared lyrics before its music boundary")
    return count


def normalized_mse(student, teacher):
    return (student.float() - teacher.float()).square().mean() / teacher.float().square().mean().clamp_min(1e-8)


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
