"""Opt-in routed particle branches on Music 3 LM attention; paired-error GAN + VIC.

Ports Yue2 ``anneal-routed-particle-error-yue2-v1`` onto MiniMax Music 3's
Qwen3 LM (prompt-state last-hidden only). Shared game math lives in
``particle_bridge_gan``. Propose-only; does not change live ``--lm_target v9``.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
import yaml

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.slider_targets import lm_faithful_plus_neu
from conceptmod.textsliders.train_lm_slider_music3 import (
    DEFAULT_MODEL,
    TARGET_REPLACE,
    _assemble,
    _assert_last_token_is_audio_start,
    _encode_static,
    _encode_train,
    _tokenize,
)
from conceptmod.textsliders.yue2_backend import sound_only

FORMAT = "conceptmod-music3-routed-particle-lm-v1"
RECIPE = dict(
    name="anneal-routed-particle-error-music3-v1",
    generator_objective="paired_error_rpgan_plus_particle_vic",
    parts=128,
    particle_dim=4,
    vicreg_weight=1.0,
    g_lr=0.0006,
    d_lr=0.0009,
    particle_lr=0.006,
    g_optimizer="Adam",
    g_weight_decay=0.0,
    grad_clip_value=None,
    critic_hidden=48,
    critic_layers=1,
    critic_heads=4,
    critic_patch=64,
    critic_tokens=8,
    critic_queries=4,
    critic="bottleneck",
    critic_score_bound=8.0,
    critic_rank=64,
    penalty_lazy_k=4,
    cap_coordinates="normalized_paired_error_plus_shared_gaussian",
    target_normalization="training_target_per_coordinate_sample_std_floor_1e-4",
    noise_start=1.0,
    noise_floor=0.03,
    noise_decay_steps=8000,
    ema=0.995,
    adv_batch=64,
    schedule="constant",
    prompt_policy="independent_D_G_with_replacement",
    routing="all_examples",
    particle_vic_batch=64,
    adapter_format=FORMAT,
    adapter_rank=8,
    adapter_width=48,
    router_width=16,
    trained_scales=[1.0],
    zero_behavior="exact_base_by_adapter_scale",
    polarity="unipolar",
    lm_target="faithful_plus_neu",
    propose_only=True,
    merge_to_trainer=False,
    yue2_reference="anneal-routed-particle-error-yue2-v1",
)


def load_prompts(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list) or not raw["rows"]:
        raise ValueError("Prompts need a nonempty rows list")
    meta = {
        "plus_label": str(raw.get("plus_label") or ""),
        "minus_label": str(raw.get("minus_label") or "Off"),
        "recommended_range": raw.get("recommended_range") or [0.0, 1.0],
    }
    if meta["recommended_range"] != [0.0, 1.0] and meta["recommended_range"] != [0, 1]:
        raise ValueError("Unipolar particle bridge uses range [0, 1]")
    meta["recommended_range"] = [0.0, 1.0]
    for value in meta.values():
        if isinstance(value, str) and value:
            sound_only(value)
    rows = []
    for item in raw["rows"]:
        if not isinstance(item, dict):
            raise ValueError("Each row must be a mapping")
        needed = {"neutral", "positive", "lyrics"}
        if not needed <= set(item):
            raise ValueError("Each unipolar row needs neutral, positive, lyrics")
        row = {key: sound_only(str(item[key]).strip()) for key in needed}
        if not all(row.values()):
            raise ValueError("Prompt values must be nonempty strings")
        if row["neutral"] == row["positive"]:
            raise ValueError("Particle bridge requires distinct neutral and positive captions")
        rows.append(row)
    return rows, meta


class Music3Backend:
    """Frozen Music 3 LM + tokenizer for prompt-state last-hidden reads."""

    def __init__(self, model_dir=DEFAULT_MODEL, device="cuda:0", dummy=False):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if not dummy else "cpu")
        self.dummy = bool(dummy)
        if dummy:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                str(self.model_dir / "language_model"), local_files_only=True
            )
            self.tokenizer = None
            self.model = _TinyLM(config.hidden_size).to(self.device)
            self.identity = {"dummy": True, "hidden_size": config.hidden_size}
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


class _TinyLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "num_hidden_layers": 1})()
        self.embed = nn.Embedding(32, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, ids):
        return self.out(self.embed(ids.clamp(0, 31).long())).mean(1, keepdim=False)


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
            bridge = shared.RoutedMLP(rank, rank)
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

    @classmethod
    def load(cls, lm, path):
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        sound_only(json.dumps(record, ensure_ascii=False))
        required = dict(
            format=FORMAT, rank=8, alpha=8.0, particles=128, particle_dim=4, bridge_width=48, router_width=16
        )
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError("Incompatible Music 3 routed-particle checkpoint")
        network = cls(lm)
        state = load_file(str(path), device="cpu")
        network.load_state_dict(state, strict=True)
        return network, record


@torch.no_grad()
def prepare(backend: Music3Backend, rows, meta, max_seq_len=4096):
    prepared = []
    for row in rows:
        neu_ids, neu_mask = backend.encode_text(row["neutral"], row["lyrics"])
        pos_ids, pos_mask = backend.encode_text(row["positive"], row["lyrics"])
        if neu_ids.shape[1] > max_seq_len or pos_ids.shape[1] > max_seq_len:
            raise ValueError("Prompt exceeds context limit")
        neu = backend.hidden(neu_ids, neu_mask)
        pos = backend.hidden(pos_ids, pos_mask)
        plus = lm_faithful_plus_neu(pos, neu, neu)
        prepared.append(
            dict(
                neutral_ids=neu_ids.cpu(),
                neutral_mask=neu_mask.cpu(),
                positive_ids=pos_ids.cpu(),
                positive_mask=pos_mask.cpu(),
                neutral=neu.cpu(),
                targets=plus.cpu(),
                guard_applied=False,
                target_shift=0.0,
            )
        )
    return prepared


def build_game(backend, network, fixed):
    device = next(network.parameters()).device
    targets = torch.cat([row["targets"] for row in fixed]).to(device)
    neutrals = torch.cat([row["neutral"] for row in fixed]).to(device)
    kind = RECIPE.get("critic", "mix")
    for key, ref in (
        ("critic", "critic"),
        ("critic_patch", "critic_patch"),
        ("critic_hidden", "critic_width"),
        ("critic_layers", "critic_layers"),
        ("critic_heads", "critic_heads"),
        ("critic_tokens", "critic_tokens"),
        ("critic_queries", "critic_queries"),
        ("critic_rank", "critic_rank"),
        ("critic_score_bound", "critic_score_bound"),
        ("d_lr", "d_lr"),
    ):
        if key in RECIPE:
            shared.REFERENCE[ref] = RECIPE[key]
    return shared.build_game(network, targets, critic=kind, neutrals=neutrals)


def update(backend, network, critic, g, d, rows, *, sampler, step, checkpointing=True):
    device = critic.target_mean.device
    targets = torch.cat([row["targets"] for row in rows]).to(device)
    predictions = {}

    def predict(index, phase):
        row = rows[index]
        ids = row["positive_ids"].to(device)
        mask = row["positive_mask"].to(device)
        with network.scaled(1.0):
            if checkpointing and phase == "g" and not backend.dummy:
                pred = torch.utils.checkpoint.checkpoint(
                    _encode_train, backend.model, ids, mask, use_reentrant=False
                ).float()
            else:
                pred = backend.hidden(ids, mask, train=(phase == "g")).float()
        if phase == "g":
            predictions[index] = pred.detach()
        return critic.normalize(pred)

    result = shared.update(
        network, critic, g, d, targets, predict, sampler=sampler, step=step
    )
    result["cos_pos"] = sum(
        float(
            torch.nn.functional.cosine_similarity(
                predictions[i] - rows[i]["neutral"].to(device),
                targets[i : i + 1] - rows[i]["neutral"].to(device),
                dim=-1,
            ).mean()
        )
        for i in result["g_rows"]
    ) / len(result["g_rows"])
    return result
