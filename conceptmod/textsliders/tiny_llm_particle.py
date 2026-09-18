"""Opt-in routed particle branches in tiny-LM attention; paired-error GAN + VIC.

Transfers YuE2's working recipe ``anneal-routed-particle-error-yue2-v1``
(``--recipe particle_bridge``) onto Qwen3-0.6B last-hidden states. The game
itself is NOT reimplemented here: ``shared`` is the exact YuE2 module
``conceptmod.textsliders.particle_bridge_gan`` (same Rp logistic on the
paired error, same noise anneal, same lazy b_cap, same particle VIC, same
EMA decay). Only the host differs: ``TinyAttention`` q/k/v/o instead of
YuE2 AR attention, and a last-caption-token readout instead of YuE2's
prefix-boundary token (see MATCH vs DRIFT in docs/tiny-llm-slider.md).

Unipolar: + caption vs raw positive teacher; scale 0 bypasses every branch
exactly. No output MSE, FM, ending, hold, or anchor — on G or anywhere.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.tiny_llm_backend import (
    ATTN_CLASS_NAMES,
    LORA_LINEAR_NAMES,
    ArchitectureMismatch,
)

FORMAT = "conceptmod-tiny-llm-particle-v1"
RECIPE = dict(
    name="anneal-routed-particle-error-tiny-llm-v1",
    source_recipe="anneal-routed-particle-error-yue2-v1",
    generator_objective="paired_error_rpgan_plus_particle_vic",
    parts=128,
    particle_dim=4,
    vicreg_weight=1.0,
    g_lr=shared.REFERENCE["g_lr"],
    d_lr=shared.REFERENCE["d_lr"],
    particle_lr=shared.REFERENCE["particle_lr"],
    g_optimizer="Adam",
    g_weight_decay=0.0,
    optimizer_betas=shared.REFERENCE["betas"],
    critic_hidden=shared.REFERENCE["width"],
    critic_layers=3,
    penalty_lazy_k=shared.REFERENCE["cap_every"],
    cap_coordinates="normalized_paired_error_plus_shared_gaussian",
    target_normalization="training_target_per_coordinate_sample_std_floor_1e-4",
    noise_start=1.0,
    noise_floor=shared.REFERENCE["noise_floor"],
    noise_decay_steps=shared.REFERENCE["noise_decay_steps"],
    ema=shared.REFERENCE["ema"],
    adv_batch=shared.REFERENCE["batch_size"],
    schedule="constant",
    prompt_policy="independent_D_G_with_replacement",
    routing="all_examples",
    particle_vic_batch=64,
    particle_vic_target_std=1.0,
    particle_vic_eps=1e-4,
    adapter_format=FORMAT,
    adapter_rank=8,
    adapter_alpha=8.0,
    adapter_width=shared.REFERENCE["width"],
    router_width=shared.REFERENCE["router_width"],
    generator_initialization="zero_output_low_rank; standard_normal_particles",
    readout="last_caption_token_hidden",
    propose_only=True,
    merge_to_trainer=False,
)


def _adapter_key(dotted: str) -> str:
    return f"lora_tiny-{dotted}".replace(".", "-")


class TinyParticleProjection(nn.Module):
    """One routed low-rank branch on a frozen attention Linear.

    Mirrors ``ParticleProjection``: ``down`` is kaiming, ``up`` starts at
    zero (base preserved at init), one shared 128x4 cloud, per-projection
    router + MLP, multiplier 0 bypasses the host exactly.
    """

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        particles: nn.Parameter,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.lora_name = name
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 0.0
        host_kwargs: dict = {}
        if hasattr(module, "weight"):
            host_kwargs["device"] = module.weight.device
            host_kwargs["dtype"] = module.weight.dtype
        self.lora_down = nn.Linear(module.in_features, rank, bias=False, **host_kwargs)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False, **host_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        # The cloud belongs to the root exactly once, not to every projection.
        object.__setattr__(self, "_particles", particles)
        self.bridge = shared.RoutedMLP(rank, rank)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0:
            return self.org_forward(x)
        features = self.lora_down(x.to(device=self.lora_down.weight.device,
                                       dtype=self.lora_down.weight.dtype))
        routed = self.bridge(features, self._particles)
        delta = self.lora_up(routed).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class TinyParticleSlider(nn.Module):
    """Wrap attention ``q/k/v/o`` Linears only, sharing one 128x4 cloud."""

    def __init__(self, model: nn.Module, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if int(rank) != 8 or float(alpha) != 8.0:
            raise ValueError("The tiny particle format pins rank/alpha 8")
        targets = _attention_targets(model)
        if any(
            isinstance(getattr(m.forward, "__self__", None), TinyParticleProjection)
            for m in targets.values()
        ):
            raise ValueError("A tiny particle slider is already attached")
        self.rank, self.alpha = int(rank), float(alpha)
        self.target_names = list(targets)
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.adapters = nn.ModuleDict()
        for name, module in targets.items():
            key = _adapter_key(name)
            adapter = TinyParticleProjection(key, module, self.particles,
                                             self.rank, self.alpha)
            self.adapters[key] = adapter
        host_dtype = _module_param_dtype(model)
        if host_dtype is not None:
            self.to(dtype=host_dtype)

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

    def save(self, path, metadata):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            targets=self.target_names,
            particles=128, particle_dim=4, bridge_width=48, router_width=16,
        )
        state = self.state_dict()
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Invalid tiny particle-slider export")
        save_file(
            {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
            str(path), metadata={"conceptmod": json.dumps(record, sort_keys=True)},
        )
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, model, path):
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            record = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
        required = dict(format=FORMAT, rank=8, alpha=8.0, particles=128,
                        particle_dim=4, bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()):
            raise ValueError("Incompatible tiny routed-particle checkpoint")
        if record.get("targets") != list(_attention_targets(model)):
            raise ValueError("Tiny particle-slider targets differ")
        state = load_file(str(path), device="cpu")
        expected = {"particles": (128, 4)}
        for name, module in _attention_targets(model).items():
            prefix = "adapters." + _adapter_key(name)
            expected.update({
                prefix + ".lora_down.weight": (8, module.in_features),
                prefix + ".lora_up.weight": (module.out_features, 8),
            })
            for branch, inputs, outputs, width in [("router", 8, 4, 16),
                                                   ("net", 12, 8, 48)]:
                sizes = [inputs, width, width, width, outputs]
                for j, (a, b) in enumerate(zip(sizes, sizes[1:])):
                    expected[f"{prefix}.bridge.{branch}.{j * 2}.weight"] = (b, a)
                    expected[f"{prefix}.bridge.{branch}.{j * 2}.bias"] = (b,)
        if set(state) != set(expected) or any(
            tuple(state[k].shape) != shape for k, shape in expected.items()
        ):
            raise ValueError("Incomplete tiny routed-particle tensors")
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite tiny routed-particle tensors")
        network = cls(model)
        network.load_state_dict(state, strict=True)
        return network, record


def _attention_targets(model: nn.Module) -> dict[str, nn.Linear]:
    targets: dict[str, nn.Linear] = {}
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
            targets[f"{name}.{child_name}" if name else child_name] = child
    if not targets:
        raise ArchitectureMismatch(
            "no attention q/k/v/o Linears found; expected Qwen3-shaped "
            f"{ATTN_CLASS_NAMES} hosts"
        )
    return targets


def _module_param_dtype(module: nn.Module):
    for param in module.parameters():
        return param.dtype
    return None


def prepare_rows(backend, rows: list[dict]) -> list[dict]:
    """Freeze per-row teachers: raw + caption and neutral last-hidden states."""
    fixed = []
    with torch.no_grad():
        for row in rows:
            plus_ids = backend.encode(row["positive"]).ids
            neu_ids = backend.encode(row["neutral"]).ids
            plus = backend.teacher_hidden(plus_ids)[:, -1].float()
            neutral = backend.teacher_hidden(neu_ids)[:, -1].float()
            fixed.append({
                "ids": plus_ids,
                "targets": plus.cpu(),
                "neutral": neutral.cpu(),
                "positive": row["positive"],
                "neutral_text": row["neutral"],
            })
    return fixed


def build_game(backend, network, fixed):
    targets = torch.cat([r["targets"] for r in fixed]).to(
        next(network.parameters()).device
    )
    return shared.build_game(network, targets)


def update(backend, network, critic, g, d, rows, *, sampler, step):
    device = critic.target_mean.device
    targets = torch.cat([r["targets"] for r in rows]).to(device)
    predictions = {}

    def predict(i, phase):
        row = rows[i]
        with network.scaled(1.0):
            pred = backend.hidden(row["ids"])[:, -1].float()
        if phase == "g":
            predictions[i] = pred.detach()
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


def resolve_particle_path(path: str):
    path = Path(path)
    if path.is_file():
        return path
    candidates = []
    if path.suffix == ".safetensors":
        candidates.append(path)
    candidates.append(Path(str(path) + "_last.safetensors"))
    candidates.append(Path(str(path) + ".safetensors"))
    if path.is_dir():
        candidates.extend(sorted(path.glob("*_last.safetensors")))
        candidates.extend(sorted(path.glob("*.safetensors")))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no tiny particle LoRA safetensors under {path} "
        "(expected {name}_last.safetensors from the particle trainer)"
    )
