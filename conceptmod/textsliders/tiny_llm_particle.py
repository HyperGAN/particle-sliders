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

from particle_sliders.tiny_native import (
    FORMAT, TinyParticleProjection, TinyParticleSlider,
    _adapter_key, _attention_targets, _module_param_dtype,
)
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
