"""Opt-in routed particle head on the frozen Bonsai GGUF readout.

Runs YuE2's working recipe ``anneal-routed-particle-error-yue2-v1``
(``--recipe particle_bridge``) on Ternary-Bonsai-2-27B last-token hidden
states served by the fork's ``llama-server --embedding --pooling last``.
The game itself is NOT reimplemented here: ``shared`` is the exact YuE2
module ``conceptmod.textsliders.particle_bridge_gan`` (same Rp logistic
on the paired error, same noise anneal, same lazy b_cap, same particle
VIC, same EMA decay).

DRIFT (honest, documented in docs/bonsai-gguf-slider.md): a GGUF is
inference-only frozen ternary — there is no gradient path through the
fork, so full LoRA-in-GGUF training is impossible and no in-attention
LoRA is claimed. The trainable slider is a torch-side residual head on
the frozen 5120-dim readout: ``down`` is kaiming, ``up`` starts at zero
(base preserved at init), one shared 128x4 cloud, router + MLP bridge,
multiplier 0 bypasses the readout exactly. Same game, different (and
explicitly weaker) adapter placement than the Qwen3-0.6B path, which
wraps live attention q/k/v/o.

Unipolar: + caption vs raw positive teacher; scale 0 bypasses exactly.
No output MSE, FM, ending, hold, or anchor — on G or anywhere.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.bonsai_gguf_backend import HIDDEN, EncodedText

from particle_sliders.bonsai_native import BonsaiParticleHead, FORMAT, RANK, ALPHA
RECIPE = dict(
    name="anneal-routed-particle-error-bonsai-gguf-v1",
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
    adapter_rank=RANK,
    adapter_alpha=ALPHA,
    adapter_width=shared.REFERENCE["width"],
    router_width=shared.REFERENCE["router_width"],
    generator_initialization="zero_output_low_rank; standard_normal_particles",
    readout="fork_server_last_token_embedding",
    adapter_placement="readout_residual_zero_init",
    propose_only=True,
    merge_to_trainer=False,
)




def prepare_rows(backend, rows: list[dict]) -> list[dict]:
    """Freeze per-row teachers: raw + caption and neutral readout states."""
    fixed = []
    with torch.no_grad():
        for row in rows:
            plus = backend.encode(row["positive"])
            neu = backend.encode(row["neutral"])
            plus_h = backend.teacher_hidden(plus).float()
            neutral = backend.teacher_hidden(neu).float()
            fixed.append({
                "enc": plus,
                "targets": plus_h.cpu(),
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
    # Frozen readout: fetch once per row per update call (the GGUF never
    # changes; D and G phases share the same frozen vectors).
    frozen: dict[int, torch.Tensor] = {}

    def predict(i, phase):
        row = rows[i]
        enc = row["enc"]
        assert isinstance(enc, EncodedText)
        if i not in frozen:
            frozen[i] = backend.hidden(enc)
        with network.scaled(1.0):
            pred = network(frozen[i]).float()
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
                targets[i:i + 1] - rows[i]["neutral"].to(device),
                dim=-1,
            ).mean()
        )
        for i in result["g_rows"]
    ) / len(result["g_rows"])
    return result


def resolve_head_path(path: str):
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
        f"no Bonsai particle-head safetensors under {path} "
        "(expected {name}_last.safetensors from the particle trainer)"
    )
