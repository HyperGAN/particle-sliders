"""Opt-in routed particle branches on Music 3 LM attention; paired-error GAN + VIC.

Ports Yue2 ``anneal-routed-particle-error-yue2-v1`` onto MiniMax Music 3's
Qwen3 LM (prompt-state last-hidden only). Shared game math lives in
``particle_bridge_gan``. Propose-only; does not change live ``--lm_target v9``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.slider_targets import lm_faithful_plus_neu
from particle_sliders.music3_native import (
    DEFAULT_MODEL, TARGET_REPLACE, FORMAT, Music3Backend, ParticleNetwork,
    Qwen3Attention, _TinyLM, _encode_train,
)
from conceptmod.textsliders.yue2_backend import sound_only

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


def build_game(backend, network, fixed, *, recipe=None):
    device = next(network.parameters()).device
    targets = torch.cat([row["targets"] for row in fixed]).to(device)
    neutrals = torch.cat([row["neutral"] for row in fixed]).to(device)
    recipe = RECIPE if recipe is None else recipe
    kind = recipe.get("critic", "mix")
    config = {}
    accepted = shared.CRITIC_CONFIG_BY_KIND.get(kind, set())
    for key, ref in (
        ("critic_patch", "patch"), ("critic_hidden", "width"),
        ("critic_layers", "layers"), ("critic_heads", "heads"),
        ("critic_tokens", "tokens"), ("critic_queries", "queries"),
        ("critic_rank", "rank"), ("critic_score_bound", "score_bound"),
    ):
        if key in recipe and (ref in accepted or ref == "score_bound"):
            config[ref] = recipe[key]
    # MLP/patch have no score-bound implementation; a positive legacy setting
    # was historically ignored for them. BISPF passes the explicit raw value 0.
    if kind in ("mlp", "patch"):
        config.pop("score_bound", None)
    return shared.build_game(network, targets, critic=kind,
                             critic_config=config, neutrals=neutrals)


def update(backend, network, critic, g, d, rows, *, sampler, step, checkpointing=True,
           bispf=None):
    device = critic.target_mean.device
    targets = torch.cat([row["targets"] for row in rows]).to(device)
    predictions = {}

    def predict(index, phase):
        row = rows[index]
        # The frozen positive-caption state is the teacher. BISPF trains the
        # adapted neutral caption, matching inference; legacy keeps its old input.
        prefix = "neutral" if bispf is not None else "positive"
        ids = row[f"{prefix}_ids"].to(device)
        mask = row[f"{prefix}_mask"].to(device)
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

    result = (bispf.step(step, predict) if bispf is not None else shared.update(
        network, critic, g, d, targets, predict, sampler=sampler, step=step
    ))
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
