"""Port the winning UNI16 600-update warm-up to native YuE2 AR states.

Reuse the release's critic, RpGAN, calibrated b_cap and exact batch-mean FM.
Only token packing, AR forwarding and the semantic/end readout are model-specific.
"""
from __future__ import annotations

import json
import time
import unicodedata

import torch
import torch.nn.functional as F

from conceptmod.textsliders.lm_adv import (
    SpanTransformerD, cap_penalty, rp_d_loss, rp_g_loss, param_grad_norm,
)
from conceptmod.textsliders.lm_gan import feature_mean_surrogate
from conceptmod.textsliders.gan_v2.critic import pad_sequences
from conceptmod.textsliders.yue2_backend import YuE2Slider

RECIPE = {
    "name": "uni16-rpgan-bcap-warmup", "reference": "docs/hub-formulation-fresh-selected.md",
    "adversarial_weight": 1.0, "fm_weight": 1.0, "end_weight": 1.0,
    "pole_weight": 0.0, "lyrichold_weight": 0.0, "cap_coefficient": 1.0, "cap_kappa": 1.0,
    "critic": {"width": 128, "n_layers": 2, "n_heads": 4, "in_mode": "scaled", "readout": "mean_last"},
    "g_betas": [0.0, 0.999], "d_betas": [0.0, 0.999], "g_weight_decay": 1e-6,
    "schedule": "constant", "gradient_clip": "value=1", "parameter_step_limit": None,
    "fm_objective": "matched_weight_batch_mean", "history_policy": "fixed_base_per_row",
    "style_positions": "matching_lyric_tokens_plus_music_start", "trained_scales": [1],
}


def lyric_positions(backend, style, lyrics):
    """Select identical lyric token bytes, excluding caption and protocol tokens."""
    from yue2.protocol import SongRequest
    text = unicodedata.normalize("NFC", SongRequest(style=style, lyrics=lyrics, cot="off").text())
    prefix = backend.prefix(style, lyrics)
    marker = "\n[Lyrics]\n"
    start = text.index(marker) + len(marker)
    end = start + len(unicodedata.normalize("NFC", lyrics))
    start, end = len(text[:start].encode()), len(text[:end].encode())
    cursor, positions = 0, []
    for position, token in enumerate(prefix[1:-3], start=1):
        raw = bytes([token]) if backend.dummy else backend.tokenizer._enc.decode_single_token_bytes(token)
        next_cursor = cursor + len(raw)
        if cursor >= start and next_cursor <= end:
            positions.append(position)
        cursor = next_cursor
    if cursor != len(text.encode()) or not positions:
        raise ValueError("Cannot locate the exact YuE2 lyric token span")
    return prefix, positions + [len(prefix) - 1]


def end_margins(model, hidden):
    """Native music-end logit minus logsumexp over exactly the codec vocabulary."""
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END
    weight = model.lm_head.weight
    semantic = F.linear(hidden, weight[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE]).float()
    end = F.linear(hidden, weight[MUSIC_END:MUSIC_END + 1]).float().squeeze(-1)
    return end - semantic.logsumexp(-1)


def preroll(backend, style, lyrics, count, seed):
    from yue2.protocol import SongRequest, Sampling, negative_prefix
    from yue2.sampling import generate_tokens
    prefix = backend.prefix(style, lyrics)
    if backend.dummy:
        return backend.continuation(prefix, count, seed), False
    request = SongRequest(style=style, lyrics=lyrics, cot="off", seed=seed)
    tokens, _, truncated = generate_tokens(backend.model, prefix,
        Sampling(min_tokens=min(200, count), max_tokens=count), seed, "semantic",
        negative=negative_prefix(request, backend.tokenizer), cfg_scale=request.guidance,
        legacy_off=True, use_cuda_graph=False)
    return tokens, not truncated


def prepare_rows(backend, rows, count, seed, max_seq_len):
    prepared = []
    for index, row in enumerate(rows):
        print(f"Preparing UNI16 prompt teachers and frozen history {index + 1}/{len(rows)}", flush=True)
        neutral, positions = lyric_positions(backend, row["neutral"], row["lyrics"])
        positive, positive_positions = lyric_positions(backend, row["positive"], row["lyrics"])
        if [neutral[i] for i in positions] != [positive[i] for i in positive_positions]:
            raise ValueError("Neutral and positive lyric/music-start tokens must match exactly")
        if max(len(neutral), len(positive)) + count > min(max_seq_len, backend.model.config.max_position_embeddings):
            raise ValueError("Prompt plus fixed history exceeds max_seq_len")
        tail, ended = preroll(backend, row["neutral"], row["lyrics"], count, seed + index)
        with torch.no_grad():
            base = backend.hidden(neutral + tail)
            neutral_span = base[:, positions].float()
            target_span = backend.hidden(positive)[:, positive_positions].float()
            margins = end_margins(backend.model, base[:, len(neutral) - 1:])
        prepared.append({"ids": neutral + tail, "prefix_len": len(neutral), "positions": positions,
            "neutral": neutral_span.cpu(), "real": (target_span - neutral_span).cpu(),
            "end_teacher": margins.cpu(), "history": tail, "history_seed": seed + index,
            "ended_naturally": ended})
    return prepared


def student(backend, row, checkpointing):
    hidden = backend.hidden(row["ids"], checkpointing=checkpointing)
    return hidden[:, row["positions"]].float() - row["neutral"].to(hidden.device), end_margins(
        backend.model, hidden[:, row["prefix_len"] - 1:])


def update(backend, network, critic, g_optimizer, d_optimizer, rows, *, checkpointing=True):
    # Keep +1 active through checkpoint recomputation during backward.
    with network.scaled(1):
        critic.requires_grad_(True)
        d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake, mask = pad_sequences([student(backend, row, False)[0] for row in rows])
            real, real_mask = pad_sequences([row["real"].to(fake.device) for row in rows])
        if not torch.equal(mask, real_mask):
            raise ValueError("Teacher and student spans are not aligned")
        penalty, cap_stats = cap_penalty(critic, real, fake, coeff=1, kappa=1,
                                        mask_real=mask, mask_fake=mask)
        d_loss = rp_d_loss(critic(real, mask), critic(fake, mask)) + penalty
        if not torch.isfinite(d_loss):
            raise FloatingPointError("Non-finite UNI16 discriminator loss")
        d_loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite UNI16 discriminator gradient")
        d_optimizer.step()
        critic.requires_grad_(False)
        with torch.no_grad():
            real_mean = critic.features(real, mask).mean(0)
            fake_mean = critic.features(fake, mask).mean(0)
            real_scores = critic(real, mask)
        g_optimizer.zero_grad(set_to_none=True)
        totals = {"g_adv": 0., "fm": 0., "end": 0., "cos_pos": 0., "mag_ratio": 0.}
        for index, row in enumerate(rows):
            delta, margins = student(backend, row, checkpointing)
            adv = rp_g_loss(critic(delta), real_scores[index:index + 1])
            fm = feature_mean_surrogate(critic.features(delta), fake_mean, real_mean)
            end = F.mse_loss(margins, row["end_teacher"].to(margins.device))
            loss = (adv + fm + end) / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite UNI16 generator loss")
            loss.backward()
            target = row["real"].to(delta.device)
            cos = F.cosine_similarity(delta[:, -1], target[:, -1], dim=-1).mean()
            ratio = delta[:, -1].norm() / target[:, -1].norm().clamp_min(1e-8)
            for key, value in (("g_adv", adv), ("fm", fm), ("end", end), ("cos_pos", cos), ("mag_ratio", ratio)):
                totals[key] += float(value.detach()) / len(rows)
        grad_norm = param_grad_norm(network.parameters())
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError("Non-finite UNI16 adapter gradient")
        torch.nn.utils.clip_grad_value_(network.parameters(), 1.0)
        g_optimizer.step()
    return dict(totals, loss=totals["g_adv"] + totals["fm"] + totals["end"],
        d_loss=float(d_loss.detach()), d_pen=float(penalty.detach()), grad_norm=grad_norm,
        d_real_grad_mean=cap_stats["real_grad_mean"], d_fake_grad_mean=cap_stats["fake_grad_mean"])


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def train_uni16(args, backend, rows):
    if len(rows) < args.adv_batch:
        raise ValueError("UNI16 needs at least adv_batch distinct prompt rows")
    signature = dict(recipe=RECIPE, rows=rows, rank=args.rank, alpha=args.alpha,
        g_lr=args.lr, d_lr=args.lr * 1.5, batch=args.adv_batch, seed=args.seed,
        endreg_frames=args.train_tokens, model_identity=backend.identity, dummy=args.dummy)
    state = torch.load(args.resume_state, map_location="cpu", weights_only=True) if args.resume_state else None
    if state is not None and state["signature"] != signature:
        raise ValueError("Resume state belongs to another recipe, prompt set or base model")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / f"{args.name}_train.jsonl"
    if log_path.exists() and state is None:
        raise FileExistsError("Run already exists; use a fresh save_dir or --resume_state")
    prepared = state["prepared"] if state else prepare_rows(backend, rows, args.train_tokens, args.seed, args.max_seq_len)
    network = YuE2Slider(backend.model, rank=args.rank, alpha=args.alpha)
    device = next(backend.model.parameters()).device
    critic = SpanTransformerD(backend.model.config.hidden_size, **RECIPE["critic"]).to(device)
    real, mask = pad_sequences([row["real"].to(device) for row in prepared])
    critic.calibrate_input_scale(real, mask)
    g_optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, betas=(0., .999), weight_decay=1e-6)
    d_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr * 1.5, betas=(0., .999))
    completed = 0
    if state:
        network.load_state_dict(state["network"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        g_optimizer.load_state_dict(state["g_optimizer"])
        d_optimizer.load_state_dict(state["d_optimizer"])
        completed = state["completed"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if completed >= args.steps:
        raise ValueError("Resume state has already reached the requested step count")
    metadata = dict(backend="yue2", recipe=RECIPE["name"], recipe_settings=signature,
        model_id=args.model_id, model_identity=backend.identity, revision=args.revision,
        dummy=args.dummy, rows=rows, seed=args.seed, lr=args.lr, hold_weight=0., train_tokens=args.train_tokens,
        cot="off", recommended_range=[0, 1], validation_status="unvalidated",
        teacher_rms=float(critic.input_scale), history_lengths=[len(row["history"]) for row in prepared])
    print(json.dumps({"recipe": RECIPE["name"], "batch": args.adv_batch, "teacher_rms": float(critic.input_scale),
                      "g_lr": args.lr, "d_lr": args.lr * 1.5, "start_step": completed}), flush=True)
    (args.save_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    start_step = completed
    with log_path.open("a" if state else "w") as log:
        for step in range(completed + 1, args.steps + 1):
            indices = [((step - 1) * args.adv_batch + i) % len(prepared) for i in range(args.adv_batch)]
            metrics = update(backend, network, critic, g_optimizer, d_optimizer,
                [prepared[i] for i in indices], checkpointing=not args.no_checkpointing)
            metrics.update(step=step, rows=indices, elapsed_sec=time.monotonic() - started)
            metrics["seconds_per_step"] = metrics["elapsed_sec"] / (step - start_step)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if step == start_step + 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(metrics), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                network.save(args.save_dir / f"{args.name}_{step}.safetensors", dict(metadata, **metrics))
                network.save(args.save_dir / f"{args.name}_last.safetensors", dict(metadata, **metrics))
                state_path = args.save_dir / f"{args.name}_state.pt"
                torch.save({"signature": signature, "prepared": prepared, "completed": step,
                    "network": _cpu(network.state_dict()), "critic": _cpu(critic.state_dict()),
                    "g_optimizer": _cpu(g_optimizer.state_dict()), "d_optimizer": _cpu(d_optimizer.state_dict()),
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
                    state_path.with_suffix(".pt.tmp"))
                state_path.with_suffix(".pt.tmp").replace(state_path)
    return metadata
