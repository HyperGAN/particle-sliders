#!/usr/bin/env python3
"""Train Music 3 with the Yue2 winning particle-bridge formulation."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conceptmod.textsliders import music3_particle_bridge as game
from conceptmod.textsliders import particle_bridge_gan as particle_game
from conceptmod.textsliders.train_lora_yue2_fresh import write_json
from conceptmod.textsliders.yue2_uni import _cpu


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    names = [
        "train_lora_music3_particle.py",
        "music3_particle_bridge.py",
        "particle_bridge_gan.py",
        "lora.py",
        "slider_targets.py",
        "train_lm_slider_music3.py",
    ]
    paths = [ROOT / "conceptmod/textsliders" / name for name in names]
    paths += [ROOT / "analysis/slider2d" / name for name in ("adv.py", "grad_regularizers.py")]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def train(args):
    recipe = dict(game.RECIPE)
    if getattr(args, "critic", None):
        recipe["critic"] = args.critic
    if getattr(args, "critic_patch", None) is not None:
        recipe["critic_patch"] = args.critic_patch
    if getattr(args, "critic_width", None) is not None:
        recipe["critic_hidden"] = args.critic_width
    if getattr(args, "critic_layers", None) is not None:
        recipe["critic_layers"] = args.critic_layers
    if getattr(args, "critic_heads", None) is not None:
        recipe["critic_heads"] = args.critic_heads
    if getattr(args, "critic_tokens", None) is not None:
        recipe["critic_tokens"] = args.critic_tokens
    if getattr(args, "critic_queries", None) is not None:
        recipe["critic_queries"] = args.critic_queries
    if getattr(args, "critic_rank", None) is not None:
        recipe["critic_rank"] = args.critic_rank
    if getattr(args, "critic_score_bound", None) is not None:
        recipe["critic_score_bound"] = args.critic_score_bound
    if getattr(args, "d_lr", None) is not None:
        recipe["d_lr"] = args.d_lr
    # build_game reads module RECIPE; keep it aligned with this run.
    game.RECIPE.update(
        {k: recipe[k] for k in recipe if k.startswith("critic") or k == "d_lr"}
    )
    if recipe["parts"] != 128 or recipe["vicreg_weight"] != 1 or recipe["particle_dim"] != 4:
        raise ValueError("Particle bridge requires 128x4 particles and VIC weight 1")
    rows, meta = game.load_prompts(args.prompts_file)
    if len(rows) < 2:
        raise ValueError("Particle bridge needs at least two prompt rows")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(args, rows, meta, recipe)


def _train_locked(args, rows, meta, recipe):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    run = args.save_dir
    path = run / "state.pt"
    print(json.dumps(dict(recipe=recipe, seed=args.seed, steps=args.steps, device=args.device)), flush=True)
    settings = dict(
        recipe=recipe,
        name=args.name,
        rows=rows,
        metadata=meta,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        checkpointing=not args.no_checkpointing,
        rank=8,
        alpha=8.0,
        dummy=args.dummy,
        model_dir=str(args.model_dir),
        sources=source_hashes(),
    )
    saved = (
        torch.load(path, map_location="cpu", weights_only=True, mmap=True) if path.exists() else None
    )
    if saved and saved["signature"]["settings"] != settings:
        raise ValueError("Resume recipe, prompts, model settings, or source differs")
    if not saved and (run / "manifest.json").exists():
        raise FileExistsError("Existing run has no recovery state; use a fresh save_dir")
    backend = game.Music3Backend(args.model_dir, device=args.device, dummy=args.dummy)
    signature = dict(settings=settings, model=backend.identity)
    if saved and saved["signature"] != signature:
        raise ValueError("Resume base weights differ")
    fixed = saved["prepared"] if saved else game.prepare(backend, rows, meta, args.max_seq_len)
    torch.manual_seed(args.seed + 1000)
    network = game.ParticleNetwork(backend.model, rank=8, alpha=8.0)
    critic, g, d = game.build_game(backend, network, fixed)
    for group in g.param_groups:
        group["lr"] = recipe["particle_lr"] if group.get("role") == "particles" else recipe["g_lr"]
    for group in d.param_groups:
        group["lr"] = recipe["d_lr"]
    device = next(network.parameters()).device
    sampler = particle_game.BridgeSampler(len(rows), args.seed)
    completed = 0
    history = []
    if saved:
        network.load_state_dict(saved["network"], strict=True)
        critic.load_state_dict(saved["critic"], strict=True)
        g.load_state_dict(saved["g_optimizer"])
        d.load_state_dict(saved["d_optimizer"])
        sampler.load_state_dict(saved["sampler"])
        completed = saved["completed"]
        history = saved["history"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if [item["step"] for item in history] != list(range(1, completed + 1)):
        raise ValueError("Non-contiguous saved update history")
    if any(
        len(item[key]) != recipe["adv_batch"] or any(not 0 <= i < len(rows) for i in item[key])
        for item in history
        for key in ("d_rows", "g_rows")
    ):
        raise ValueError("Invalid saved particle-bridge prompt batches")
    ema = (
        {key: value.to(device) for key, value in saved["ema"].items()}
        if saved
        else particle_game.initialize_ema(network)
    )
    metadata = dict(
        backend="music3",
        recipe=recipe["name"],
        recipe_settings=signature,
        model_dir=str(args.model_dir),
        model_identity=backend.identity,
        dummy=args.dummy,
        rows=rows,
        prompt_metadata=meta,
        recommended_range=[0, 1],
        polarity="unipolar",
        trained_scales=recipe["trained_scales"],
        zero_behavior="exact_base_by_adapter_scale",
        validation_status="experimental",
        teacher_rms=float(critic.input_scale),
    )
    write_json(run / "manifest.json", signature)
    std = critic.target_std.detach().float().cpu()
    write_json(
        run / "teacher-audit.json",
        dict(
            rows=[{key: row[key] for key in ("guard_applied", "target_shift")} for row in fixed],
            teacher_rms=float(critic.input_scale),
            batch=recipe["adv_batch"],
            rank=8,
            alpha=8.0,
            recipe=recipe,
            normalization=dict(
                training_rows=len(rows),
                dimensions=std.numel(),
                std_min=float(std.min()),
                std_median=float(std.median()),
                std_max=float(std.max()),
                floor_count=int((std <= particle_game.REFERENCE["target_std_floor"]).sum()),
            ),
        ),
    )
    start = time.monotonic()
    start_step = completed
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def status(phase, **extra):
        write_json(
            run / "status.json",
            dict(
                status=phase,
                completed=completed,
                total=args.steps,
                until=args.until,
                elapsed_sec=time.monotonic() - start,
                start_step=start_step,
                gpu=os.getenv("CUDA_VISIBLE_DEVICES"),
                **extra,
            ),
        )

    def save():
        if any(
            not torch.isfinite(value).all()
            for value in [*network.state_dict().values(), *critic.state_dict().values()]
        ):
            raise FloatingPointError("Non-finite checkpoint")
        if any(not torch.isfinite(value).all() for value in ema.values()):
            raise FloatingPointError("Non-finite EMA")
        blob = dict(
            signature=signature,
            prepared=fixed,
            completed=completed,
            history=history,
            network=_cpu(network.state_dict()),
            critic=_cpu(critic.state_dict()),
            g_optimizer=_cpu(g.state_dict()),
            d_optimizer=_cpu(d.state_dict()),
            sampler=sampler.state_dict(),
            ema=_cpu(ema),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        temporary = path.with_suffix(".pt.tmp")
        torch.save(blob, temporary)
        temporary.replace(path)
        counted = [i for item in history for i in item["d_rows"] + item["g_rows"]]
        record = dict(metadata, step=completed, prompt_draws=len(counted), row_counts=dict(Counter(counted)))
        export = run / f"{args.name}_last.safetensors"
        tmp = run / f"{args.name}_pending.safetensors"
        network.save(tmp, dict(record, weights_kind="ema"), state=ema)
        tmp.replace(export)
        tmp.with_suffix(".json").replace(export.with_suffix(".json"))
        live = run / f"{args.name}_live_last.safetensors"
        network.save(tmp, dict(record, weights_kind="live"))
        tmp.replace(live)
        tmp.with_suffix(".json").replace(live.with_suffix(".json"))
        if completed and (completed % args.save_every == 0 or completed == args.until):
            pinned = run / f"state-step{completed}.pt"
            if not pinned.exists():
                os.link(path, pinned)
                os.link(export, run / f"{args.name}_step{completed}.safetensors")
                write_json(run / f"{args.name}_step{completed}.json", dict(record, weights_kind="ema"))
                os.link(live, run / f"{args.name}_live_step{completed}.safetensors")
                write_json(run / f"{args.name}_live_step{completed}.json", dict(record, weights_kind="live"))
        status(
            "paused"
            if stop
            else "complete"
            if completed >= args.steps
            else "checkpoint_ready"
            if completed >= args.until
            else "training",
            prompt_draws=record["prompt_draws"],
            row_counts=record["row_counts"],
        )

    try:
        if completed >= args.until:
            status("complete" if completed >= args.steps else "checkpoint_ready")
            return
        if not saved:
            save()
        with (run / f"updates-from-{completed}-{time.time_ns()}.jsonl").open("w") as log:
            while completed < args.until and not stop:
                step = completed + 1
                begin = time.monotonic()
                status("training", next_step=step)
                for group in d.param_groups:
                    group["lr"] = float(recipe["d_lr"])
                metrics = game.update(
                    backend,
                    network,
                    critic,
                    g,
                    d,
                    fixed,
                    sampler=sampler,
                    step=step,
                    checkpointing=not args.no_checkpointing,
                )
                particle_game.update_ema(ema, network)
                completed = step
                record = dict(
                    metrics,
                    step=step,
                    step_seconds=time.monotonic() - begin,
                    g_lr=g.param_groups[0]["lr"],
                    d_lr=d.param_groups[0]["lr"],
                    particle_lr=g.param_groups[1]["lr"],
                )
                history.append(record)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                write_json(run / "progress.json", record)
                print(json.dumps(record), flush=True)
                if completed % args.save_every == 0 and completed != args.until:
                    save()
            save()
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="music3-particle")
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--until", type=int)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--model_dir", type=Path, default=game.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--no_checkpointing", action="store_true")
    parser.add_argument(
        "--critic",
        choices=["mlp", "patch", "mix", "bottleneck", "lowrank", "hybrid", "query", "bquery"],
        default=None,
    )
    parser.add_argument("--critic_patch", type=int, default=None)
    parser.add_argument("--critic_width", type=int, default=None)
    parser.add_argument("--critic_layers", type=int, default=None)
    parser.add_argument("--critic_heads", type=int, default=None)
    parser.add_argument("--critic_tokens", type=int, default=None)
    parser.add_argument("--critic_queries", type=int, default=None)
    parser.add_argument("--critic_rank", type=int, default=None)
    parser.add_argument("--critic_score_bound", type=float, default=None)
    parser.add_argument("--d_lr", type=float, default=None)
    args = parser.parse_args(argv)
    args.until = args.steps if args.until is None else args.until
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("Invalid checkpoint name")
    if not 1 <= args.until <= args.steps or min(args.save_every, args.max_seq_len) < 1:
        parser.error("Invalid budget")
    if not 0 <= args.seed < 2**63:
        parser.error("Invalid seed")
    for key in (
        "critic_patch", "critic_width", "critic_layers", "critic_heads",
        "critic_tokens", "critic_queries", "critic_rank",
    ):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"{key} must be ≥ 1")
    if args.critic_score_bound is not None and args.critic_score_bound < 0:
        parser.error("critic_score_bound must be ≥ 0")
    if args.d_lr is not None and not (args.d_lr > 0):
        parser.error("d_lr must be > 0")
    return args


if __name__ == "__main__":
    train(parse_args())
