#!/usr/bin/env python3
"""Opt-in tiny-LM routed-particle slider trainer (YuE2 working recipe).

Test target for slider ideas without loading Music 3, YuE2, H3 or a big LM.
Runs YuE2's CURRENT working game ``anneal-routed-particle-error`` via the
shared ``particle_bridge_gan`` module — Rp paired-error GAN on
``e = T(student) - T(positive)`` plus particle VIC — on Qwen3-0.6B-Base
last-hidden states. Default Music 3 trainers are unchanged
(``--lm_target v9`` / ``--pole_mode hidden``).

Pinned model: ``Qwen/Qwen3-0.6B-Base`` (596M params, 2025, Apache-2.0).
``--dummy`` is the CI / CPU path: tiny random stand-in of the same config
shape, no Hub, no GPU. Live runs need ``transformers>=4.51`` and pass
``--allow_hub`` once to download the weights.

Game numbers are read from ``particle_bridge_gan.REFERENCE``, never copied:
G/D/particle LR 0.0006/0.0009/0.006, Adam betas (0, 0.999), constant,
128x4 cloud, VIC coeff 1, lazy b_cap every 4th update x4, EMA 0.995, noise
anneal 0.03 over 8000 (never compressed to the step budget). No output MSE,
FM, ending, hold, or anchor. Unipolar (+ vs raw positive); the
``unconditional`` prompt row is an unscored canary. Propose-only: does not
flip Music ARM_B, live ``--lm_target``, locked AdvConfig, or YuE2 defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.tiny_llm_backend import (
    DEFAULT_MODEL,
    MODEL_LICENSE,
    MODEL_PARAMS_TOTAL,
    MODEL_RELEASE,
    TinyLLMBackend,
    hidden_delta_metrics,
)
from conceptmod.textsliders.tiny_llm_particle import (
    FORMAT,
    RECIPE,
    TinyParticleSlider,
    build_game,
    prepare_rows,
    resolve_particle_path,
    update,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-tiny-llm.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-tiny-llm.yaml"
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opt-in tiny-LM particle-bridge trainer")
    p.add_argument(
        "--recipe",
        choices=["particle_bridge"],
        default="particle_bridge",
        help="Only the YuE2 working game (choices rejects anything else)",
    )
    p.add_argument("--name", type=str, default="tiny-llm-particle")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--allow_hub",
        action="store_true",
        help="Allow downloading the pinned weights (live path only)",
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_tiny_lora",
        type=str,
        default=None,
        help="Dir or .safetensors with tiny particle lora_tiny-* keys",
    )
    p.add_argument(
        "--no_report",
        action="store_true",
        help="Skip the hidden-delta report after train / load",
    )
    p.add_argument(
        "--report_scales",
        type=str,
        default="0,0.5,1",
        help="LoRA scales for the hidden-delta report, comma-separated",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock causal LM; no Hub, no GPU, no Qwen weights",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _make_parser()
    initial, _ = parser.parse_known_args(argv)
    if initial.config_file:
        config = yaml.safe_load(Path(initial.config_file).read_text())
        if not isinstance(config, dict):
            raise ValueError("config_file must be a mapping")
        flat = _flatten_config(config)
        allowed = {a.dest for a in parser._actions} - {"help", "config_file"}
        unknown = set(flat) - allowed
        if unknown:
            parser.error(f"config_file has unsupported keys: {sorted(unknown)}")
        parser.set_defaults(**flat)
    return parser.parse_args(argv)


def load_slider_rows(prompts_file: str) -> list[dict]:
    raw = yaml.safe_load(Path(prompts_file).read_text()) or []
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("rows") or [raw]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = str(item.get("positive") or "")
        neu = str(item.get("neutral") or "")
        if not pos or not neu or pos == neu:
            raise ValueError(f"row needs distinct positive/neutral: {item!r}")
        rows.append({
            "positive": pos,
            "neutral": neu,
            "unconditional": str(item.get("unconditional") or ""),
            "target": str(item.get("target") or neu),
        })
    if len(rows) < 2:
        raise ValueError(
            f"particle bridge needs at least two prompt rows ({prompts_file})"
        )
    return rows


def parse_report_scales(text: str) -> list[float]:
    scales = [float(x) for x in str(text).split(",") if str(x).strip()]
    if not scales:
        raise ValueError("--report_scales is empty")
    return scales


def build_backend(args: argparse.Namespace) -> TinyLLMBackend:
    if args.dummy:
        return TinyLLMBackend(device="cpu", model_id=args.model_id, dummy=True)
    return TinyLLMBackend(
        device=args.device,
        model_id=args.model_id,
        allow_hub=bool(args.allow_hub),
        dummy=False,
    )


def _shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def emit_report(
    backend: TinyLLMBackend,
    network: TinyParticleSlider,
    args: argparse.Namespace,
    save_dir: Path,
    rows: list[dict],
) -> list[dict]:
    """Hidden-delta diagnostic at each scale (not part of the game)."""
    scales = parse_report_scales(getattr(args, "report_scales", "0,0.5,1"))
    out_dir = Path(save_dir) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            plus_ids = backend.encode(row["positive"]).ids
            neu_ids = backend.encode(row["neutral"]).ids
            teacher_plus = backend.teacher_hidden(plus_ids)
            teacher_zero = backend.teacher_hidden(neu_ids)
            with network.scaled(0.0):
                base_zero = backend.hidden(neu_ids)
            assert torch.allclose(base_zero, teacher_zero), "scale 0 must be exact base"
            for scale in scales:
                with network.scaled(float(scale)):
                    student_plus = backend.hidden(plus_ids)
                    student_zero = backend.hidden(neu_ids)
                metrics = hidden_delta_metrics(
                    student_plus, student_zero, teacher_plus, teacher_zero
                )
                records.append({
                    "row": i,
                    "positive": row["positive"],
                    "neutral": row["neutral"],
                    "scale": float(scale),
                    "shared_prefix_tokens": _shared_prefix_len(plus_ids, neu_ids),
                    **metrics,
                })
    (out_dir / "hidden_delta.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} hidden-delta rows under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: TinyLLMBackend | None = None) -> dict:
    if args.recipe != "particle_bridge":
        raise ValueError("tiny-llm only runs --recipe particle_bridge")
    rows = load_slider_rows(args.prompts_file)
    backend = backend or build_backend(args)
    torch.manual_seed(int(args.seed))
    fixed = prepare_rows(backend, rows)
    loaded_lora = None
    if getattr(args, "load_tiny_lora", None):
        resolved = resolve_particle_path(args.load_tiny_lora)
        network, _ = TinyParticleSlider.load(backend.model, resolved)
        loaded_lora = str(resolved)
        print(f"loaded tiny particle weights from {loaded_lora}")
    else:
        network = TinyParticleSlider(backend.model, rank=8, alpha=8.0)
    critic, g, d = build_game(backend, network, fixed)
    sampler = shared.BridgeSampler(len(rows), int(args.seed))
    ema = shared.initialize_ema(network)

    history = []
    for update_index in range(int(args.steps)):
        step = update_index + 1  # noise anneal is 1-indexed like the YuE2 loop
        metrics = update(
            backend, network, critic, g, d, fixed,
            sampler=sampler, step=step,
        )
        shared.update_ema(ema, network)
        rec = dict(metrics, step=step)
        history.append(rec)
        if step == 1 or step % 10 == 0 or step == int(args.steps):
            print(
                f"tiny-llm particle step {step}: "
                f"g_adv={rec['g_adv']:.4f} vic={rec['particle_vic']:.4f} "
                f"d_loss={rec['d_loss']:.4f} cos_pos={rec['cos_pos']:.4f} "
                f"sigma={rec['noise_std']:.4f}"
            )

    ref = shared.REFERENCE
    sidecar = {
        "name": args.name,
        "backend": "tiny_llm",
        "recipe": RECIPE["name"],
        "source_recipe": RECIPE["source_recipe"],
        "game_module": "conceptmod.textsliders.particle_bridge_gan",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "model_params_total": MODEL_PARAMS_TOTAL,
        "model_release": MODEL_RELEASE,
        "model_license": MODEL_LICENSE,
        "format": FORMAT,
        "stack": "causal_lm_hidden",
        "plus_neu": True,
        "minus_teacher": False,
        "unipolar": True,
        "output_mse": False,
        "lora_only": True,
        "lora_linears": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "rank": 8,
        "alpha": 8.0,
        "particles": [128, 4],
        "g_lr": ref["g_lr"],
        "d_lr": ref["d_lr"],
        "particle_lr": ref["particle_lr"],
        "optimizer_betas": list(ref["betas"]),
        "schedule": "constant",
        "ema": ref["ema"],
        "vic_coeff": ref["vic_coeff"],
        "cap_coeff": ref["cap_coeff"],
        "cap_kappa": ref["cap_kappa"],
        "cap_every": ref["cap_every"],
        "noise_floor": ref["noise_floor"],
        "noise_decay_steps": ref["noise_decay_steps"],
        "propose_only": True,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device if not args.dummy else "cpu",
        "load_tiny_lora": loaded_lora,
        "dummy": bool(args.dummy),
        "history": history[-8:],
    }
    save_dir = Path(args.save_dir or f"models/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = save_dir / f"{args.name}_last.json"
    record = dict(sidecar, step=int(args.steps))
    # `_last` is EMA (final weights, like the YuE2 campaign); `_live_*`
    # preserves raw weights. Both go through the same format validation.
    _save_state_as(network, ema, save_dir / f"{args.name}_last.safetensors", record)
    network.save(save_dir / f"{args.name}_live_last.safetensors",
                 dict(record, weights_kind="live"))
    report_records: list[dict] = []
    if not getattr(args, "no_report", False):
        report_records = emit_report(backend, network, args, save_dir, rows)
        sidecar["report"] = {
            "method": "hidden_delta",
            "scales": parse_report_scales(getattr(args, "report_scales", "0,0.5,1")),
            "n": len(report_records),
        }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {sidecar_path}")
    return sidecar


def _save_state_as(network, state, path, record) -> None:
    """Export EMA weights through the same format/validation as live saves."""
    from safetensors.torch import save_file

    if set(state) != set(network.state_dict()):
        raise ValueError("EMA state does not match the particle network")
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Non-finite EMA weights")
    stamped = dict(
        record, format=FORMAT, rank=network.rank, alpha=network.alpha,
        targets=network.target_names,
        particles=128, particle_dim=4, bridge_width=48, router_width=16,
    )
    save_file(
        {k: v.detach().float().cpu().contiguous() for k, v in state.items()},
        str(path), metadata={"conceptmod": json.dumps(stamped, sort_keys=True)},
    )
    Path(str(path)).with_suffix(".json").write_text(json.dumps(stamped, indent=2) + "\n")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return train(args)


def _flatten_config(config: dict) -> dict:
    """Map the YAML card onto CLI names.

    Game numbers (LRs, cloud, VIC, noise, EMA) are pinned in
    ``particle_bridge_gan.REFERENCE`` and are NOT configurable here; the
    card only carries identity + budget + prompts.
    """
    flat: dict = {}
    model = config.get("pretrained_model") or {}
    if isinstance(model, dict) and model.get("name_or_path"):
        flat["model_id"] = model["name_or_path"]
    train_cfg = config.get("train") or {}
    if isinstance(train_cfg, dict):
        for key in ("steps", "save_every", "seed"):
            if train_cfg.get(key) is not None:
                flat[key] = train_cfg[key]
        if train_cfg.get("iterations") is not None:
            flat.setdefault("steps", train_cfg["iterations"])
        if train_cfg.get("recipe") is not None:
            flat["recipe"] = train_cfg["recipe"]
    save = config.get("save") or {}
    if isinstance(save, dict) and save.get("name"):
        flat["name"] = save["name"]
    if config.get("prompts_file"):
        flat["prompts_file"] = config["prompts_file"]
    if config.get("device"):
        flat["device"] = config["device"]
    return flat


if __name__ == "__main__":
    main()
