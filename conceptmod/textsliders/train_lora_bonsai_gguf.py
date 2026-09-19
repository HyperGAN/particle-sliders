#!/usr/bin/env python3
"""Opt-in Bonsai-GGUF routed-particle slider trainer (YuE2 working recipe).

Test/smoke target for slider ideas against Ternary-Bonsai-2-27B-PTQ1_0
without loading Music 3, YuE2, H3 or a big transformers LM. Runs YuE2's
CURRENT working game ``anneal-routed-particle-error`` via the shared
``particle_bridge_gan`` module — Rp paired-error GAN on
``e = T(student) - T(positive)`` plus particle VIC — on frozen
fork-server last-token readouts. Default Music 3 trainers are unchanged
(``--lm_target v9`` / ``--pole_mode hidden``).

Pinned weights: ``prism-ml/Ternary-Bonsai-2-27B-gguf`` /
``Ternary-Bonsai-2-27B-PTQ1_0.gguf`` (5,946,648,928 bytes). The F16 pack
has no code path here and is never downloaded. Stock llama.cpp is
rejected: only the ``PrismML-Eng/llama.cpp`` fork can open PTQ1_0 (run
source of truth: ``PrismML-Eng/Bonsai-demo``).

``--dummy`` is the CI / CPU path: seeded readouts of width 5120, no
weights, no binary, no server. Live runs need the fork ``llama-server``
on the pinned file (see docs/bonsai-gguf-slider.md) plus
``--server_url``. Full LoRA-in-GGUF training is impossible (frozen
inference-only ternary; the fork has no gradient path), so the trained
slider is a torch-side residual head on the readout (scale 0 bypasses
exactly), not in-attention LoRA.

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
from conceptmod.textsliders.bonsai_gguf_backend import (
    BASE_MODEL,
    BonsaiGGUFBackend,
    CONTEXT,
    FILE_BYTES,
    FILENAME,
    HIDDEN,
    LAYERS,
    MODEL_LICENSE,
    REPO,
    download_weights,
    resolve_weights,
)
from conceptmod.textsliders.bonsai_gguf_particle import (
    FORMAT,
    RECIPE,
    BonsaiParticleHead,
    build_game,
    prepare_rows,
    resolve_head_path,
    update,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-bonsai-gguf.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-bonsai-gguf.yaml"
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Opt-in Bonsai-GGUF particle trainer")
    p.add_argument(
        "--recipe",
        choices=["particle_bridge"],
        default="particle_bridge",
        help="Only the YuE2 working game (choices rejects anything else)",
    )
    p.add_argument("--name", type=str, default="bonsai-gguf-particle")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to the pinned PTQ1_0 file (live path)")
    p.add_argument("--server_url", type=str, default=None,
                   help="Fork llama-server base URL (live path)")
    p.add_argument("--allow_download", action="store_true",
                   help="Allow downloading ONLY the pinned PTQ1_0 file")
    p.add_argument("--weights_dir", type=str, default="models/bonsai-gguf",
                   help="Download dir for --allow_download")
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_head",
        type=str,
        default=None,
        help="Dir or .safetensors with a Bonsai particle head",
    )
    p.add_argument(
        "--no_report",
        action="store_true",
        help="Skip the readout-delta report after train / load",
    )
    p.add_argument(
        "--report_scales",
        type=str,
        default="0,0.5,1",
        help="Head scales for the readout-delta report, comma-separated",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Seeded CPU stand-in readouts; no weights, no fork, no server",
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


def build_backend(args: argparse.Namespace) -> BonsaiGGUFBackend:
    if args.dummy:
        return BonsaiGGUFBackend(device="cpu", dummy=True)
    weights = args.weights
    if weights is None and args.allow_download:
        weights = download_weights(args.weights_dir, allow_download=True)
    if weights is None:
        raise ValueError(
            "live Bonsai readout needs --weights (pinned "
            f"{FILENAME}) or --allow_download, or pass --dummy; "
            "see docs/bonsai-gguf-slider.md"
        )
    resolve_weights(weights)
    return BonsaiGGUFBackend(
        device=args.device,
        weights=weights,
        server_url=args.server_url,
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
    backend: BonsaiGGUFBackend,
    network: BonsaiParticleHead,
    args: argparse.Namespace,
    save_dir: Path,
    rows: list[dict],
) -> list[dict]:
    """Readout-delta diagnostic at each scale (not part of the game)."""
    scales = parse_report_scales(getattr(args, "report_scales", "0,0.5,1"))
    out_dir = Path(save_dir) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            plus = backend.encode(row["positive"])
            neu = backend.encode(row["neutral"])
            teacher_plus = backend.teacher_hidden(plus)
            teacher_zero = backend.teacher_hidden(neu)
            with network.scaled(0.0):
                base_zero = network(backend.hidden(neu))
            assert torch.equal(base_zero, teacher_zero), "scale 0 must be exact base"
            for scale in scales:
                with network.scaled(float(scale)):
                    student_plus = network(backend.hidden(plus))
                    student_zero = network(backend.hidden(neu))
                s_delta = (student_plus - student_zero).float().reshape(-1)
                t_delta = (teacher_plus - teacher_zero).float().reshape(-1)
                cos = torch.nn.functional.cosine_similarity(
                    s_delta, t_delta, dim=0).item()
                l2 = float(torch.linalg.vector_norm(s_delta - t_delta).item())
                t_norm = float(torch.linalg.vector_norm(t_delta).item())
                records.append({
                    "row": i,
                    "positive": row["positive"],
                    "neutral": row["neutral"],
                    "scale": float(scale),
                    "shared_prefix_tokens": _shared_prefix_len(plus.ids, neu.ids),
                    "delta_cos": cos,
                    "delta_l2": l2,
                    "teacher_delta_norm": t_norm,
                    "rel_l2": l2 / max(t_norm, 1e-8),
                })
    (out_dir / "readout_delta.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} readout-delta rows under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: BonsaiGGUFBackend | None = None) -> dict:
    if args.recipe != "particle_bridge":
        raise ValueError("bonsai-gguf only runs --recipe particle_bridge")
    rows = load_slider_rows(args.prompts_file)
    backend = backend or build_backend(args)
    torch.manual_seed(int(args.seed))
    fixed = prepare_rows(backend, rows)
    loaded_head = None
    if getattr(args, "load_head", None):
        resolved = resolve_head_path(args.load_head)
        network, _ = BonsaiParticleHead.load(resolved)
        loaded_head = str(resolved)
        print(f"loaded Bonsai particle head from {loaded_head}")
    else:
        network = BonsaiParticleHead()
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
                f"bonsai-gguf particle step {step}: "
                f"g_adv={rec['g_adv']:.4f} vic={rec['particle_vic']:.4f} "
                f"d_loss={rec['d_loss']:.4f} cos_pos={rec['cos_pos']:.4f} "
                f"sigma={rec['noise_std']:.4f}"
            )

    ref = shared.REFERENCE
    sidecar = {
        "name": args.name,
        "backend": "bonsai_gguf",
        "recipe": RECIPE["name"],
        "source_recipe": RECIPE["source_recipe"],
        "game_module": "conceptmod.textsliders.particle_bridge_gan",
        "model_repo": REPO,
        "model_file": FILENAME,
        "model_file_bytes": FILE_BYTES,
        "base_model": BASE_MODEL,
        "model_arch": "qwen35",
        "model_hidden": HIDDEN,
        "model_layers": LAYERS,
        "model_context": CONTEXT,
        "model_license": MODEL_LICENSE,
        "format": FORMAT,
        "stack": "gguf_readout",
        "plus_neu": True,
        "minus_teacher": False,
        "unipolar": True,
        "output_mse": False,
        "lora_only": False,
        "adapter_placement": "readout_residual_zero_init",
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
        "load_head": loaded_head,
        "dummy": bool(args.dummy),
        "history": history[-8:],
    }
    save_dir = Path(args.save_dir or f"models/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = save_dir / f"{args.name}_last.json"
    record = dict(sidecar, step=int(args.steps))
    _save_state_as(network, ema, save_dir / f"{args.name}_last.safetensors", record)
    network.save(save_dir / f"{args.name}_live_last.safetensors",
                 dict(record, weights_kind="live"))
    report_records: list[dict] = []
    if not getattr(args, "no_report", False):
        report_records = emit_report(backend, network, args, save_dir, rows)
        sidecar["report"] = {
            "method": "readout_delta",
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
        hidden=HIDDEN, particles=128, particle_dim=4,
        bridge_width=48, router_width=16,
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
    # Identity (repo/file) is pinned in code, never configurable.
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
    if config.get("server_url"):
        flat["server_url"] = config["server_url"]
    if config.get("weights"):
        flat["weights"] = config["weights"]
    return flat


if __name__ == "__main__":
    main()
