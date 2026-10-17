#!/usr/bin/env python3
"""Report archived Bonsai readout heads through the installed shared package.

Positive-step legacy particle_bridge training is retired. Use an explicitly
reviewed shared ConditionalPG09 worker for a new cohort; this CLI never restarts
Bonsai's failed held-out recipe. --steps 0 --load_head performs inference only,
retaining the input artifact and its original metadata without re-exporting it.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.bonsai_gguf_backend import (
    BonsaiGGUFBackend,
    FILENAME,
    download_weights,
    resolve_weights,
)
from conceptmod.textsliders.legacy_inference import (
    resolve_export, report_destination, inference_receipt, reporting_operation, source_digest,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-bonsai-gguf.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-bonsai-gguf.yaml"
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference-only Bonsai shared-package head reports")
    p.add_argument(
        "--recipe",
        choices=["particle_bridge"],
        default="particle_bridge",
        help="Historical recipe label only; positive-step legacy training is retired",
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
    p.add_argument('--pg_source_profile',choices=('release','alz-continuous-v1'),default='release',
                   help='Explicit reviewed PG source identity; no automatic profile selection')
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_head",
        type=str,
        default=None,
        help="Explicit .safetensors export (or a directory with exactly one export)",
    )
    p.add_argument(
        "--no_report",
        action="store_true",
        help="Load/validate and record provenance without fetching readouts",
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
    if not scales or any(not isfinite(value) for value in scales):
        raise ValueError("--report_scales must contain finite scales")
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
    records: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            plus = backend.encode(row["positive"])
            neu = backend.encode(row["neutral"])
            teacher_plus = backend.teacher_hidden(plus)
            teacher_zero = backend.teacher_hidden(neu)
            if any(not torch.isfinite(value).all() for value in (teacher_plus, teacher_zero)):
                raise FloatingPointError('nonfinite frozen Bonsai report readout')
            with network.scaled(0.0):
                base_zero = network(backend.hidden(neu))
            assert torch.equal(base_zero, teacher_zero), "scale 0 must be exact base"
            for scale in scales:
                with network.scaled(float(scale)):
                    student_plus = network(backend.hidden(plus))
                    student_zero = network(backend.hidden(neu))
                if any(not torch.isfinite(value).all() for value in (student_plus, student_zero)):
                    raise FloatingPointError('nonfinite Bonsai report response')
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
    payload = json.dumps(records, indent=2, allow_nan=False)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "readout_delta.json").write_text(payload)
    print(f"wrote {len(records)} readout-delta rows under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: BonsaiGGUFBackend | None = None, observer=None) -> dict:
    # Gate before prompts, model/download/readout access, output or optimizer work.
    if type(args.steps) is not int or args.steps < 0:
        raise ValueError("--steps must be a nonnegative integer")
    if args.steps:
        raise ValueError("legacy Bonsai particle_bridge training is retired; use an explicitly "
                         "reviewed particle_sliders.pg09.ConditionalPG09 worker/new cohort; "
                         "this CLI supports only --steps 0 --load_head inference")
    if observer is not None:
        raise ValueError("legacy training observers cannot run in inference-only mode")
    if not getattr(args, "load_head", None):
        raise ValueError("--steps 0 requires --load_head with an existing export")
    parse_report_scales(args.report_scales)
    source = resolve_export(args.load_head, label="--load_head")
    source_sha256 = source_digest(source)
    directory, receipt_path = report_destination(
        args.save_dir or f"models/{args.name}", args.name, source)
    from conceptmod.textsliders.text_package import require_text_package
    package_identity = require_text_package(pg_source_profile=args.pg_source_profile)
    from particle_sliders.bonsai_native import BonsaiParticleHead
    network, metadata = BonsaiParticleHead.load(source)
    json.dumps(metadata, allow_nan=False)
    receipt = inference_receipt(args, source, metadata, family="bonsai_gguf",
                                expected_sha256=source_sha256)
    receipt["head_class"] = type(network).__module__ + "." + type(network).__name__
    receipt["load_head"] = str(source)
    receipt["base_identity_verified"] = False
    receipt['package_identity'] = package_identity
    with reporting_operation(receipt_path, receipt):
        if not getattr(args, "no_report", False):
            rows = load_slider_rows(args.prompts_file)
            backend = backend or build_backend(args)
            network.to(device=backend.device, dtype=torch.float32)
            records = emit_report(backend, network, args, directory, rows)
            receipt["report"] = dict(method="readout_delta", scales=parse_report_scales(args.report_scales),
                                     n=len(records), interpretation="diagnostic only; no transfer qualification")
    print(f"wrote {receipt_path}")
    return receipt


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return train(args)


def _flatten_config(config: dict) -> dict:
    """Map the YAML card onto CLI names.

    Historical positive-step cards are parsed for explicit retirement errors.
    Reporting reads identity/prompts without constructing a training game.
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
