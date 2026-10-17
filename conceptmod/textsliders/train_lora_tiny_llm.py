#!/usr/bin/env python3
"""Report archived TinyQwen exports using the installed shared package.

The historical ``particle_bridge`` optimizer path is retired. Use the separately
reviewed shared PG0.9 worker for new training; this CLI only accepts
``--steps 0 --load_tiny_lora`` and never rewrites the loaded export.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

if TYPE_CHECKING:
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import TinyParticleSlider

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-tiny-llm.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-tiny-llm.yaml"
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report an archived tiny-LM particle export")
    p.add_argument(
        "--recipe",
        choices=["particle_bridge"],
        default="particle_bridge",
        help="Archived recipe identity; this CLI no longer trains it",
    )
    p.add_argument("--name", type=str, default="tiny-llm-particle")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--steps", type=int, default=20,
                   help="Only 0 is supported; positive legacy training is retired")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument('--pg_source_profile',choices=('release','alz-continuous-v1'),default='release',
                   help='Explicit reviewed PG source identity; no automatic profile selection')
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
        help="Load and record export provenance without hidden-delta readouts",
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
    if not scales or any(not math.isfinite(value) for value in scales):
        raise ValueError("--report_scales must contain finite scales")
    return scales


def build_backend(args: argparse.Namespace) -> TinyLLMBackend:
    from particle_sliders.tiny_llm_backend import TinyLLMBackend

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
    from particle_sliders.tiny_llm_backend import hidden_delta_metrics

    scales = parse_report_scales(getattr(args, "report_scales", "0,0.5,1"))
    out_dir = Path(save_dir) / "report"
    records: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            plus_ids = backend.encode(row["positive"]).ids
            neu_ids = backend.encode(row["neutral"]).ids
            with network.scaled(0.0):
                teacher_plus = backend.teacher_hidden(plus_ids)
                teacher_zero = backend.teacher_hidden(neu_ids)
                base_zero = backend.hidden(neu_ids)
            if any(not torch.isfinite(value).all()
                   for value in (teacher_plus, teacher_zero, base_zero)):
                raise FloatingPointError("Non-finite Tiny hidden response")
            assert torch.equal(base_zero, teacher_zero), "scale 0 must be exact base"
            for scale in scales:
                with network.scaled(float(scale)):
                    student_plus = backend.hidden(plus_ids)
                    student_zero = backend.hidden(neu_ids)
                if any(not torch.isfinite(value).all() for value in (student_plus, student_zero)):
                    raise FloatingPointError("Non-finite Tiny hidden response")
                metrics = hidden_delta_metrics(
                    student_plus, student_zero, teacher_plus, teacher_zero
                )
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Non-finite Tiny hidden-delta metric")
                records.append({
                    "row": i,
                    "positive": row["positive"],
                    "neutral": row["neutral"],
                    "scale": float(scale),
                    "shared_prefix_tokens": _shared_prefix_len(plus_ids, neu_ids),
                    **metrics,
                })
    payload = json.dumps(records, indent=2, allow_nan=False)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "hidden_delta.json").write_text(payload)
    print(f"wrote {len(records)} hidden-delta rows under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: TinyLLMBackend | None = None, observer=None) -> dict:
    """Retired training entry point; retain explicit archived-export reports only."""
    if type(args.steps) is not int or args.steps != 0:
        raise ValueError(
            "Legacy tiny-LM particle_bridge training is retired; use the reviewed "
            "shared-package PG0.9 worker. This CLI requires --steps 0 --load_tiny_lora."
        )
    if observer is not None:
        raise ValueError("Observers/resume are unsupported for the inference-only tiny-LM CLI")
    if not getattr(args, "load_tiny_lora", None):
        raise ValueError("--steps 0 requires --load_tiny_lora with an archived export")
    if args.recipe != "particle_bridge":
        raise ValueError("Archived tiny-LM exports use --recipe particle_bridge")

    from conceptmod.textsliders.legacy_inference import (
        inference_receipt, report_destination, resolve_export, reporting_operation, source_digest,
    )

    source = resolve_export(args.load_tiny_lora, label="--load_tiny_lora")
    source_sha256 = source_digest(source)
    save_dir, receipt_path = report_destination(
        args.save_dir or f"models/{args.name}", args.name, source
    )
    from conceptmod.textsliders.text_package import require_text_package
    package_identity = require_text_package(pg_source_profile=args.pg_source_profile)
    # Validate report options before constructing a host or reading prompt rows.
    if not getattr(args, "no_report", False):
        parse_report_scales(getattr(args, "report_scales", "0,0.5,1"))
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = json.loads((handle.metadata() or {}).get("conceptmod", "{}"))
    if not isinstance(metadata, dict):
        raise ValueError("Tiny export metadata must be a mapping")
    json.dumps(metadata, allow_nan=False)
    protocol_loader = None
    if "protocol" in metadata:
        from conceptmod.textsliders.tiny_llm_pg09 import PROTOCOL, load_export

        if metadata["protocol"] != PROTOCOL:
            raise ValueError("Unsupported Tiny export protocol; archived PG states cannot be reinterpreted")
        protocol_loader = load_export
    from particle_sliders.tiny_llm_backend import TinyLLMBackend
    from particle_sliders.tiny_native import TinyParticleSlider

    if backend is not None and type(backend) is not TinyLLMBackend:
        raise ValueError("Report-only loading requires the installed package's TinyLLMBackend")
    base_identity_verified = False
    # Loading the archived adapter must not consume a caller's CPU training RNG.
    with torch.random.fork_rng(devices=[]):
        if backend is None:
            torch.manual_seed(int(args.seed))
            backend = build_backend(args)
        if protocol_loader is not None:
            network, metadata = protocol_loader(backend, source,pg_source_profile=args.pg_source_profile)
            base_identity_verified = True
        else:
            if "model_fingerprint" in metadata:
                from conceptmod.textsliders.tiny_llm_pg09 import tensor_digest

                if tensor_digest(backend.model.state_dict()) != metadata["model_fingerprint"]:
                    raise ValueError("Frozen base fingerprint differs from archived export")
                base_identity_verified = True
            network, metadata = TinyParticleSlider.load(
                backend.model, source, adapter_dtype=torch.float32
            )
    receipt = inference_receipt(args, source, metadata, family="tiny_llm",
                                expected_sha256=source_sha256)
    receipt.update(base_identity_verified=base_identity_verified,
                   protocol_validated=protocol_loader is not None,package_identity=package_identity)
    with reporting_operation(receipt_path, receipt):
        if not getattr(args, "no_report", False):
            rows = load_slider_rows(args.prompts_file)
            records = emit_report(backend, network, args, save_dir, rows)
            receipt['report'] = dict(method="hidden_delta", scales=parse_report_scales(args.report_scales),
                                     n=len(records), qualification="diagnostic only")
    print(f"wrote inference-only receipt {receipt_path}")
    return receipt


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return train(args)


def _flatten_config(config: dict) -> dict:
    """Map the YAML card onto CLI names.

    Historical budget fields are parsed for a clear retirement error. The
    remaining card fields select an archived export report identity and prompts.
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
