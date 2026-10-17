#!/usr/bin/env python3
"""Fail-closed MiniMax-H3 entry point, retaining CPU diagnostic fixtures.

H3 has no admitted shared ParticleGAN response/critic protocol. The historical
UNI/MSE optimizer and unbound sample-only route are removed. Parsing old flags
is retained so stale commands fail with the actual migration gates, even when
PyTorch, model code and Hub dependencies are not installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Historical parser defaults only. No native backend is imported for rejection.
DEFAULT_MODEL = "MiniMaxAI/MiniMax-H3"
DEFAULT_VARIANT = "FL2VA"
DEFAULT_WORKFLOW = "t2va"
DEFAULT_LORA_UP_INIT_STD = 0.02
DEFAULT_SAMPLE_DURATION = 5.0
H3_FPS = 24.0
DEFAULT_HOLD_MODE = "non_concept"
HOLD_MODES = ("non_concept", "attributes")
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-minimax-h3.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-minimax-h3.yaml"

H3_MIGRATION_BLOCKED = (
    "MiniMax-H3 execution is closed: no reviewed native jointly packed AV "
    "response/critic protocol is admitted for the shared ParticleGAN API. "
    "Exact staged source/assets/variant, the applicable MiniMax-H3 community "
    "license, measured hardware fit and coordinator runtime admission remain "
    "pending. The legacy supervised UNI/MSE trainer, adapter-only resume and "
    "unbound inference/sample export are unavailable. CPU architecture "
    "diagnostics remain available with --dummy --diag; they do not qualify "
    "native training, inference, rendering or quality."
)


class H3MigrationBlocked(RuntimeError):
    """The required native H3 protocol and runtime admission do not exist yet."""


def require_h3_cpu_fixture(dummy: bool) -> None:
    """Reject native execution before loading components or mutating state."""
    if dummy is not True:
        raise H3MigrationBlocked(H3_MIGRATION_BLOCKED)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniMax-H3 migration gate; CPU diagnostics only")
    p.add_argument("--name", type=str, default="age-minimax-h3-uni")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=["FL2VA", "Ref2VA"])
    p.add_argument(
        "--workflow",
        type=str,
        default=DEFAULT_WORKFLOW,
        choices=["t2va", "fl2va", "ref2va"],
        help="Historical native workflow identity; native execution is closed.",
    )
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--attributes",
        type=str,
        default="",
        help="comma-separated unused attributes pinned on both poles",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument(
        "--lora_up_init_std",
        type=float,
        default=DEFAULT_LORA_UP_INIT_STD,
        help=(
            "CPU fixture LoRA-up initialization standard deviation (default 0.02). "
            "This does not enable training."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--encoder_device",
        type=str,
        default=None,
        help=(
            "Historical placement argument; ignored by CPU diagnostics. "
            "Native hardware placement is not admitted."
        ),
    )
    p.add_argument("--short_side", type=int, default=768)
    p.add_argument("--guidance", type=float, default=0.0, help="CFG-distilled: stay 0")
    p.add_argument("--hold_weight", type=float, default=1.0)
    p.add_argument(
        "--hold_mode",
        type=str,
        default=DEFAULT_HOLD_MODE,
        choices=list(HOLD_MODES),
        help=(
            "Token hold. non_concept (default): pin every plus token that "
            "is not a concept word (ids in plus but not in neu) to the "
            "matching encode(neu) row — Music 3 lyric-hold analog. "
            "attributes: old v1/v2 leak — only yaml unused attributes "
            "(male/female); shared subject tokens stay free and Omni LoRA "
            "rewrites identity (clothes/props/vase)."
        ),
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_h3_lora",
        type=str,
        default=None,
        help="CPU diagnostic fixture adapter only; training resume and inference are closed",
    )
    p.add_argument(
        "--no_sample",
        action="store_true",
        help="Historical argument; training and sample export are closed",
    )
    p.add_argument(
        "--sample_scales",
        type=str,
        default="0,1",
        help="LoRA scales for t2va clips, comma-separated (default 0,1; add 0.5)",
    )
    p.add_argument(
        "--sample_duration",
        type=float,
        default=DEFAULT_SAMPLE_DURATION,
        help="Seconds of t2va (default 5; H3 snaps to 17n+5 frames, min ~5s)",
    )
    p.add_argument(
        "--sample_fps",
        type=float,
        default=H3_FPS,
        help="Output fps (H3 clock is 24)",
    )
    p.add_argument(
        "--sample_short_side",
        type=int,
        default=None,
        help="Sample canvas short side (default: --short_side, 768 → 1344x768)",
    )
    p.add_argument(
        "--sample_max_rows",
        type=int,
        default=None,
        help="Cap unique yaml targets sampled (default: all)",
    )
    p.add_argument("--sample_seed", type=int, default=None)
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU architecture fixture; only --dummy --diag is executable",
    )
    p.add_argument(
        "--diag",
        action="store_true",
        help=(
            "CPU architecture diagnostic only; requires --dummy. "
            "Writes {name}_diag.json under --save_dir; no native qualification."
        ),
    )
    return p.parse_args(argv)


def load_slider_rows(prompts_file: str, cli_attributes: str) -> list[dict]:
    import yaml

    from conceptmod.textsliders.minimax_h3_uni import pin_unused_attributes

    raw = yaml.safe_load(Path(prompts_file).read_text()) or []
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("rows") or [raw]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = str(item.get("positive") or item.get("target") or "")
        neu = str(item.get("neutral") or item.get("target") or "")
        attrs = item.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [a.strip() for a in attrs.split(",") if a.strip()]
        if cli_attributes:
            attrs = list(attrs) + [a.strip() for a in cli_attributes.split(",") if a.strip()]
        seen: set[str] = set()
        uniq = []
        for a in attrs:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        for pos_p, neu_p in pin_unused_attributes(pos, neu, uniq):
            rows.append({
                "positive": pos_p,
                "neutral": neu_p,
                "unconditional": str(item.get("unconditional") or item.get("negative") or ""),
                "target": str(item.get("target") or neu),
                "attributes": uniq,
            })
    if not rows:
        raise ValueError(f"no slider rows in {prompts_file}")
    return rows


def parse_sample_scales(text: str) -> list[float]:
    scales = [float(x) for x in str(text).split(",") if str(x).strip()]
    if not scales:
        raise ValueError(" --sample_scales is empty")
    return scales


def sample_prompts_from_rows(rows: list[dict], max_rows: int | None = None) -> list[str]:
    """Unique yaml ``target`` (subject, no plus lighting) for the scale grid.

    This is the neu subject, not the plus caption. Plus-oracle clips
    (plus lighting at scale 0 as a side-by-side reference) are not
    emitted here.
    """
    seen: list[str] = []
    for row in rows:
        target = str(row.get("target") or row.get("neutral") or "").strip()
        if target and target not in seen:
            seen.append(target)
    if max_rows is not None:
        seen = seen[: max(0, int(max_rows))]
    if not seen:
        raise ValueError("no sample prompts in slider rows")
    return seen


def build_backend(args: argparse.Namespace):
    """Build only the CPU diagnostic fixture; never load native components."""
    require_h3_cpu_fixture(getattr(args, "dummy", False))
    from conceptmod.textsliders.minimax_h3_backend import MiniMaxH3Backend

    return MiniMaxH3Backend(
        device="cpu", encoder_device=None, model_id=args.model_id,
        variant=args.variant, workflow=args.workflow,
        short_side=min(int(args.short_side), 32), lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_up_init_std=float(getattr(args, "lora_up_init_std", DEFAULT_LORA_UP_INIT_STD)),
        dummy=True,
    )


def write_h3_sample_mp4(*args, **kwargs) -> None:
    """Closed: a failed codec must never be replaced by a fake MP4 container."""
    raise H3MigrationBlocked(H3_MIGRATION_BLOCKED)


def emit_h3_samples(*args, **kwargs):
    """Closed until native AV rendering and artifact binding are reviewed."""
    raise H3MigrationBlocked(H3_MIGRATION_BLOCKED)


def train(args: argparse.Namespace, backend=None) -> dict:
    """Closed for every backend, step count and adapter-only load option."""
    raise H3MigrationBlocked(H3_MIGRATION_BLOCKED)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.diag:
        require_h3_cpu_fixture(args.dummy)
        from conceptmod.textsliders.diag_minimax_h3_uni import run_diag

        return run_diag(args)
    return train(args)


if __name__ == "__main__":
    try:
        main()
    except H3MigrationBlocked as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
