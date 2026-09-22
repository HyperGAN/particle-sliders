#!/usr/bin/env python3
"""Opt-in Krea2 turbo-bbox concept-slider trainer.

Base model ``jimmycarter/krea2-turbo-bbox``: a transformer-only
distilled finetune. Load

    Krea2Transformer2DModel.from_pretrained(
        model_id, subfolder="epoch-14-step-73184/transformer")

into ``Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", transformer=tf)``.

Train and sample defaults are the distilled card: **8 steps, CFG 0,
mu=1.15**. They are not the Raw card (CFG 4.5 / 28) from
``train_lora_krea.py``. Stock Raw / Turbo run files stay on that
trainer; this one refuses them.

UNI contract shared with stock Krea (where it still applies):

- student +1 → + concept velocity ``v(pos)`` (CFG off, so not the
  Raw ``v(pos) + 4.5 * (v(pos) - v(''))`` teacher)
- student scale 0 → neutral velocity
- minus is a canary only
- unused prompt tokens hold to encode(neu); concept words are not held
- bare captions (attributes are not prefixed)
- fruit-bowl control prompt is verify-only

``--dummy`` never loads Hub weights. CI uses that.
Live load is offline-safe unless ``--allow_hub``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.krea2_bbox import (
    assert_krea2_only,
    assert_krea2_skeleton,
    live_train_card,
    live_train_command,
    resolve_krea2_bbox_card,
    resolve_krea2_bbox_source,
)
from conceptmod.textsliders.slider_targets import (
    KREA2_BBOX_CFG,
    KREA2_BBOX_MODEL,
    KREA2_BBOX_MU,
    KREA2_BBOX_SKELETON,
    KREA2_BBOX_STEPS,
    KREA2_BBOX_SUBFOLDER,
    KREA_CONTROL_PROMPT,
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_EMBED_COSINE_WEIGHT,
    KREA_EMBED_LATE_LAYER_START,
    KREA_EMBED_LATE_WEIGHT,
    KREA_EMBED_REL_L2_WEIGHT,
    KREA_HOLD_WEIGHT,
    KREA_LM_TARGET_CHOICES,
    KREA_LM_TARGET_DEFAULT,
    KREA_LORA_TARGET_CHOICES,
    KREA_ORACLE_EMBED_COS,
    KREA_ORACLE_SHOTS,
    KREA_RAW_CFG,
    KREA_RAW_STEPS,
    KREA_RECIPE_CHOICES,
    KREA_RECIPE_DEFAULT,
    KREA_SAMPLE_SCALES,
    KREA_SMILE_HOLD_WEIGHT,
    KREA_TE_DIT_MASK_CHOICES,
    KREA_TE_DIT_MASK_DEFAULT,
    force_krea_embed_lora_targets,
    krea_embed_requires_te,
    krea_oracle_readout,
    resolve_krea_lm_target,
)
from conceptmod.textsliders.train_lora_krea import (
    DummyKreaBackend,
    _sample_z,
    emit_inprocess_samples,
    emit_oracle_grid,
    krea_step_loss,
    load_prompts,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-krea2-bbox.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-krea2-bbox.yaml"
DEFAULT_SAVE_DIR = Path("models/krea2-bbox-slider")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="smile-krea2-bbox")
    parser.add_argument("--rank", type=int, default=KREA_DEFAULT_RANK)
    parser.add_argument("--resolution", type=int, default=KREA_DEFAULT_RESOLUTION)
    parser.add_argument("--model_id", type=str, default=KREA2_BBOX_MODEL)
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default=KREA2_BBOX_SUBFOLDER,
        help="Hub or local-clone transformer subfolder "
        f"(default {KREA2_BBOX_SUBFOLDER})",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default=None,
        help="local Comfy .safetensors or a diffusers transformer directory. "
        "Default is the Hub id + --transformer_subfolder.",
    )
    parser.add_argument(
        "--skeleton_model",
        type=str,
        default=KREA2_BBOX_SKELETON,
        help="VAE + text encoder + scheduler. Default krea/Krea-2-Raw. "
        "The bbox repo does not ship these.",
    )
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=KREA2_BBOX_STEPS,
        help=f"distilled sample steps (default {KREA2_BBOX_STEPS}, not Raw {KREA_RAW_STEPS})",
    )
    parser.add_argument(
        "--sample_guidance",
        type=float,
        default=KREA2_BBOX_CFG,
        help=f"distilled CFG (default {KREA2_BBOX_CFG:g}, not Raw {KREA_RAW_CFG:g}). "
        "Train-time +1 teacher uses this same scale.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=KREA2_BBOX_MU,
        help=f"distilled timestep shift (default {KREA2_BBOX_MU:g}). Not a CFG scale.",
    )
    parser.add_argument(
        "--hold_weight",
        type=float,
        default=KREA_SMILE_HOLD_WEIGHT,
        help="unused-token hold → encode(neu). Default "
        f"{KREA_SMILE_HOLD_WEIGHT:g} matches the bare smile yaml. "
        f"An age-style yaml on the stock Krea trainer keeps {KREA_HOLD_WEIGHT:g}.",
    )
    parser.add_argument(
        "--lora_targets",
        type=str,
        default=KREA_DEFAULT_LORA_TARGETS,
        help="dit (default), te / text_encoder, or dit+te. "
        f"Choices: {', '.join(KREA_LORA_TARGET_CHOICES)}.",
    )
    parser.add_argument(
        "--lm_target",
        type=str,
        default=KREA_LM_TARGET_DEFAULT,
        help="v (default): velocity UNI. embed: TE-only stacked embeds. "
        f"Choices/aliases: {', '.join(KREA_LM_TARGET_CHOICES)}.",
    )
    parser.add_argument(
        "--embed_cosine_weight",
        type=float,
        default=KREA_EMBED_COSINE_WEIGHT,
    )
    parser.add_argument(
        "--embed_rel_l2_weight",
        type=float,
        default=KREA_EMBED_REL_L2_WEIGHT,
    )
    parser.add_argument(
        "--embed_late_weight",
        type=float,
        default=KREA_EMBED_LATE_WEIGHT,
    )
    parser.add_argument(
        "--embed_late_layer_start",
        type=int,
        default=KREA_EMBED_LATE_LAYER_START,
    )
    parser.add_argument(
        "--recipe",
        choices=list(KREA_RECIPE_CHOICES),
        default=KREA_RECIPE_DEFAULT,
        help="uni (default) or embed_uni (alias for --lm_target embed). "
        "Not a Raw/Turbo switch.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU backend, never loads Krea / Hub weights",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="permit Hub download of the bbox transformer and the Raw skeleton",
    )
    parser.add_argument("--control_prompt", type=str, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--load_te_lora", type=str, default=None)
    parser.add_argument(
        "--te_dit_mask",
        type=str,
        default=KREA_TE_DIT_MASK_DEFAULT,
        choices=list(KREA_TE_DIT_MASK_CHOICES),
    )
    parser.add_argument(
        "--print_card",
        action="store_true",
        help="print the distilled train card and CLI, then exit",
    )
    return parser.parse_args(argv)


def _prompts_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = _REPO_ROOT / path
    return candidate if candidate.exists() else path


def _load_live_backend(args: argparse.Namespace, device: torch.device):
    """Lazy live loader. ``--dummy`` never calls this."""
    from conceptmod.textsliders.krea2_bbox_live import load_live_krea2_backend

    return load_live_krea2_backend(args, device)


def train(args: argparse.Namespace) -> dict | Path:
    assert_krea2_only(args.model_id, getattr(args, "transformer", None))
    assert_krea2_skeleton(str(args.skeleton_model))
    if args.print_card:
        card = live_train_card(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            subfolder=args.transformer_subfolder,
            skeleton=args.skeleton_model,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            sample_guidance=float(args.sample_guidance),
            mu=float(args.mu),
            hold_weight=float(args.hold_weight),
            lora_targets=str(args.lora_targets),
        )
        print(json.dumps(card, indent=2))
        print()
        print(live_train_command(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            subfolder=args.transformer_subfolder,
            skeleton=args.skeleton_model,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            sample_guidance=float(args.sample_guidance),
            mu=float(args.mu),
            hold_weight=float(args.hold_weight),
            lora_targets=str(args.lora_targets),
            save_dir=args.save_dir,
        ))
        return card

    prompts_path = _prompts_path(Path(args.prompts_file))
    prompts, meta = load_prompts(prompts_path)
    lm_target = resolve_krea_lm_target(
        getattr(args, "lm_target", KREA_LM_TARGET_DEFAULT),
        getattr(args, "recipe", KREA_RECIPE_DEFAULT),
    )
    card = resolve_krea2_bbox_card(
        args.sample_steps,
        args.sample_guidance,
        args.mu,
    )
    weights = resolve_krea2_bbox_source(
        args.model_id,
        subfolder=args.transformer_subfolder,
        transformer=getattr(args, "transformer", None),
        skeleton=args.skeleton_model,
    )
    requested_targets = str(getattr(args, "lora_targets", KREA_DEFAULT_LORA_TARGETS))
    lora_spec = force_krea_embed_lora_targets(
        requested_targets,
        lm_target=lm_target,
        recipe=getattr(args, "recipe", None),
    )
    if lm_target == "embed":
        krea_embed_requires_te(lora_spec)
        if requested_targets not in ("te", "text_encoder", "encoder"):
            print(
                f"note: --lm_target embed forces --lora_targets te "
                f"(was {requested_targets!r}); DiT stays frozen"
            )
        args.lora_targets = lora_spec.label
        args.lm_target = lm_target

    steps = int(args.steps)
    if args.dummy:
        steps = min(steps, 2)
        device = torch.device("cpu")
        backend = DummyKreaBackend(
            dim=8,
            rank=min(2, int(args.rank)),
            seed=int(args.seed),
            lora_targets=lora_spec.label,
        )
    else:
        device = torch.device(
            f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu"
        )
        backend = _load_live_backend(args, device)

    skip_train = bool(getattr(args, "load_te_lora", None))
    if skip_train:
        if hasattr(backend, "load_te_adapter"):
            backend.load_te_adapter(args.load_te_lora)
        steps = 0

    torch.manual_seed(int(args.seed))
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    last_stats: dict[str, float] = {}
    guidance = float(card["sample_guidance"])
    control_prompt = str(
        getattr(args, "control_prompt", None) or meta.control_prompt or KREA_CONTROL_PROMPT
    )

    print(
        f"train krea2-turbo-bbox name={args.name} recipe={args.recipe} "
        f"lm_target={lm_target} rank={args.rank} res={args.resolution} "
        f"model={args.model_id} source={weights['source']} "
        f"subfolder={weights['subfolder']!r} "
        f"sample_steps={card['sample_steps']} cfg={guidance} "
        f"mu={card['mu']} dummy={bool(args.dummy)} "
        f"allow_hub={bool(args.allow_hub)} "
        f"lora_targets={lora_spec.label} hold_weight={float(args.hold_weight)} "
        f"minus_teacher=off unused_hold=on control={control_prompt!r}"
    )
    if meta.bare_captions and abs(float(args.hold_weight) - KREA_HOLD_WEIGHT) < 1e-12:
        print(
            f"note: bare-caption yaml with --hold_weight {KREA_HOLD_WEIGHT:g}; "
            f"the smile card uses {KREA_SMILE_HOLD_WEIGHT:g}"
        )

    progress = tqdm(range(steps), desc="train-krea2-bbox")
    for step in progress:
        prompt = prompts[step % len(prompts)]
        if hasattr(backend, "begin_step"):
            backend.begin_step()
        z = _sample_z(backend, device)
        loss, stats = krea_step_loss(
            backend,
            prompt,
            z,
            guidance=guidance,
            hold_weight=float(args.hold_weight),
            lm_target=lm_target,
            recipe=str(getattr(args, "recipe", KREA_RECIPE_DEFAULT)),
            embed_cosine_weight=float(args.embed_cosine_weight),
            embed_rel_l2_weight=float(args.embed_rel_l2_weight),
            embed_late_weight=float(args.embed_late_weight),
            embed_late_layer_start=int(args.embed_late_layer_start),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_stats = stats
        progress.set_postfix({"loss": f"{stats['loss']:.4f}"})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **stats}) + "\n")

    samples = emit_inprocess_samples(
        backend,
        args,
        save_dir,
        prompts,
        dummy=bool(args.dummy),
        control_prompt=control_prompt,
        card=card,
    )
    oracle = emit_oracle_grid(
        backend,
        args,
        save_dir,
        prompts,
        dummy=bool(args.dummy),
        card=card,
    )
    if hasattr(backend, "save_trained") and not args.dummy and not skip_train:
        backend.save_trained(save_dir / f"{args.name}_lora")
    dit_lora_path = None
    te_lora_path = None
    if lora_spec.train_dit and lora_spec.train_te:
        dit_lora_path = f"{args.name}_lora/dit_lora"
        te_lora_path = f"{args.name}_lora/te_lora"
    elif lora_spec.train_dit:
        dit_lora_path = f"{args.name}_lora"
    elif lora_spec.train_te:
        te_lora_path = f"{args.name}_lora/te_lora"

    sidecar = {
        "kind": "krea2_turbo_bbox",
        "recipe": "embed_uni" if lm_target == "embed" else "uni",
        "lm_target": lm_target,
        "dit_velocity_supervised": lm_target != "embed",
        "name": args.name,
        "model_id": args.model_id,
        "transformer_subfolder": args.transformer_subfolder,
        "transformer": getattr(args, "transformer", None),
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "official": (
            "train and sample on jimmycarter/krea2-turbo-bbox "
            f"({KREA2_BBOX_STEPS} steps, CFG {KREA2_BBOX_CFG:g}, "
            f"mu={KREA2_BBOX_MU:g})"
        ),
        "not_raw_card": {
            "raw_cfg": KREA_RAW_CFG,
            "raw_steps": KREA_RAW_STEPS,
        },
        "minus_teacher": False,
        "minus_canary": True,
        "token_hold": "unused_to_neu",
        "lyric_hold": False,
        "dummy": bool(args.dummy),
        "allow_hub": bool(args.allow_hub),
        "lora_targets": lora_spec.label,
        "dit_lora": lora_spec.train_dit,
        "te_lora": lora_spec.train_te,
        "dit_lora_path": dit_lora_path,
        "te_lora_path": te_lora_path,
        "encoder_lora": lora_spec.encoder_lora,
        "hold_weight": float(args.hold_weight),
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "concept_words": meta.concept_words,
        "control_prompt": control_prompt,
        "bare_captions": bool(meta.bare_captions),
        "recommended_range": meta.recommended_range,
        "weights": weights,
        "sample_grid": {
            "scales": list(KREA_SAMPLE_SCALES),
            "gate": "smile-first",
            "crop_purity": False,
            "count": len(samples),
            "dir": "samples",
        },
        "oracle_grid": {
            "dir": "samples/oracle",
            "shots": list(KREA_ORACLE_SHOTS),
            "embed_cos_threshold": float(KREA_ORACLE_EMBED_COS),
            "count": len(oracle),
            "readout": krea_oracle_readout(),
        },
        "train_guidance": guidance,
        "load_te_lora": getattr(args, "load_te_lora", None),
        "skipped_train": bool(skip_train),
        "last": last_stats,
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
        **card,
        "skeleton": str(args.skeleton_model),
        "mu": float(card["mu"]),
        "sample_steps": int(card["sample_steps"]),
        "sample_guidance": float(card["sample_guidance"]),
        "is_distilled": True,
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def main(argv: list[str] | None = None) -> dict | Path:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
