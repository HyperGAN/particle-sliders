#!/usr/bin/env python3
"""Render a matched YuE2 slider-scale comparison through the native pipeline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conceptmod.textsliders.yue2_backend import (
    DEFAULT_MODEL, DEFAULT_VAE, YuE2Slider, file_digest, require_yue2, sound_only,
)


def render(pipe, network, *, style, lyrics, scale, seed, cot="off", abc=None,
           semantic_sampling=None, adapter_identity=None):
    """Only AR composition is adapted; the same frozen NAR/VAE renders each cut."""
    from yue2.pipeline import SongResult
    from yue2.protocol import SongRequest
    from yue2.storage import identity
    if pipe.backend != "torch-eager" or pipe.quantization != "none" or pipe.offload_ar:
        raise ValueError("Slider rendering requires torch-eager, no quantization and no AR offload")
    request = SongRequest(style=sound_only(style), lyrics=sound_only(lyrics), seed=seed,
                          cot=cot, abc=sound_only(abc) if abc else None)
    start = time.perf_counter()
    # Each stage must use the original hooked model; CUDA graphs and vLLM can
    # bypass Python LoRA forwards. Acoustic synthesis also packs base weights.
    with network.scaled(scale):
        plan = pipe.plan(request=request)
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling)
    with network.scaled(0):
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic)
        nar_seconds = time.perf_counter() - nar_start
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
    config = pipe.effective_config(request, semantic_sampling=semantic_sampling)
    config["conceptmod"] = {"adapter": adapter_identity, "scale": scale,
                            "stages": ["plan", "semantic"], "acoustic_scale": 0}
    timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
              "vae_seconds": time.perf_counter() - vae_start, "e2e_seconds": time.perf_counter() - start}
    request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
    return SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--style", required=True, help="Neutral sound description")
    parser.add_argument("--lyrics_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=DEFAULT_MODEL)
    parser.add_argument("--vae_id", default=DEFAULT_VAE)
    parser.add_argument("--revision")
    parser.add_argument("--vae_revision")
    parser.add_argument("--cache_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory_budget_gib", type=float, default=20)
    parser.add_argument("--scales", default="0,0.5,1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cot", choices=["off", "melody", "full"], default="off")
    parser.add_argument("--abc_file", type=Path)
    parser.add_argument("--max_tokens", type=int, default=9000)
    parser.add_argument("--allow_hub", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.scales = [float(s) for s in args.scales.split(",")]
    except ValueError:
        parser.error("scales must be comma-separated numbers")
    if not args.scales or not all(math.isfinite(s) for s in args.scales):
        parser.error("scales must be finite")
    if len(set(args.scales)) != len(args.scales):
        parser.error("scales must be unique")
    if args.max_tokens < 1 or not 0 <= args.seed < 2**63:
        parser.error("max_tokens must be positive; seed must be in [0, 2**63)")
    if args.abc_file and args.cot == "off":
        parser.error("abc_file requires cot=melody or full")
    return args


def main(argv=None):
    args = parse_args(argv)
    require_yue2()
    from yue2 import YuE2Pipeline
    lyrics = sound_only(args.lyrics_file.read_text())
    style = sound_only(args.style)
    abc = sound_only(args.abc_file.read_text()) if args.abc_file else None
    if not lyrics.strip() or not style.strip():
        raise ValueError("style and lyrics must be nonempty")
    # Fail before loading weights if any destination would overwrite a render.
    outputs = [args.output_dir / f"scale_{scale:g}" for scale in args.scales]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Use a fresh output directory for this comparison")
    digest = file_digest(args.weights)
    with YuE2Pipeline.from_pretrained(args.model_id, vae=args.vae_id,
            revision=args.revision, vae_revision=args.vae_revision, cache_dir=args.cache_dir,
            local_files_only=not args.allow_hub, device=args.device, backend="torch-eager",
            memory_budget_gib=args.memory_budget_gib, quantization="none", offload_ar=False) as pipe:
        model = pipe._load_model()
        network, record = YuE2Slider.load(model, args.weights)
        if record.get("model_identity") != pipe.weights["mot"]:
            raise ValueError("Slider was trained on different base model weights")
        for scale, output in zip(args.scales, outputs):
            result = render(pipe, network, style=style, lyrics=lyrics, scale=scale,
                seed=args.seed, cot=args.cot, abc=abc, adapter_identity=digest,
                semantic_sampling={"max_tokens": args.max_tokens, "min_tokens": min(200, args.max_tokens)})
            result.save_artifacts(output)
            print(json.dumps({"output": str(output), "scale": scale, "truncated": result.truncated}), flush=True)


if __name__ == "__main__":
    main()
