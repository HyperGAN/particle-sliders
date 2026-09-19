#!/usr/bin/env python3
"""Render selected step checkpoints with matched prompts and seeds in one load.

Uses the existing listening renderer, keeps short/silent outputs as evidence,
and reuses identical base/reference audio across checkpoints. No seed retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "conceptmod/textsliders/data/prompts-gender-uni-v2.yaml")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--memory-fraction", type=float,
                        help="optional CUDA allocator limit while sharing the assigned GPU")
    args = parser.parse_args()
    import torch
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("--memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    from safetensors.torch import load_file
    from conceptmod.textsliders import generate_listen as G
    from conceptmod.textsliders.lora import LoRANetwork
    from lm_evaluate import checkpoint_metadata, topology
    torch.set_num_threads(4)
    weights = [path.resolve() for path in args.weights]
    names = [path.stem for path in weights]
    if len(set(names)) != len(names):
        raise ValueError("Checkpoint stems must be distinct")
    metas = [checkpoint_metadata(path) for path in weights]
    shape = topology(metas[0][0])
    if any(topology(meta[0]) != shape for meta in metas):
        raise ValueError("Checkpoint topologies differ")
    row = G._load_prompt_row(args.prompts, args.row)
    spec = dict(prompts=str(args.prompts.resolve()), prompts_sha256=file_hash(args.prompts),
                row=args.row, seeds=args.seeds, duration=args.duration, scales=[0., 1.],
                seed_retries=0, checkpoints=[dict(path=str(path), sha256=file_hash(path), steps=meta[2])
                                           for path, meta in zip(weights, metas)])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "render_spec.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Render specification changed; use a fresh output directory")
    manifest.write_text(json.dumps(spec, indent=2)+"\n")
    pipe = G._load_pipeline(G.DEFAULT_MODEL_DIR, "cuda:0")
    network = LoRANetwork(pipe.language_model, multiplier=0., **shape).to("cuda:0")
    if not network.unet_loras:
        raise ValueError("No LoRA modules attached")
    network.eval()
    shared = {}
    lines = ["# Repaired smoke checkpoint comparison", "",
             "Same row, lyrics, duration and generation seeds for every checkpoint. "
             "All adapters use scale +1. These are listening candidates, not quality rankings.", "",
             "| Checkpoint | Updates | Seed 7 | Seed 23 |", "| --- | ---: | --- | --- |"]
    # The table below is built dynamically for arbitrary seed lists.
    lines[-2] = "| Checkpoint | Updates | " + " | ".join(f"Seed {seed}" for seed in args.seeds) + " |"
    lines[-1] = "| --- | ---: | " + " | ".join("---" for _ in args.seeds) + " |"
    for weights_path, (meta, metadata_path, steps) in zip(weights, metas):
        network.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        links = []
        for seed in args.seeds:
            folder = args.out / f"{weights_path.stem}-s{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            opts = SimpleNamespace(name=weights_path.stem, out_dir=str(folder), scales="0,1",
                                   plus_label=meta["plus_label"], minus_label=meta["minus_label"],
                                   seed=seed, duration=args.duration, kind="lm")
            stats = {}
            for dest, prompt, scale in G._jobs(opts, row):
                effective_scale = float(scale or 0.)
                identity = (prompt, row["lyrics"], seed, effective_scale,
                            str(weights_path) if effective_scale else "base")
                if dest.exists():
                    stats[dest.name] = G._inspect_wav(dest)
                elif identity in shared:
                    shutil.copyfile(shared[identity], dest)
                    stats[dest.name] = G._inspect_wav(dest)
                else:
                    network.set_lora_slider(effective_scale)
                    generator = torch.Generator("cuda:0").manual_seed(seed)
                    print(f"Rendering {weights_path.stem}, seed {seed}: {dest.name}", flush=True)
                    with torch.inference_mode(), network:
                        audio = pipe(prompt=prompt, lyrics=row["lyrics"], audio_duration=args.duration,
                                     generator=generator, output="audios")[0]
                    stats[dest.name] = G._write_wav(dest, audio, int(pipe.sampling_rate), args.duration,
                                                  accept_short=True, accept_silent=True)
                shared[identity] = dest
            G._write_readme(opts, weights_path, row, [0., 1.], stats, meta["rank"], meta["alpha"], unit_scale=1.)
            (folder / "checkpoint.json").write_text(json.dumps(dict(
                weights=str(weights_path), sha256=file_hash(weights_path), steps=steps,
                metadata_source=str(metadata_path), seed=seed), indent=2)+"\n")
            links.append(f"[Listen]({folder.name}/02_slider_{opts.plus_label}_plus1.wav)")
        lines.append(f"| {weights_path.stem} | {steps} | " + " | ".join(links) + " |")
    lines += ["", "Each folder also contains slider-off and positive-caption references. "
              "Compare intended voice change, lyric order/additions, song continuity and artifacts. "
              "A 20-second duration cap does not test natural long-form endings.", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    print(f"Saved {args.out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
