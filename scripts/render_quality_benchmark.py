#!/usr/bin/env python3
"""Render a fixed smoke/paired comparison once per prompt and seed on GPU 1.

One pipeline load. Identical prompt-only/zero WAVs are reused across candidates.
No seed retries, threshold changes, checkpoint edits or training happen here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slider_selection.features import digest, file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/listen/quality-heldout")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--duration", type=float, default=20.)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from conceptmod.textsliders import generate_listen as G
    from conceptmod.textsliders.lora import LoRANetwork
    torch.set_num_threads(4)
    fixtures = ROOT / "slider_selection/heldout-prompts.json"
    rows = json.loads(fixtures.read_text())["rows"]
    checkpoints = {
        "smoke": ROOT / "models/gan-bcap-repair/repaired-tx-smoke/repaired-tx-smoke_last.safetensors",
        "paired": ROOT / "models/gan-bcap-repair/gender-paired-mined-bcap/gender-paired-mined-bcap_last.safetensors",
    }
    metas = {name: G._sidecar(path) for name, path in checkpoints.items()}
    shape_fields = ["rank", "alpha", "target_replace", "train_method", "delimiter", "prefix"]
    if any([meta.get(key) for key in shape_fields] != [metas["smoke"].get(key) for key in shape_fields]
           for meta in metas.values()):
        raise ValueError("Checkpoint topologies differ")
    spec = {"fixtures_sha256": file_hash(fixtures), "seeds": args.seeds, "duration": args.duration,
            "checkpoints": {name: file_hash(path) for name, path in checkpoints.items()},
            "scales": [0., .5, 1.], "seed_retries": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "benchmark.json"
    if frozen.exists() and json.loads(frozen.read_text()) != spec:
        raise ValueError("Existing render benchmark differs; use another output directory")
    write_json(frozen, spec)
    pipe = G._load_pipeline(G.DEFAULT_MODEL_DIR, "cuda:0")
    meta = metas["smoke"]
    network = LoRANetwork(pipe.language_model, rank=meta["rank"], alpha=meta["alpha"], multiplier=0.,
                          target_replace=meta.get("target_replace") or G.LM_REPLACE,
                          train_method=meta.get("train_method") or "full",
                          delimiter=meta.get("delimiter") or "-", prefix=meta.get("prefix") or "lora_te").to("cuda:0")
    if not network.unet_loras:
        raise ValueError("No adapter modules attached")
    network.eval()
    shared = {}
    for name, weights in checkpoints.items():
        network.load_state_dict(load_file(str(weights)), strict=True)
        for fixture in rows:
            neutral_vocal = "One lead singer, plain studio timbre, ordinary phrasing, no backing vocals. Sung, not rapped."
            positive_vocal = "One female lead singer with an adult mezzo voice, a chest-to-head mix and raised formants. Plain studio timbre, ordinary phrasing, no backing vocals. Sung, not rapped."
            def caption(vocal):
                return f"Global Metadata:\n{fixture['metadata']}\nVocal Details:\n{vocal}\nArrangement:\n{fixture['arrangement']}"
            row = {"neutral": caption(neutral_vocal), "target": caption(neutral_vocal),
                   "negative": caption(neutral_vocal), "positive": caption(positive_vocal), "lyrics": fixture["lyrics"]}
            for seed in args.seeds:
                folder = args.out / f"{name}-{fixture['id']}-s{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                opts = SimpleNamespace(name=f"{name}-{fixture['id']}", out_dir=str(folder), scales="0,0.5,1",
                                       plus_label="Female", minus_label="Off", seed=seed, duration=args.duration, kind="lm")
                stats = {}
                for dest, prompt, scale in G._jobs(opts, row):
                    effective = float(scale or 0.)
                    identity = digest([prompt, row["lyrics"], seed, args.duration, effective,
                                       spec["checkpoints"][name] if effective else "base"])
                    if dest.exists():
                        stats[dest.name] = G._inspect_wav(dest)
                    elif identity in shared:
                        shutil.copyfile(shared[identity], dest)
                        stats[dest.name] = G._inspect_wav(dest)
                    else:
                        network.set_lora_slider(effective)
                        generator = torch.Generator("cuda:0").manual_seed(seed)
                        print(f"Rendering {name}/{fixture['id']}/seed{seed}/{dest.name}", flush=True)
                        with torch.inference_mode(), network:
                            audio = pipe(prompt=prompt, lyrics=row["lyrics"], audio_duration=args.duration,
                                         generator=generator, output="audios")[0]
                        stats[dest.name] = G._write_wav(dest, audio, int(pipe.sampling_rate), args.duration,
                                                      accept_silent=True, accept_short=True)
                    shared[identity] = dest
                G._write_readme(opts, weights, row, [0., .5, 1.], stats, meta["rank"], meta["alpha"], unit_scale=1.)
                write_json(folder / "evaluation.json", {"candidate": name, "recipe_family": "gan-smoke" if name == "smoke" else "gan-paired",
                           "prompt_family": fixture["id"], "seed": seed, "benchmark_sha256": file_hash(frozen),
                           "checkpoint_sha256": spec["checkpoints"][name], "scope": "development; outside training"})
    print("Heldout render benchmark complete", flush=True)


if __name__ == "__main__":
    main()
