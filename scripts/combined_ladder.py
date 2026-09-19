#!/usr/bin/env python3
"""Render a combined LM+TF ladder for one axis through dual-host composition.

Attaches the TF LoRA (transformer host) and LM LoRA (language_model host)
simultaneously, multipliers resolved exactly like app/sliders.py
(scale * unit from each sidecar), then renders a fixed grid:

    (tf, lm) in [(-2,0), (-1,0), (0,0), (+1,0), (+2,0), (0,-2), (0,+2), (+2,+2)]

Usage:
  python scripts/combined_ladder.py --axis energy \
      --tf_weights models/energy-tf-v10/energy-tf-v10_derateXXX.safetensors \
      --out_dir eval/listen/v10-combined/energy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ[k] = f"{_HF_HOME}/hub" if k != "HF_HOME" else _HF_HOME
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
import torch
import yaml

_REPO = Path("/ml2/music/sliders-conceptmod")
sys.path.insert(0, str(_REPO))

from conceptmod.textsliders.infer_music3 import _load_pipeline, _to_wav_array
from conceptmod.textsliders.lora import LoRANetwork

TF_REPLACE = ["MiniMaxMusic3Attention", "MiniMaxMusic3TransformerBlock", "MiniMaxMusic3Transformer1DModel"]
LM_REPLACE = ["Qwen3Attention"]
GRID = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, -2), (0, 2), (2, 2)]


def sidecar(weights: Path) -> dict:
    p = weights.with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


def unit_of(meta: dict) -> float:
    u = meta.get("unit_scale")
    try:
        u = float(u)
        if u > 0 and u == u:
            return u
    except (TypeError, ValueError):
        pass
    return 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--tf_weights", required=True)
    ap.add_argument("--lm_weights", default=None,
                    help=f"default: {{axis}}-lm-v9/{{axis}}-lm-v9_last.safetensors")
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()

    root = Path(_REPO)
    tf_w = Path(a.tf_weights)
    lm_w = Path(a.lm_weights or root / "models" / f"{a.axis}-lm-v9" / f"{a.axis}-lm-v9_last.safetensors")
    prompts_file = Path(a.prompts_file or root / "conceptmod/textsliders/data" / f"prompts-{a.axis}-v7.yaml")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_meta, lm_meta = sidecar(tf_w), sidecar(lm_w)
    tf_unit, lm_unit = unit_of(tf_meta), unit_of(lm_meta)

    row = yaml.safe_load(prompts_file.read_text())["rows"][0]
    caption, lyrics = row.get("neutral") or row.get("target"), row.get("lyrics", "")

    device = "cuda:0"
    pipe = _load_pipeline(Path("/ml2/music/models/MiniMax-Music3"), device)

    def attach(host, w, meta, replace, prefix):
        net = LoRANetwork(
            host, rank=int(meta.get("rank") or 8), alpha=float(meta.get("alpha") or 8),
            multiplier=1.0, target_replace=list(meta.get("target_replace") or replace),
            train_method=str(meta.get("train_method") or "full"),
            delimiter=str(meta.get("delimiter") or "-"), prefix=str(prefix),
        ).to(device)
        from safetensors.torch import load_file

        missing, _unexp = net.load_state_dict(load_file(str(w), device="cpu"), strict=False)
        if missing:
            raise RuntimeError(f"{w.name}: {len(missing)} missing keys")
        return net

    tf_net = attach(pipe.transformer, tf_w, tf_meta, TF_REPLACE, "lora_unet")
    lm_net = attach(pipe.language_model, lm_w, lm_meta, LM_REPLACE, "lora_te")
    sr = int(pipe.sampling_rate)

    manifest = []
    for tf_s, lm_s in GRID:
        tf_m, lm_m = tf_s * tf_unit, lm_s * lm_unit
        tf_net.set_lora_slider(tf_m)
        lm_net.set_lora_slider(lm_m)
        gen = torch.Generator(device).manual_seed(int(a.seed))
        tag = f"tf{tf_s:+g}_lm{lm_s:+g}".replace("+", "p").replace("-", "m")
        dest = out_dir / f"{tag}.wav"
        print(f"render {dest.name}  tf_mult={tf_m:g} lm_mult={lm_m:g}", flush=True)
        with tf_net, lm_net:
            audio = pipe(
                prompt=caption, lyrics=lyrics, audio_duration=float(a.duration),
                generator=gen, output="audios",
            )[0]
        sf.write(str(dest), np.ascontiguousarray(_to_wav_array(audio)), sr, format="WAV")
        wav, _sr = sf.read(str(dest))
        rms = float(np.sqrt(np.mean(wav.mean(axis=1) ** 2))) if wav.ndim > 1 else float(np.sqrt(np.mean(wav**2)))
        manifest.append({"file": dest.name, "tf": tf_s, "lm": lm_s, "rms": round(rms, 5)})
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}/MANIFEST.json ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
