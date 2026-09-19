#!/usr/bin/env python3
"""Held-out Music 3 routed-particle listens: Off / Half / On / +caption."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
from pathlib import Path

# Clear empty inherited CVD before any Music3 import that setdefaults it.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Do NOT import generate_listen — its module-level code rewrites
# CUDA_VISIBLE_DEVICES from --device (e.g. "cuda:0"), which hides every GPU.
from conceptmod.textsliders.infer_music3 import _load_pipeline, _to_wav_array
from conceptmod.textsliders.music3_particle_bridge import ParticleNetwork, load_prompts
from conceptmod.textsliders.train_lora_yue2_fresh import write_json

TAKES = [
    ("off", 0.0, "neutral"),
    ("half", 0.5, "neutral"),
    ("on", 1.0, "neutral"),
    ("on-caption", 0.0, "positive"),
]
MIN_RMS = 1e-3
DURATION_TOLERANCE = 0.90


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_wav(path: Path, audio, sample_rate: int, requested: float) -> tuple[float, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _to_wav_array(audio)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.wav")
    sf.write(str(tmp), array, sample_rate, format="WAV")
    duration = float(len(array) / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(array.astype(np.float64)))))
    if rms < MIN_RMS:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"rejected {path.name}: silent rms={rms:.6f}")
    if duration < requested * DURATION_TOLERANCE:
        print(f"KEEPING SHORT RENDER {path.name} duration={duration:.2f}s", flush=True)
    tmp.replace(path)
    print(f"wrote {path} duration={duration:.2f}s rms={rms:.4f}", flush=True)
    return duration, rms


def page(output: Path, row_map: dict[int, dict], seeds, label: str, critic: str, weights_name: str):
    label_e = html.escape(label)
    critic_e = html.escape(critic)
    sidecar = Path(weights_name).with_suffix(".json").name
    display = {
        "off": "Off",
        "half": "Half",
        "on": label_e,
        "on-caption": f"{label_e} caption",
    }
    cards = []
    for row_i in sorted(row_map):
        row = row_map[row_i]
        for seed in seeds:
            cells = []
            for name, _scale, _field in TAKES:
                relative = Path(f"row-{row_i}-seed-{seed}") / name
                meta = output / relative / "evaluation.json"
                if meta.exists():
                    data = json.loads(meta.read_text())
                    cells.append(
                        f"<div><b>{display[name]}</b> · {data['duration']:.1f}s"
                        f'<audio controls preload="none" src="{relative}/audio.wav"></audio></div>'
                    )
                else:
                    cells.append(f"<div><b>{display[name]}</b> · pending</div>")
            cards.append(
                f"<section><h2>Prompt {row_i + 1}, seed {seed}</h2>"
                f"<p>{html.escape(row['neutral'])}</p>"
                + "".join(cells)
                + "</section>"
            )
    document = (
        "<!doctype html><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Music 3 · {label_e} · {critic_e}</title>"
        "<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}"
        "section{padding:20px;border:1px solid #46505b;margin:20px 0}"
        "audio{display:block;width:100%;margin:10px 0}"
        "section>div{display:inline-block;vertical-align:top;width:46%;margin:1%}"
        "a{color:#a9d7ff}.muted{color:#a8bacb}</style>"
        f"<p><a href=\"../\">← critic sweep</a></p>"
        f"<h1>Music 3 · {label_e} · {critic_e} samples</h1>"
        f"<p class=\"muted\">Routed particle bridge · EMA @ 1800 · transformer critic={critic_e}. "
        f"0 = Off · 1 = {label_e}. Matched held-out prompts.</p>"
        f'<p><a href="{html.escape(weights_name)}">Download slider</a>'
        f' · <a href="{html.escape(sidecar)}">Training details</a></p>'
        + "".join(cards)
    )
    temporary = output / f"index.html.{os.getpid()}.tmp"
    temporary.write_text(document)
    temporary.replace(output / "index.html")


def discover_rows(output: Path, all_rows, row_map: dict[int, dict]) -> dict[int, dict]:
    for existing in sorted(output.glob("row-*-seed-*")):
        try:
            idx = int(existing.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if idx not in row_map and idx < len(all_rows):
            row_map[idx] = all_rows[idx]
    return row_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--prompts_file",
        type=Path,
        default=ROOT / "analysis/uni16_fresh3400_20260912/prompts/metal-eval.yaml",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1709, 2903])
    parser.add_argument("--rows", type=int, nargs="+", default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--model_dir", type=Path, default=Path("/ml2/music/models/MiniMax-Music3"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--critic_label", default="mix_t16_w128_l2")
    args = parser.parse_args(argv)
    if args.duration < 1 or not args.seeds or any(not 0 <= s < 2**63 for s in args.seeds):
        parser.error("Invalid duration or seeds")

    all_rows, meta = load_prompts(args.prompts_file)
    row_indices = list(range(len(all_rows))) if args.rows is None else list(args.rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_map = discover_rows(args.output_dir, all_rows, {i: all_rows[i] for i in row_indices})
    label = meta.get("plus_label") or "Metal"

    digest = file_digest(args.weights)
    weight_dest = args.output_dir / args.weights.name
    sidecar_src = args.weights.with_suffix(".json")
    if not weight_dest.exists() or file_digest(weight_dest) != digest:
        shutil.copy2(args.weights, weight_dest)
    if sidecar_src.exists():
        shutil.copy2(sidecar_src, args.output_dir / sidecar_src.name)
    page(args.output_dir, row_map, args.seeds, label, args.critic_label, args.weights.name)

    print(
        json.dumps(
            dict(
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                device=args.device,
                available=torch.cuda.is_available(),
                count=torch.cuda.device_count(),
            )
        ),
        flush=True,
    )
    pipe = _load_pipeline(args.model_dir, args.device)
    network, record = ParticleNetwork.load(pipe.language_model, args.weights)
    if record.get("recipe") != "anneal-routed-particle-error-music3-v1":
        raise ValueError(f"Unexpected recipe {record.get('recipe')!r}")
    sample_rate = int(pipe.sampling_rate)
    write_json(
        args.output_dir / "status.json",
        dict(stage="rendering", weights_sha256=digest, critic=args.critic_label, rows=row_indices),
    )

    for row_i in row_indices:
        row = all_rows[row_i]
        for seed in args.seeds:
            for name, scale, field in TAKES:
                dest = args.output_dir / f"row-{row_i}-seed-{seed}" / name
                request = dict(
                    weights_sha256=digest,
                    eval_row=row_i,
                    scale=scale,
                    field=field,
                    seed=seed,
                    duration=args.duration,
                    lyrics=row["lyrics"],
                    prompt=row[field],
                )
                if (dest / "evaluation.json").exists():
                    prior = json.loads((dest / "evaluation.json").read_text())
                    if prior.get("request") != request:
                        raise ValueError(f"Existing render request differs: {dest}")
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                generator = torch.Generator(args.device).manual_seed(int(seed))
                print(
                    json.dumps(dict(row=row_i, seed=seed, take=name, scale=scale, field=field)),
                    flush=True,
                )
                with network.scaled(scale):
                    audio = pipe(
                        prompt=row[field],
                        lyrics=row["lyrics"],
                        audio_duration=float(args.duration),
                        generator=generator,
                        output="audios",
                    )[0]
                wav_path = dest / "audio.wav"
                duration, rms = write_wav(wav_path, audio, sample_rate, float(args.duration))
                flat = _to_wav_array(audio).reshape(-1)
                stats = dict(
                    request=request,
                    duration=duration,
                    rms=rms,
                    clipped_fraction=float(np.mean(np.abs(flat) >= 0.999)) if flat.size else 0.0,
                    peak=float(np.max(np.abs(flat))) if flat.size else 0.0,
                    audio_sha256=file_digest(wav_path),
                )
                write_json(dest / "evaluation.json", stats)
                row_map = discover_rows(args.output_dir, all_rows, row_map)
                page(args.output_dir, row_map, args.seeds, label, args.critic_label, args.weights.name)
                print(
                    json.dumps(dict(row=row_i, seed=seed, take=name, duration=duration, rms=rms)),
                    flush=True,
                )

    write_json(
        args.output_dir / "status.json",
        dict(stage="complete", weights_sha256=digest, critic=args.critic_label, rows=row_indices),
    )
    page(args.output_dir, row_map, args.seeds, label, args.critic_label, args.weights.name)
    print(json.dumps(dict(done=True, output=str(args.output_dir))), flush=True)


if __name__ == "__main__":
    main()
