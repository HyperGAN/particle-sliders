#!/usr/bin/env python3
"""Same-seed ending A/B for a full slider STACK (multiple LoRAs at once).

render_end_ab.py applies one LoRA per variant; the studio applies a whole rack
(TF + LM components together), and the documented blow-through failure mode is
a property of stacked LM halves. This variant takes repeatable
--component label=weights:scale entries, wraps every checkpoint (host read from
each sidecar), applies them simultaneously, and reports length / tail loudness
per seed:

  python scripts/render_end_stack.py --song 5ec87fed --duration 90 \
      --seeds 7,23,77 \
      --component gender_tf=models/gender-tf-v6/gender-tf-v6_unit_last.safetensors:-1.3 \
      --component gender_lm=models/gender-lm-v6/gender-lm-v6_last.safetensors:-1.3 \
      --out_dir eval/listen/v7-endreg-stack-90s --device 1

Verdicts match render_end_ab.py: ended naturally means clearly short of the
requested duration; a hot tail sitting at the cap means <|audio_end|> never
sampled in time.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/ml2/music")
LIBRARY = APP_ROOT / "library"


def _early_visible_device(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


import os  # noqa: E402  (after _early_visible_device import-time use below)

os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_device(sys.argv[1:])

for path in (str(ROOT), str(APP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import DEFAULT_MODEL_DIR, _load_prompt_row  # noqa: E402
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402


def _parse_component(spec: str) -> tuple[str, Path, float]:
    """'label=path:scale' -> (label, absolute weights path, scale)."""
    label, _, rest = spec.partition("=")
    path_str, _, scale_str = rest.rpartition(":")
    if not label or not path_str or not scale_str:
        raise SystemExit(f"--component must look like label=path:scale, got {spec!r}")
    weights = Path(path_str)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        raise SystemExit(f"component '{label}': weights not found at {weights}")
    return label, weights, float(scale_str)


def _song_text(song_prefix: str) -> tuple[str, str, float]:
    matches = sorted(glob.glob(str(LIBRARY / f"{song_prefix}*" / "meta.json")))
    if len(matches) != 1:
        raise SystemExit(f"--song {song_prefix!r} matched {len(matches)} library songs, need exactly 1")
    meta = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    return str(meta["caption"]), str(meta["lyrics"]), float(meta.get("duration_sec") or 60.0)


def _save_and_measure(dest: Path, audio, sample_rate: int, requested: float) -> dict:
    if torch.is_tensor(audio):
        wav = audio.detach().T.float().cpu().numpy()
    else:
        wav = np.asarray(audio, dtype=np.float32).T
    wav = np.ascontiguousarray(wav)
    sf.write(str(dest), wav, sample_rate)
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    overall = float(np.sqrt(np.mean(mono**2)))
    tail = mono[-int(sample_rate * 0.5):]
    tail_ratio = float(np.sqrt(np.mean(tail**2)) / overall) if overall > 0 else 0.0
    duration = float(mono.shape[0] / sample_rate)
    return {
        "seconds": duration,
        "rms": overall,
        "tail_ratio": tail_ratio,
        "ended": bool(duration < requested - 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--song", help="library song id (prefix ok): reuse its caption and lyrics")
    source.add_argument("--prompts_file", help="prompt YAML row whose neutral caption drives every render")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--component", action="append", default=[], help="label=weights_path:scale (repeatable)")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if args.song:
        caption, lyrics, _ = _song_text(args.song)
    else:
        row = _load_prompt_row(Path(args.prompts_file), row=args.row)
        caption = str(row.get("neutral") or row["target"])
        lyrics = str(row.get("lyrics") or "")

    components = [_parse_component(spec) for spec in args.component]
    if not components:
        raise SystemExit("no --component given: this script exists to test stacks")
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    sample_rate = int(pipe.sampling_rate)

    nets: list[tuple[str, object, float]] = []
    for label, weights, scale in components:
        network, _meta = _wrap_sidecar(pipe, device, weights, "transformer")
        nets.append((label, network, scale))

    rows_md: list[str] = []
    stats_all: list[dict] = []
    for seed in seeds:
        for tag, pairs in (("base", []), ("stack", [(net, scale) for (_label, net, scale) in nets])):
            name = f"{seed:02d}_{tag}.wav"
            print(f"rendering {name} ({len(pairs)} components, {args.duration:g}s cap)...", flush=True)
            generator = torch.Generator(device).manual_seed(seed)
            with _apply(*pairs):
                audio = pipe(
                    prompt=caption,
                    lyrics=lyrics,
                    audio_duration=float(args.duration),
                    generator=generator,
                    output="audios",
                )[0]
            stats = _save_and_measure(out_dir / name, audio, sample_rate, float(args.duration))
            stats.update({"seed": seed, "tag": tag})
            stats_all.append(stats)
            verdict = "ended naturally" if stats["ended"] else "TRUNCATED at cap"
            print(f"  {name}: {stats['seconds']:.2f}s tail/overall {stats['tail_ratio']:.3f} - {verdict}", flush=True)
            rows_md.append(
                f"| `{name}` | {tag} | {stats['seconds']:.2f} | {stats['tail_ratio']:.3f} "
                f"| {'yes' if stats['ended'] else '**no - cap**'} |"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(stats_all, indent=2), encoding="utf-8")
    comp_lines = "\n".join(f"- `{label}` @ {scale:+g}" for label, _w, scale in components)
    (out_dir / "LISTEN.md").write_text(
        "\n".join(
            [
                "# Stacked-slider ending A/B",
                "",
                f"Requested {args.duration:g}s cap; seeds {seeds}.",
                "",
                "## Rack",
                "",
                comp_lines,
                "",
                "| file | tag | seconds | tail/overall | ended naturally |",
                "|------|-----|--------:|-------------:|-----------------|",
                *rows_md,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'LISTEN.md'}", flush=True)


if __name__ == "__main__":
    main()
