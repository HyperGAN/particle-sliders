#!/usr/bin/env python3
"""Stack triage: find which component turns a full rack into noise.

Renders, in ONE pipeline session, from a fixed caption+lyrics at one seed:
  base (no LoRA)
  each component SOLO at its rack multiplier
  CUMULATIVE builds adding components in rack order

Prints rms / spectral centroid / implied-BPM per clip so the poison shows up
as the step where level or spectrum explodes.

  python scripts/render_stack_triage.py --song 5ec87fed --duration 20 \
      --component label=weights:scale ... --out_dir eval/listen/v7-triage --device 0
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
    return "0"


import os  # noqa: E402

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

sys.path.insert(0, str(ROOT / "scripts"))
from tf_leak_metrics import implied_bpm  # noqa: E402


def _parse_component(spec: str) -> tuple[str, Path, float]:
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


def _song_text(song_prefix: str) -> tuple[str, str]:
    matches = sorted(glob.glob(str(LIBRARY / f"{song_prefix}*" / "meta.json")))
    if len(matches) != 1:
        raise SystemExit(f"--song {song_prefix!r} matched {len(matches)} library songs")
    meta = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    return str(meta["caption"]), str(meta["lyrics"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--song")
    source.add_argument("--prompts_file")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--component", action="append", default=[])
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if args.song:
        caption, lyrics = _song_text(args.song)
    else:
        row = _load_prompt_row(Path(args.prompts_file), row=args.row)
        caption = str(row.get("neutral") or row["target"])
        lyrics = str(row.get("lyrics") or "")

    comps = [_parse_component(s) for s in args.component]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    sr = int(pipe.sampling_rate)

    nets = [(label, *_wrap_sidecar(pipe, device, w, "transformer")) for label, w, _s in comps]
    by_label = {label: net for label, net, _m in nets}
    scales = {label: s for label, _w, s in comps}

    def render(name: str, pairs: list[tuple[object, float]]) -> dict:
        dest = out_dir / f"{name}.wav"
        gen = torch.Generator(device).manual_seed(int(args.seed))
        with _apply(*pairs):
            audio = pipe(prompt=caption, lyrics=lyrics, audio_duration=float(args.duration),
                         generator=gen, output="audios")[0]
        wav = audio.detach().T.float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio, dtype=np.float32).T
        sf.write(str(dest), np.ascontiguousarray(wav), sr)
        mono = wav.mean(axis=1) if wav.ndim > 1 else wav
        rms = float(np.sqrt(np.mean(mono**2)))
        mag = np.abs(np.fft.rfft(mono[: sr * 8]))
        freqs = np.fft.rfftfreq(sr * 8, 1 / sr)
        centroid = float((mag * freqs).sum() / max(mag.sum(), 1e-9))
        tempo = implied_bpm(dest)
        row = {"clip": name, "rms": round(rms, 5), "centroid_hz": round(centroid, 1),
               "bpm": round(tempo["bpm"], 1), "onset_rate": round(tempo["onset_rate"], 2)}
        print(json.dumps(row), flush=True)
        return row

    rows: list[dict] = [render("00_base", [])]

    for label, net, _meta in nets:
        rows.append(render(f"solo_{label}", [(net, scales[label])]))

    active: list[tuple[object, float]] = []
    for i, (label, net, _meta) in enumerate(nets, start=1):
        active.append((net, scales[label]))
        rows.append(render(f"cum_{i:02d}_{'_'.join(l for l, _n, _m in nets[:i])}", list(active)))

    (out_dir / "triage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'triage.json'}", flush=True)


if __name__ == "__main__":
    main()
