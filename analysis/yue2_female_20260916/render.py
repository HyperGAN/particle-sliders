"""Matched, held-out listening check for the first YuE2 UNI16 female slider."""
from pathlib import Path
import html
import json
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders.infer_yue2 import render
from conceptmod.textsliders.yue2_backend import YuE2Slider, file_digest
from yue2 import YuE2Pipeline

RUN = ROOT / "models/female-yue2-uni16-600-20260916"
WEIGHTS = RUN / "female-yue2-uni16-600_600.safetensors"
OUT = ROOT / "eval/listen/yue2-female-uni16-600-20260916"
PROMPTS = ROOT / "conceptmod/textsliders/data/prompts-yue2-female-eval.yaml"
TRAIN = ROOT / "conceptmod/textsliders/data/prompts-yue2-female.yaml"
LABELS = {"off": "Off · 0", "on": "Female slider · 1", "caption": "Female caption · slider off"}
TITLES = ["Small band", "Upright piano"]


def write_page(records):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>YuE2 female slider · 600 steps</title>',
        '<style>body{font:17px system-ui;background:#101419;color:#edf0f3;max-width:1100px;margin:40px auto;padding:0 20px}p{line-height:1.6;color:#bdc7d2}h1{font-size:30px}section{margin:32px 0;padding:22px;background:#1b222b;border-radius:12px}.clips{display:flex;flex-wrap:wrap;gap:20px}.clip{flex:1;min-width:260px}audio{width:100%}summary{cursor:pointer;color:#a5c8ff}pre{white-space:pre-wrap;font:14px/1.6 system-ui}a{color:#a5c8ff}</style>',
        '<h1>YuE2 female slider · 600 steps</h1>',
        '<p>UNI16 warmup formulation, rank 8. Off and on use the same neutral prompt, lyrics, and seed. The caption reference uses an explicit female vocal prompt with the slider off.</p>',
        '<p>Held-out lyrics and arrangements. Each preview is capped at 750 semantic tokens (about 30 seconds); capped clips are excerpts, not complete songs. These are listening candidates, with no claim of an audible improvement yet.</p>',
        '<p><a href="evaluation.json">Render metadata</a> · <a href="audio-diagnostics.json">Audio diagnostic scores</a></p>']
    for row in range(2):
        for seed in (7, 23):
            group = [r for r in records if r["row"] == row and r["seed"] == seed]
            if not group:
                continue
            parts.append(f'<section><h2>{TITLES[row]} · seed {seed}</h2><div class="clips">')
            for r in group:
                parts.append(f'<div class="clip"><h3>{LABELS[r["variant"]]}</h3><audio controls preload="none" src="{r["path"]}/song.wav"></audio><p>{r["duration_seconds"]:.1f} seconds</p></div>')
            parts.append('</div><details><summary>Prompt and lyrics</summary><pre>' + html.escape(group[0]["neutral"] + "\n\n" + group[0]["lyrics"]) + '</pre></details></section>')
    parts.append('<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(b!==a)b.pause()})))</script></html>')
    (OUT / "index.html").write_text("\n".join(parts))


def main():
    rows = yaml.safe_load(PROMPTS.read_text())["rows"]
    training = yaml.safe_load(TRAIN.read_text())["rows"]
    assert not ({r["lyrics"] for r in rows} & {r["lyrics"] for r in training})
    OUT.mkdir(parents=True, exist_ok=True)
    digest = file_digest(WEIGHTS)
    records, comparisons = [], []
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae",
            local_files_only=True, device="cuda:0", backend="torch-eager",
            memory_budget_gib=18, quantization="none", offload_ar=False) as pipe:
        network, metadata = YuE2Slider.load(pipe._load_model(), WEIGHTS)
        if metadata["model_identity"] != pipe.weights["mot"]:
            raise ValueError("Base weights differ from training")
        for row_index, row in enumerate(rows[:2]):
            for seed in (7, 23):
                waves, tokens = {}, {}
                for variant in (("off", "on", "caption") if seed == 7 else ("off", "on")):
                    name = f"row{row_index}-seed{seed}-{variant}"
                    dest = OUT / name
                    if dest.exists():
                        raise FileExistsError(dest)
                    result = render(pipe, network,
                        style=row["positive"] if variant == "caption" else row["neutral"],
                        lyrics=row["lyrics"], scale=1 if variant == "on" else 0, seed=seed,
                        adapter_identity=digest, semantic_sampling={"min_tokens": 200, "max_tokens": 750})
                    data = result.audio
                    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all() or not len(data):
                        raise ValueError("Invalid stereo audio")
                    rms = float(np.sqrt(np.mean(data.astype(np.float64)**2)))
                    if rms <= 1e-8:
                        raise ValueError("Silent audio")
                    result.save_artifacts(dest)
                    result.save(dest / "song.wav")
                    record = dict(row=row_index, seed=seed, variant=variant, path=name,
                        neutral=row["neutral"], lyrics=row["lyrics"],
                        duration_seconds=len(data)/result.sample_rate, sample_rate=result.sample_rate,
                        rms=rms, peak=float(np.max(np.abs(data))),
                        clipped_fraction=float(np.mean(np.abs(data) >= 1)),
                        wav_sha256=file_digest(dest / "song.wav"), request_identity=result.request_identity,
                        truncated=result.truncated, semantic_tokens=len(result.semantic.tokens), timing=result.timing)
                    records.append(record)
                    waves[variant], tokens[variant] = data.copy(), result.semantic.tokens
                    report = dict(checkpoint=str(WEIGHTS), checkpoint_sha256=digest,
                        evaluation_prompts_sha256=file_digest(PROMPTS), training_steps=600,
                        max_tokens=750, held_out_lyrics=True, records=records, comparisons=comparisons)
                    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                    write_page(records)
                    print(json.dumps({k: record[k] for k in ("path", "duration_seconds", "rms", "truncated")}), flush=True)
                n = min(len(waves["off"]), len(waves["on"]))
                a, b = waves["off"][:n].astype(np.float64), waves["on"][:n].astype(np.float64)
                comparisons.append(dict(row=row_index, seed=seed,
                    waveform_exact_equal=np.array_equal(waves["off"], waves["on"]),
                    semantic_exact_equal=tokens["off"] == tokens["on"],
                    waveform_difference_rms=float(np.sqrt(np.mean((a-b)**2))),
                    waveform_correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
                report["comparisons"] = comparisons
                (OUT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "clips": len(records), "output": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
