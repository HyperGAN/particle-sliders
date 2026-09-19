#!/usr/bin/env python3
"""Build local side-by-side audio players from a completed render manifest."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--title", default="Training length: listen and compare")
    parser.add_argument("--reference-label", default="Previous listening reference")
    parser.add_argument("--checkpoint-labels", nargs="+",
                        help="optional display labels in manifest checkpoint order")
    args = parser.parse_args()
    spec = json.loads((args.folder / "render_spec.json").read_text())
    prompt_rows = yaml.safe_load(Path(spec['prompts']).read_text())
    if isinstance(prompt_rows, dict):
        prompt_rows = prompt_rows['rows']
    lyrics = str(prompt_rows[spec['row'] % len(prompt_rows)]['lyrics']).replace('\\n', '\n')
    seeds = spec["seeds"]
    if args.checkpoint_labels is not None and len(args.checkpoint_labels) != len(spec["checkpoints"]):
        parser.error("--checkpoint-labels must provide one label per checkpoint")
    parts = [f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(args.title)}</title>',
             '<style>body{font:16px system-ui,sans-serif;background:#f3f4f5;color:#20242b;margin:0;padding:32px}main{max-width:1050px;margin:auto;background:white;padding:28px;border-radius:12px}h1{margin-top:0}p{line-height:1.55;color:#4b5563}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:14px 12px;border-bottom:1px solid #e5e7eb}th{font-size:14px;color:#4b5563}audio{width:100%;min-width:210px;height:38px}small{display:block;color:#6b7280;margin-top:5px}.scroll{overflow:auto}a{color:#185cb2}</style>',
             f'<main><h1>{html.escape(args.title)}</h1>',
             '<p>Compare the overall music and the intended voice change. Each column keeps the prompt, lyrics and generation seed fixed. Every candidate uses slider +1.</p>',
             f'<details><summary>Supplied lyric lines</summary><pre style="white-space:pre-wrap">{html.escape(lyrics)}</pre></details>',
             '<p><a href="README.md">Listening notes and checkpoint details</a></p>',
             '<div class="scroll"><table><thead><tr><th>Checkpoint</th>']
    parts += [f'<th>Seed {seed}</th>' for seed in seeds]
    parts.append('</tr></thead><tbody>')

    def audio_row(label, stem, filename):
        parts.append(f'<tr><td><strong>{html.escape(label)}</strong></td>')
        for seed in seeds:
            relative = Path(f'{stem}-s{seed}') / filename
            if not (args.folder / relative).is_file():
                raise FileNotFoundError(args.folder / relative)
            parts.append(f'<td><audio controls preload="none" src="{html.escape(str(relative))}"></audio></td>')
        parts.append('</tr>')

    for index, checkpoint in enumerate(spec["checkpoints"]):
        stem = Path(checkpoint["path"]).stem
        label = (args.checkpoint_labels[index] if args.checkpoint_labels is not None else
                 args.reference_label if index == 0 else f'{checkpoint["steps"]} updates — new run')
        audio_row(label, stem, '02_slider_Female_plus1.wav')
    reference = Path(spec["checkpoints"][0]["path"]).stem
    audio_row('Slider off', reference, '01_slider_neutral_base_zero.wav')
    audio_row('Positive-caption reference', reference, '03_REF_prompt_Female_no_slider.wav')
    parts += ['</tbody></table></div>',
              f'<p>{spec["duration"]:g}-second duration cap; no seed retries. Listen for clarity, phrasing, coherence, voice character and artifacts. No checkpoint has been selected by the automated diagnostics.</p>',
              '</main><script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()})))</script></html>']
    output = args.folder / "index.html"
    output.write_text('\n'.join(parts))
    print(output)


if __name__ == "__main__":
    main()
