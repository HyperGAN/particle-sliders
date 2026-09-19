#!/usr/bin/env python3
"""Inspect fixed-clip ASR transcripts with the existing lyric-recall diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lm_score import WhisperBackend, lyric_recall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'analysis/gan_bcap/asr_cpu_fp32_cache')
    args = p.parse_args()
    torch.set_num_threads(4)
    spec = json.loads((args.folder / 'render_spec.json').read_text())
    prompt_file = Path(spec['prompts'])
    assert hashlib.sha256(prompt_file.read_bytes()).hexdigest() == spec['prompts_sha256']
    rows = yaml.safe_load(prompt_file.read_text())
    if isinstance(rows, dict):
        rows = rows['rows']
    sheet = rows[spec['row']]['lyrics']
    backend = WhisperBackend(args.cache, device='cpu')
    report = dict(model=backend.model_id, device='cpu', dtype='float32',
                  folder=str(args.folder.resolve()), lyrics=sheet,
                  render_spec_sha256=hashlib.sha256((args.folder/'render_spec.json').read_bytes()).hexdigest(),
                  backend_sha256=hashlib.sha256((ROOT/'scripts/lm_score.py').read_bytes()).hexdigest(),
                  limitations=['ASR may mistranscribe singing, hallucinate words or miss audible gibberish.',
                               'Existing bag-word recall does not check order, extras, repeats or voice quality.',
                               'A 20-second cap can omit valid later lyrics; missing sheet words are not automatically errors.',
                               'Transcripts and recall do not override listening feedback.'], records=[])
    first = Path(spec['checkpoints'][0]['path']).stem
    jobs = []
    for seed in spec['seeds']:
        for role, filename in [('off', '01_slider_neutral_base_zero.wav'),
                               ('positive_reference', '03_REF_prompt_Female_no_slider.wav')]:
            jobs.append((role, None, seed, args.folder/f'{first}-s{seed}'/filename))
        for checkpoint in spec['checkpoints']:
            stem = Path(checkpoint['path']).stem
            jobs.append(('slider_plus1', checkpoint, seed,
                         args.folder/f'{stem}-s{seed}'/'02_slider_Female_plus1.wav'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for role, checkpoint, seed, audio in jobs:
        result = backend.measure(audio, spec['duration'])
        record = dict(role=role, checkpoint=checkpoint, seed=seed, audio=str(audio.resolve()),
                      sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                      transcript=result['text'], lyric_recall=lyric_recall(result['text'], sheet),
                      seconds=result['duration'])
        report['records'].append(record)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(audio.parent.name, role, round(record['lyric_recall'], 3), repr(record['transcript']), flush=True)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
