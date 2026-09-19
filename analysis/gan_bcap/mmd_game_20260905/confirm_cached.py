"""Run the predeclared extra-fixture comparison, outside leaderboard aggregation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .game import load_game, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--subdir', default='confirmation')
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    root = home.parents[2]
    folder = home/'rounds'/args.round/args.subdir
    plan = json.loads((folder/'plan.json').read_text())
    protocol, state = load_game(home)
    if plan['prompts_sha256'] != sha(plan['prompts']):
        raise ValueError('Predeclared confirmation prompts changed')
    candidate = state['entries'][plan['candidate_id']]
    reference = state['entries'][plan['reference']]
    if reference['weights'] != plan['reference_weights']:
        raise ValueError('Predeclared reference changed')
    entries = [candidate, reference]+[state['entries'][name] for name in plan.get('extra_references', [])]
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('Duplicate confirmation entry')
    for entry in entries:
        if entry['id'] in plan.get('extra_references', []) and entry['weights'] != plan.get('extra_reference_weights', {}).get(entry['id']):
            raise ValueError('Predeclared extra reference changed')
        if sha(entry['weights']) != entry['weights_sha256']:
            raise ValueError('Scored checkpoint changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1',
               HF_HOME='/ml2/music/.cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    reports = []
    for row in plan['rows']:
        destination = folder/f'row-{row}'
        report = folder/f'scores-row-{row}.json'
        from .confirmation_cache import reuse_confirmation
        for source in args.reuse:
            copied = reuse_confirmation(source, destination, plan, entries, row,
                                         root/'analysis/gan_bcap/render_v2.py')
            print(f'Reused {len(copied)} verified audio files for row {row}', flush=True)
        commands = [
            [sys.executable, str(root/'analysis/gan_bcap/render_v2.py'), '--weights',
             *[entry['weights'] for entry in entries], '--out', str(destination),
             '--prompts', plan['prompts'], '--row', str(row), '--seeds',
             *map(str, plan['seeds']), '--duration', str(plan['duration'])],
            [sys.executable, str(root/'analysis/gan_bcap/autonomous_audio.py'), '--folders',
             str(destination), '--concept', protocol['concept'], '--output', str(report)],
        ]
        for stage, command in zip(('render', 'score'), commands):
            start = time.monotonic()
            print(stage, row, flush=True)
            with (folder/f'{stage}-row-{row}.log').open('a') as stream:
                result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with (folder/'execution.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(command=command, seconds=time.monotonic()-start, returncode=result.returncode))+'\n')
            result.check_returncode()
        reports.extend(dict(row=row, **record) for record in json.loads(report.read_text())['records'])

    # These diagnostics complement the judge and never feed the leaderboard.
    # Full-mix harmony and rhythm are proxies, not isolated vocal melodies.
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.functional import resample
    torch.set_num_threads(4)
    from scripts.lm_score import WhisperBackend
    backend = WhisperBackend(root/'analysis/gan_bcap/asr_cpu_fp32_cache', device='cpu')
    def features(record):
        path = Path(record['candidate']['audio'])
        data, rate = sf.read(path, dtype='float32', always_2d=True)
        signal = resample(torch.from_numpy(data.mean(axis=1)), rate, 11025)
        spectrum = torch.stft(signal, n_fft=2048, hop_length=512,
                              window=torch.hann_window(2048), return_complex=True)
        frequencies = np.fft.rfftfreq(2048, d=1/11025)
        magnitude = spectrum.abs().numpy()
        band = (frequencies >= 55) & (frequencies <= 4000)
        pitch_classes = np.rint(69+12*np.log2(frequencies[band]/440)).astype(int) % 12
        chroma = np.bincount(pitch_classes, weights=(magnitude[band]**2).mean(axis=1), minlength=12)
        onset = np.maximum(np.diff(magnitude, axis=1), 0).mean(axis=0)
        onset = onset-onset.mean()
        autocorrelation = np.correlate(onset, onset, mode='full')[len(onset)-1:][5:44]
        embedding = np.asarray(backend.measure(path, plan['duration'])['emb'])
        return dict(chroma=chroma, onset_autocorrelation=autocorrelation, whisper_embedding=embedding)
    def cosine_distance(a, b):
        norm = float(np.linalg.norm(a)*np.linalg.norm(b))
        return float(1-np.dot(a, b)/norm) if norm > 1e-12 else None
    diversity = []
    for row in plan['rows']:
        for entry in entries:
            pair = sorted([record for record in reports if record['row']==row and record['checkpoint']['path']==entry['weights']], key=lambda record: record['seed'])
            if len(pair) != 2:
                raise ValueError('Expected the exact two predeclared seeds')
            a, b = map(features, pair)
            diversity.append(dict(row=row, entry=entry['id'], seeds=plan['seeds'],
                distances={key:cosine_distance(a[key], b[key]) for key in a},
                scores=[record['heuristic_score'] for record in pair],
                production=[record['candidate']['production'] for record in pair],
                lyric_recall=[record['candidate']['lyric_diagnostics']['recall'] for record in pair],
                hf14k_fraction=[record['candidate']['hf14k_fraction'] for record in pair]))
    from analysis.gan_bcap.autonomous_audio import summarize
    summary = dict(status='complete', plan=plan, ranking=summarize(reports), records=reports,
                   diversity=diversity, diversity_limitations=[
                       'Only two samples per condition: not a test of full mode coverage.',
                       'Whisper embedding distance reflects content and timbre; full-mix chroma and onset autocorrelation describe harmony and rhythm.',
                       'Chroma bins use equal-tempered pitch classes from 55 to 4000 Hz; rhythm uses centered spectral-flux autocorrelation at lags 5 through 43, with 512-sample hops at 11025 Hz.',
                       'Larger distances can reflect errors or noise. Inspect the accompanying production, lyrics and artifacts; no diversity pass threshold was fitted.',
                       'Automated measurements; human listening verdict remains unmeasured.'])
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(ranking=summary['ranking'], diversity=diversity), indent=2), flush=True)


if __name__ == '__main__':
    main()
