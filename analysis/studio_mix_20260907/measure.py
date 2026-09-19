"""CPU diagnostics for the first 20 seconds, separate from listening preference."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/ml2/music/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/ml2/music/.cache/huggingface/hub'

import json
from pathlib import Path
import sys
import time
import traceback

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT.parent/'.cache/slider-quality/python')]
from slider_selection import features as F
from scripts.lm_score import WhisperBackend
from app.rewriter import _artist_name_hit

# Fixed descriptions from the existing uni16 evaluation, frozen for this screen.
F.CONCEPTS = {
    'female': ('a song with a feminine sounding lead singing voice', 'a song with a masculine sounding lead singing voice'),
    'pop': ('contemporary pop music with melodic hooks and polished drums', 'a song with a plain accompaniment'),
    'country': ('country music with acoustic guitar, twangy electric guitar and pedal steel', 'a song with a plain accompaniment'),
    'indie-rock': ('indie rock music with electric guitars, moving bass and a live drum kit', 'a song with a plain accompaniment'),
    'house': ('house music with four on the floor kick, offbeat open hats and rolling bass', 'a song with a plain accompaniment'),
    'acoustic-folk': ('acoustic folk music with fingerpicked guitar, upright bass and brushed percussion', 'a song with a plain accompaniment'),
}


def main():
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    spec = json.loads((WORK/'screen.json').read_text())
    measurer = F.AudioMeasurer(WORK/'measurement-cache', 'cpu')
    asr = WhisperBackend(WORK/'asr-cache', device='cpu')
    out = WORK/'measurements.json'
    result = json.loads(out.read_text()) if out.exists() else dict(
        status='running', screen_sha256=F.file_hash(WORK/'screen.json'), records={},
        protocol=dict(description='CPU CLAP and aesthetics on RMS 0.1 float32 copies of first 20 seconds; ASR on original excerpts.',
                      concepts=F.CONCEPTS,
                      limitation='ASR and embedding diagnostics are imperfect and do not establish listening preference. Full-sheet recall is duration-dependent.'))
    if result['screen_sha256'] != F.file_hash(WORK/'screen.json'):
        raise ValueError('Screen changed')
    result['source_sha256'] = {str(p):F.file_hash(p) for p in [Path(__file__), ROOT/'slider_selection/features.py', ROOT/'scripts/lm_score.py']}
    while True:
        if not (WORK/'renders.json').exists():
            time.sleep(10)
            continue
        renders = json.loads((WORK/'renders.json').read_text())
        for job in spec['jobs']:
            rec = renders['records'].get(job['id'], {})
            if rec.get('status') != 'complete':
                continue
            if job['id'] in result['records']:
                old = result['records'][job['id']]
                if old.get('excerpt_sha256') == rec['excerpt_sha256']:
                    continue
            path = Path(rec['excerpt'])
            print(f'MEASURE {job["id"]} {job["label"]}', flush=True)
            try:
                data, rate = sf.read(path, dtype='float32', always_2d=True)
                rms = float(np.sqrt(np.mean(data.astype('float64')**2)))
                normalized = WORK/'measurement-audio'/f'{rec["excerpt_sha256"]}.wav'
                normalized.parent.mkdir(exist_ok=True)
                if not normalized.exists():
                    sf.write(normalized, data*(.1/max(rms, 1e-12)), rate, subtype='FLOAT')
                measured = measurer.measure(normalized)
                transcript = asr.measure(path, 20)['text']
                lyrics = F.lyric_features(job['lyrics'], transcript)
                # A hallucinated proper name must not become a listening note.
                if _artist_name_hit('', transcript):
                    transcript = '[Transcript omitted by project name validation]'
                windows = measured['windows']
                lengths = [w['end_s']-w['start_s'] for w in windows]
                denominator = sum(lengths)

                def average(block, key):
                    return sum(n*w[block][key] for n,w in zip(lengths,windows))/denominator

                result['records'][job['id']] = dict(
                    status='complete', label=job['label'], excerpt_sha256=rec['excerpt_sha256'],
                    concepts={k:average('concept', k) for k in F.CONCEPTS},
                    aesthetics={k:average('aesthetics', k) for k in windows[0]['aesthetics']},
                    lyric_diagnostics=lyrics, transcript=transcript, rms=rms,
                    hf14k_fraction=F.fullband_features(path)['hf14k_fraction'],
                    measurement_provenance=measured['provenance'])
                print(f'MEASURED {job["id"]} phrase={lyrics["phrase_accuracy"]:.3f} recall={lyrics["recall"]:.3f}', flush=True)
            except Exception as exc:
                result['records'][job['id']] = dict(status='error', excerpt_sha256=rec['excerpt_sha256'],
                                                  error=str(exc), traceback=traceback.format_exc())
                print(f'MEASUREMENT ERROR {job["id"]}: {exc}', flush=True)
            F.write_json(out, result)
        if renders['status'] in ('complete', 'complete_with_errors', 'interrupted_cuda_error'):
            break
        time.sleep(10)
    result['status'] = ('complete' if len(result['records']) == 25
                        and all(r['status'] == 'complete' for r in result['records'].values())
                        else 'complete_with_errors')
    F.write_json(out, result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
