"""Pinned CPU Audiobox CE; other axes and raw levels are separate diagnostics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np

from .specs import WORKSPACE, RewardSpec, sha, digest


def scoring_windows(samples, rate, full_song=False):
    if samples <= 0 or rate <= 0:
        raise ValueError('Empty audio or invalid rate')
    size = 10*rate
    if not full_song:
        if samples < 2*size:
            raise ValueError('Short-screen output contains less than 20 seconds')
        return [(0, size), (size, 2*size)]
    return [(start, min(samples, start+size)) for start in range(0, samples, size)]


class CEReward:
    def __init__(self, spec=None):
        self.spec = spec or RewardSpec()
        self.model = None

    @staticmethod
    def provenance():
        sys.path.insert(0, str(WORKSPACE/'.cache/slider-quality/python'))
        import audiobox_aesthetics
        import torchaudio
        from huggingface_hub import hf_hub_download
        source = Path(audiobox_aesthetics.__file__).parent
        weights = [Path(hf_hub_download('facebook/audiobox-aesthetics', filename=name,
                       revision=RewardSpec().revision, local_files_only=True))
                   for name in ('config.json', 'model.safetensors')]
        paths = sorted(source.rglob('*.py')) + weights
        paths += [Path(torchaudio.__file__).parent/'functional/functional.py']
        return {str(path): sha(path) for path in paths}

    def load(self):
        sys.path.insert(0, str(WORKSPACE/'.cache/slider-quality/python'))
        import torch
        from audiobox_aesthetics.model.aes import AesMultiOutput
        torch.set_num_threads(4)
        self.model = AesMultiOutput.from_pretrained(self.spec.scorer, revision=self.spec.revision,
                                                   local_files_only=True).cpu().eval()

    def _score(self, data, rate, windows):
        import torch
        from torchaudio.functional import resample
        mono = torch.from_numpy(data.mean(axis=1))
        results = []
        with torch.inference_mode():
            for start, end in windows:
                wav = resample(mono[start:end], rate, 16000)[None, None]
                output = self.model({'wav': wav, 'mask': torch.ones_like(wav, dtype=torch.bool)})
                axes = {key: float(value.item()*self.model.target_transform[key]['std']
                                   + self.model.target_transform[key]['mean']) for key, value in output.items()}
                if not all(np.isfinite(value) for value in axes.values()):
                    raise ValueError('Nonfinite scorer result')
                if not self.spec.valid_range[0] <= axes['CE'] <= self.spec.valid_range[1]:
                    raise ValueError('CE outside declared range')
                results.append(dict(start_s=start/rate, end_s=end/rate, axes=axes))
        return results

    def measure(self, path, *, full_song=False):
        import soundfile as sf
        started = time.monotonic()
        result = dict(valid=False, scalar=None, audio_sha256=sha(path), reward_spec_sha256=digest(asdict(self.spec)))
        try:
            data, rate = sf.read(path, dtype='float32', always_2d=True)
            if not len(data) or not np.isfinite(data).all():
                raise ValueError('Empty or nonfinite waveform')
            rms = float(np.sqrt(np.mean(data.astype('float64')**2)))
            result['diagnostics'] = dict(duration_s=len(data)/rate, sample_rate=rate, rms=rms,
                peak=float(np.abs(data).max()), clipped_fraction=float((np.abs(data) >= .999).mean()),
                silent_fraction=float((np.abs(data) < 1e-5).mean()))
            if rms < 1e-5:
                raise ValueError('Silent waveform')
            windows = scoring_windows(len(data), rate, full_song)
            source = data if full_song else data[:20*rate]
            normalization_rms = float(np.sqrt(np.mean(source.astype('float64')**2)))
            normalized = (source*(.1/normalization_rms)).astype('float32')
            if self.model is None:
                self.load()
            scores = self._score(normalized, rate, windows)
            raw = self._score(data, rate, windows)
            weights = [w['end_s']-w['start_s'] for w in scores]
            result.update(valid=True, scalar=float(np.average([w['axes']['CE'] for w in scores], weights=weights)),
                          windows=scores, raw_windows=raw, normalization_rms=normalization_rms,
                          full_song=full_song)
            result['diagnostics']['axes'] = {axis: float(np.average([w['axes'][axis] for w in scores], weights=weights))
                                           for axis in scores[0]['axes']}
            result['diagnostics']['sections_ce'] = {name: scores[i]['axes']['CE'] for name, i in
                                                    [('beginning', 0), ('middle', len(scores)//2), ('ending', -1)]}
        except Exception as exc:
            result.update(valid=False, scalar=None, error=f'{type(exc).__name__}: {exc}')
        result['seconds'] = time.monotonic()-started
        return result
