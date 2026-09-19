#!/usr/bin/env python3
"""Tiny tempo-proxy metrics for the tf-leak gates and H1-style analyses.

numpy-only (no librosa): half-wave-rectified spectral flux, adaptive peak
picking, then implied BPM = 60 / median inter-onset interval. Good enough to
detect "the slider moved the beat", which is all these gates ask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _flux(signal: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
    hop, win = 512, 1024
    frames = max(1, (len(signal) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    mags = np.abs(np.fft.rfft(signal[idx] * np.hanning(win), axis=1))
    diff = np.clip(mags[1:] - mags[:-1], 0, None).sum(axis=1)
    return diff, hop / sr


def _peaks(flux: np.ndarray, dt: float) -> np.ndarray:
    if len(flux) < 3:
        return np.array([], dtype=int)
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(flux, kernel, mode="same")
    thresh = smooth.mean() + smooth.std()
    cand = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]) & (smooth[1:-1] > thresh))[0] + 1
    keep: list[int] = []
    for p in cand:
        if not keep or (p - keep[-1]) * dt >= 0.15:
            keep.append(int(p))
    return np.array(keep, dtype=int)


def implied_bpm(path: str | Path) -> dict[str, float]:
    """-> {'bpm': median-IOI BPM (0 if <3 onsets), 'onset_rate': onsets/sec}."""
    wav, sr = sf.read(str(path))
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    flux, dt = _flux(np.asarray(mono, dtype=np.float32), float(sr))
    peaks = _peaks(flux, dt)
    times = peaks * dt
    rate = len(times) / max(len(mono) / sr, 1e-9)
    if len(times) < 3:
        return {"bpm": 0.0, "onset_rate": rate}
    ioi = np.median(np.diff(times))
    return {"bpm": float(60.0 / ioi) if ioi > 0 else 0.0, "onset_rate": float(rate)}
