#!/usr/bin/env python3
"""Measure gain, hiss and repetition controls with the frozen audio readouts.

These are development probes, not human labels or an optimizer score.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / ".cache/slider-quality/python"))
import numpy as np
import soundfile as sf
from slider_selection.features import AudioMeasurer, file_hash, write_json


def main():
    source = ROOT / "eval/listen/gan-bcap-repair/repaired-tx-smoke/01_slider_neutral_base_zero.wav"
    out = ROOT / "eval/slider-quality/attacks"
    out.mkdir(parents=True, exist_ok=True)
    base, rate = sf.read(source, always_2d=True)
    rng = np.random.default_rng(319)
    noise = rng.normal(size=base.shape)
    spectrum = np.fft.rfft(noise, axis=0)
    frequency = np.fft.rfftfreq(len(noise), 1/rate)
    spectrum[(frequency < 14000) | (frequency > 20000)] = 0
    noise = np.fft.irfft(spectrum, n=len(base), axis=0)
    noise *= np.sqrt(.02 * np.mean(base**2) / np.mean(noise**2))
    segment = base[int(6*rate):int(6.25*rate)]
    repeat = np.tile(segment, (int(np.ceil(len(base)/len(segment))), 1))[:len(base)]
    repeat *= np.sqrt(np.mean(base**2) / max(1e-12, np.mean(repeat**2)))
    variants = {"gain_up_3db": base*10**(3/20), "gain_down_3db": base*10**(-3/20),
                "high_hiss_2pct": base+noise, "repeat_quarter_second": repeat}
    paths = {"original": source}
    for name, audio in variants.items():
        paths[name] = out / f"{name}.wav"
        sf.write(paths[name], audio, rate, subtype="PCM_16")
    measurer = AudioMeasurer(ROOT / "eval/slider-quality/audio-cache", "cuda:0")
    rows = []
    for name, path in paths.items():
        measured = measurer.measure(path)
        rows.append({"name": name, "sha256": file_hash(path), "rms": measured["rms"],
                     "hf14k_fraction": measured["fullband"]["hf14k_fraction"],
                     "flatness": measured["fullband"]["flatness"],
                     "female_margin": float(np.mean([w["concept"]["female"] for w in measured["windows"]])),
                     "energy_margin": float(np.mean([w["concept"]["loud"] for w in measured["windows"]])),
                     "enjoyment": min(w["aesthetics"]["CE"] for w in measured["windows"]),
                     "production_quality": min(w["aesthetics"]["PQ"] for w in measured["windows"])})
        print(name, {k:round(v,4) for k,v in rows[-1].items() if isinstance(v,float)}, flush=True)
    write_json(out / "measurements.json", {"source_sha256": file_hash(source), "rows": rows,
               "note": "Probes of frozen readout behavior only; no fabricated listening labels."})


if __name__ == "__main__":
    main()
