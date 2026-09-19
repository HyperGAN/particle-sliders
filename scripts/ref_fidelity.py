#!/usr/bin/env python3
"""Score a listen ladder's caption-swap fidelity: how close each slider clip
is to its pole's no-slider REF clip (same seed) vs the base clip.

For each scale: rms_ratio_to_ref = rms(slider@scale)/rms(REF_pole),
spec_cos = cosine of log-mel-ish band energies vs REF_pole.
A perfectly caption-swap-faithful slider reads rms~1.0 and spec_cos~1.0.

Usage: ref_fidelity.py <ladder_dir>   (needs generate_listen naming)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def bands(p):
    w, sr = sf.read(str(p))
    m = w.mean(axis=1) if w.ndim > 1 else w
    spec = np.abs(np.fft.rfft(m)) + 1e-9
    freqs = np.fft.rfftfreq(len(m), 1 / sr)
    edges = [0, 120, 400, 1200, 3500, 8000, 22050]
    out = [np.sqrt(np.sum(spec[(freqs >= lo) & (freqs < hi)] ** 2)) for lo, hi in zip(edges, edges[1:])]
    return float(np.sqrt(np.mean(m ** 2))), np.array(out)


def main() -> None:
    d = Path(sys.argv[1])
    wavs = sorted(d.glob("*.wav"))
    base = next(w for w in wavs if "zero" in w.stem)
    refs = {("plus", "plus"): None, ("minus", "minus"): None}
    for w in wavs:
        s = w.stem.lower()
        if "_ref_" in s:
            pole = "plus" if "plus" in s.split("_ref_")[1] else "minus"
            refs[pole] = w
    b_rms, b_bands = bands(base)
    # generate_listen names REFs by label (not by plus/minus); they are emitted
    # plus-first, so sort order identifies the poles.
    ref_wavs = sorted(w for w in wavs if "_ref_" in w.stem.lower())
    r_rms = {"plus": bands(ref_wavs[0])[0], "minus": bands(ref_wavs[1])[0]}
    r_bands = {"plus": bands(ref_wavs[0])[1], "minus": bands(ref_wavs[1])[1]}
    print(f"{'scale':>6} {'rms/base':>8} {'rms/REF':>8} {'spec_cos/REF':>12}")
    for w in wavs:
        s = w.stem.lower()
        if "minus" not in s and "plus" in s:
            sign, pole, mag = 1.0, "plus", float(s.split("plus")[-1].split("_")[0])
        elif "minus" in s:
            sign, pole, mag = -1.0, "minus", float(s.split("minus")[-1].split("_")[0])
        else:
            continue
        rms, bd = bands(w)
        cos = float(
            np.dot(bd, r_bands[pole]) / (np.linalg.norm(bd) * np.linalg.norm(r_bands[pole]) + 1e-9)
        )
        print(f"{sign*mag:+5g}  {rms/max(b_rms,1e-9):8.3f} {rms/max(r_rms[pole],1e-9):8.3f} {cos:12.3f}")


if __name__ == "__main__":
    main()
