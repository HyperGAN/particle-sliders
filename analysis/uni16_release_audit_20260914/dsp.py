"""Original-waveform diagnostics; musical repetition is not inferred from loops."""
from collections import Counter
import hashlib
import numpy as np
import soundfile as sf


def longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def measure(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if not y.size or not np.isfinite(y).all():
        raise ValueError("Empty or non-finite audio")
    rms = float(np.sqrt(np.mean(y.astype(np.float64)**2)))
    channel_rms = np.sqrt(np.mean(y.astype(np.float64)**2, axis=0))
    mono = y.mean(1)
    mono_ratio = float(np.sqrt(np.mean(mono.astype(np.float64)**2)) / max(rms, 1e-12))
    frame = max(1, round(sr * .05)); count = len(y) // frame
    levels = np.sqrt(np.mean(y[:count*frame].reshape(count, frame, -1).astype(np.float64)**2, axis=(1, 2)))
    active = np.flatnonzero(levels > max(1e-4, rms * .02))
    interior = levels[active[0]:active[-1]+1] if active.size else levels
    silent = interior < max(1e-4, rms * .005)
    # Exact repeated half-second PCM blocks are an observable diagnostic. A
    # musical loop can be intentional, so only excess relative to controls warns.
    block = max(1, round(sr * .5)); hashes = []
    for start in range(0, len(y)-block+1, block):
        chunk = y[start:start+block]
        if np.sqrt(np.mean(chunk.astype(np.float64)**2)) > max(1e-4, rms * .02):
            hashes.append(hashlib.sha256(chunk.tobytes()).hexdigest())
    repeated = sum(n-1 for n in Counter(hashes).values()) / max(1, len(hashes))
    return dict(rms=rms, duration=len(y)/sr, channels=y.shape[1],
        channel_rms=channel_rms.tolist(), channel_balance=float(min(channel_rms)/max(max(channel_rms), 1e-12)),
        mono_rms_ratio=mono_ratio, interior_silence_seconds=longest(silent)*.05,
        exact_repeated_block_fraction=repeated, repeated_block_seconds=.5,
        crest_factor=float(np.abs(y).max()/max(rms, 1e-12)))


def warnings(candidate, neutral, positive):
    result = []
    if candidate["mono_rms_ratio"] < .1 and min(neutral["mono_rms_ratio"], positive["mono_rms_ratio"]) > .3:
        result.append("mono_cancellation")
    if candidate["channels"] == 2 and candidate["channel_balance"] < .01 and min(neutral["channel_balance"], positive["channel_balance"]) > .2:
        result.append("one_channel_nearly_silent")
    if candidate["interior_silence_seconds"] > max(neutral["interior_silence_seconds"], positive["interior_silence_seconds"]) + 2.:
        result.append("long_interior_gap")
    if candidate["exact_repeated_block_fraction"] > max(neutral["exact_repeated_block_fraction"], positive["exact_repeated_block_fraction"]) + .25:
        result.append("exact_audio_blocks_repeat")
    return result


def components(candidate, neutral, positive, rule):
    limit=rule["component_clip"]
    result={k:max(-limit,min(limit,(candidate[k]-neutral[k])/rule["scales"][k]))
            for k in ("concept","enjoyment","production")}
    excess=max(0.,candidate["hf14k_fraction"]-max(neutral["hf14k_fraction"],positive["hf14k_fraction"])-.002)
    result["artifacts"]=-min(limit,excess/rule["scales"]["artifacts"])
    return result


def diagnostics(candidate, neutral, positive):
    """Original technical thresholds, allowing explicitly missing lyric checks."""
    import math
    keys=("rms","duration","concept","hf14k_fraction","clipped_fraction")
    if any(not math.isfinite(r[k]) for r in (candidate,neutral,positive) for k in keys):
        raise ValueError("Non-finite diagnostic component")
    failures=[]
    if candidate["rms"]<.02*max(neutral["rms"],1e-12):failures.append("near_silence")
    if candidate["duration"]<.5*min(neutral["duration"],positive["duration"]):failures.append("short_output")
    if candidate["clipped_fraction"]>max(neutral["clipped_fraction"],positive["clipped_fraction"])+.01:failures.append("excess_clipping")
    if candidate["hf14k_fraction"]>max(neutral["hf14k_fraction"],positive["hf14k_fraction"])+.02:failures.append("excess_high_frequency_energy")
    lyric_status="not_rescored"
    if all(r.get("lyrics") is not None for r in (candidate,neutral,positive)):
        reference=min(neutral["lyrics"],positive["lyrics"])
        lyric_status="unmeasurable" if reference<.2 else "measurable_proxy"
        if reference>=.2 and candidate["lyrics"]<reference-.15:failures.append("lyric_proxy_regression")
    return dict(failures=failures,technical_screen_passed=not failures,lyric_status=lyric_status,
        absolute_description_margin=candidate["concept"],relative_description_gain=candidate["concept"]-neutral["concept"],
        positive_reference_margin=positive["concept"],reaches_positive_description_reference=candidate["concept"]>=positive["concept"]-.02,
        condition_interpretation="Description similarity, not a calibrated voice classifier")
