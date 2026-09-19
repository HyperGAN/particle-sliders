"""Frozen audio measurements; no learned thresholds or checkpoint ranking here."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

CLAP = ("laion/clap-htsat-unfused", "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a")
AES = ("facebook/audiobox-aesthetics", "9b1dd8e5df9af7216e836a98974fe3b82c56ded6")
VERSION = "slider-audio-v1"
DSP_VERSION = "stereo-power-2048-hop1024-v1"

# Fixed sound descriptions; the candidate cannot write its own judge prompt.
CONCEPTS = {
    "female": ("a song with a feminine sounding lead singing voice", "a song with a masculine sounding lead singing voice"),
    "male": ("a song with a masculine sounding lead singing voice", "a song with a feminine sounding lead singing voice"),
    "loud": ("a band playing intensely with forceful drums and energetic singing", "a band playing gently with restrained drums and quiet singing"),
    "quiet": ("a band playing gently with restrained drums and quiet singing", "a band playing intensely with forceful drums and energetic singing"),
    "fast": ("a song played at a fast musical tempo", "a song played at a slow musical tempo"),
    "slow": ("a song played at a slow musical tempo", "a song played at a fast musical tempo"),
    "grit": ("a song with a rough raspy gritty singing voice", "a song with a smooth clean singing voice"),
    "distorted": ("music with distorted electric guitars and saturated sound", "music with clean acoustic instruments and undistorted sound"),
    "joy": ("a song with joyful happy expressive singing", "a song with sad somber expressive singing"),
    "hurt": ("a song with anguished pained emotional singing", "a song with relaxed untroubled singing"),
    "rap": ("a song with rhythmic spoken rap vocals", "a song with sustained melodic singing"),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r"\[[^\]]*\]", " ", text.lower()))


def lyric_features(sheet: str, transcript: str) -> dict[str, float]:
    """Order-aware diagnostic that permits complete phrase repeats.

    Align the transcript against any sequence of whole written lines using
    word-level edit distance. Every transcribed word must be accounted for.
    A repeated word cannot jump back to itself unless it is a complete line.
    Separate sheet coverage prevents repeating one line from scoring perfectly.
    This is a text diagnostic, not proof of vocal intelligibility.
    """
    lines = [words(line) for line in re.split(r"/|\n", sheet)]
    lines = [line for line in lines if line]
    expected, actual = [word for line in lines for word in line], words(transcript)
    if not expected:
        raise ValueError("A nonempty lyric sheet is required")
    vocabulary = set(expected)
    recall = sum(word in set(actual) for word in expected) / len(expected)
    precision = sum(word in vocabulary for word in actual) / max(1, len(actual))
    # dp[j]: minimum edit cost explaining exactly j transcript words by complete
    # phrases. Incomplete phrases still incur deletion cost; this is diagnostic.
    n = len(actual)
    dp = list(range(n + 1))
    for start in range(n):
        for line in lines:
            previous = [dp[start] + j for j in range(n - start + 1)]
            for i, token in enumerate(line, 1):
                current = [dp[start] + i]
                for j, got in enumerate(actual[start:], 1):
                    current.append(min(previous[j] + 1, current[-1] + 1,
                                       previous[j - 1] + (token != got)))
                previous = current
                if i == len(line):
                    for j in range(1, len(current)):
                        dp[start + j] = min(dp[start + j], current[j])
    error = dp[-1] / max(1, n)
    return {"recall": recall, "precision": precision,
            "phrase_accuracy": max(0.0, 1.0 - error),
            "extra_word_rate": 1.0 - precision}


def windows(length: int, rate: int, seconds: int = 10) -> list[tuple[int, int]]:
    """Cover all samples, merging a tiny tail through an overlapping last window."""
    if length <= 0 or rate <= 0:
        raise ValueError("Empty audio or invalid sample rate")
    size = seconds * rate
    starts = list(range(0, max(1, length - size + 1), size))
    if starts[-1] + size < length:
        starts.append(max(0, length - size))
    return [(start, min(length, start + size)) for start in sorted(set(starts))]


def fullband_features(path: Path) -> dict:
    """Stereo power avoids losing out-of-phase content in a mono downmix."""
    import soundfile as sf
    data, rate = sf.read(path, always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Empty or nonfinite audio")
    if len(data) < 2048:
        data = np.pad(data, ((0, 2048-len(data)), (0, 0)))
    frames = np.lib.stride_tricks.sliding_window_view(data, 2048, axis=0)[::1024]
    power = np.abs(np.fft.rfft(frames * np.hanning(2048), axis=-1)) ** 2
    spectrum = power.mean(axis=(0, 1))
    total = max(float(spectrum.sum()), 1e-20)
    frequencies = np.fft.rfftfreq(2048, 1/rate)
    flatness = float(np.exp(np.mean(np.log(spectrum + 1e-20))) / max(float(spectrum.mean()), 1e-20))
    return {"version": DSP_VERSION, "sample_rate": rate,
            "hf14k_fraction": float(spectrum[frequencies >= 14000].sum()/total),
            "flatness": flatness}


class AudioMeasurer:
    def __init__(self, cache: Path, device: str):
        self.cache, self.device = cache, device
        self.clap = self.aes = self.processor = None

    def load(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        from audiobox_aesthetics.model.aes import AesMultiOutput
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.processor = ClapProcessor.from_pretrained(CLAP[0], revision=CLAP[1])
        self.clap = ClapModel.from_pretrained(CLAP[0], revision=CLAP[1]).to(self.device).eval()
        self.aes = AesMultiOutput.from_pretrained(AES[0], revision=AES[1]).to(self.device).eval()
        descriptions = sorted({text for pair in CONCEPTS.values() for text in pair})
        with torch.inference_mode():
            inputs = self.processor.tokenizer(descriptions, padding=True, return_tensors="pt").to(self.device)
            output = self.clap.get_text_features(**inputs)
            embedding = output.pooler_output if hasattr(output, "pooler_output") else output
        self.text_embeddings = dict(zip(descriptions, embedding.float().cpu().numpy()))

    def measure(self, path: Path) -> dict:
        import importlib.metadata
        provenance = {"version": VERSION, "clap": CLAP, "aesthetics": AES,
                      "descriptions": CONCEPTS, "window_seconds": 10, "dtype": "float32",
                      "device_type": self.device.split(":")[0],
                      "packages": {name: importlib.metadata.version(name)
                                   for name in ("torch", "transformers", "audiobox-aesthetics")}}
        sha = file_hash(path)
        cache_path = self.cache / f"{sha}-{digest(provenance)[:16]}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text())
            if result.get("fullband", {}).get("version") != DSP_VERSION:
                result["fullband"] = fullband_features(path)
                write_json(cache_path, result)
            return result
        import torch
        import soundfile as sf
        from torchaudio.functional import resample
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite audio: {path}")
        if self.clap is None:
            self.load()
        mono = torch.from_numpy(data.mean(axis=1))
        frame_results = []
        with torch.inference_mode():
            for start, end in windows(len(data), sr):
                segment = mono[start:end]
                audio48 = resample(segment, sr, 48000).numpy()
                # Each segment fits one CLAP window: no random long-clip crop.
                inputs = self.processor.feature_extractor(audio48, sampling_rate=48000,
                                                          return_tensors="pt").to(self.device)
                output = self.clap.get_audio_features(**inputs)
                embedding = output.pooler_output if hasattr(output, "pooler_output") else output
                embedding = embedding[0].float().cpu().numpy()
                signal = resample(segment, sr, 16000).to(self.device)[None, None, :]
                output = self.aes({"wav": signal, "mask": torch.ones_like(signal, dtype=torch.bool)})
                quality = {axis: float(output[axis].item() * self.aes.target_transform[axis]["std"]
                                       + self.aes.target_transform[axis]["mean"]) for axis in output}
                margins = {name: float(embedding @ self.text_embeddings[pos]
                                      - embedding @ self.text_embeddings[neg])
                           for name, (pos, neg) in CONCEPTS.items()}
                frame_results.append({"start_s": start / sr, "end_s": end / sr,
                                      "concept": margins, "aesthetics": quality})
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
        result = {"sha256": sha, "provenance": provenance, "duration_s": len(data) / sr,
                  "rms": rms, "clipped_fraction": float(np.mean(np.abs(data) >= .999)),
                  "tail_ratio": float(np.sqrt(np.mean(data[-min(len(data), int(1.5 * sr)):].astype(np.float64) ** 2)) / max(rms, 1e-12)),
                  "windows": frame_results, "fullband": fullband_features(path)}
        write_json(cache_path, result)
        return result


FEATURE_NAMES = ("concept_delta", "lyric_recall", "lyric_precision", "phrase_accuracy",
                 "enjoyment", "production_quality", "quality_delta", "song_distance",
                 "level_excursion", "hf14k_excess", "flatness_excess")


def ladder_features(record: dict, measured: dict[str, dict]) -> dict:
    """Worst declared pole/setting; no endpoint can hide behind the other one."""
    baseline = measured[record["baseline"]]
    base_q = min(w["aesthetics"]["PQ"] for w in baseline["windows"])
    rows = []
    for clip in record["clips"]:
        audio = measured[clip["path"]]
        concept = clip["concept"].lower()
        if concept not in CONCEPTS:
            raise ValueError(f"No frozen concept description for {concept}")
        lyrics = lyric_features(record["lyrics"], clip["transcript"])
        rows.append({
            "concept_delta": min(w["concept"][concept] for w in audio["windows"])
                             - float(np.mean([w["concept"][concept] for w in baseline["windows"]])),
            "lyric_recall": lyrics["recall"], "lyric_precision": lyrics["precision"],
            "phrase_accuracy": lyrics["phrase_accuracy"],
            "enjoyment": min(w["aesthetics"]["CE"] for w in audio["windows"]),
            "production_quality": min(w["aesthetics"]["PQ"] for w in audio["windows"]),
            "quality_delta": min(w["aesthetics"]["PQ"] for w in audio["windows"]) - base_q,
            "song_distance": float(clip["same_song"]),
            "level_excursion": abs(math.log(max(audio["rms"], 1e-12) / max(baseline["rms"], 1e-12))),
            "hf14k_excess": max(0., audio["fullband"]["hf14k_fraction"] - baseline["fullband"]["hf14k_fraction"]),
            "flatness_excess": max(0., audio["fullband"]["flatness"] - baseline["fullband"]["flatness"]),
        })
    if not rows:
        raise ValueError("No nonzero operating settings")
    features = {name: (max if name in {"song_distance", "level_excursion", "hf14k_excess", "flatness_excess"} else min)(row[name] for row in rows)
                for name in FEATURE_NAMES}
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Nonfinite features")
    return features
