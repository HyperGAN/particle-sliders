#!/usr/bin/env python3
"""LM-slider render-path scorer (the LM half of the scoring contract).

SCORING.md Amendment 6 scopes the transformer contract to transformer sliders:
an LM half changes the arrangement by design, so same-seed onset/envelope
identity against the scale-0 clip is ~0 for *any* working LM slider and G3/G4/G6
reject shipped, ears-approved checkpoints. This module is the promised
LM-specific instrument. Its channels are the ones that matched ears on eight
axes in docs/lm-uni-v2-garble.md ("What to measure instead of ears"):

  * whisper lyric recall of the yaml line on the rendered clip (the garble gate;
    train-time c+ / p% / loss are blind to it),
  * whisper encoder-embedding geometry against the caption-swap REF pair
    (direction + "is this still one song"),
  * level / ending DSP with the repo's frozen rms convention.

Gates are VETOES, never terms a strong effect can trade against (SCORING.md).

FROZEN 2026-09-02. One calibration pass against eval/lm_score_labels.json set
every threshold here; they are not tuned on candidates again. That pass also
DEMOTED two channels to diagnostics because they reject ears-approved sliders:
the U6 null (rejects 6 of 6 ears-PASS folders) and U3's rank-monotonicity term
(anti-correlates with ears). What is left is a no-harm certificate -- a PASS
means "still sings the line, still the same song, not silent, not reversed",
NOT "the slider works". See LM-SCORING.md, "What the calibration pass decided".

    PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
    $PY scripts/lm_score.py selftest
    HF_HOME=/ml2/trained/huggingface CUDA_VISIBLE_DEVICES=0 \
      $PY scripts/lm_score.py score eval/listen/uni-v2/grit-lm-uni-v2 \
        --cache_dir eval/lm_score_cache --device cuda:0 --out board.tsv

`score` writes lm_scores.json into every folder it reads. Measurements are
cached by file content hash, so re-scoring is free and threshold changes do not
cost GPU time.

Contract: LM-SCORING.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MEASURE_VERSION = 1  # bump to invalidate the measurement cache
MAX_NEW_TOKENS = 128

# Fallback lyric sheet, matching scripts/blindspot_whisper.py's EXPECTED, used
# only when a folder has no LISTEN.md lyrics line.
DEFAULT_LYRICS = "I can feel it in the air tonight / Louder now or fade away"
DEFAULT_REQUESTED_S = 20.0

# ---------------------------------------------------------------------------
# Thresholds. FROZEN 2026-09-02 by the calibration pass documented in
# LM-SCORING.md ("Validation against locked verdicts" / "What the calibration
# pass decided"), against the 11 PASS/FAIL folders of eval/lm_score_labels.json.
# BORDERLINE folders were reported, never used to place a threshold. Never
# tuned on candidates after this date.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # U1 level -- unchanged by calibration; no labeled folder fires it, it is
    # kept for the SCORING.md D-pole-uni silence attack (selftest "silence").
    "rms_ratio_min": 0.02,        # vs the folder's own scale-0 clip
    "d_ln_rms_max": 1.15,         # ~10 dB, at |s| <= 1
    # U2 lyric hold -- VERIFIED on the labeled set: fires on 3/3 ears-FAIL
    # garbles (recall 0.00 / 0.00 / 0.31) and on 0/6 ears-PASS folders
    # (worst PASS margin: gender-lm-v9 +1 recall 0.909 on hold_ref 0.4545).
    "recall_floor": 0.40,
    "recall_drop": 0.35,          # vs max(base, REF+)
    # U4 same-song -- RE-DERIVED from all labeled folders. max same_song/
    # cross_anchor over |s| <= 1: ears-PASS top out at 0.800
    # (tempo-lm-uni-v2), the two replacement-style ears-FAILs sit at 1.012
    # (grit-lm-uni-v2) and 1.086 (gender-lm-v16). Gap midpoint 0.906 -> 0.90.
    "same_song_frac": 0.90,       # x cross_anchor (different-song embedding scale)
    # U5 ending -- unchanged; every listen ladder here is 20 s, so U5 only warns.
    "tail_ratio_max": 0.5,
    "natural_end_margin_s": 0.5,  # duration < requested - margin = ended on its own
    "ending_veto_duration_s": 45.0,  # below this U5 warns; the real gate is 60-90 s
    # U6 null -- DIAGNOSTIC ONLY after calibration (see VETO_GATES). Kept so the
    # numbers that demoted it stay on the board.
    "null_pct": 95.0,
    "null_floor": 0.01,           # denominator floor for the logged E_emb
    # Diagnostic-only reference values. NOT gates: the calibration pass measured
    # both against the labeled corpus and neither separates PASS from FAIL.
    "spearman_ref": 0.8,          # ladder rank monotonicity of emb proj
    "ref_axis_dist_ref": 0.08,    # "close REF pair" (docs/lm-pair-exam.md)
}

GATE_NAMES = ("U0_shape", "U1_level", "U2_lyric_hold", "U3_direction",
              "U4_same_song", "U5_ending", "U6_null")

# Which gates can reject. U6 is computed and logged but cannot veto: the
# calibration pass measured it rejecting 4 of 4 ears-PASS bipolar ladders and
# all 6 ears-PASS ladders overall (LM-SCORING.md, "U6 has no power").
VETO_GATES = ("U0_shape", "U1_level", "U2_lyric_hold", "U3_direction",
              "U4_same_song", "U5_ending")

BAND_EDGES = [0, 120, 400, 1200, 3500, 8000, 22050]  # scripts/ref_fidelity.py

DEFAULT_POOL_GLOBS = (
    "eval/listen/uni-v2/*/01_slider_neutral_base_zero.wav",
    "eval/listen/uni-v2/*/05_REF_prompt_Off_no_slider.wav",
    "eval/listen/uni-lyric/*/01_slider_neutral_base_zero.wav",
    "eval/listen/uni-lyric/*/05_REF_prompt_Off_no_slider.wav",
)


# ---------------------------------------------------------------------------
# folder parsing
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    """One rendered wav in a listen ladder."""

    name: str
    role: str                 # "slider" | "ref_plus" | "ref_minus" | "ref_off"
    scale: float | None = None
    path: Path | None = None


@dataclass
class Ladder:
    folder: Path
    name: str
    kind: str                 # "uni" | "bipolar"
    clips: list[Clip]
    lyrics: str
    requested_s: float
    plus_label: str = ""
    minus_label: str = ""
    seed: int | None = None
    notes: list[str] = field(default_factory=list)

    def by_role(self, role: str) -> Clip | None:
        for c in self.clips:
            if c.role == role:
                return c
        return None

    def slider(self, scale: float) -> Clip | None:
        for c in self.clips:
            if c.role == "slider" and c.scale == scale:
                return c
        return None

    @property
    def scales(self) -> list[float]:
        return sorted(c.scale for c in self.clips if c.role == "slider")


def strip_sections(text: str) -> str:
    """Drop [verse]/[chorus]-style section tags from a lyric sheet."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def lyric_recall(hyp: str, sheet: str) -> float:
    """Word recall of the lyric sheet in a transcript (blindspot_whisper.py)."""
    exp = norm_words(strip_sections(sheet))
    if not exp:
        return 0.0
    hw = set(norm_words(hyp))
    return sum(w in hw for w in exp) / len(exp)


def parse_listen_md(folder: Path) -> tuple[str, float, int | None]:
    """Return (lyric sheet, requested seconds, seed) from LISTEN.md."""
    md = folder / "LISTEN.md"
    lyrics, requested, seed = DEFAULT_LYRICS, DEFAULT_REQUESTED_S, None
    if md.is_file():
        text = md.read_text(encoding="utf-8")
        m = re.search(r"^-\s*lyrics:\s*`(.+?)`\s*$", text, re.M)
        if m:
            lyrics = m.group(1)
        m = re.search(r"requested:\s*([0-9.]+)\s*s", text)
        if m:
            requested = float(m.group(1))
        m = re.search(r"seed:\s*(\d+)", text)
        if m:
            seed = int(m.group(1))
    return lyrics, requested, seed


_SLIDER_RE = re.compile(r"^\d+_slider_(?P<label>.+?)_(?P<sign>plus|minus)(?P<mag>[0-9.]+)$")
_REF_RE = re.compile(r"^\d+_REF_prompt_(?P<label>.+?)_no_slider$", re.I)


def parse_ladder(folder: Path, names: list[str] | None = None) -> Ladder:
    """Parse a generate_listen.py folder into a Ladder.

    `names` overrides the wav listing (stub backends carry no audio on disk).
    """
    folder = Path(folder)
    if names is None:
        names = sorted(p.name for p in folder.glob("*.wav"))
    clips: list[Clip] = []
    plus_label = minus_label = ""
    ref_names: list[str] = []
    for name in names:
        stem = name[:-4] if name.lower().endswith(".wav") else name
        if "zero" in stem.lower() and "_slider_" in stem:
            clips.append(Clip(name, "slider", 0.0, folder / name))
            continue
        m = _SLIDER_RE.match(stem)
        if m:
            sign = 1.0 if m.group("sign") == "plus" else -1.0
            scale = sign * float(m.group("mag"))
            if sign > 0:
                plus_label = plus_label or m.group("label")
            else:
                minus_label = minus_label or m.group("label")
            clips.append(Clip(name, "slider", scale, folder / name))
            continue
        if _REF_RE.match(stem):
            ref_names.append(name)
    if not ref_names:
        raise ValueError(f"{folder}: no REF clips found")
    kind = "bipolar" if any(c.role == "slider" and c.scale < 0 for c in clips) else "uni"

    # REF roles. generate_listen names REFs by caption label, not by pole, and
    # emits them plus-first (see scripts/ref_fidelity.py); match the label to
    # the slider labels first and fall back to that emission order.
    def label_of(name: str) -> str:
        return _REF_RE.match(name[:-4] if name.lower().endswith(".wav") else name).group("label")

    ref_names = sorted(ref_names)
    assigned: dict[str, str] = {}
    for name in ref_names:
        lab = label_of(name)
        if lab.lower() == "off":
            assigned[name] = "ref_off"
        elif plus_label and lab.lower() == plus_label.lower():
            assigned[name] = "ref_plus"
        elif minus_label and lab.lower() == minus_label.lower():
            assigned[name] = "ref_minus"
    unassigned = [n for n in ref_names if n not in assigned]
    want = ["ref_plus"] + (["ref_minus"] if kind == "bipolar" else ["ref_off"])
    for role in want:
        if role in assigned.values() or not unassigned:
            continue
        assigned[unassigned.pop(0)] = role
    for name in ref_names:
        clips.append(Clip(name, assigned.get(name, "ref_other"), None, folder / name))

    lyrics, requested, seed = parse_listen_md(folder)
    lad = Ladder(folder=folder, name=folder.name, kind=kind, clips=clips,
                 lyrics=lyrics, requested_s=requested,
                 plus_label=plus_label, minus_label=minus_label, seed=seed)
    if lad.slider(0.0) is None:
        raise ValueError(f"{folder}: no scale-0 clip")
    if lad.by_role("ref_plus") is None:
        raise ValueError(f"{folder}: no plus REF clip")
    if kind == "bipolar" and lad.by_role("ref_minus") is None:
        raise ValueError(f"{folder}: bipolar ladder without a minus REF")
    return lad


# ---------------------------------------------------------------------------
# measurement backends
# ---------------------------------------------------------------------------

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rms_pc(x: np.ndarray) -> float:
    """The repo's one rms convention: per-channel power over the de-interleaved
    array (SCORING.md Amendment 2 — mono downmix mixes level with stereo width)."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def dsp_features(data: np.ndarray, sr: int, requested_s: float) -> dict:
    """Level / ending / spectrum channels for one clip."""
    if data.ndim == 1:
        data = data[:, None]
    dur = data.shape[0] / float(sr)
    overall = rms_pc(data)
    tail_n = min(data.shape[0], int(1.5 * sr))
    tail = rms_pc(data[-tail_n:]) if tail_n else 0.0
    mono = data.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono)) + 1e-9
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    bands = [float(np.sqrt(np.sum(spec[(freqs >= lo) & (freqs < hi)] ** 2)))
             for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:])]
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    return {
        "rms": overall,
        "duration": dur,
        "tail_rms": tail,
        "tail_ratio": float(tail / overall) if overall > 0 else 0.0,
        "ln_rms": float(math.log(max(overall, 1e-12))),
        "log_bands": [float(math.log(b + 1e-9)) for b in bands],
        "centroid": centroid,
        "natural_end": bool(dur < requested_s - THRESHOLDS["natural_end_margin_s"]),
    }


class StubBackend:
    """Injected measurements — no audio, no model. Used by selftest and tests."""

    model_id = "stub"

    def __init__(self, table: dict[str, dict]):
        self.table = table

    def measure(self, path: Path, requested_s: float) -> dict:
        key = str(path)
        if key not in self.table:
            key = Path(path).name
        if key not in self.table:
            raise KeyError(f"stub backend has no measurement for {path}")
        m = dict(self.table[key])
        m.setdefault("duration", requested_s)
        m.setdefault("natural_end", bool(m["duration"] < requested_s - 0.5))
        m.setdefault("tail_ratio", 0.3)
        m.setdefault("ln_rms", float(math.log(max(m.get("rms", 0.1), 1e-12))))
        m.setdefault("centroid", 2000.0)
        m.setdefault("log_bands", [0.0] * 6)
        m.setdefault("text", "")
        return m


class WhisperBackend:
    """whisper-large-v3-turbo transcripts + encoder embeddings + DSP, cached.

    Cache key is (file content hash, MEASURE_VERSION, model id), so re-runs and
    threshold changes cost nothing and two folders sharing a byte-identical clip
    (the uni zero renders do) are measured once.
    """

    model_id = WHISPER_MODEL

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None
        self._proc = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self._proc = WhisperProcessor.from_pretrained(self.model_id)
        self._model = (WhisperForConditionalGeneration
                       .from_pretrained(self.model_id, dtype=dtype)
                       .to(self.device).eval())

    def _load_audio(self, path: Path):
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return data, sr

    def _to_16k(self, data: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional as AF

        mono = torch.from_numpy(data.mean(axis=1).astype(np.float32))
        if sr != 16000:
            mono = AF.resample(mono, sr, 16000)
        return mono.numpy()

    def measure(self, path: Path, requested_s: float) -> dict:
        path = Path(path)
        digest = sha1_file(path)
        cache = self.cache_dir / f"{digest}-v{MEASURE_VERSION}.json"
        if cache.is_file():
            m = json.loads(cache.read_text(encoding="utf-8"))
            # natural_end depends on the folder's requested duration, not the file
            m["natural_end"] = bool(m["duration"] < requested_s
                                    - THRESHOLDS["natural_end_margin_s"])
            return m
        import torch

        data, sr = self._load_audio(path)
        m = dsp_features(data, sr, requested_s)
        audio = self._to_16k(data, sr)
        self._load()
        feats = self._proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, self._model.dtype)
        with torch.no_grad():
            enc = self._model.model.encoder(feats).last_hidden_state[0]  # (1500, d)
            n_real = max(1, int(len(audio) / 16000 / 30.0 * enc.shape[0]))
            emb = enc[:n_real].mean(dim=0).float().cpu().numpy()
            ids = self._model.generate(feats, language="en", task="transcribe",
                                       do_sample=False, num_beams=1,
                                       max_new_tokens=MAX_NEW_TOKENS)
        m["text"] = self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        m["emb"] = [float(v) for v in emb]
        m["sha1"] = digest
        m["model_id"] = self.model_id
        cache.write_text(json.dumps(m), encoding="utf-8")
        return m


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _vec(m: dict) -> np.ndarray:
    return np.asarray(m["emb"], dtype=np.float64)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho with average ranks; 0.0 when a side is constant."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den > 0 else 0.0


def substitution_null(zero_vec: np.ndarray, v: np.ndarray,
                      pool: list[np.ndarray]) -> list[float]:
    """proj of each foreign pool clip substituted in place of the candidate."""
    vv = float(np.dot(v, v))
    if vv <= 0:
        return []
    return [float(np.dot(p - zero_vec, v) / vv) for p in pool]


def null_stats(zero_emb: np.ndarray, v: np.ndarray, pool: list[np.ndarray]) -> dict:
    """Null calibration from a pool of foreign no-slider clips.

    cross_anchor: median cosine distance between distinct pool clips — the
    "different song, same-ish caption" scale for U4.
    null95: 95th percentile of proj obtained by substituting each pool clip in
    place of the +1 clip — the U6 floor. Same construction as SCORING.md G6
    (pseudo-candidates built from renders the slider never touched).
    """
    vv = float(np.dot(v, v))
    projs = [float(np.dot(p - zero_emb, v) / vv) for p in pool] if vv > 0 else []
    dists, paired = [], []
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            if i < j:
                dists.append(1.0 - cos(pool[i], pool[j]))
            if vv > 0:
                paired.append(float(np.dot(pool[i] - pool[j], v) / vv))
    pct = THRESHOLDS["null_pct"]
    return {
        "n_pool": len(pool),
        "cross_anchor": float(np.median(dists)) if dists else float("nan"),
        "pool_dist_to_zero_median": (float(np.median([1.0 - cos(p, zero_emb) for p in pool]))
                                     if pool else float("nan")),
        "null95": float(np.percentile(projs, pct)) if projs else float("nan"),
        "null_median": float(np.median(projs)) if projs else float("nan"),
        "null_max": float(np.max(projs)) if projs else float("nan"),
        # Diagnostic only. The paired null re-rolls BOTH ends of the difference
        # (p_i - p_j), so it is centred at 0; the substitution null shares the
        # candidate's anchor at the zero clip and is therefore the matched
        # comparison for proj. On uni ladders the two differ by ~0.5 (the
        # squared distance of the zero clip from the pool mean, in ||v||^2
        # units) — see LM-SCORING.md "What the first calibration run measured".
        "null95_paired": float(np.percentile(paired, pct)) if paired else float("nan"),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_ladder(lad: Ladder, meas: dict[str, dict], pool: list[np.ndarray],
                 thresholds: dict | None = None, model_id: str = "unknown",
                 pool_dsp: list[np.ndarray] | None = None) -> dict:
    """Pure data -> verdict. No file or model I/O, so selftest can feed attacks."""
    T = dict(THRESHOLDS if thresholds is None else thresholds)
    zero = meas[lad.slider(0.0).name]
    ref_plus = meas[lad.by_role("ref_plus").name]
    ref_minus = meas[lad.by_role("ref_minus").name] if lad.by_role("ref_minus") else None
    ref_off = meas[lad.by_role("ref_off").name] if lad.by_role("ref_off") else None

    e0 = _vec(zero)
    # Concept direction from the caption-swap REFs. CAVEAT: ||v|| is a
    # single-seed caption-swap magnitude, and this campaign measured that
    # magnitude to be unstable across seeds (SCORING.md ruling 2: the dust swap's
    # centroid ranges -50%..+92% over 10 seeds). Only its DIRECTION is
    # certifiable, so every gate below uses sign, rank, monotonicity or
    # null-calibrated units — never a raw proj magnitude.
    v = _vec(ref_plus) - (_vec(ref_minus) if ref_minus is not None else e0)
    vnorm = float(np.linalg.norm(v))
    vhat = v / vnorm if vnorm > 0 else v

    # The same projection in DSP space (6 log bands + ln rms + ln centroid,
    # scripts/ref_fidelity.py's bands). After the 2026-09-02 calibration pass
    # this is HALF of the U3 sign veto (the emb-OR-dsp rule) and the direction
    # term of the ranking scalar E, because the whisper-embedding proj sign at
    # the unit rung contradicts ears on 4 of 6 ears-PASS folders while the DSP
    # sign agrees with all 6. It is still a caption-swap FRACTION, so it never
    # sets a threshold on its own magnitude -- only its sign gates.
    def _dsp(m: dict) -> np.ndarray:
        return np.asarray(list(m["log_bands"]) + [m["ln_rms"], math.log(max(m["centroid"], 1e-9))],
                          dtype=np.float64)

    d0 = _dsp(zero)
    dv = _dsp(ref_plus) - (_dsp(ref_minus) if ref_minus is not None else d0)
    dvv = float(np.dot(dv, dv))

    scales = lad.scales
    ch: dict[str, dict] = {}
    for s in scales:
        m = meas[lad.slider(s).name]
        e = _vec(m)
        d = e - e0
        dd = _dsp(m) - d0
        proj = float(np.dot(d, vhat) / vnorm) if vnorm > 0 else 0.0
        resid = d - (proj * vnorm) * vhat
        ch[f"{s:+g}"] = {
            "scale": s,
            "file": lad.slider(s).name,
            "rms": m["rms"],
            "rms_ratio": float(m["rms"] / zero["rms"]) if zero["rms"] > 0 else float("inf"),
            "d_ln_rms": float(m["ln_rms"] - zero["ln_rms"]),
            "centroid_ratio": float(m["centroid"] / zero["centroid"]) if zero["centroid"] else 0.0,
            "duration": m["duration"],
            "natural_end": bool(m["natural_end"]),
            "tail_ratio": float(m["tail_ratio"]),
            "lyric_recall": round(lyric_recall(m.get("text", ""), lad.lyrics), 4),
            "text": m.get("text", ""),
            "proj": proj,
            "proj_cos": cos(d, vhat),
            "offaxis": float(np.linalg.norm(resid) / vnorm) if vnorm > 0 else 0.0,
            "same_song": 1.0 - cos(e, e0),
            "dsp_proj": float(np.dot(dd, dv) / dvv) if dvv > 0 else 0.0,
            "dsp_cos": cos(dd, dv),
        }

    recall_base = ch["+0"]["lyric_recall"]
    recall_ref_plus = round(lyric_recall(ref_plus.get("text", ""), lad.lyrics), 4)
    recall_ref_minus = (round(lyric_recall(ref_minus.get("text", ""), lad.lyrics), 4)
                        if ref_minus is not None else None)
    recall_ref_off = (round(lyric_recall(ref_off.get("text", ""), lad.lyrics), 4)
                      if ref_off is not None else None)

    nulls = null_stats(e0, v, pool)
    # The same substitution null taken in DSP space, so the demotion of U6 is
    # visible in both spaces on the board rather than asserted. On the labeled
    # corpus the DSP null is no more powerful than the embedding one: only
    # 2 of 6 ears-PASS folders clear it at the unit plus rung.
    dsp_null = substitution_null(d0, dv, list(pool_dsp or []))
    nulls["null95_dsp"] = (float(np.percentile(dsp_null, THRESHOLDS["null_pct"]))
                           if dsp_null else float("nan"))
    nulls["null_median_dsp"] = float(np.median(dsp_null)) if dsp_null else float("nan")
    unit_scales = [s for s in scales if 0 < abs(s) <= 1.0]  # the product setting
    nonzero = [s for s in scales if s != 0]

    # The product setting is |s| = 1. The rung a gate is read at is the one
    # closest to 1 from below on each side; a ladder with no rung in 0 < |s| <= 1
    # was never rendered at the product setting and cannot be evaluated against
    # the contract (U0_shape). For reporting we then fall back to the nearest
    # rung the ladder does have, and say so.
    unit_plus = max((s for s in scales if 0 < s <= 1.0), default=None)
    unit_minus = min((s for s in scales if -1.0 <= s < 0), default=None)
    plus_fallback = minus_fallback = False
    if unit_plus is None:
        unit_plus = min((s for s in scales if s > 0), default=None)
        plus_fallback = unit_plus is not None
    if unit_minus is None and lad.kind == "bipolar":
        unit_minus = max((s for s in scales if s < 0), default=None)
        minus_fallback = unit_minus is not None
    if not unit_scales:  # U0 ladders still need numbers on the board
        unit_scales = [s for s in (unit_plus, unit_minus) if s is not None]

    def at(s):
        return ch[f"{s:+g}"] if s is not None else None

    gates: dict[str, dict] = {}

    # U0 shape ---------------------------------------------------------------
    # NOT a slider failure: the ladder does not contain the product setting.
    shape_ok = not plus_fallback and not minus_fallback and unit_plus is not None
    gates["U0_shape"] = {
        "pass": bool(shape_ok),
        "unit_plus": unit_plus, "unit_minus": unit_minus,
        "plus_fallback": plus_fallback, "minus_fallback": minus_fallback,
        "scales": scales,
        "detail": ("the ladder must have a rung at 0 < |s| <= 1 (the product "
                   "setting) on every pole it claims; otherwise the contract "
                   "has no data and the verdict is UNSCOREABLE"),
    }

    # U1 level -------------------------------------------------------------
    bad_floor = [s for s in nonzero if ch[f"{s:+g}"]["rms_ratio"] < T["rms_ratio_min"]]
    bad_swing = [s for s in unit_scales if abs(ch[f"{s:+g}"]["d_ln_rms"]) > T["d_ln_rms_max"]]
    gates["U1_level"] = {
        "pass": not bad_floor and not bad_swing,
        "silent_scales": bad_floor, "loud_scales": bad_swing,
        "detail": f"rms_ratio_min={T['rms_ratio_min']}, |d_ln_rms|<={T['d_ln_rms_max']}",
    }

    # U2 lyric hold --------------------------------------------------------
    hold_ref = max(recall_base, recall_ref_plus)
    bad_lyrics = [s for s in unit_scales
                  if ch[f"{s:+g}"]["lyric_recall"] < T["recall_floor"]
                  or ch[f"{s:+g}"]["lyric_recall"] < hold_ref - T["recall_drop"]]
    gates["U2_lyric_hold"] = {
        "pass": not bad_lyrics, "failed_scales": bad_lyrics,
        "hold_ref": hold_ref, "recall_base": recall_base,
        "recall_ref_plus": recall_ref_plus,
        "detail": f"recall>={T['recall_floor']} and >= max(base,REF+)-{T['recall_drop']}",
    }

    # U3 direction ---------------------------------------------------------
    # FROZEN 2026-09-02. Three things changed here, all forced by the labeled
    # corpus and all costing power rather than buying it:
    #  * the veto reads the PLUS pole only. The product claim in LM-SCORING.md
    #    is "at +1 it leans toward the + REF"; the minus pole's sign is logged.
    #    Gating it would reject ears-PASS gender-lm-20s, whose -2 clip runs
    #    +0.156 (emb) and +0.715 (DSP) -- both toward Female.
    #  * the sign test is emb-proj OR dsp-proj, not emb alone. The whisper
    #    embedding's sign at the unit rung is wrong on 4 of 6 ears-PASS folders
    #    (v18 -0.016, gender-lm-v9 -0.009, energy-v3 -0.296, gender-20s -0.080);
    #    the DSP sign is right on 6 of 6. Neither is trusted alone: DSP is blind
    #    on non-spectral concepts (gender F0), emb is a speech-content space.
    #  * the Spearman rank requirement is REMOVED. Over the labeled set it
    #    anti-correlates with ears -- ears-PASS rho = +1.0/+0.5/+0.90/-0.40/
    #    -0.30/-1.0 against ears-FAIL rho = +1.0/+1.0/+1.0/+1.0/-0.70. It is
    #    still computed and logged.
    # HONEST LIMIT: with these rules U3 fires on 0 of 5 ears-FAIL folders. It
    # has NO measured power on this corpus; it is retained only as a floor
    # against a flagrantly reversed slider (selftest "reversed").
    cp, cm = at(unit_plus), at(unit_minus)
    proj_plus = cp["proj"] if cp else None
    proj_minus = cm["proj"] if cm else None
    dsp_plus = cp["dsp_proj"] if cp else None
    dsp_minus = cm["dsp_proj"] if cm else None
    rho = spearman(scales, [ch[f"{s:+g}"]["proj"] for s in scales])
    rho_dsp = spearman(scales, [ch[f"{s:+g}"]["dsp_proj"] for s in scales])
    sign_ok = bool(cp is not None
                   and ((proj_plus is not None and proj_plus > 0)
                        or (dsp_plus is not None and dsp_plus > 0)))
    minus_sign_ok = (None if cm is None else
                     bool((proj_minus is not None and proj_minus < 0)
                          or (dsp_minus is not None and dsp_minus < 0)))
    gates["U3_direction"] = {
        "pass": sign_ok,
        "unit_plus": unit_plus, "unit_minus": unit_minus,
        "proj_plus": proj_plus, "proj_minus": proj_minus,
        "dsp_proj_plus": dsp_plus, "dsp_proj_minus": dsp_minus,
        "sign_ok": sign_ok,
        # logged, NOT gated (see comment above)
        "minus_sign_ok": minus_sign_ok,
        "spearman": rho, "spearman_dsp": rho_dsp,
        "detail": ("plus pole only: proj>0 OR dsp_proj>0 at the unit rung. "
                   "minus-pole sign and Spearman rho are diagnostics -- neither "
                   "separated ears-PASS from ears-FAIL on 2026-09-02"),
    }

    # U4 same-song ---------------------------------------------------------
    anchor = nulls["cross_anchor"]
    limit = T["same_song_frac"] * anchor if anchor == anchor else float("nan")
    bad_song = [s for s in unit_scales if ch[f"{s:+g}"]["same_song"] > limit] \
        if limit == limit else []
    gates["U4_same_song"] = {
        "pass": bool(limit == limit and not bad_song),
        "failed_scales": bad_song, "limit": limit, "cross_anchor": anchor,
        "detail": f"cos-dist to zero <= {T['same_song_frac']} x cross_anchor at |s|<=1",
    }

    # U5 ending ------------------------------------------------------------
    hot = [s for s in unit_scales
           if not ch[f"{s:+g}"]["natural_end"]
           and ch[f"{s:+g}"]["tail_ratio"] > T["tail_ratio_max"]]
    veto_enabled = lad.requested_s >= T["ending_veto_duration_s"]
    gates["U5_ending"] = {
        "pass": (not hot) or (not veto_enabled),
        "warn": bool(hot), "hot_tail_scales": hot, "veto_enabled": bool(veto_enabled),
        "ending": "hot_tail" if hot else "ok",
        "detail": ("natural end or tail_ratio<=%.2f; WARN below %.0f s (short clips hit "
                   "the cap mid-song), veto at 60-90 s"
                   % (T["tail_ratio_max"], T["ending_veto_duration_s"])),
    }

    # U6 null -- DIAGNOSTIC ONLY, never a veto (frozen 2026-09-02) -----------
    # uni: v = emb(REF+) - emb(zero), so the candidate statistic and the
    #   substitution null share the zero anchor and both carry
    #   ||emb(zero) - pool mean||^2. The MEDIAN foreign clip projects +0.84
    #   (grit) / +0.46 (gender) against candidate proj[+1] of 0.59 / 0.57: no
    #   uni slider can clear it. Status "not_applicable" until same-caption
    #   re-rolled zero renders exist (the >=3-seed promotion stage provides
    #   them). The paired null is NOT substituted in: it re-rolls both ends and
    #   is unmatched to the candidate statistic. It stays logged.
    # bipolar: the null IS centred (median +0.09 v18, -0.11 v16), but the
    #   statistic it thresholds is the emb proj -- the channel that contradicts
    #   ears at the unit rung. It rejects 4 of 4 ears-PASS bipolar ladders
    #   (v18 -0.016 vs 0.244; gender-lm-v9 -0.009 vs 0.577; energy-v3 -0.296 vs
    #   0.121; gender-20s -0.080 vs 0.185). Taken in DSP space it rejects 4 of 6
    #   ears-PASS folders instead. Status "no_power".
    n95 = nulls["null95"]
    null_ok = n95 == n95 and proj_plus is not None and proj_plus >= n95
    if lad.kind == "bipolar":
        null_ok = null_ok and proj_minus is not None and -proj_minus >= n95
    n95d = nulls["null95_dsp"]
    gates["U6_null"] = {
        "pass": True,               # never vetoes; see VETO_GATES
        "veto": False,
        "would_pass": bool(null_ok),
        "status": "not_applicable" if lad.kind == "uni" else "no_power",
        "null95": n95, "null95_dsp": n95d,
        "dsp_beats_null": bool(n95d == n95d and dsp_plus is not None
                               and dsp_plus >= n95d),
        "detail": ("DIAGNOSTIC after the 2026-09-02 calibration pass: the "
                   "substitution null shares the candidate's zero anchor on uni "
                   "ladders, and on bipolar ladders it thresholds the emb proj, "
                   "which rejects every ears-PASS folder. Logged, never a veto"),
    }

    # ranking scalar among gate-passers ------------------------------------
    # NOT CERTIFIED. The null denominator the drafted spec used is unusable
    # (U6 above), so E's direction term is the DSP caption-swap fraction --
    # exactly the kind of raw magnitude LM-SCORING.md forbids as a GATE. It is
    # never a gate, never a certification, and never comparable across seeds;
    # it exists to put gate-passers in a deterministic order. The old
    # null-unit embedding scalar is kept as E_emb so the two can be compared.
    den = max(n95 if n95 == n95 else 0.0, T["null_floor"])
    sides: dict[str, dict] = {}
    for side, s in (("plus", unit_plus), ("minus", unit_minus)):
        c = at(s)
        if c is None:
            continue
        sgn = 1.0 if side == "plus" else -1.0
        # lyric_factor is defined on the +1 clip in the spec; on a bipolar
        # ladder each side is graded by its own clip's recall so a garbled minus
        # pole cannot hide behind a clean plus pole.
        lf = min(1.0, c["lyric_recall"] / max(recall_base, T["recall_floor"]))
        sides[side] = {"scale": s, "proj": c["proj"], "dsp_proj": c["dsp_proj"],
                       "proj_null_units": (c["proj"] * sgn) / den,
                       "lyric_factor": lf,
                       "E": (c["dsp_proj"] * sgn) * lf,
                       "E_emb": ((c["proj"] * sgn) / den) * lf}
    E = min((d["E"] for d in sides.values()), default=0.0)
    E_emb = min((d["E_emb"] for d in sides.values()), default=0.0)
    # A negative E means a pole that moves the wrong way in DSP space; the
    # squash is taken on the clamped value rather than walking through the
    # pole at E = -2.
    squashed = max(E, 0.0) / (max(E, 0.0) + 2.0)

    fired = [g for g, d in gates.items()
             if g in VETO_GATES and not d["pass"]]
    return {
        "folder": str(lad.folder),
        "name": lad.name,
        "kind": lad.kind,
        "seeds": 1,  # single-seed listen ladders: every verdict here is PROVISIONAL
        "seed": lad.seed,
        "requested_s": lad.requested_s,
        "lyrics": lad.lyrics,
        "plus_label": lad.plus_label,
        "minus_label": lad.minus_label,
        "whisper_model": model_id,
        "measure_version": MEASURE_VERSION,
        "channels": ch,
        "refs": {
            "ref_plus": {"file": lad.by_role("ref_plus").name,
                         "lyric_recall": recall_ref_plus,
                         "text": ref_plus.get("text", "")},
            "ref_minus": ({"file": lad.by_role("ref_minus").name,
                           "lyric_recall": recall_ref_minus,
                           "text": ref_minus.get("text", "")} if ref_minus else None),
            "ref_off": ({"file": lad.by_role("ref_off").name,
                         "lyric_recall": recall_ref_off,
                         "text": ref_off.get("text", "")} if ref_off else None),
        },
        "v_norm": vnorm,
        # How far apart the two caption ends are in embedding space. A "close"
        # REF pair (docs/lm-pair-exam.md) gives a short axis, and every proj
        # channel measured against it is correspondingly noisy. The 2026-09-02
        # calibration pass tested this as a candidate U0 veto and REFUTED it:
        # ears-PASS energy-v3-lm-20s has the SMALLEST pair in the whole labeled
        # corpus (0.030) while ears-FAIL energy-lm-v16 has the largest (0.144).
        # Logged, not gated.
        "ref_axis_dist": (1.0 - cos(_vec(ref_plus), _vec(ref_minus))) if ref_minus is not None
        else (1.0 - cos(_vec(ref_plus), e0)),
        "null": nulls,
        "unit_plus": unit_plus,
        "unit_minus": unit_minus,
        "gates": gates,
        "gates_fired": fired,
        "verdict": ("UNSCOREABLE" if not gates["U0_shape"]["pass"]
                    else ("PASS" if not fired else "FAIL")),
        "warn": [g for g, d in gates.items() if d.get("warn")],
        "sides": sides,
        "E": E,
        "E_basis": "dsp_proj x lyric_factor, min over poles (UNCERTIFIED ranking heuristic)",
        "E_emb": E_emb,
        "E_squashed": squashed,
        "thresholds": T,
        "veto_gates": list(VETO_GATES),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def collect_pool(patterns: list[str], exclude: set[str], lyrics: str | None = None) -> list[Path]:
    """Foreign no-slider clips for the null, deduped by content hash.

    Clips whose folder sings a different lyric sheet are dropped: whisper
    embeddings are dominated by sung content, so a different-lyrics clip would
    inflate both the null and the same-song anchor with something the slider
    cannot be blamed for. The null must vary the caption, not the words.
    """
    want = set(norm_words(strip_sections(lyrics))) if lyrics else None
    # FROZEN 2026-09-02: OFF by default. The calibration pass measured it both
    # ways. Same-lyrics pooling keeps a PASS/FAIL gap on U4 (ears-PASS top out
    # at 0.870, the replacement FAILs sit at 1.127 / 1.225), but it EMPTIES the
    # pool for every folder whose sheet the pool does not contain -- 2 of the 11
    # labeled PASS/FAIL folders (rapslow-lm-uni-v2, and the ears-PASS
    # v9-ritual/gender-lm-v9) and the whole v9-ritual wave. U4 and the null then
    # silently cease to exist on exactly the ladders that most need checking.
    # The mixed pool (13-14 clips, cross_anchor 0.083-0.092) is the frozen one;
    # the flag stays available for a future pool that covers every sheet.
    seen: dict[str, Path] = {}
    for pat in patterns:
        for p in sorted(REPO_ROOT.glob(pat)):
            if not p.is_file():
                continue
            if want is not None:
                got = set(norm_words(strip_sections(parse_listen_md(p.parent)[0])))
                if got != want:
                    continue
            digest = sha1_file(p)
            if digest in exclude or digest in seen:
                continue
            seen[digest] = p
    return list(seen.values())


def cmd_score(args) -> int:
    folders = [resolve(f) for f in args.folders]
    stub_pool: list[np.ndarray] = []
    stub_pool_dsp: list[np.ndarray] = []
    if args.backend == "stub":
        blob = json.loads(resolve(args.stub).read_text(encoding="utf-8"))
        # {"files": {...}, "pool": [[emb], ...]} or a bare {name: measurement} map
        files = blob.get("files", blob)
        stub_pool = [np.asarray(p, dtype=np.float64) for p in blob.get("pool", [])]
        stub_pool_dsp = [np.asarray(p, dtype=np.float64) for p in blob.get("pool_dsp", [])]
        backend = StubBackend(files)
    else:
        backend = WhisperBackend(resolve(args.cache_dir), device=args.device)

    results = []
    for folder in folders:
        names = None
        if args.backend == "stub":
            names = sorted(k for k in backend.table if k.endswith(".wav"))
        try:
            lad = parse_ladder(folder, names=names)
        except (ValueError, StopIteration) as e:
            print(f"SKIP {folder}: {e}", file=sys.stderr)
            continue
        meas = {c.name: backend.measure(c.path, lad.requested_s) for c in lad.clips
                if c.role != "ref_other"}
        if args.backend == "stub":
            pool, pool_dsp, pool_paths = stub_pool, stub_pool_dsp, []
        else:
            own = {m["sha1"] for m in meas.values() if "sha1" in m}
            pool_paths = collect_pool(list(args.null_pool or DEFAULT_POOL_GLOBS), own,
                                      lyrics=lad.lyrics if args.pool_same_lyrics else None)
            pool_meas = [backend.measure(p, lad.requested_s) for p in pool_paths]
            pool = [np.asarray(m["emb"], dtype=np.float64) for m in pool_meas]
            pool_dsp = [np.asarray(list(m["log_bands"])
                                   + [m["ln_rms"], math.log(max(m["centroid"], 1e-9))],
                                   dtype=np.float64) for m in pool_meas]
        res = score_ladder(lad, meas, pool, model_id=backend.model_id, pool_dsp=pool_dsp)
        res["null_pool_files"] = [str(p.relative_to(REPO_ROOT)) for p in pool_paths]
        out = folder / "lm_scores.json"
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        results.append(res)
        print_row(res)
    if args.out:
        write_board(results, resolve(args.out))
    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if all(r["verdict"] == "PASS" for r in results) else 1


def _fmt(x, spec="6.3f"):
    return "   nan" if x is None or x != x else format(x, spec)


def _unit_ch(res: dict, side: str) -> dict:
    """The channel at the rung a gate is read at (usually +-1)."""
    s = res.get("unit_plus" if side == "plus" else "unit_minus")
    return res["channels"].get(f"{s:+g}", {}) if s is not None else {}


def print_row(res: dict) -> None:
    ch = res["channels"]
    p1, m1 = _unit_ch(res, "plus"), _unit_ch(res, "minus")
    anchor = res["null"]["cross_anchor"]
    frac = (p1.get("same_song") / anchor) if p1.get("same_song") is not None and anchor else None
    print(f"{res['name']:<28s} {res['verdict']:<11s} "
          f"rec0={_fmt(ch['+0']['lyric_recall'], '4.2f')} "
          f"recU+={_fmt(p1.get('lyric_recall'), '4.2f')} "
          f"recREF+={_fmt(res['refs']['ref_plus']['lyric_recall'], '4.2f')} "
          f"proj+={_fmt(p1.get('proj'), '+6.3f')} "
          f"dsp+={_fmt(p1.get('dsp_proj'), '+6.3f')} "
          f"dsp-={_fmt(m1.get('dsp_proj'), '+6.3f')} "
          f"songfrac={_fmt(frac, '5.3f')} "
          f"E={_fmt(res['E'], '6.2f')} Esq={_fmt(res['E_squashed'], '5.2f')} "
          f"{','.join(res['gates_fired']) or '-'}"
          + (f" WARN:{','.join(res['warn'])}" if res["warn"] else ""))


BOARD_COLS = ("name", "kind", "verdict", "gates_fired", "warn", "unit_plus", "unit_minus",
              "recall_0", "recall_up", "recall_um", "recall_ref_plus",
              "proj_up", "proj_um", "dsp_proj_up", "dsp_proj_um",
              "same_song_up", "cross_anchor", "same_song_frac_up",
              "null95", "null95_dsp", "u6_would_pass", "ref_axis_dist", "spearman",
              "d_ln_rms_up", "offaxis_up", "E", "E_emb", "E_squashed")


def write_board(results: list[dict], out: Path) -> None:
    rows = []
    for r in results:
        ch = r["channels"]
        p1, m1 = _unit_ch(r, "plus"), _unit_ch(r, "minus")
        anchor = r["null"]["cross_anchor"]
        ss = p1.get("same_song")
        rows.append({
            "name": r["name"], "kind": r["kind"], "verdict": r["verdict"],
            "gates_fired": ",".join(r["gates_fired"]) or "-",
            "warn": ",".join(r["warn"]) or "-",
            "unit_plus": r.get("unit_plus"), "unit_minus": r.get("unit_minus"),
            "recall_0": ch["+0"]["lyric_recall"], "recall_up": p1.get("lyric_recall"),
            "recall_um": m1.get("lyric_recall"),
            "recall_ref_plus": r["refs"]["ref_plus"]["lyric_recall"],
            "proj_up": p1.get("proj"), "proj_um": m1.get("proj"),
            "dsp_proj_up": p1.get("dsp_proj"), "dsp_proj_um": m1.get("dsp_proj"),
            "same_song_up": ss, "cross_anchor": anchor,
            "same_song_frac_up": (ss / anchor) if (ss is not None and anchor) else None,
            "null95": r["null"]["null95"], "null95_dsp": r["null"].get("null95_dsp"),
            "u6_would_pass": r["gates"]["U6_null"]["would_pass"],
            "ref_axis_dist": r["ref_axis_dist"],
            "spearman": r["gates"]["U3_direction"]["spearman"],
            "d_ln_rms_up": p1.get("d_ln_rms"), "offaxis_up": p1.get("offaxis"),
            "E": r["E"], "E_emb": r.get("E_emb"), "E_squashed": r["E_squashed"],
        })
    lines = ["\t".join(BOARD_COLS)]
    for row in rows:
        lines.append("\t".join(
            "" if row[c] is None else (f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]))
            for c in BOARD_COLS))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote board -> {out}")


# ---------------------------------------------------------------------------
# synthetic attacks (selftest + tests/test_lm_score.py)
# ---------------------------------------------------------------------------

EMB_DIM = 8
LYRIC_SHEET = "[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away"
CLEAN_TEXT = "I can feel it in the air tonight, louder now or fade away."
GARBLE_TEXT = "Those are those are those are those are"
OTHER_SONG_TEXT = "Wheels on the highway burning through another town"

_CENTER = np.zeros(EMB_DIM)
_CENTER[0] = 10.0             # the shared "these are all 20 s pop renders" mass
_U = np.zeros(EMB_DIM)
_U[1] = 1.0
_VLEN = 3.0  # ||v||: the caption-swap distance


def _foreign(seed: int, radius: float = 2.5) -> np.ndarray:
    """One unrelated no-slider render: shared mass + an idiosyncratic offset.

    Fixed seed per call (no shared RNG state), so a case built in isolation is
    identical to the same case built inside the full suite.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(size=EMB_DIM)
    r[0] = 0.0
    r[1] *= 0.12              # only a little of the caption-swap direction
    return _CENTER + r / np.linalg.norm(r) * radius


_E0 = _foreign(7)             # the folder's own zero clip is one such render


def _pool_embs(n: int = 14, seed: int = 1234) -> list[np.ndarray]:
    """Foreign no-slider clips: the null pool."""
    return [_foreign(seed + i) for i in range(n)]


def _m(emb: np.ndarray, *, rms: float = 0.12, text: str = CLEAN_TEXT,
       duration: float = 20.02, tail_ratio: float = 0.35, band: float = 0.0) -> dict:
    """One synthetic measurement.

    `band` is the clip's position on the synthetic DSP concept axis (the six
    log band energies move together). The plus REF sits at band = 1.0 and the
    zero clip at 0.0, so `dsp_proj` of a clip is exactly its `band` -- which is
    what U3's emb-OR-dsp sign rule and the ranking scalar E read.
    """
    return {"emb": [float(x) for x in emb], "rms": rms, "text": text,
            "duration": duration, "tail_ratio": tail_ratio,
            "ln_rms": float(math.log(rms)), "centroid": 2000.0,
            "log_bands": [float(band)] * 6, "natural_end": duration < 19.5}


def _pool_dsps(n: int = 14, seed: int = 1234) -> list[np.ndarray]:
    """DSP vectors for the null pool: foreign clips scattered around band 0."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        b = float(rng.normal(scale=0.3))
        out.append(np.asarray([b] * 6 + [math.log(0.12), math.log(2000.0)]))
    return out


def _uni_files(label: str = "Concept") -> list[str]:
    return ["01_slider_neutral_base_zero.wav", f"02_slider_{label}_plus1.wav",
            f"03_slider_{label}_plus2.wav", f"04_REF_prompt_{label}_no_slider.wav",
            "05_REF_prompt_Off_no_slider.wav"]


def _bipolar_files(plus: str = "Loud", minus: str = "Quiet") -> list[str]:
    return [f"01_slider_{minus}_minus2.wav", f"02_slider_{minus}_minus1.wav",
            "03_slider_neutral_base_zero.wav", f"04_slider_{plus}_plus1.wav",
            f"05_slider_{plus}_plus2.wav", f"06_REF_prompt_{plus}_no_slider.wav",
            f"07_REF_prompt_{minus}_no_slider.wav"]


def _write_folder(root: Path, name: str, files: list[str], requested: float = 20.0) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "LISTEN.md").write_text(
        f"# {name}\n\n- lyrics: `{LYRIC_SHEET}`\n"
        f"- requested: {requested}s  seed: 7  rank: 8  alpha: 8.0  kind: lm\n",
        encoding="utf-8")
    return folder


def uni_case(kind: str) -> tuple[list[str], dict[str, dict]]:
    """One synthetic uni ladder per known attack. Returns (files, measurements)."""
    files = _uni_files()
    zero, p1, p2, refp, refoff = files
    t = {zero: _m(_E0), refp: _m(_E0 + _VLEN * _U, band=1.0), refoff: _m(_E0)}
    if kind == "healthy":
        t[p1] = _m(_E0 + 1.5 * _U, band=0.5)
        t[p2] = _m(_E0 + 2.4 * _U, band=0.8)
    elif kind == "garble":
        # lyric loop at +1 with a LARGER on-axis move than the healthy slider:
        # without U2 this attack ranks first (docs/lm-uni-v2-garble.md, grit).
        t[p1] = _m(_E0 + 2.7 * _U, text=GARBLE_TEXT, band=0.9)
        t[p2] = _m(_E0 + 3.2 * _U, text=GARBLE_TEXT, band=1.1)
    elif kind == "silence":
        t[p1] = _m(_E0 + 1.5 * _U, rms=0.00012, band=0.5)
        t[p2] = _m(_E0 + 2.4 * _U, rms=0.00010, band=0.8)
    elif kind == "song_replace":
        # healthy level and a real on-axis move, but a foreign recording: the
        # +1 embedding sits a full cross-anchor away from the zero clip
        # (SCORING.md G4's D-pop-uni, which read healthy on every spectral
        # feature while rendering a different song).
        far = _foreign(99, radius=3.0) + 1.5 * _U
        t[p1] = _m(far, text=OTHER_SONG_TEXT, band=0.5)
        t[p2] = _m(far + 0.5 * _U, text=OTHER_SONG_TEXT, band=0.6)
    elif kind == "noop":
        t[p1] = _m(_E0 + 0.02 * _U, band=0.01)
        t[p2] = _m(_E0 + 0.04 * _U, band=0.02)
    elif kind == "reversed":
        t[p1] = _m(_E0 - 1.5 * _U, band=-0.5)
        t[p2] = _m(_E0 - 2.4 * _U, band=-0.8)
    elif kind == "nonmonotone":
        t[p1] = _m(_E0 + 1.5 * _U, band=0.5)
        t[p2] = _m(_E0 + 0.3 * _U, band=0.1)
    elif kind == "hot_tail":
        t[p1] = _m(_E0 + 1.5 * _U, duration=20.02, tail_ratio=0.9, band=0.5)
        t[p2] = _m(_E0 + 2.4 * _U, duration=20.02, tail_ratio=0.9, band=0.8)
    else:
        raise ValueError(kind)
    return files, t


def bipolar_case(kind: str) -> tuple[list[str], dict[str, dict]]:
    files = _bipolar_files()
    m2, m1, zero, p1, p2, refp, refm = files
    t = {zero: _m(_E0),
         refp: _m(_E0 + 0.5 * _VLEN * _U, band=0.5),
         refm: _m(_E0 - 0.5 * _VLEN * _U, band=-0.5)}
    if kind == "healthy":
        for f, k in ((m2, -2.4), (m1, -1.5), (p1, 1.5), (p2, 2.4)):
            t[f] = _m(_E0 + k * _U, band=0.33 * k)
    elif kind == "dead_minus":
        for f, k in ((m2, -0.02), (m1, -0.01), (p1, 1.5), (p2, 2.4)):
            t[f] = _m(_E0 + k * _U, band=0.33 * k)
    elif kind == "no_unit_rung":
        # the shape of eval/listen/gender-lm-20s: +-2 only, no rung at the
        # product setting, so the contract has no data (U0_shape -> UNSCOREABLE)
        for f, k in ((m2, -2.4), (p2, 2.4)):
            t[f] = _m(_E0 + k * _U, band=0.33 * k)
        files = [m2, zero, p2, refp, refm]
        t = {k: v for k, v in t.items() if k in files}
    else:
        raise ValueError(kind)
    return files, t


def build_case(root: Path, name: str, kind: str, shape: str = "uni"):
    files, table = (uni_case(kind) if shape == "uni" else bipolar_case(kind))
    folder = _write_folder(root, name, files)
    lad = parse_ladder(folder, names=files)
    return lad, table


def cmd_selftest(_args=None) -> int:
    """Feed every known LM attack to the gates and assert the right veto fires.

    Some assertions encode LOSSES from the 2026-09-02 calibration pass: U6 was
    demoted to a diagnostic and the Spearman rank rule was dropped, so the no-op
    and non-monotone attacks are no longer vetoed by anything. They stay in the
    suite, asserting the blind spot explicitly rather than quietly losing it.
    """
    pool = _pool_embs()
    pool_dsp = _pool_dsps()
    tmp = Path(tempfile.mkdtemp(prefix="lm_score_selftest_"))

    def run(kind, shape="uni"):
        lad, table = build_case(tmp, f"{shape}-{kind}", kind, shape)
        return score_ladder(lad, table, pool, model_id="stub", pool_dsp=pool_dsp)

    healthy = run("healthy")
    assert healthy["verdict"] == "PASS", healthy["gates_fired"]
    assert healthy["E"] > 0, healthy["E"]
    assert 0 < healthy["E_squashed"] < 1

    bip = run("healthy", "bipolar")
    assert bip["verdict"] == "PASS", bip["gates_fired"]
    assert bip["E"] > 0 and bip["kind"] == "bipolar"

    garble = run("garble")
    assert garble["gates_fired"] == ["U2_lyric_hold"], garble["gates_fired"]
    assert garble["channels"]["+1"]["lyric_recall"] == 0.0
    # ... and it would have won the ranking without that veto:
    assert (garble["sides"]["plus"]["dsp_proj"]
            > healthy["sides"]["plus"]["dsp_proj"]), "garble must out-rank healthy raw"
    assert garble["E"] < healthy["E"], "the lyric factor must price the garble"

    silence = run("silence")
    assert "U1_level" in silence["gates_fired"], silence["gates_fired"]

    repl = run("song_replace")
    assert "U4_same_song" in repl["gates_fired"], repl["gates_fired"]
    assert "U2_lyric_hold" in repl["gates_fired"], repl["gates_fired"]

    # KNOWN BLIND SPOT, frozen 2026-09-02: U6 was the only veto that could
    # reject a no-op, and it rejects every ears-PASS folder in the labeled
    # corpus, so it is now a diagnostic. A no-op slider passes the gates and is
    # separated only by the ranking scalar.
    noop = run("noop")
    assert noop["verdict"] == "PASS", noop["gates_fired"]
    assert not noop["gates"]["U6_null"]["would_pass"], "U6 still reads the no-op"
    assert noop["gates"]["U6_null"]["status"] == "not_applicable"
    assert noop["E"] < 0.1 < healthy["E"], (noop["E"], healthy["E"])

    rev = run("reversed")
    assert "U3_direction" in rev["gates_fired"], rev["gates_fired"]
    assert not rev["gates"]["U3_direction"]["sign_ok"]

    # KNOWN BLIND SPOT: rank monotonicity was dropped from U3 because it
    # anti-correlates with ears on the labeled corpus. Still logged.
    nonmono = run("nonmonotone")
    assert "U3_direction" not in nonmono["gates_fired"], nonmono["gates_fired"]
    assert nonmono["gates"]["U3_direction"]["spearman"] < THRESHOLDS["spearman_ref"]

    hot = run("hot_tail")
    assert hot["verdict"] == "PASS", hot["gates_fired"]   # 20 s: WARN, not veto
    assert hot["warn"] == ["U5_ending"], hot["warn"]

    dead = run("dead_minus", "bipolar")
    assert dead["verdict"] == "PASS", dead["gates_fired"]  # no veto reads a dead pole
    assert dead["E"] < healthy["E"], "min() over sides must price a dead pole"
    assert dead["sides"]["minus"]["E"] < 0.05, dead["sides"]["minus"]

    shape = run("no_unit_rung", "bipolar")
    assert shape["verdict"] == "UNSCOREABLE", shape["verdict"]
    assert shape["gates_fired"] == ["U0_shape"], shape["gates_fired"]
    assert shape["unit_plus"] == 2.0 and shape["unit_minus"] == -2.0

    print("lm_score selftest OK (garble, silence, song-replace, no-op[diagnostic], "
          "reversed, non-monotone[diagnostic], hot-tail, dead-pole, no-unit-rung, "
          "healthy uni + bipolar)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("folders", nargs="+")
    p.add_argument("--cache_dir", default="eval/lm_score_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="combined TSV board")
    p.add_argument("--json", action="store_true", help="dump full JSON to stdout")
    p.add_argument("--null_pool", action="append", default=None,
                   help="glob (repo-relative) of foreign zero/Off clips; repeatable")
    p.add_argument("--pool_same_lyrics", action="store_true",
                   help="drop null-pool clips whose folder sings a different lyric sheet")
    p.add_argument("--backend", choices=("whisper", "stub"), default="whisper")
    p.add_argument("--stub", default=None, help="measurements JSON for --backend stub")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    return cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
