"""Immutable experiment contracts and auditable, atomic artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
MODEL = WORKSPACE / 'models/MiniMax-Music3'
DEFAULT_RUN = ROOT / 'analysis/reward_sliders_20260907'
SCORER_REVISION = '9b1dd8e5df9af7216e836a98974fe3b82c56ded6'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_tensor(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validate_families(rows):
    sys.path.insert(0, str(WORKSPACE))
    from app.rewriter import _artist_name_hit
    ids, sheets = set(), set()
    for row in rows:
        if row['family'] in ids or row['lyrics'] in sheets:
            raise ValueError('Duplicate family or sheet; variants must share a split')
        ids.add(row['family']); sheets.add(row['lyrics'])
        if row['split'] not in ('train', 'dev', 'test', 'transfer'):
            raise ValueError('Unknown split')
        if not row['caption'].strip() or not row['lyrics'].strip():
            raise ValueError('Empty caption or lyrics')
        if _artist_name_hit('', json.dumps(row)):
            raise ValueError(f"Name validation rejected {row['family']}")


@dataclass(frozen=True)
class RewardSpec:
    identifier: str = 'audiobox-ce-v1'
    orientation: str = 'higher'
    scorer: str = 'facebook/audiobox-aesthetics'
    revision: str = SCORER_REVISION
    preprocessing: str = 'one float32 stereo copy, RMS 0.1, mono mean, resample to 16000 Hz'
    window_rule: str = 'two disjoint 10-second windows in the first 20 seconds'
    aggregation: str = 'arithmetic mean of CE only'
    valid_range: tuple = (1., 10.)
    invalid_policy: str = 'retain failed/missing/silent/short/nonfinite/out-of-range arms; no seed rerolls'
    full_song_rule: str = 'normalize whole song once; duration-weighted disjoint 10-second windows including actual tail'
    hashes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureSpec:
    model: str = str(MODEL)
    layers: tuple = (11, 23)
    hook: str = 'decoder block residual output before subsequent blocks'
    residual_width: int = 4096
    cfg_branches: tuple = ('conditional', 'unconditional')
    position_policy: str = 'exclude entire prefill, including audio_start; include subsequent feedback positions'
    frame_rate: float = 25.
    mapping: str = ('feedback position prompt_length + k predicts emitted frame k; prefill predicts discarded warmup; '
                    'nominal windows 0:250 and 250:500; acoustic context uses 200-frame windows with 100-frame hop, '
                    'overlap 344 latent frames, waveform crops 86 left/258 right; no exact token reward labels')
    dtype: str = 'bfloat16 residuals and feedback, float32 pooled statistics'
    normalization: str = 'training-family mean center; no coordinate whitening; training median residual L2'
    hashes: dict = field(default_factory=dict)


@dataclass
class Observation:
    id: str
    family: str
    split: str
    seed: int
    cell_hash: str
    arm: dict
    status: str = 'pending'
    audio: str | None = None
    audio_sha256: str | None = None
    trajectory: str | None = None
    trajectory_sha256: str | None = None
    reward: dict | None = None
    provenance: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    error: str | None = None

    def json(self):
        return asdict(self)
