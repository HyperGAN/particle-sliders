"""Prompt construction and content-addressed provenance. No model imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


VARIATIONS = ("candlelit", "moonlit", "theatrical")
QUALITY = "masterpiece, best quality, safe, solo"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class Shared:
    character: str
    outfit: str
    pose: str
    framing: str
    scene: str
    medium: str = "anime illustration"
    quality: str = QUALITY

    def prompt(self):
        return ", ".join(s for s in (self.quality, self.character, self.outfit, self.pose,
                                    self.framing, self.scene, self.medium) if s)


@dataclass(frozen=True)
class Definition:
    id: str
    variation: str
    split: str
    lighting: str
    neutral_lighting: str = ""

    def __post_init__(self):
        if self.variation not in VARIATIONS or self.split not in ("train", "eval"):
            raise ValueError("Unknown variation or definition split")
        if not self.lighting or self.lighting == self.neutral_lighting:
            raise ValueError("A definition needs a distinct positive lighting clause")


def paired_prompts(shared: Shared, definition: Definition):
    base = shared.prompt()
    return {
        "shared": asdict(shared),
        "neutral_lighting": definition.neutral_lighting,
        "positive_lighting": definition.lighting,
        "neutral": base + (", " + definition.neutral_lighting if definition.neutral_lighting else ""),
        "positive": base + ", " + definition.lighting,
    }


def validate_pair(row):
    shared = Shared(**row["shared"])
    expected = paired_prompts(shared, Definition(row["definition"], row["variation"],
        row["definition_split"], row["positive_lighting"], row["neutral_lighting"]))
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(f"Pair {row.get('id')} changes shared content: {key}")


def normalized_strengths(mix, energy):
    if not math.isfinite(energy) or not 0 <= energy <= 1:
        raise ValueError("Energy must be finite and in [0, 1]")
    if set(mix) - set(VARIATIONS):
        raise ValueError("Unknown slider")
    values = {key: float(mix.get(key, 0)) for key in VARIATIONS}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Mix values must be finite and nonnegative")
    # Rescale before summing to avoid overflow for large, valid proportions.
    largest = max(values.values())
    if energy == 0 or largest == 0:
        return dict.fromkeys(VARIATIONS, 0.0)
    total = math.fsum(v / largest for v in values.values())
    return {key: energy * (v / largest) / total for key, v in values.items()}
