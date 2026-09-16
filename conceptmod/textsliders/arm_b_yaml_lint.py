"""Propose-only YAML hygiene hooks for Music Arm B (Reoc0 / FLP).

``PROPOSE_ONLY = True`` — everything here is a proposed heuristic, not a
proven trainer default. Nothing in this module gates a train; the
fail-closed Arm B gates live in ``tests/test_music_arm_b_gates.py``
(against the trainer ``ARM_B`` row + ``--adv_preset arm_b`` /
``--require_arm_b``).

Handoff hygiene rows covered:

- Reoc0 (``e_on_content=0``): the leftover ê must not restate row
  content. Without an encoder on CPU this is a lexical proxy: content
  words shared between the leak pair and the slider pair are flagged.
- FLP (no lyric/caption row unused-ê primary, ``se < max(su, sc)``):
  for a row embedding, similarity to ê must lose to similarity to û
  (slider) or ĉ (caption/neutral). The comparison logic is implemented
  over caller-supplied vectors — encoder wiring is explicitly out of
  scope, so the ``encode`` side stays unproven.

No torch, no GPU, no Music 3 weights.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import yaml

PROPOSE_ONLY = True

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "over",
        "the", "to", "vs", "with",
    }
)

_WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    """Lowercased alphanumeric words minus stopwords (lexical proxy set)."""
    return {w for w in _WORD.findall(str(text).lower()) if w not in _STOPWORDS}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-python cosine (no numpy/torch so the lint stays dependency-light)."""
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return num / den if den > 0.0 else 0.0


def flp_sims(
    row_vec: Sequence[float],
    e_vec: Sequence[float],
    u_vec: Sequence[float],
    c_vec: Sequence[float],
) -> dict[str, float]:
    """Per-row similarities: se (unused-ê), su (slider û), sc (caption ĉ)."""
    return {
        "se": cosine(row_vec, e_vec),
        "su": cosine(row_vec, u_vec),
        "sc": cosine(row_vec, c_vec),
    }


def check_flp(sims: Mapping[str, float], *, where: str = "row") -> list[str]:
    """Flag a row where unused-ê is primary: ``se >= max(su, sc)``."""
    try:
        se, su, sc = float(sims["se"]), float(sims["su"]), float(sims["sc"])
    except (KeyError, TypeError, ValueError):
        return [f"{where}: FLP needs se/su/sc sims, got {dict(sims)!r}"]
    if se >= max(su, sc):
        return [
            f"{where}: unused-ê primary (se={se:.3f} >= max(su={su:.3f}, sc={sc:.3f}))"
        ]
    return []


def check_reoc0(doc: Mapping[str, object]) -> list[str]:
    """Lexical proxy for ``e_on_content=0`` (propose-only).

    Flags content words the leak pair shares with the slider pair —
    shared words are where ê restates the axis instead of packing only
    the unused leftover. Undeclared pairs are vacuous (no finding).
    """
    slider = f"{doc.get('slider_positive') or ''} {doc.get('slider_negative') or ''}"
    leak = f"{doc.get('leak_positive') or ''} {doc.get('leak_negative') or ''}"
    if not slider.strip() or not leak.strip():
        return []
    shared = sorted(content_words(slider) & content_words(leak))
    if shared:
        return [f"reoc0: leak restates slider words: {', '.join(shared)}"]
    return []


def lint_prompts_yaml(path: str | Path) -> list[str]:
    """Structural lint for a prompts YAML (propose-only).

    Checks the file parses, has non-empty rows, and every row carries
    the target/positive/negative/neutral captions plus lyrics (the
    lyric span the Arm B holds need). Returns finding strings.
    """
    findings: list[str] = []
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"{path}: unreadable prompts yaml: {exc}"]
    if not isinstance(doc, Mapping):
        return [f"{path}: top level must be a mapping"]
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        return [f"{path}: no rows"]
    for i, row in enumerate(rows):
        where = f"{path}:row{i}"
        if not isinstance(row, Mapping):
            findings.append(f"{where}: row must be a mapping")
            continue
        for key in ("target", "positive", "negative", "neutral"):
            if not row.get(key):
                findings.append(f"{where}: missing {key}")
        if not row.get("lyrics"):
            findings.append(f"{where}: empty lyrics (lyric-span holds need yaml lyrics)")
    findings.extend(f"{path}: {m}" for m in check_reoc0(doc))
    return findings


__all__ = [
    "PROPOSE_ONLY",
    "check_flp",
    "check_reoc0",
    "content_words",
    "cosine",
    "flp_sims",
    "lint_prompts_yaml",
]
