"""Uni-v1 prompts: plus-only, raw h+, no leftover-gate, no opposite pole."""

from __future__ import annotations

from pathlib import Path

import yaml

from conceptmod.textsliders.train_lm_slider_music3 import (
    parse_args,
    resolve_leak_axis_captions,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "conceptmod" / "textsliders" / "data"
AXES = (
    "energy",
    "gender",
    "tempo",
    "distortion",
    "breath",
    "rhyme",
    "triphop",
    "live",
)
ATTRS = ["A man is singing.", "A woman is singing."]
PLUS_WORDS = {
    "energy": ("loud", "slammed"),
    "gender": ("female", "woman"),
    "tempo": ("fast",),
    "distortion": ("distorted",),
    "breath": ("inhale",),
    "rhyme": ("dense rhyme",),
    "triphop": ("dusty", "vinyl", "hazy"),
    "live": ("bleed", "crowd", "human time"),
}


def _load(axis: str) -> dict:
    path = DATA / f"prompts-{axis}-uni-v1.yaml"
    assert path.is_file(), path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_uni_v1_omits_leak_so_teacher_is_raw_h():
    for axis in AXES:
        blob = _load(axis)
        assert "leak_positive" not in blob, axis
        assert "leak_negative" not in blob, axis
        assert "leak" not in blob, axis
        assert resolve_leak_axis_captions(
            leak_positive=None, leak_negative=None, prompts_meta=blob
        ) is None


def test_uni_v1_has_no_opposite_pole():
    for axis in AXES:
        blob = _load(axis)
        assert blob.get("minus_label") == "Off", axis
        lo, hi = blob["recommended_range"]
        assert float(lo) == 0.0, axis
        assert float(hi) == 2.0, axis
        for i, row in enumerate(blob["rows"]):
            assert row["negative"] == row["target"], f"{axis} row {i}"
            assert row["neutral"] == row["target"], f"{axis} row {i}"
            assert row["positive"] != row["target"], f"{axis} row {i}"


def test_uni_v1_caption_shape():
    for axis in AXES:
        blob = _load(axis)
        if axis == "gender":
            assert "attributes" not in blob
        for i, row in enumerate(blob["rows"]):
            if axis == "gender":
                assert "attributes" not in row, f"{axis} row {i}"
            else:
                assert row.get("attributes") == ATTRS, f"{axis} row {i}"
            for key in ("target", "positive", "negative", "neutral"):
                text = row[key]
                assert "Global Metadata:" in text, f"{axis} {key}"
                assert "Vocal Details:" in text, f"{axis} {key}"
                assert "Arrangement:" in text, f"{axis} {key}"
                assert "Sung, not rapped." in text, f"{axis} {key}"
                if axis != "gender":
                    assert "A man is singing." not in text, f"{axis} {key}"
                    assert "A woman is singing." not in text, f"{axis} {key}"
            pos = row["positive"].lower()
            for word in PLUS_WORDS[axis]:
                assert word in pos, f"{axis} row {i} missing {word!r}"
            if axis == "gender":
                assert "female" in row["positive"].lower()
                assert "male" not in row["negative"].lower()
                assert "man is singing" not in row["negative"].lower()


def test_uni_v1_keeps_v4_lyrics():
    for axis in AXES:
        uni = _load(axis)
        v4 = yaml.safe_load((DATA / f"prompts-{axis}-v4.yaml").read_text())
        assert [row["lyrics"] for row in uni["rows"]] == [
            row["lyrics"] for row in v4["rows"]
        ]
