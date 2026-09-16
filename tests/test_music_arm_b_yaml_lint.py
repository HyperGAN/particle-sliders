"""Propose-only YAML lint hooks for Music Arm B (Reoc0 / FLP).

Everything asserted here is marked propose_only: heuristics over prompts
YAML, not proven trainer defaults. The fail-closed gates live in
``test_music_arm_b_gates.py``; this file must never gate a train.
"""

from __future__ import annotations

from pathlib import Path

from conceptmod.textsliders.arm_b_yaml_lint import (
    PROPOSE_ONLY,
    check_flp,
    check_reoc0,
    flp_sims,
    lint_prompts_yaml,
)

DATA = Path("conceptmod/textsliders/data/prompts-energy-v4.yaml")


def test_lint_is_marked_propose_only():
    assert PROPOSE_ONLY is True


def test_energy_v4_passes_structural_lint():
    assert lint_prompts_yaml(DATA) == []


def test_missing_lyrics_and_captions_are_flagged(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "rows:\n"
        "  - target: t\n    positive: p\n    negative: n\n    neutral: z\n",
        encoding="utf-8",
    )
    findings = lint_prompts_yaml(bad)
    assert any("lyrics" in f for f in findings)
    empty = tmp_path / "empty.yaml"
    empty.write_text("rows: []\n", encoding="utf-8")
    assert any("no rows" in f for f in lint_prompts_yaml(empty))


def test_reoc0_flags_leak_restating_slider_words():
    clean = {
        "slider_positive": "Extremely high energy, aggressive and loud.",
        "slider_negative": "Extremely quiet and calm, almost silent.",
        "leak_positive": "Pop-punk mix, BPM 168.",
        "leak_negative": "Ambient lullaby mix, BPM 52.",
    }
    assert check_reoc0(clean) == []
    restated = dict(clean, leak_positive="Loud aggressive slammed mix.")
    flagged = check_reoc0(restated)
    assert len(flagged) == 1 and "reoc0" in flagged[0]
    assert "loud" in flagged[0] and "aggressive" in flagged[0]
    # Undeclared pairs are vacuous, not failures.
    assert check_reoc0({}) == []


def test_flp_flags_unused_e_primary_with_fake_encoder():
    e_vec, u_vec, c_vec = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    caption_row = (0.1, 1.0)  # near û: se < max(su, sc)
    assert check_flp(flp_sims(caption_row, e_vec, u_vec, c_vec)) == []
    leaked_row = (1.0, 0.1)  # near ê: unused-ê primary
    flagged = check_flp(flp_sims(leaked_row, e_vec, u_vec, c_vec))
    assert len(flagged) == 1 and "unused-ê primary" in flagged[0]


def test_flp_rejects_malformed_sims():
    assert check_flp({"se": 0.1}) != []
