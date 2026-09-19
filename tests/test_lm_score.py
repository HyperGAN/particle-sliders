"""LM render-path scorer: the attack suite, on CPU, with no whisper load.

Same cases as `scripts/lm_score.py selftest`, asserted one at a time so a
regression names the gate it broke. Every gate here exists because a measured
failure is on disk (docs/lm-uni-v2-garble.md, docs/lm-pair-exam.md, SCORING.md).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import lm_score as L  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return L._pool_embs()


@pytest.fixture(scope="module")
def pool_dsp():
    return L._pool_dsps()


@pytest.fixture(scope="module")
def tmproot():
    return Path(tempfile.mkdtemp(prefix="test_lm_score_"))


def run(tmproot, pool, kind, shape="uni"):
    lad, table = L.build_case(tmproot, f"{shape}-{kind}", kind, shape)
    return L.score_ladder(lad, table, pool, model_id="stub",
                          pool_dsp=L._pool_dsps())


def test_healthy_uni_passes(tmproot, pool):
    r = run(tmproot, pool, "healthy")
    assert r["verdict"] == "PASS", r["gates_fired"]
    assert r["kind"] == "uni"
    assert r["E"] > 0
    assert 0.0 < r["E_squashed"] < 1.0
    assert r["seeds"] == 1  # single-seed ladders: the verdict is provisional


def test_healthy_bipolar_passes(tmproot, pool):
    r = run(tmproot, pool, "healthy", "bipolar")
    assert r["verdict"] == "PASS", r["gates_fired"]
    assert r["kind"] == "bipolar"
    assert r["channels"]["-1"]["proj"] < 0 < r["channels"]["+1"]["proj"]
    assert r["E"] > 0


def test_garble_fires_lyric_hold_and_would_have_won(tmproot, pool):
    """grit-lm-uni-v2: +1 recall 0.00 against base 0.62, on a big on-axis move."""
    healthy = run(tmproot, pool, "healthy")
    r = run(tmproot, pool, "garble")
    assert r["gates_fired"] == ["U2_lyric_hold"], r["gates_fired"]
    assert r["channels"]["+1"]["lyric_recall"] == 0.0
    assert (r["sides"]["plus"]["dsp_proj"]
            > healthy["sides"]["plus"]["dsp_proj"])
    assert r["E"] < healthy["E"]  # the lyric factor also prices it down


def test_silence_fires_level(tmproot, pool):
    r = run(tmproot, pool, "silence")
    assert "U1_level" in r["gates_fired"], r["gates_fired"]
    assert r["gates"]["U1_level"]["silent_scales"] == [1.0, 2.0]


def test_song_replacement_fires_same_song_and_lyrics(tmproot, pool):
    r = run(tmproot, pool, "song_replace")
    assert "U4_same_song" in r["gates_fired"], r["gates_fired"]
    assert "U2_lyric_hold" in r["gates_fired"], r["gates_fired"]
    assert (r["channels"]["+1"]["same_song"]
            > r["null"]["cross_anchor"] * L.THRESHOLDS["same_song_frac"])


def test_noop_is_a_known_blind_spot_after_the_2026_09_02_freeze(tmproot, pool):
    """U6 was the only veto that read a no-op, and it rejects every ears-PASS
    folder in eval/lm_score_labels.json, so the calibration pass demoted it to a
    diagnostic. A no-op now passes the gates; only the ranking scalar sees it.
    This test asserts the LOSS so it cannot be lost silently."""
    healthy = run(tmproot, pool, "healthy")
    r = run(tmproot, pool, "noop")
    assert r["verdict"] == "PASS", r["gates_fired"]
    assert r["gates"]["U6_null"]["would_pass"] is False  # the channel still reads it
    assert r["gates"]["U6_null"]["veto"] is False
    assert r["gates"]["U6_null"]["status"] == "not_applicable"  # uni: shared anchor
    assert r["E"] < 0.1 < healthy["E"]


def test_reversed_fires_direction(tmproot, pool):
    r = run(tmproot, pool, "reversed")
    assert "U3_direction" in r["gates_fired"], r["gates_fired"]
    assert not r["gates"]["U3_direction"]["sign_ok"]


def test_nonmonotone_is_logged_not_gated_after_the_freeze(tmproot, pool):
    """Rank monotonicity was dropped from U3: over the labeled corpus it
    anti-correlates with ears (PASS rho +1.0/+0.5/+0.90/-0.40/-0.30/-1.0 vs
    FAIL rho +1.0/+1.0/+1.0/+1.0/-0.70). Still computed, still on the board."""
    r = run(tmproot, pool, "nonmonotone")
    assert "U3_direction" not in r["gates_fired"], r["gates_fired"]
    assert r["gates"]["U3_direction"]["spearman"] < L.THRESHOLDS["spearman_ref"]


def test_hot_tail_warns_but_does_not_veto_at_20s(tmproot, pool):
    r = run(tmproot, pool, "hot_tail")
    assert r["verdict"] == "PASS", r["gates_fired"]
    assert r["warn"] == ["U5_ending"]
    assert r["gates"]["U5_ending"]["ending"] == "hot_tail"
    assert not r["gates"]["U5_ending"]["veto_enabled"]


def test_hot_tail_vetoes_at_long_duration(tmproot, pool):
    lad, table = L.build_case(tmproot, "uni-hot_tail_long", "hot_tail", "uni")
    lad.requested_s = 90.0
    r = L.score_ladder(lad, table, pool, model_id="stub", pool_dsp=L._pool_dsps())
    assert "U5_ending" in r["gates_fired"], r["gates_fired"]


def test_dead_pole_is_priced_by_min_over_sides(tmproot, pool):
    """No veto reads a dead pole after the freeze -- U3 gates the plus pole only
    (gating the minus pole rejects ears-PASS gender-lm-20s). The ranking scalar
    is what prices it, via min() over sides."""
    healthy = run(tmproot, pool, "healthy", "bipolar")
    r = run(tmproot, pool, "dead_minus", "bipolar")
    assert r["verdict"] == "PASS", r["gates_fired"]
    assert r["sides"]["minus"]["E"] < 0.05
    assert r["E"] < healthy["E"]


def test_ladder_without_a_unit_rung_is_unscoreable(tmproot, pool):
    """eval/listen/gender-lm-20s shape: +-2 only. The contract is stated at the
    product setting |s| = 1, and this ladder has no rung there."""
    r = run(tmproot, pool, "no_unit_rung", "bipolar")
    assert r["verdict"] == "UNSCOREABLE", r["verdict"]
    assert r["gates_fired"] == ["U0_shape"], r["gates_fired"]
    assert r["gates"]["U0_shape"]["plus_fallback"] is True
    assert r["unit_plus"] == 2.0 and r["unit_minus"] == -2.0


def test_u3_accepts_either_sign_channel(tmproot, pool):
    """The frozen U3 rule: emb proj > 0 OR dsp_proj > 0 at the unit plus rung.
    Motivated by v18/energy-lm-v18 (ears PASS, emb proj[+1] = -0.016,
    dsp_proj[+1] = +0.566)."""
    lad, table = L.build_case(tmproot, "uni-embwrong", "healthy", "uni")
    import numpy as np
    # keep the DSP move, reverse the embedding move: U3 must still pass
    table["02_slider_Concept_plus1.wav"]["emb"] = [
        float(x) for x in (L._E0 - 1.5 * L._U)]
    r = L.score_ladder(lad, table, pool, model_id="stub", pool_dsp=L._pool_dsps())
    g = r["gates"]["U3_direction"]
    assert g["proj_plus"] < 0 < g["dsp_proj_plus"]
    assert g["pass"] is True


def test_lyric_recall_strips_section_tags():
    sheet = "[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away"
    assert L.lyric_recall("verse chorus", sheet) == 0.0
    assert L.lyric_recall("I can feel it in the air tonight "
                          "louder now or fade away", sheet) == 1.0


def test_parse_uni_and_bipolar_folders(tmproot):
    uni = L._write_folder(tmproot, "parse-uni", L._uni_files("Grit"))
    lad = L.parse_ladder(uni, names=L._uni_files("Grit"))
    assert lad.kind == "uni" and lad.scales == [0.0, 1.0, 2.0]
    assert lad.by_role("ref_plus").name.endswith("Grit_no_slider.wav")
    assert lad.by_role("ref_off").name.endswith("Off_no_slider.wav")
    assert lad.requested_s == 20.0

    bi = L._write_folder(tmproot, "parse-bi", L._bipolar_files("Loud", "Quiet"))
    lad = L.parse_ladder(bi, names=L._bipolar_files("Loud", "Quiet"))
    assert lad.kind == "bipolar" and lad.scales == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert lad.by_role("ref_plus").name.endswith("Loud_no_slider.wav")
    assert lad.by_role("ref_minus").name.endswith("Quiet_no_slider.wav")


def test_rms_convention_is_per_channel_power():
    import numpy as np

    # a hard-panned signal: mono downmix would read ~3 dB low / cancel
    x = np.stack([np.ones(100), -np.ones(100)], axis=1)
    assert L.rms_pc(x) == pytest.approx(1.0)


def test_selftest_entrypoint_runs():
    assert L.cmd_selftest(None) == 0
