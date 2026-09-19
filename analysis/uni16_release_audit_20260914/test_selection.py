"""Decision regressions: preference changes must not weaken quality/flag gates."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection import select
from test_audit import candidate

POLICY = dict(tolerance=.2, sensitivity_tolerances=[.1, .2, .3])
PROTOCOL = dict(close_call_gap=.1)


def checkpoint(step, ce, pq, flags=()):
    result = candidate(f"step{step}", [0.] * 4, flags)
    for clip in result["clips"]:
        clip["candidate"].update(enjoyment=ce, production=pq)
    return result


def pick(candidates, flags=()):
    return select(candidates, PROTOCOL, POLICY, flags)


def test_later_preference_has_fixed_bounds_and_cannot_chain_to_worse_quality():
    result = pick([checkpoint(600, 8., 8.), checkpoint(2000, 7.85, 7.85),
                   checkpoint(3400, 7.7, 7.7)])
    assert result["recommendation"]["label"] == "step2000"
    assert result["eligible_checkpoints"] == ["step2000", "step600"]
    assert result["recommendation"]["close_call"]
    assert result["sensitivity"][0]["recommendation"] == "step600"
    assert result["sensitivity"][2]["recommendation"] == "step3400"


@pytest.mark.parametrize("ce,pq", [(7.79, 8.2), (8.2, 7.79)])
def test_better_score_on_one_axis_cannot_pay_for_regression_on_the_other(ce, pq):
    result = pick([checkpoint(600, 8., 8.), checkpoint(3400, ce, pq)])
    assert result["recommendation"]["label"] == "step600"


def test_unresolved_quality_tradeoff_requires_listening():
    result = pick([checkpoint(600, 8.4, 8.), checkpoint(3400, 8., 8.4)])
    assert result["recommendation"] is None
    assert result["decision_status"] == "quality_tradeoff"


def test_style_and_lyrics_cannot_change_eligibility_or_preference():
    candidates = [checkpoint(600, 8., 8.), checkpoint(3400, 7.9, 7.9)]
    before = pick(candidates)
    modified = copy.deepcopy(candidates)
    for i, row in enumerate(modified):
        for clip in row["clips"]:
            clip["score"] = 1000. if i == 0 else -1000.
            clip["candidate"].update(concept=10. if i == 0 else -10., lyrics=0.)
            clip["lyric_flags"] = ["lyric_proxy_regression"]
    after = pick(modified)
    for field in ("recommendation", "eligible_checkpoints", "sensitivity", "quality_anchors"):
        assert before[field] == after[field]
    assert [r["label"] for r in before["ranking"]] == [r["label"] for r in after["ranking"]]
    assert all("score" not in r and all("score" not in c for c in r["clips"])
               for r in after["ranking"])


def test_technical_and_waveform_flags_exclude_later_candidate_and_its_anchor():
    for flags, waveform in [(["high_frequency_excess"], []),
                            ([], [dict(checkpoint="step3400")])]:
        result = pick([checkpoint(600, 7., 8.), checkpoint(3400, 9., 9., flags)], waveform)
        assert result["recommendation"]["label"] == "step600"
        assert result["quality_anchors"]["enjoyment"]["value"] == 7.


def test_legacy_step_is_not_treated_as_an_extension_of_the_new_run():
    legacy = checkpoint(660, 8., 8.)
    legacy["label"] = "published660"
    assert pick([legacy, checkpoint(600, 7.9, 7.9)])["recommendation"]["label"] == "step600"
    assert pick([legacy, checkpoint(600, 7.7, 7.7)])["recommendation"]["label"] == "published660"


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_invalid_measurement_cannot_be_recommended_or_export_nonfinite_json(value):
    bad = checkpoint(3400, 9., 9.)
    bad["clips"][0]["candidate"]["enjoyment"] = value
    result = pick([checkpoint(600, 8., 8.), bad])
    assert result["recommendation"]["label"] == "step600"
    assert result["ranking"][-1]["risk"] == 3
    json.dumps(result, allow_nan=False)


def test_matched_cases_and_duplicate_audio_checks_are_preserved():
    bad = checkpoint(3400, 9., 9.)
    bad["clips"][1]["fixture"] = bad["clips"][0]["fixture"]
    with pytest.raises(ValueError, match="unique"):
        pick([bad])
    bad = checkpoint(3400, 9., 9.)
    bad["clips"][1]["candidate"]["sha256"] = bad["clips"][0]["candidate"]["sha256"]
    assert pick([bad])["recommendation"] is None


def test_future_checkpoint_is_selected_without_mutating_historical_scores():
    candidates = [checkpoint(600, 8., 8.)]
    before = copy.deepcopy(candidates)
    assert pick(candidates)["recommendation"]["label"] == "step600"
    assert candidates == before
    candidates.append(checkpoint(3400, 7.9, 7.9))
    result = pick(candidates)
    assert result["recommendation"]["label"] == "step3400"
    assert result["quality_comparisons"][0]["metrics"]["enjoyment"]["mean_delta"] == pytest.approx(-.1)
    assert candidates[0] == before[0]


def test_empty_audit_stays_pending():
    result = pick([])
    assert result["recommendation"] is None and result["decision_status"] == "waiting"
