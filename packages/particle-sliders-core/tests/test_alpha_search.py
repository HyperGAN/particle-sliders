"""The saved-adapter search ranks complete, qualified native output gains."""

import math

import pytest

from particle_sliders.alpha_search import AlphaObservation, matched_effect_mean, search_alpha
from particle_sliders.candidate_failure import FileRecord


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "adapter.raw"
    path.write_bytes(b"fixed saved adapter")
    return FileRecord.from_path(path)


def test_matched_effect_mean_keeps_signed_continuous_and_binary_scores():
    observations = iter((("a", .5, .75), ("b", True, False), ("c", 0, 0)))
    assert matched_effect_mean(case_ids=("c", "b", "a"), observations=observations) == -.25


@pytest.mark.parametrize(("roster", "observations"), [
    ((), ()),
    (("a", "a"), (("a", 0, 1),)),
    (("a", "b"), (("a", 0, 1),)),
    (("a",), (("a", 0, 1), ("b", 0, 1))),
    (("a", "b"), (("a", 0, 1), ("a", 0, 1), ("b", 0, 1))),
])
def test_matched_effect_mean_requires_exact_roster(roster, observations):
    with pytest.raises(ValueError, match="case"):
        matched_effect_mean(case_ids=roster, observations=observations)


@pytest.mark.parametrize(("off", "on"), [
    (0, math.nan),
    (math.inf, 0),
    (0, "1"),
    (0, None),
    (-1.7e308, 1.7e308),
])
def test_matched_effect_mean_requires_finite_scores_and_differences(off, on):
    with pytest.raises(ValueError, match="finite"):
        matched_effect_mean(case_ids=("a",), observations=[("a", off, on)])


def test_nonmonotonic_grid_and_bounded_refinement(checkpoint):
    measured = {0.: 0., 1.: .4, 2.: .2, 4.: -.2,
                .5: .8, 1.5: .3, .25: .7, .75: .9}
    calls = []

    def evaluate(alpha):
        calls.append(alpha)
        return {"case": AlphaObservation(0., measured[alpha], "pass")}

    result = search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(4, 1, 0, 2, 2.),
                          evaluate=evaluate, refinement_rounds=2)
    assert calls == [0., 1., 2., 4., .5, 1.5, .25, .75]
    assert result.checkpoint == checkpoint
    assert result.case_ids == ("case",)
    assert tuple(trial.alpha for trial in result.trials) == tuple(calls)
    assert [(trial.alpha, trial.mean_gain) for trial in result.best] == [(.75, .9)]
    assert result.trials[3].mean_gain == -.2


def test_equal_case_signed_mean_and_separate_integrity(checkpoint):
    passing = {"a": AlphaObservation(.75, .5, "pass"), "b": AlphaObservation(.25, .375, "pass")}
    values = {
        .5: passing,
        1.: {"a": AlphaObservation(.1, .9, "unmeasured"), "b": AlphaObservation(0, 1, "pass")},
        1.5: {"a": AlphaObservation(0, 1, "fail")},
        2.: {"a": AlphaObservation(.1, None, "pass"), "b": AlphaObservation(0, 1, "pass")},
    }
    result = search_alpha(checkpoint=checkpoint, case_ids=("b", "a"), alphas=tuple(values),
                          evaluate=values.__getitem__, refinement_rounds=0)
    assert [row.status for row in result.trials] == ["pass", "unmeasured", "fail", "unmeasured"]
    assert result.best == (result.trials[0],)
    assert result.best[0].mean_gain == -.0625
    assert result.trials[1].mean_gain == .9
    assert result.trials[2].mean_gain is None
    assert result.trials[2].observations == (("b", AlphaObservation(None, None, "unmeasured")),
                                             ("a", AlphaObservation(0, 1, "fail")))
    assert result.trials[3].mean_gain is None


def test_tied_leaders_refine_widest_brackets_and_all_stay_best(checkpoint):
    def evaluate(alpha):
        score = 1. if alpha in (.5, 2.) else 0.
        return {"case": AlphaObservation(0, score, "pass")}

    result = search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(.5, 1., 2., 4.),
                          evaluate=evaluate)
    assert tuple(trial.alpha for trial in result.trials) == (.5, 1., 2., 4., 3., 1.5)
    assert tuple(trial.alpha for trial in result.best) == (.5, 2.)


def test_no_eligible_measurement_retains_failed_and_unknown_trials(checkpoint):
    def evaluate(alpha):
        if alpha == 1.:
            return {"case": AlphaObservation(0, 1, "fail")}
        return {}

    result = search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1., 2.),
                          evaluate=evaluate, refinement_rounds=4)
    assert result.best == ()
    assert len(result.trials) == 2
    assert result.trials[0].mean_gain == 1.
    assert result.trials[1].mean_gain is None


def test_extreme_and_adjacent_float_brackets_do_not_overflow_or_duplicate(checkpoint):
    def evaluate(alpha):
        return {"case": AlphaObservation(0, -abs(alpha), "pass")}

    result = search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(-1.7e308, 1.7e308),
                          evaluate=evaluate)
    assert result.best[0].alpha == 0.
    next_alpha = math.nextafter(1., math.inf)
    result = search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1., next_alpha),
                          evaluate=evaluate, refinement_rounds=2)
    assert tuple(trial.alpha for trial in result.trials) == (1., next_alpha)


def test_extra_cases_and_untyped_observations_are_rejected(checkpoint):
    for invalid in ({"other": AlphaObservation(0, 1, "pass")}, {"case": (0, 1, "pass")}, None):
        with pytest.raises(ValueError, match="(?i)observa"):
            search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,),
                         evaluate=lambda _: invalid)


@pytest.mark.parametrize(("field", "invalid"), [
    ("case_ids", ()),
    ("case_ids", ("test", "test")),
    ("case_ids", "test"),
    ("alphas", ()),
    ("alphas", (math.nan,)),
    ("alphas", (True,)),
    ("refinement_rounds", -1),
])
def test_invalid_plan_cannot_start_evaluation(checkpoint, field, invalid):
    def unexpected(_):
        pytest.fail("invalid plan invoked evaluator")

    config = dict(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,), evaluate=unexpected)
    config[field] = invalid
    with pytest.raises(ValueError):
        search_alpha(**config)


def test_bad_scores_and_unqualified_verdicts_are_rejected():
    for values in ((0, math.inf, "pass"), (0, 1, "unknown"), (-1.7e308, 1.7e308, "pass")):
        with pytest.raises(ValueError):
            AlphaObservation(*values)


def test_checkpoint_mutation_and_callback_error_are_visible(checkpoint):
    def unavailable(_):
        raise OSError("scorer unavailable")

    with pytest.raises(OSError, match="scorer unavailable"):
        search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,), evaluate=unavailable)

    def mutate(_):
        with open(checkpoint.path, "wb") as stream:
            stream.write(b"different checkpoint")
        return {"case": AlphaObservation(0, 1, "pass")}

    with pytest.raises(ValueError, match="bytes differ"):
        search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,), evaluate=mutate)
    with pytest.raises(ValueError, match="bytes differ"):
        search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,), evaluate=unavailable)


def test_callback_error_keeps_concurrent_checkpoint_mutation_visible(checkpoint):
    def interrupt(_):
        with open(checkpoint.path, "wb") as stream:
            stream.write(b"different checkpoint")
        raise RuntimeError("native renderer interrupted")

    with pytest.raises(RuntimeError, match="native renderer interrupted") as caught:
        search_alpha(checkpoint=checkpoint, case_ids=("case",), alphas=(1.,), evaluate=interrupt)
    assert any("Checkpoint verification also failed" in note for note in caught.value.__notes__)
