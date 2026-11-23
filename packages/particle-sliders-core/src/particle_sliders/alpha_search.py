"""Bounded scalar strength selection on matched finished outputs."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .candidate_failure import FileRecord
from .parameter_search.metric import mean


Integrity = Literal["pass", "fail", "unmeasured"]


def _finite_float(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class AlphaObservation:
    """One case's matched scores and qualified categorical integrity judgment."""

    off: float | None
    on: float | None
    integrity: Integrity

    def __post_init__(self):
        if self.integrity not in ("pass", "fail", "unmeasured"):
            raise ValueError("integrity must be pass, fail, or unmeasured")
        for key, value in (("off", self.off), ("on", self.on)):
            if value is not None:
                object.__setattr__(self, key, _finite_float(value, key))
        if self.off is not None and self.on is not None and not isfinite(self.on - self.off):
            raise ValueError("matched ON minus OFF gain must be finite")


@dataclass(frozen=True)
class AlphaTrial:
    alpha: float
    status: Integrity
    mean_gain: float | None
    observations: tuple[tuple[str, AlphaObservation], ...]


@dataclass(frozen=True)
class AlphaSearchResult:
    checkpoint: FileRecord
    case_ids: tuple[str, ...]
    trials: tuple[AlphaTrial, ...]
    best: tuple[AlphaTrial, ...]


def _evaluate(alpha, cases, evaluate):
    measured = evaluate(alpha)
    if not isinstance(measured, Mapping) or measured.keys() - set(cases):
        raise ValueError("expected observations for declared case IDs")
    missing = AlphaObservation(None, None, "unmeasured")
    observations = tuple((case, measured.get(case, missing)) for case in cases)
    if any(not isinstance(item, AlphaObservation) for _, item in observations):
        raise ValueError("each observed case requires an AlphaObservation")
    gains = [item.on - item.off for _, item in observations
             if item.off is not None and item.on is not None]
    mean_gain = mean(gains) if len(gains) == len(cases) else None
    if any(item.integrity == "fail" for _, item in observations):
        status = "fail"
    elif mean_gain is None or any(item.integrity != "pass" for _, item in observations):
        status = "unmeasured"
    else:
        status = "pass"
    return AlphaTrial(alpha, status, mean_gain, observations)


def _best(trials):
    passing = [trial for trial in trials.values() if trial.status == "pass"]
    if not passing:
        return ()
    maximum = max(trial.mean_gain for trial in passing)
    return tuple(sorted((trial for trial in passing if trial.mean_gain == maximum),
                        key=lambda trial: trial.alpha))


def _refinement_points(trials, leaders):
    alphas = sorted(trials)
    leading_alphas = {trial.alpha for trial in leaders}
    intervals = [(low, high) for low, high in zip(alphas, alphas[1:])
                 if low in leading_alphas or high in leading_alphas]
    # Half-widths remain finite even when the bracket spans both float extremes.
    intervals.sort(key=lambda pair: (-(pair[1] / 2 - pair[0] / 2), pair[0]))
    points = []
    for low, high in intervals:
        midpoint = low / 2 + high / 2
        if low < midpoint < high:
            points.append(midpoint)
            if len(points) == 2:
                break
    return points


def search_alpha(
    *,
    checkpoint: FileRecord,
    case_ids: Iterable[str],
    alphas: Iterable[float],
    evaluate: Callable[[float], Mapping[str, AlphaObservation]],
    refinement_rounds: int = 1,
) -> AlphaSearchResult:
    """Measure a finite grid and return all passing maxima among tested points.

    The caller owns native rendering, scoring, and integrity qualification. Each
    call uses the same saved RAW adapter, cases, OFF outputs, and protocol. Missing
    or explicitly unmeasured cases keep the candidate unmeasured, unless any
    case carries a qualified fail. Evaluator exceptions propagate.

    Distinct grid points run in ascending order. Each refinement round explores
    up to two previously unmeasured midpoints in the widest adjacent intervals
    touching any tied leader, preferring the lower interval for equal widths.
    This keeps evaluations inside the original grid endpoints.
    """
    if not isinstance(checkpoint, FileRecord):
        raise ValueError("a pinned checkpoint FileRecord is required")
    if not callable(evaluate):
        raise ValueError("an evaluator callback is required")
    if type(refinement_rounds) is not int or refinement_rounds < 0:
        raise ValueError("refinement_rounds must be a nonnegative integer")
    if isinstance(case_ids, str):
        raise ValueError("provide an iterable of distinct case IDs")
    cases = tuple(case_ids)
    if not cases or any(not isinstance(case, str) or not case.strip() for case in cases) or len(set(cases)) != len(cases):
        raise ValueError("provide nonempty, distinct case IDs")
    grid = sorted({_finite_float(alpha, "alpha") for alpha in alphas})
    if not grid:
        raise ValueError("provide at least one alpha")

    checkpoint.verify()
    try:
        trials = {alpha: _evaluate(alpha, cases, evaluate) for alpha in grid}
        for _ in range(refinement_rounds):
            points = _refinement_points(trials, _best(trials))
            if not points:
                break
            trials.update((alpha, _evaluate(alpha, cases, evaluate)) for alpha in points)
    except BaseException as exc:
        try:
            checkpoint.verify()
        except (OSError, ValueError) as verification_error:
            exc.add_note(f"Checkpoint verification also failed: {verification_error}")
        raise
    checkpoint.verify()
    return AlphaSearchResult(checkpoint, cases, tuple(trials.values()), _best(trials))
