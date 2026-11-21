"""CFP, CLABS and matched-budget controls from Right Token, Right Time."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from random import Random

import torch


def contrastive_sae_direction(target_codes, neutral_codes, decoder, *, top_k):
    """Return the unit direction in Eqs. (2)-(3).

    Supply one row of token-averaged SAE codes per corpus example, and one
    decoder row per feature. Rank by signed target-minus-neutral difference.
    """
    if (decoder.ndim != 2 or target_codes.ndim != 2 or neutral_codes.ndim != 2
            or target_codes.shape[1] != decoder.shape[0]
            or neutral_codes.shape[1] != decoder.shape[0]
            or not target_codes.shape[0] or not neutral_codes.shape[0]):
        raise ValueError("expected nonempty example-by-feature codes and a feature-by-hidden decoder")
    if not 1 <= top_k <= decoder.shape[0]:
        raise ValueError("top_k must be in the range of the number of SAE features")
    target = _float_activations(target_codes).mean(dim=0)
    neutral = _float_activations(neutral_codes).mean(dim=0).to(target.device)
    differential = target - neutral
    if not torch.isfinite(differential).all():
        raise ValueError("feature differentials must be finite")
    indices = differential.argsort(descending=True, stable=True)[:top_k].to(decoder.device)
    direction = _float_activations(decoder.index_select(0, indices)).sum(dim=0)
    length = direction.norm()
    if not torch.isfinite(length) or length <= 0:
        raise ValueError("selected decoder rows must have a finite, nonzero sum")
    return direction / length


def _float_activations(tensor):
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def activation_projection(residual, direction, *, strength):
    """Read Eq. (5)'s alignment before editing, with Eq. (4)'s signed probe."""
    if not isfinite(strength) or strength == 0:
        raise ValueError("alignment requires a finite, nonzero steering strength")
    hidden = _float_activations(residual)
    probe = _direction_for(hidden, direction)
    return (hidden @ probe) * (1 if strength > 0 else -1)


def _direction_for(hidden, direction):
    if direction.ndim != 1 or hidden.ndim < 1 or direction.shape[0] != hidden.shape[-1]:
        raise ValueError("direction must be a vector matching the last residual dimension")
    return direction.to(device=hidden.device, dtype=hidden.dtype)


def steer_residual(residual, direction, *, strength, gate, preserve_norm=False, projection_bounds=None):
    """Apply Eq. (1) with a unit direction, optionally with Appendix B corrections.

    A Boolean gate broadcasts across positions, never across hidden features.
    If requested, restore the original norm first and then clamp the projection
    onto the unsigned direction. Ungated positions retain the input exactly.
    """
    if not isfinite(strength):
        raise ValueError("steering strength must be finite")
    if strength == 0 or gate is False:
        return residual
    hidden = _float_activations(residual)
    unit = _direction_for(hidden, direction)
    edited = hidden + strength * unit
    if preserve_norm:
        edited_norm = edited.norm(dim=-1, keepdim=True)
        nonzero = edited_norm > 0
        divisor = torch.where(nonzero, edited_norm, 1)
        restored = edited * (hidden.norm(dim=-1, keepdim=True) / divisor)
        edited = torch.where(nonzero, restored, hidden)
    if projection_bounds is not None:
        low, high = projection_bounds
        if not isfinite(low) or not isfinite(high) or low > high:
            raise ValueError("projection bounds must be finite and ordered")
        projection = edited @ unit
        edited = edited + (projection.clamp(low, high) - projection).unsqueeze(-1) * unit
    if gate is True:
        return edited.to(residual.dtype)
    if gate.dtype != torch.bool:
        raise TypeError("gate must be a Boolean or Boolean tensor")
    mask = torch.broadcast_to(gate.to(device=residual.device), residual.shape[:-1])
    return torch.where(mask.unsqueeze(-1), edited.to(residual.dtype), residual)


@dataclass(frozen=True)
class GateUsage:
    """Counts over eligible steps, including steps on which the gate stayed off."""

    fired: int
    eligible: int

    def __post_init__(self):
        if (type(self.fired) is not int or type(self.eligible) is not int
                or not 0 <= self.fired <= self.eligible):
            raise ValueError("gate usage requires integer counts with 0 <= fired <= eligible")

    @property
    def rate(self):
        if not self.eligible:
            raise ValueError("at least one eligible step is required to measure a firing rate")
        return self.fired / self.eligible


@dataclass(frozen=True)
class GateDecision:
    fired: bool
    probability: float


@dataclass(frozen=True, kw_only=True)
class CLABSParameters:
    """Calibration projection statistics and declared PI settings for Eqs. (6)-(9)."""

    mean: float
    std: float
    epsilon: float
    setpoint: float
    proportional_gain: float
    integral_gain: float
    beta: float

    def __post_init__(self):
        if (not all(isfinite(value) for value in (self.mean, self.std, self.epsilon,
                self.setpoint, self.proportional_gain, self.integral_gain, self.beta))
                or self.std < 0 or self.epsilon <= 0 or not isfinite(self.std + self.epsilon)
                or not 0 < self.beta <= 1
                or min(self.proportional_gain, self.integral_gain) < 0):
            raise ValueError("invalid CLABS statistics or controller parameters")


class ActivationGate:
    """Request-local gate state. Advance once per eligible native generation step."""

    def __init__(self, method, parameter, *, seed=None, clabs=None):
        if method not in ("cfp", "clabs", "random") or not isfinite(parameter):
            raise ValueError("specify cfp, clabs or random and a finite threshold, bias or rate")
        if method == "random" and not 0 <= parameter <= 1:
            raise ValueError("random gate rate must be in [0, 1]")
        if (method == "clabs") != isinstance(clabs, CLABSParameters):
            raise ValueError("CLABS requires CLABSParameters, which only CLABS uses")
        if method != "cfp" and type(seed) is not int:
            raise ValueError("stochastic gates require an explicit integer seed")
        self._method = method
        self._parameter = float(parameter)
        self._clabs = clabs
        self._random = Random(seed)
        self._integral = 0.0
        self._fired = 0
        self._eligible = 0

    @property
    def usage(self):
        return GateUsage(self._fired, self._eligible)

    def step(self, score=None):
        """Read pre-intervention alignment and return the current decision (Eqs. 5-10)."""
        if self._method != "random":
            if score is None:
                raise ValueError("activation-dependent gates require a finite scalar score")
            score = float(score)
            if not isfinite(score):
                raise ValueError("activation-dependent gates require a finite scalar score")
        probability = self._parameter
        if self._method == "cfp":
            probability = float(score <= self._parameter)
            fired = bool(probability)
        else:
            if self._method == "clabs":
                config = self._clabs
                alignment = (score - config.mean) / (config.std + config.epsilon)
                error = config.setpoint - alignment
                integral = (1 - config.beta) * self._integral + config.beta * error
                feedback = config.proportional_gain * error + config.integral_gain * integral
                probability += feedback
                if not all(isfinite(value) for value in (error, integral, probability)):
                    raise ValueError("CLABS feedback overflowed")
                probability = min(1.0, max(0.0, probability))
                self._integral = integral
            fired = self._random.random() < probability
        self._eligible += 1
        self._fired += fired
        return GateDecision(fired, probability)


@dataclass(frozen=True)
class GateCalibration:
    parameter: float
    target_rate: float
    usage: GateUsage


def calibrate_gate_budget(
    rollout: Callable[[float], GateUsage], *, target_rate, bounds, tolerance, max_iterations,
):
    """Bisect a CFP threshold or CLABS bias using fresh active calibration rollouts.

    Each call resets gate state and reuses the same held-out calibration cases
    and seeds. It must run with candidate steering enabled and return pooled
    fired/eligible counts. Report failure if no measured rate meets tolerance.
    """
    low, high = map(float, bounds)
    if not (0 < target_rate < 1 and 0 <= tolerance < min(target_rate, 1 - target_rate)
            and isfinite(low) and isfinite(high) and low < high
            and type(max_iterations) is int and max_iterations > 0):
        raise ValueError("invalid target rate, tolerance, parameter bounds or iteration limit")

    def measure(parameter):
        return GateCalibration(parameter, target_rate, rollout(parameter))

    lower, upper = measure(low), measure(high)
    best = min((lower, upper), key=lambda trial: abs(trial.usage.rate - target_rate))
    if abs(best.usage.rate - target_rate) <= tolerance:
        return best
    if not lower.usage.rate < target_rate < upper.usage.rate:
        raise ValueError(f"active rollout rates {lower.usage.rate:g} and {upper.usage.rate:g} "
                         f"do not bracket target {target_rate:g}")
    for _ in range(max_iterations):
        midpoint = low / 2 + high / 2
        if midpoint == low or midpoint == high:
            break
        trial = measure(midpoint)
        distance = abs(trial.usage.rate - target_rate)
        if distance <= tolerance:
            return trial
        if distance < abs(best.usage.rate - target_rate):
            best = trial
        if trial.usage.rate < target_rate:
            low = midpoint
        else:
            high = midpoint
    raise ValueError(f"gate calibration did not meet target {target_rate:g} ± {tolerance:g}; "
                     f"closest active rollout was {best.usage.rate:g} at {best.parameter:g}")
