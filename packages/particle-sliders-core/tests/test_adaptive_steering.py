from fractions import Fraction
from random import Random

import pytest
import torch

from particle_sliders.adaptive_steering import (
    ActivationGate,
    CLABSParameters,
    GateUsage,
    activation_projection,
    calibrate_gate_budget,
    contrastive_sae_direction,
    steer_residual,
)


def test_signed_sae_ranking_and_pre_edit_alignment():
    target = torch.tensor([[3., 0., 5.], [5., 2., 5.]])
    neutral = torch.tensor([[2., 20., 4.], [2., 20., 4.]])
    direction = contrastive_sae_direction(target, neutral, torch.eye(3), top_k=2)
    torch.testing.assert_close(direction, torch.tensor([1., 0., 1.]) / 2**0.5)
    residual = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    torch.testing.assert_close(activation_projection(residual, direction, strength=-7),
                               -(residual @ direction))
    with pytest.raises(ValueError, match="nonzero sum"):
        contrastive_sae_direction(target, neutral, torch.zeros((3, 3)), top_k=2)


def test_fixed_direction_intervention_and_paper_correction_order():
    residual = torch.tensor([[0., 3.], [0., -3.]], dtype=torch.bfloat16)
    unit = torch.tensor([1., 0.])
    mask = torch.tensor([True, False])
    torch.testing.assert_close(steer_residual(residual, unit, strength=4, gate=mask),
                               torch.tensor([[4., 3.], [0., -3.]], dtype=residual.dtype))
    corrected = steer_residual(residual, unit, strength=4, gate=mask, preserve_norm=True,
                               projection_bounds=(0, 2))
    torch.testing.assert_close(corrected[0], torch.tensor([2., 1.8], dtype=residual.dtype))
    assert torch.equal(corrected[1], residual[1])
    assert steer_residual(residual, unit, strength=4, gate=False) is residual
    assert steer_residual(residual, unit, strength=0, gate=True) is residual
    canceled = steer_residual(unit, unit, strength=-1, gate=True, preserve_norm=True)
    torch.testing.assert_close(canceled, unit)


def test_cfp_bisects_steering_active_trajectories():
    traces = {}

    def run_rollout(threshold):
        gate = ActivationGate("cfp", threshold)
        previous_edit = 0
        scores = []
        for base_alignment in (-3, -2, -1, 1, 2, 3):
            scores.append(base_alignment + 0.25 * previous_edit)
            previous_edit = gate.step(scores[-1]).fired
        traces[threshold] = scores
        return gate.usage

    calibration = calibrate_gate_budget(run_rollout, target_rate=0.5, bounds=(-5, 5),
                                        tolerance=0, max_iterations=4)
    assert calibration.parameter == 0
    assert calibration.usage == GateUsage(fired=3, eligible=6)
    assert traces[0] != traces[-5]
    assert ActivationGate("cfp", calibration.parameter).step(0).fired
    with pytest.raises(ValueError, match="do not bracket"):
        calibrate_gate_budget(run_rollout, target_rate=0.5, bounds=(4, 5),
                              tolerance=0, max_iterations=4)
    with pytest.raises(ValueError, match="closest active rollout"):
        calibrate_gate_budget(run_rollout, target_rate=0.4, bounds=(-5, 5),
                              tolerance=0, max_iterations=4)


@pytest.mark.parametrize("fires", [198, 200])
@pytest.mark.parametrize("parameter", [-1, 0, 1])
def test_calibration_accepts_inclusive_one_firing_boundary(fires, parameter):
    eligible = 398
    measured = GateUsage(fired=fires, eligible=eligible)

    def rollout(candidate):
        if candidate == parameter:
            return measured
        return GateUsage(fired=0 if candidate < parameter else eligible, eligible=eligible)

    result = calibrate_gate_budget(rollout, target_rate=0.5, bounds=(-1, 1),
                                   tolerance=1 / eligible, max_iterations=1)
    assert result.parameter == parameter
    assert result.usage == measured


@pytest.mark.parametrize("fires", [197, 201])
def test_calibration_rejects_firing_counts_outside_boundary(fires):
    eligible = 398

    def rollout(candidate):
        if candidate == 0:
            return GateUsage(fired=fires, eligible=eligible)
        return GateUsage(fired=0 if candidate < 0 else eligible, eligible=eligible)

    with pytest.raises(ValueError, match="did not meet target"):
        calibrate_gate_budget(rollout, target_rate=0.5, bounds=(-1, 1),
                              tolerance=1 / eligible, max_iterations=1)


@pytest.mark.parametrize("target,fires", [
    (Fraction(1, 5), 59),
    (Fraction(1, 5), 61),
    (Fraction(4, 5), 239),
    (Fraction(4, 5), 241),
])
def test_calibration_checks_exact_one_firing_boundaries(target, fires):
    usage = GateUsage(fires, 300)
    rollouts = {-1.0: GateUsage(0, 300), 0.0: usage, 1.0: GateUsage(300, 300)}
    result = calibrate_gate_budget(rollouts.__getitem__, target_rate=target, bounds=(-1, 1),
                                   tolerance=Fraction(1, 300), max_iterations=1)
    assert result.parameter == 0
    assert result.usage is usage
    assert type(result.target_rate) is float
    assert result.target_rate == float(target)


def test_calibration_rejects_rate_below_exact_boundary():
    rollouts = {-1.0: GateUsage(0, 300), 0.0: GateUsage(58, 300), 1.0: GateUsage(300, 300)}
    with pytest.raises(ValueError, match="did not meet target"):
        calibrate_gate_budget(rollouts.__getitem__, target_rate=Fraction(1, 5), bounds=(-1, 1),
                              tolerance=Fraction(1, 300), max_iterations=1)


def test_float_calibration_keeps_binary64_boundary_behavior():
    rollouts = {-1.0: GateUsage(0, 300), 0.0: GateUsage(59, 300), 1.0: GateUsage(300, 300)}
    with pytest.raises(ValueError, match="did not meet target"):
        calibrate_gate_budget(rollouts.__getitem__, target_rate=0.2, bounds=(-1, 1),
                              tolerance=1 / 300, max_iterations=1)


def test_exact_calibration_cannot_skip_a_lower_endpoint_at_the_boundary():
    def rollout(_):
        pytest.fail("a potentially qualifying lower endpoint requires a measurement")

    with pytest.raises(ValueError, match="strictly below the accepted rate interval"):
        calibrate_gate_budget(rollout, target_rate=Fraction(1, 5), bounds=(-1, 1),
                              tolerance=Fraction(1, 300), max_iterations=1,
                              lower_usage_bound=GateUsage(59, 300))


@pytest.mark.parametrize("upper_fired, expected_parameter", [(8, 1), (12, 0)])
def test_calibration_can_resume_with_a_proven_lower_bound(upper_fired, expected_parameter):
    upper = GateUsage(upper_fired, 16)
    midpoint = GateUsage(8, 16)
    trials = []

    def rollout(parameter):
        trials.append(parameter)
        return {1: upper, 0: midpoint}[parameter]

    result = calibrate_gate_budget(rollout, target_rate=0.5, bounds=(-1, 1),
                                   tolerance=0, max_iterations=4,
                                   lower_usage_bound=GateUsage(4, 16))
    assert result.parameter == expected_parameter
    assert trials == ([1] if expected_parameter == 1 else [1, 0])
    assert result.usage is (upper if expected_parameter == 1 else midpoint)


def test_calibration_rejects_a_bound_that_can_meet_inclusive_tolerance():
    def rollout(_):
        pytest.fail("a potentially qualifying lower endpoint requires a measurement")

    with pytest.raises(ValueError, match="strictly below the accepted rate interval"):
        calibrate_gate_budget(rollout, target_rate=0.5, bounds=(-1, 1),
                              tolerance=1/16, max_iterations=4,
                              lower_usage_bound=GateUsage(7, 16))


def test_calibration_with_a_lower_bound_requires_a_measured_upper_bracket():
    def rollout(parameter):
        assert parameter == 1
        return GateUsage(6, 16)

    with pytest.raises(ValueError, match="upper rollout.*does not bracket"):
        calibrate_gate_budget(rollout, target_rate=0.5, bounds=(-1, 1),
                              tolerance=0, max_iterations=4,
                              lower_usage_bound=GateUsage(4, 16))


def test_clabs_updates_ema_before_sampling_and_resets_per_request():
    config = CLABSParameters(mean=1.5, std=1.0, epsilon=1e-6, setpoint=0.75,
                             proportional_gain=1.0, integral_gain=0.2, beta=0.2)
    bias = 0.3
    gate = ActivationGate("clabs", bias, clabs=config, seed=123)
    restart = ActivationGate("clabs", bias, clabs=config, seed=123)
    independent_rng = Random(123)
    scores = [1.2, 2.2, 2.2, 3.5, 2.3]
    history = 0
    fired = 0
    for score in scores:
        error = 0.75 - (score - 1.5) / 1.000001
        history = 0.8 * history + 0.2 * error
        expected = min(1, max(0, bias + error + 0.2 * history))
        decision = gate.step(score)
        assert decision.probability == pytest.approx(expected)
        assert decision.fired == (independent_rng.random() < expected)
        assert restart.step(score) == decision
        fired += decision.fired
    assert gate.usage == GateUsage(fired=fired, eligible=len(scores))
    with pytest.raises(ValueError, match="finite scalar"):
        gate.step(float("nan"))
    assert gate.usage.eligible == len(scores)


def test_random_control_requires_seed_and_zero_gain_clabs_is_random():
    config = CLABSParameters(mean=0, std=1, epsilon=1e-6, setpoint=1,
                             proportional_gain=0, integral_gain=0, beta=1)
    control = ActivationGate("random", 0.5, seed=19)
    clabs = ActivationGate("clabs", 0.5, clabs=config, seed=19)
    assert [control.step() for _ in range(40)] == [clabs.step(i) for i in range(40)]
    assert control.usage == clabs.usage
    with pytest.raises(ValueError, match="explicit integer seed"):
        ActivationGate("random", 0.5)
