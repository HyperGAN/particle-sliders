"""CPU gate for the Right Token, Right Time claim (arXiv:2611.18274):
steering is a per-token ALLOCATION problem -- at matched total
intervention, steering only the tokens least aligned with the target
direction matches or beats steering every token on target behavior,
while causing less off-target displacement.

Paper: "Right Token, Right Time: From Per-Token Gating to Trajectory-Aware
Activation Steering" (Eshraghi et al., 2026). Its central mechanism: a
fixed steering budget spent uniformly is wasted on tokens that are
already aligned with the target direction and under-serves the tokens
that lag; the paper's CFP rule gates steering to the least-aligned
tokens (budget fraction 0.5 in its headline Gemma 2 2B result) and beats
open-loop steer-all and random gating at matched intervention.

Toy operationalization: sequences of T=16 token states h in R^12, steered
along a fixed direction v. The "prompted" target brings each token's
alignment up to a target level TAU along a PROBE direction u (cos(u, v)
= 0.8, u != v): the ground-truth gain of a token is its deficit below
TAU along u, clipped to [0, 2.5] -- tokens already above the level need
no steering (gain 0, and they are where off-target displacement is
measured). Crucially, no arm ever sees u or the gains. The CFP gate uses
only deployment-available information -- each token's current alignment
h . v and the same target level TAU along v -- selects the k = T/2
least-aligned tokens, assigns each its alignment deficit along v, and
rescales so the total intervention equals the shared budget B exactly.
Because u != v, the v-deficit is a noisy estimate of the true need:
selection is enriched for needy tokens but is NOT the label set.

Arms, all at exactly matched total intervention (sum of applied
coefficients == B per sequence, B = T * E[gain] under the world model):

  (a) steer-all:   uniform coefficient B / T on every token;
  (b) CFP gate:    deficit-weighted steering on the k least-aligned;
  (c) random gate: the same deficit-weighted rule on k random tokens
                   (the paper's open-loop random gating baseline) --
                   isolates selection from the within-gate rule.

Assertions per seed: CFP target MSE <= 0.8x steer-all, CFP off-target
displacement (mean squared movement on zero-gain tokens) <= 0.5x
steer-all, and CFP target MSE <= 0.5x random-gate. Skips cleanly when
the installed particlegan does not provide the EMPF recipe.
"""
import math

import pytest
import torch

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 12
T = 16               # tokens per sequence
BUDGET_FRAC = 0.5    # the paper's headline budget: gate half the tokens
TAU = 0.3            # target alignment level (along u for truth, v for arms)
CAP = 2.5
COS_UV = 0.8         # probe/steering correlation: informative, not identity
N_SEQ = 512


def _expected_gain():
    """E[clip(TAU - s, 0, CAP)] for s ~ N(0, 1), ignoring the CAP tail
    (P(gain > CAP) ~ 1%); the shared per-token budget for every arm."""
    return (TAU * 0.5 * (1.0 + math.erf(TAU / math.sqrt(2.0)))
            + math.exp(-TAU * TAU / 2.0) / math.sqrt(2.0 * math.pi))


def _world(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(D, generator=g)
    v = v / v.norm()
    w = torch.randn(D, generator=g)
    w = w - (w @ v) * v
    w = w / w.norm()
    u = COS_UV * v + math.sqrt(1.0 - COS_UV ** 2) * w
    return v, u


def _run_seed(seed):
    v, u = _world(seed)
    ge = torch.Generator().manual_seed(555 + seed)
    h = torch.randn(N_SEQ, T, D, generator=ge)
    gain = torch.clamp(TAU - h @ u, 0.0, CAP)          # ground truth (arms never see it)
    target = h + gain.unsqueeze(-1) * v
    align = h @ v                                       # deployment-available
    budget = T * _expected_gain()
    k = int(T * BUDGET_FRAC)

    def gated(idx):
        raw = torch.clamp(TAU - torch.gather(align, 1, idx), 0.0, CAP)
        coeff = raw * (budget / raw.sum(dim=1, keepdim=True).clamp(min=1e-6))
        return torch.zeros(N_SEQ, T).scatter(1, idx, coeff)

    cfp_idx = align.topk(k, largest=False).indices
    gr = torch.Generator().manual_seed(777 + seed)
    rnd_idx = torch.argsort(torch.rand(N_SEQ, T, generator=gr), dim=1)[:, :k]
    arms = {
        "all": torch.full((N_SEQ, T), budget / T),
        "cfp": gated(cfp_idx),
        "random": gated(rnd_idx),
    }
    zero_gain = gain <= 0.0
    out = {}
    for name, coeff in arms.items():
        # Matched intervention is a construction property -- verify it.
        assert torch.allclose(coeff.sum(dim=1),
                              torch.full((N_SEQ,), budget), atol=1e-4)
        steered = h + coeff.unsqueeze(-1) * v
        mse = ((steered - target) ** 2).sum(-1).mean().item()
        off = ((coeff ** 2) * zero_gain).sum().item() / zero_gain.sum().item()
        out[name] = (mse, off)
    out["none_mse"] = (gain ** 2).mean().item()
    return out


def test_budgeted_gating_beats_steer_all_at_matched_intervention():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = _run_seed(seed)
        all_mse, all_off = r["all"]
        cfp_mse, cfp_off = r["cfp"]
        rnd_mse, _ = r["random"]
        # Steering at all must beat doing nothing.
        assert cfp_mse < r["none_mse"] and all_mse < r["none_mse"]
        # The paper's claim, at exactly matched total intervention:
        # gating to the least-aligned tokens is at least as good on
        # target behavior (measured: ~0.67x steer-all) ...
        assert cfp_mse <= 0.8 * all_mse, (
            f"seed {seed}: CFP target MSE {cfp_mse:.3f} vs steer-all {all_mse:.3f}")
        # ... causes clearly less off-target displacement on the tokens
        # that needed no steering (measured: ~0.32x steer-all) ...
        assert cfp_off <= 0.5 * all_off, (
            f"seed {seed}: CFP off-target {cfp_off:.3f} vs steer-all {all_off:.3f}")
        # ... and the selection itself carries the effect: the same
        # rule on random tokens collapses (measured: CFP ~0.26x random).
        assert cfp_mse <= 0.5 * rnd_mse, (
            f"seed {seed}: CFP target MSE {cfp_mse:.3f} vs random gate {rnd_mse:.3f}")
