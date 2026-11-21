"""CPU gate for the activation-difference-coherence claim
(arXiv:2505.22637): whether a harvested steering direction will
steer at all is predicted, BEFORE any steering is attempted, by the
coherence of the per-example activation differences it is harvested
from -- the mean pairwise cosine of the differences, a statistic
computable from the same paired states the harvest consumes,
without knowing the true behavior direction.

Paper: "Understanding (Un)Reliability of Steering Vectors in
Language Models" (Braun, Eickhoff, Krueger, Bahrainian &
Krasheninnikov). Reached via the reference list of "The Sharp
Directions Are Against You" (arXiv:2610.16084, this repo's PR #237)
-- a May-2025 predecessor a normal date-window scan would never
surface. Abstract-verified findings asserted here in toy form:
steering effects have high per-sample variance and often go the
wrong way; steering vectors harvested from different prompt types
differ directionally; and the load-bearing one -- higher cosine
similarity between the training set's per-example activation
differences predicts more effective steering, and datasets whose
positive/negative activations are better separated are more
steerable. The paper's exact model/task point values are NOT
asserted -- only the predictive structure, in a toy where the true
direction is known.

Toy operationalization: paired states in R^128 share a per-example
centre c_i ~ N(0, I); the positive/negative states are

    pos_i = c_i + u/2 + eta_i+,   neg_i = c_i - u/2 + eta_i-,
    eta ~ N(0, (sigma/sqrt(2))^2 I),

around a unit ground-truth behavior direction u, so the per-example
activation difference is delta_i = pos_i - neg_i = u + noise with
per-coordinate noise variance sigma^2. The dispersion knob is
sigma, swept over SIGMAS = (0.05, 0.2, 0.5, 1.4) -- from coherent
(the differences point nearly the same way) to incoherent (each
difference is mostly its own noise). The harvested direction is
the plain mean difference, normalized -- exactly what
fit_direction computes in
packages/particle-sliders-core/src/particle_sliders/textsliders/
reward_sliders/directions.py (mean of per-cell highest-minus-lowest
activation means, then unit-normalized).

Pre-harvest statistic (knows nothing about u): coherence =
mean pairwise cosine of the delta_i. Secondary covariate, also
u-free: class separability = mean/std of the delta_i projected on
the harvested direction.

Readout world (family of PR #231 / #237 toys): one behavior axis u
plus the orthogonal side subspace. Steering a held-out state by a
unit-norm harvested direction v at matched norm gives behavior
gain v.u and side-effect displacement ||P_side v|| =
sqrt(1 - (v.u)^2). The PRIMARY efficacy metric, fixed for the
whole test, is

    E(v) = (v . u) - sqrt(1 - (v . u)^2)

-- net steering: behavior gained minus collateral displacement at
the same steering norm. E > 0 means the step buys more behavior
than side effect; E < 0 means the "steering" is mostly side
effect.

Controls:
  (i) shuffled pairing: positives from the COHERENT (sigma = 0.05)
      dataset re-paired with negatives drawn from an independent
      batch (fresh centres) -- the marginals still contain the
      signal, but the per-example differences are dominated by
      centre mismatch, so coherence collapses and efficacy must
      sit at the floor;
  (ii) a random unit direction: efficacy at the floor by
      construction.

Measured on pop-os (particlegan 0.10.1), 3 seeds: coherence by
sigma level 0.759-0.766 (sigma 0.05), 0.165-0.176 (0.2),
0.032-0.036 (0.5), -0.001..0.012 (1.4); efficacy by level
0.884-0.900, 0.526-0.597, -0.022..0.130, -0.781..-0.396.
Spearman(coherence, efficacy) = 1.00 on every seed; endpoint
margin (coherent minus incoherent efficacy) 1.29-1.67. Shuffled
control: coherence 0.003-0.008, efficacy -0.667..-0.599; random
control: efficacy -1.008..-0.938. Separability covariate: 17.4-23.6
on the coherent arm vs 2.12-3.23 on the incoherent arm.
Assertions below encode exactly these margins, no more.

What is NOT asserted: any real model's coherence threshold. The
toy has isotropic noise and a single true direction; real
per-example differences also vary because prompts genuinely
differ (the paper's prompt-type finding), which this toy folds
into the same dispersion knob. The claim tested is the ordering
-- coherence ranks efficacy before steering -- not a universal
cutoff value.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family).
Pure numpy/torch geometry -- no GAN machinery, because the claim
is about harvested directions, not training; particlegan is only
the skip gate.
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
    pytest.skip("particlegan unavailable", allow_module_level=True)
try:
    from particlegan import get_recipe  # noqa: E402
except Exception:  # a particlegan without the recipe API at all
    get_recipe = None


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    if get_recipe is None:
        pytest.skip("installed particlegan has no recipe API")
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 128           # activation dimension
N = 32            # paired examples per dispersion level
SIGMAS = (0.05, 0.2, 0.5, 1.4)   # dispersion: coherent -> incoherent


def _unit(x):
    return x / np.linalg.norm(x)


def _coherence(deltas):
    """Mean pairwise cosine of the per-example differences.

    Computable from the paired states alone -- no knowledge of u.
    """
    n = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
    gram = n @ n.T
    iu = np.triu_indices(len(deltas), 1)
    return float(gram[iu].mean())


def _efficacy(v, u):
    """Primary metric: behavior gain minus side displacement,
    unit steering norm."""
    c = float(v @ u)
    return c - float(np.sqrt(max(0.0, 1.0 - c * c))), c


def _spearman(x, y):
    """Spearman rank correlation (no ties in this toy)."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _paired_deltas(rng, u, sigma):
    """pos/neg paired states around shared centres; returns the
    per-example differences and the states themselves."""
    centres = rng.standard_normal((N, D))
    sd = sigma / np.sqrt(2.0)
    pos = centres + 0.5 * u + rng.normal(0.0, sd, (N, D))
    neg = centres - 0.5 * u + rng.normal(0.0, sd, (N, D))
    return pos - neg, pos, neg


def _harvest(deltas):
    """The plain mean difference, normalized -- fit_direction's
    estimator in toy form."""
    return _unit(deltas.mean(0))


def run_seed(seed):
    rng = np.random.default_rng(13000 + seed)
    u = _unit(rng.standard_normal(D))

    levels = []
    coherent_pos = None
    for sigma in SIGMAS:
        deltas, pos, _ = _paired_deltas(rng, u, sigma)
        if sigma == SIGMAS[0]:
            coherent_pos = pos
        v = _harvest(deltas)
        eff, cos_u = _efficacy(v, u)
        proj = deltas @ v
        levels.append(dict(
            sigma=sigma, coherence=_coherence(deltas), eff=eff,
            cos_u=cos_u, sep=float(proj.mean() / proj.std())))

    # Control (i): shuffled pairing -- coherent positives re-paired
    # with negatives from an independent batch (fresh centres).
    sd = SIGMAS[0] / np.sqrt(2.0)
    centres2 = rng.standard_normal((N, D))
    neg2 = centres2 - 0.5 * u + rng.normal(0.0, sd, (N, D))
    deltas_shuf = coherent_pos - neg2
    eff_shuf, _ = _efficacy(_harvest(deltas_shuf), u)

    # Control (ii): a random unit direction.
    eff_rand, _ = _efficacy(_unit(rng.standard_normal(D)), u)

    return dict(levels=levels,
                coh_shuf=_coherence(deltas_shuf), eff_shuf=eff_shuf,
                eff_rand=eff_rand)


def test_difference_coherence_predicts_steering_efficacy():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        lv = r["levels"]
        coh = [x["coherence"] for x in lv]
        eff = [x["eff"] for x in lv]

        # Trap guard: the knob must actually span coherent to
        # incoherent -- otherwise the test tests nothing.
        assert coh[0] >= 0.6, (
            f"seed {seed}: coherent-arm coherence {coh[0]:.3f} "
            f"-- dispersion knob broken at the low end")
        assert abs(coh[-1]) <= 0.05, (
            f"seed {seed}: incoherent-arm coherence {coh[-1]:.3f} "
            f"-- dispersion knob broken at the high end")

        # KEY ASSERTION: the u-free coherence statistic tracks
        # realized efficacy across the dispersion levels.
        rho = _spearman(coh, eff)
        assert rho >= 0.8, (
            f"seed {seed}: Spearman(coherence, efficacy) {rho:.2f} "
            f"across levels {list(zip(coh, eff))}")

        # Endpoint margin: coherent harvests steer, incoherent
        # harvests do not.
        assert eff[0] - eff[-1] >= 1.0, (
            f"seed {seed}: endpoint efficacy margin "
            f"{eff[0] - eff[-1]:.3f} (high {eff[0]:.3f}, "
            f"low {eff[-1]:.3f})")
        assert eff[0] > 0.5, (
            f"seed {seed}: coherent efficacy {eff[0]:.3f} -- the "
            f"harvest does not net-steer even when coherent")
        assert eff[-1] <= -0.3, (
            f"seed {seed}: incoherent efficacy {eff[-1]:.3f} -- "
            f"the harvest still net-steers when incoherent")

        # The incoherent endpoint sits at the control floor.
        assert eff[-1] <= r["eff_shuf"] + 0.4, (
            f"seed {seed}: incoherent efficacy {eff[-1]:.3f} vs "
            f"shuffled control {r['eff_shuf']:.3f}")
        assert eff[-1] <= r["eff_rand"] + 0.75, (
            f"seed {seed}: incoherent efficacy {eff[-1]:.3f} vs "
            f"random control {r['eff_rand']:.3f}")

        # Controls behave: shuffled pairing destroys coherence
        # despite signal-bearing marginals, and both controls sit
        # below zero net steering while the coherent arm clears
        # every control by a wide margin.
        assert abs(r["coh_shuf"]) <= 0.05, (
            f"seed {seed}: shuffled coherence {r['coh_shuf']:.3f}")
        assert r["eff_shuf"] <= -0.4, (
            f"seed {seed}: shuffled efficacy {r['eff_shuf']:.3f}")
        assert r["eff_rand"] <= -0.6, (
            f"seed {seed}: random efficacy {r['eff_rand']:.3f}")
        assert eff[0] - r["eff_shuf"] >= 1.2, (
            f"seed {seed}: coherent-vs-shuffled margin "
            f"{eff[0] - r['eff_shuf']:.3f}")
        assert eff[0] - r["eff_rand"] >= 1.4, (
            f"seed {seed}: coherent-vs-random margin "
            f"{eff[0] - r['eff_rand']:.3f}")

        # Secondary covariate (recorded in the docstring table):
        # class separability along the harvested direction also
        # degrades with dispersion -- high arm clearly above low.
        assert lv[0]["sep"] > 3.0 * lv[-1]["sep"], (
            f"seed {seed}: separability high {lv[0]['sep']:.2f} "
            f"vs low {lv[-1]['sep']:.2f}")
