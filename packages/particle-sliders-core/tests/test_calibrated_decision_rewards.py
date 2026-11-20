"""CPU gate for the calibrated decision-reward claim
(arXiv:2611.25135): a decision-model judge whose probabilities are
(a) debiased over option orders and (b) calibrated on ~100 labels
has roughly half the ECE of the raw judge, stays calibrated on
shifted / policy-generated samples, and -- the paper's stated
boundary -- turns overconfident where the base model mostly fails.

Paper: "Calibrated Decision Rewards: Training Language Models with
a Judge That Knows How Sure It Is" (Agnihotri & Kapoor). A softmax
decision judge is systematically miscalibrated and sensitive to the
order in which options are presented. The paper's recipe averages
the judge's distribution over both presentation orders (removing
the order bias) and fits a calibration map on a small labelled set
(~100 labels). Headline result: ECE 0.074 for the debiased +
calibrated judge vs 0.141 for the raw judge on the same backbone --
roughly a halving -- with the calibration holding on shifted,
policy-generated samples. The stated boundary: on distributions
where the base model mostly fails, the calibrated judge turns
overconfident (its mean predicted probability exceeds the empirical
success rate by a wide margin).

Toy operationalization: items with KNOWN ground-truth correctness
probabilities p*(x), drawn from a known Beta generative process;
outcomes y ~ Bernoulli(p*). A synthetic judge emits, for the
correct option presented in position A or B,

    q(x, order) = clip( sigmoid(TEMP * logit(p*) + OFF)
                        + kink(p*) + (DELTA / 2) * s(order)
                        + noise ),   s = +1 if the correct option
                        sits in position B, -1 if in position A,

with TEMP = 2.0, OFF = 0 (a pure temperature / overconfidence
distortion of the logit, Platt-family so a Platt map can in
principle undo it), DELTA = 0.14 (the option-order gap: the
judge favours whichever option sits in position B, so the
correct option's probability shifts by +-DELTA / 2 with its
position -- the two-option form of an order bias), and a low-p* kink,
kink(p*) = 0.16 * clip((0.35 - p*) / 0.35, 0, 1): the judge
additionally overrates items the base model mostly fails. The
kink is NOT Platt-family (it is localised in p*, not linear in
logit q), so a global Platt map fitted on a mixed distribution
cannot remove it -- that mismatch is the toy form of the paper's
boundary. Noise sd 0.02. All parameters are fixed before any
measurement; the kink is an assumption of the toy, stated here,
not a result.

Arms (calibration maps fitted per seed on exactly N_CAL = 100
labelled in-distribution items, Platt / logistic on logit(q) with
a mild ridge toward the identity map):
  (1) raw:   q at a single, randomly assigned presentation order;
  (2) deb:   order-debiased only -- q averaged over orders A, B;
  (3) cal:   Platt map fitted on raw q, applied to raw q;
  (4) full:  Platt map fitted on debiased q, applied to debiased q
             (the paper's recipe).

Metrics per seed (3 seeds), N_EVAL = 4000 items per distribution:
  ECE: 10 equal-mass bins whose edges are FIXED once per seed from
       the raw judge's quantiles on a separate in-distribution
       reference pool (edges fit pre-shift and reused for every arm
       and distribution -- bins re-fit inside a shifted pool would
       manufacture spurious ECE, the cycle-8 funnel lesson);
  order-flip rate: fraction of items whose thresholded decision
       (q > 0.5) changes between presentation orders, per arm --
       for the debiased arms the decision is a function of the
       order-averaged q, so the flip rate is 0 by construction;
       the non-trivial content is that the raw and calibrated-only
       arms flip often while the debiased arms cannot;
  order gap: mean |q_B - q_A| over the same items (the raw
       judge's per-item order sensitivity, ~DELTA by construction;
       the debiased score has no order left to gap over);
  mean bias: mean predicted - empirical rate;
  boundary overconfidence: mean predicted - empirical rate of the
       full arm on the mostly-fail slice vs the mixed slice.

Distributions: in-distribution p* ~ Beta(2.5, 2.5); shifted
(policy-generated) p* ~ Beta(3.0, 1.6) -- the policy's own samples
skew toward items it is more likely to get right; mostly-fail
p* ~ Beta(1.0, 7.0) (empirical rate ~0.12).

Measured on pop-os (particlegan 0.10.1, 3 seeds): in-distribution
ECE 0.090-0.097 (raw) vs 0.038-0.054 (full), ratio 0.42-0.59
(paper: 0.074 vs 0.141, ratio 0.52); shifted ECE 0.082-0.104 (raw)
vs 0.026-0.049 (full), ratio 0.32-0.50; order-flip rate
0.114-0.130 (raw) and 0.115-0.128 (calibrated only) vs 0 (full, by
construction), raw order gap 0.135-0.136; boundary overconfidence
of the full arm +0.092..+0.101 on the mostly-fail slice vs
-0.050..+0.009 on the mixed slice. Calibrated-only ECE
(0.030-0.061 in-distribution, 0.022-0.073 shifted) is comparable
to the full arm's: in this toy the ECE gain is carried by
calibration, and debiasing's measured contribution is the
elimination of order flips / order gap, not further ECE --
assertions encode exactly that division. Assertions below encode
exactly these margins, no more.

What is NOT asserted: the paper's real-model numbers (the exact
0.074 / 0.141 ECE points, any benchmark or downstream RL result) --
the toy has no language model; only the margin STRUCTURE is
claimed (full-arm ECE roughly half the raw ECE on both the
in-distribution and shifted pools, debiasing removes the order
effect calibration alone leaves standing, and the calibrated judge
is materially more overconfident where the base model mostly
fails). The shifted pool here skews EASIER (policy-generated
samples); a harder shift is not asserted. The boundary is driven
by the planted low-p* kink: the toy shows a Platt map cannot fix
a localised, non-Platt distortion concentrated where calibration
labels are sparse, not that every real judge fails the same way.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure numpy -- no GAN
machinery, because the claim is about judge probabilities, not
training dynamics.
"""
import numpy as np
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


TEMP = 2.0         # judge temperature distortion of logit(p*)
OFF = 0.0          # judge logit offset
DELTA = 0.14       # option-order bias (correct option in position B)
KINK = 0.16        # extra optimism where p* is low (non-Platt)
NOISE = 0.02       # judge noise sd on the probability scale
N_CAL = 100        # calibration labels (the paper's ~100)
N_EVAL = 4000      # items per distribution per seed
N_REF = 5000       # reference pool for the fixed ECE bin edges
N_BINS = 10
DIST_IN = (2.5, 2.5)
DIST_SHIFT = (3.0, 1.6)
DIST_FAIL = (1.0, 7.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _judge(pstar, in_b, rng):
    base = _sigmoid(TEMP * _logit(pstar) + OFF)
    kink = KINK * np.clip((0.35 - pstar) / 0.35, 0.0, 1.0)
    q = base + kink + np.where(in_b, 0.5 * DELTA, -0.5 * DELTA)
    q = q + rng.normal(0.0, NOISE, size=len(pstar))
    return np.clip(q, 0.01, 0.99)


def _fit_platt(q, y):
    """Logistic map on logit(q), Newton steps, ridge to identity."""
    z = _logit(q)
    th = np.array([1.0, 0.0])
    x = np.stack([z, np.ones_like(z)], axis=1)
    for _ in range(50):
        p = _sigmoid(th[0] * z + th[1])
        r = p - y
        w = p * (1.0 - p) + 1e-6
        g = x.T @ r / len(z) + 0.02 * (th - np.array([1.0, 0.0]))
        h = (x * w[:, None]).T @ x / len(z) + 0.02 * np.eye(2)
        step = np.linalg.solve(h, g)
        th = th - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(th[0]), float(th[1])


def _apply_platt(q, a, b):
    return _sigmoid(a * _logit(q) + b)


def _ece(pred, y, edges):
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)
    tot = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() > 0:
            tot += float(m.mean()) * abs(float(pred[m].mean())
                                         - float(y[m].mean()))
    return tot


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)

    # Calibration: exactly N_CAL labelled in-distribution items.
    p_cal = rng.beta(*DIST_IN, N_CAL)
    y_cal = (rng.random(N_CAL) < p_cal).astype(float)
    in_b_cal = rng.random(N_CAL) < 0.5
    q_cal_raw = _judge(p_cal, in_b_cal, rng)
    q_cal_deb = 0.5 * (_judge(p_cal, np.zeros(N_CAL, bool), rng)
                       + _judge(p_cal, np.ones(N_CAL, bool), rng))
    a_raw, b_raw = _fit_platt(q_cal_raw, y_cal)
    a_deb, b_deb = _fit_platt(q_cal_deb, y_cal)

    # Fixed ECE bin edges: raw-judge quantiles on a separate
    # in-distribution reference pool, fit BEFORE any shift.
    p_ref = rng.beta(*DIST_IN, N_REF)
    q_ref = _judge(p_ref, rng.random(N_REF) < 0.5, rng)
    edges = np.quantile(q_ref, np.linspace(0.0, 1.0, N_BINS + 1))
    edges[0], edges[-1] = 0.0, 1.0

    out = {}
    for name, dist in (("in", DIST_IN), ("shift", DIST_SHIFT),
                       ("fail", DIST_FAIL)):
        p = rng.beta(*dist, N_EVAL)
        y = (rng.random(N_EVAL) < p).astype(float)
        in_b = rng.random(N_EVAL) < 0.5
        q_raw = _judge(p, in_b, rng)
        q_a = _judge(p, np.zeros(N_EVAL, bool), rng)
        q_b = _judge(p, np.ones(N_EVAL, bool), rng)
        q_deb = 0.5 * (q_a + q_b)
        arms = dict(
            raw=q_raw,
            deb=q_deb,
            cal=_apply_platt(q_raw, a_raw, b_raw),
            full=_apply_platt(q_deb, a_deb, b_deb),
        )
        out[name] = dict(
            ece={k: _ece(v, y, edges) for k, v in arms.items()},
            bias={k: float(v.mean() - y.mean())
                  for k, v in arms.items()},
            flip_raw=float(np.mean((q_a > 0.5) != (q_b > 0.5))),
            flip_cal=float(np.mean(
                (_apply_platt(q_a, a_raw, b_raw) > 0.5)
                != (_apply_platt(q_b, a_raw, b_raw) > 0.5))),
            flip_full=0.0,  # decision uses order-averaged q
            gap_raw=float(np.mean(np.abs(q_b - q_a))),
            emp=float(y.mean()),
        )
    return out


def test_debiased_calibrated_judge_halves_ece():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        for dist in ("in", "shift"):
            e = r[dist]["ece"]
            # Trap guard: the raw judge must be genuinely
            # miscalibrated (paper: 0.141) on both pools.
            assert e["raw"] > 0.075, (
                f"seed {seed} {dist}: raw ECE {e['raw']:.3f} -- "
                f"the planted distortion did not miscalibrate")
            # The paper's recipe roughly halves ECE (paper ratio
            # 0.52); debiasing + calibration beats calibration
            # alone and debiasing alone.
            assert e["full"] < 0.65 * e["raw"], (
                f"seed {seed} {dist}: full ECE {e['full']:.3f} vs "
                f"raw {e['raw']:.3f}")
            assert e["full"] < 0.07, (
                f"seed {seed} {dist}: full ECE {e['full']:.3f} -- "
                f"the recipe should land at the paper's level "
                f"(0.074) on both pools")
            assert e["full"] < e["deb"], (
                f"seed {seed} {dist}: full ECE {e['full']:.3f} vs "
                f"deb-only {e['deb']:.3f}")


def test_debiasing_removes_order_effect_calibration_leaves():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)["in"]
        # The raw judge's decisions flip with presentation order,
        # and Platt calibration alone does NOT fix that (it is a
        # monotone remap of a single-order score).
        assert r["flip_raw"] > 0.08, (
            f"seed {seed}: raw flip rate {r['flip_raw']:.3f}")
        assert r["flip_cal"] > 0.08, (
            f"seed {seed}: cal-only flip rate {r['flip_cal']:.3f}")
        # The debiased decision is a function of the order-averaged
        # probability, so it cannot flip -- 0 by construction; the
        # content is the contrast with the two arms above.
        assert r["flip_full"] == 0.0
        # Per-item order sensitivity: the same item's probability
        # moves by ~DELTA when its correct option swaps position.
        # (The debiased score is order-averaged, so it has no
        # per-order scores left to gap -- gap 0 by construction,
        # like the flip rate; the content is the raw arm's gap.)
        assert r["gap_raw"] > 0.10, (
            f"seed {seed}: raw order gap {r['gap_raw']:.3f}")


def test_calibrated_judge_overconfident_where_base_mostly_fails():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        over_fail = r["fail"]["bias"]["full"]
        over_in = r["in"]["bias"]["full"]
        # Boundary (paper): on the mostly-fail slice the calibrated
        # judge is materially overconfident, far more than on the
        # mixed slice it was calibrated on -- the Platt map cannot
        # remove the localised low-p* distortion.
        assert r["fail"]["emp"] < 0.2, (
            f"seed {seed}: fail-slice rate {r['fail']['emp']:.3f}")
        assert over_fail > 0.07, (
            f"seed {seed}: fail overconfidence {over_fail:.3f}")
        assert over_fail > over_in + 0.05, (
            f"seed {seed}: fail overconfidence {over_fail:.3f} vs "
            f"mixed {over_in:.3f}")
