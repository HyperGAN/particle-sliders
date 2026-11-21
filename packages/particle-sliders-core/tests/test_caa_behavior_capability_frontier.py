"""CPU gate for the behavior-capability frontier claim of
contrastive activation addition (CAA) steering (arXiv:2611.06001):
choosing the steering strength on a measured behavior-capability
frontier under an explicit capability-loss budget finds operating
points a fixed/default strength misses; the fraction of operating
points that jointly survive (behavior gained AND capability within
budget) rises with the budget; and -- the paper's warning -- an
operating point can satisfy every protected capability probe within
budget while still degrading an UNTESTED capability, because passing
the protected probes does not certify the untested ones.

Paper: "How Far Can You Steer? Behavior-Capability Frontiers in
Contrastive Activation Addition" (Raj). CAA steering is evaluated
as a frontier: select interventions maximizing behavioral gain
subject to an explicit bound on capability degradation. Across 4
instruction-tuned models and 7 behaviors, at capability-loss budgets
of 1%/3%/5%, validation-selected behavioral effects generalize to
held-out data (rho = 0.945); joint behavior-capability survival
among nonzero operating points rises from 37.5% at the 1% budget to
72.7% at 3%, with limited additional improvement at 5%. Controls
show the effects depend on learned directions, not generic
perturbations. The warning: interventions satisfying protected
benchmarks can still degrade untested capabilities or transfer
unevenly to free-form generation.

Toy operationalization: a scripted activation geometry in R^24 with
a KNOWN ground truth, so every quantity the paper measures on real
models is computed exactly here. Per seed, a random orthonormal
basis supplies one behavior axis b, three PROTECTED capability axes
P (the probe suite used during selection), and one UNTESTED
capability axis u (never measured during selection). The learned
CAA direction is

    v = normalize(b + gamma_p * (protected combination)
                    + gamma_u * u),     gamma_p ~ 0.25-0.35,
                                        gamma_u ~ 0.55-0.75,

i.e. the steering vector mostly does the behavior, leaks a little
into the protected axes, and leaks MORE into the untested axis --
the realistic case the warning is about: contrastive means are
estimated against the behaviors/probes at hand, and nothing in that
estimation penalizes a side component along a capability nobody
probed. Steering by alpha * v gives, per behavior j with
responsiveness k_j:

    behavior gain   G_j(a) = 1 - exp(-k_j * a * (v . b))
    protected loss  L_p(a) = A_p * a^2 * sum_j (v . p_j)^2
    untested loss   L_u(a) = A_u * a^2 * (v . u)^2

with A_u > A_p (the untested capability is at least as sensitive
per unit projection). Losses are quadratic in a (a perturbation of
size a along an axis degrades a capability quadratically to first
order); gain saturates in a. Six behaviors per seed (k_j spread)
mirror the paper's multi-behavior sweep. Each behavior also has a
held-out responsiveness k_j' = k_j * (1 +/- 5%), so selection is
done on validation quantities and the headline comparison is
re-checked on held-out quantities, as in the paper.

Arms:
  (a) frontier selection: for each budget B in {1%, 3%, 5%}, pick
      the grid strength maximizing validation behavior gain subject
      to L_p(a) <= B (the paper's selection rule);
  (b) fixed default strength: a = 0.6 for every budget, the
      set-it-once default a practitioner ships without a frontier;
  (c) random-direction control: the mean behavior gain of 8 random
      unit directions at the frontier-selected strength -- the
      paper's control that effects come from the LEARNED direction,
      not from perturbing activations at all.

Survival at budget B is the paper's joint statistic: the fraction
of (behavior, nonzero grid strength) operating points with
G_j(a) >= 0.5 AND L_p(a) <= B.

Iteration log (the loop that produced this gate):
  1. Analytic geometry as above, first pass: all three claims
     separate cleanly on every seed. Frontier (3%) gain 0.795-0.895
     vs fixed-default 0.615-0.665 (margin +0.18 to +0.23 at
     protected loss <= 3%); survival strictly increasing per seed
     (1%: 0.069-0.083 -> 3%: 0.250-0.333 -> 5%: 0.333-0.417);
     untested loss at the 3%-selected point 0.121-0.214, i.e.
     4.0-7.1x the budget it passed on protected probes. One
     refinement followed: the random control is scored on the MEAN
     of 8 random directions -- in prototyping a single lucky random
     direction reached gain 0.66 on one seed (its behavior-axis
     projection happened to be large), while the mean stays
     <= 0.16; the control claim is about generic perturbations,
     so the mean is the honest statistic.
  Assertions below encode roughly half the thinnest observed
  margins, not the observed values themselves.

Measured on pop-os (particlegan 0.10.1), 4 seeds: frontier (3%
budget) behavior gain 0.795-0.895 (held-out 0.800-0.893) at
protected loss 0.020-0.030, vs fixed-default gain 0.615-0.665
(held-out 0.620-0.663) and random-direction mean gain -0.398 to
+0.156; survival 0.069-0.083 (1%) -> 0.250-0.333 (3%) ->
0.333-0.417 (5%); untested loss at the selected point
0.121-0.214 against the 0.03 protected budget it satisfied.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe.
"""
import numpy as np
import pytest

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 24
N_BEHAVIORS = 6
N_RANDOM = 8
BUDGETS = (0.01, 0.03, 0.05)
ALPHA_GRID = np.arange(0.25, 3.01, 0.25)   # nonzero operating points
FIXED_ALPHA = 0.6                          # the fixed/default strength
GAIN_TAU = 0.5                             # "behavior gained" threshold


def make_world(seed):
    """One seed's geometry: axes, learned direction, responsiveness.

    Everything the arms share within a seed is drawn here, so the
    comparison is paired: same direction, same probes, same
    behaviors -- only the strength-selection rule differs."""
    rng = np.random.default_rng(70000 + seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    b = Q[:, 0]
    prot = Q[:, 1:4]                       # 3 protected capability axes
    unt = Q[:, 4]                          # 1 untested capability axis
    gamma_p = float(rng.uniform(0.25, 0.35))
    gamma_u = float(rng.uniform(0.55, 0.75))
    pw = rng.standard_normal(3)
    pw /= np.linalg.norm(pw)
    v = b + gamma_p * (prot @ pw) + gamma_u * unt
    v /= np.linalg.norm(v)
    ks = rng.uniform(1.6, 2.8, N_BEHAVIORS)
    ks_held = ks * rng.uniform(0.95, 1.05, N_BEHAVIORS)
    return {
        "b": b, "prot": prot, "unt": unt, "v": v, "ks": ks,
        "ks_held": ks_held,
        "A_p": float(rng.uniform(0.30, 0.40)),
        "A_u": float(rng.uniform(0.40, 0.50)),
    }


def gain(world, alpha, k, vec=None):
    """Behavior gain of steering alpha * vec for responsiveness k."""
    vec = world["v"] if vec is None else vec
    proj = float(vec @ world["b"])
    return float(1.0 - np.exp(-k * alpha * proj))


def protected_loss(world, alpha, vec=None):
    vec = world["v"] if vec is None else vec
    proj2 = float(np.sum((world["prot"].T @ vec) ** 2))
    return world["A_p"] * alpha ** 2 * proj2


def untested_loss(world, alpha, vec=None):
    vec = world["v"] if vec is None else vec
    return world["A_u"] * alpha ** 2 * float(vec @ world["unt"]) ** 2


def frontier_select(world, budget):
    """The paper's selection rule: maximize mean validation behavior
    gain over the grid subject to protected loss <= budget."""
    best_alpha, best_gain = 0.0, -np.inf
    for alpha in ALPHA_GRID:
        if protected_loss(world, alpha) <= budget:
            g = float(np.mean([gain(world, alpha, k)
                               for k in world["ks"]]))
            if g > best_gain:
                best_alpha, best_gain = float(alpha), g
    return best_alpha, best_gain


def survival(world, budget):
    """Fraction of (behavior, nonzero strength) operating points
    with behavior gained AND protected loss within budget."""
    ok = total = 0
    for k in world["ks"]:
        for alpha in ALPHA_GRID:
            total += 1
            if gain(world, alpha, k) >= GAIN_TAU and \
                    protected_loss(world, alpha) <= budget:
                ok += 1
    return ok / total


def run_seed(seed):
    _empf_recipe()
    world = make_world(seed)
    out = {"world": world, "frontier": {}, "survival": {}}
    for budget in BUDGETS:
        alpha_star, g = frontier_select(world, budget)
        out["frontier"][budget] = {
            "alpha": alpha_star,
            "gain": g,
            "gain_held": float(np.mean(
                [gain(world, alpha_star, k) for k in world["ks_held"]])),
            "ploss": protected_loss(world, alpha_star),
            "uloss": untested_loss(world, alpha_star),
        }
        out["survival"][budget] = survival(world, budget)
    out["fixed"] = {
        "gain": float(np.mean([gain(world, FIXED_ALPHA, k)
                                for k in world["ks"]])),
        "gain_held": float(np.mean([gain(world, FIXED_ALPHA, k)
                                     for k in world["ks_held"]])),
        "ploss": protected_loss(world, FIXED_ALPHA),
        "uloss": untested_loss(world, FIXED_ALPHA),
    }
    # Random-direction control at the 3%-budget selected strength.
    alpha_star = out["frontier"][0.03]["alpha"]
    rng = np.random.default_rng(90000 + seed)
    rand_gains = []
    for _ in range(N_RANDOM):
        rv = rng.standard_normal(D)
        rv /= np.linalg.norm(rv)
        rand_gains.append(float(np.mean(
            [gain(world, alpha_star, k, vec=rv) for k in world["ks"]])))
    out["random_gain_mean"] = float(np.mean(rand_gains))
    return out


def test_caa_behavior_capability_frontier():
    for seed in range(4):
        r = run_seed(seed)
        fr, fx = r["frontier"], r["fixed"]
        sel = fr[0.03]

        # (1) Frontier selection under an explicit budget beats the
        # fixed default strength: strictly more behavior gain while
        # the selected point itself respects the budget the fixed
        # default never consulted. Demonstrated margin +0.18 to
        # +0.23 gain; asserted at +0.10. The advantage is re-checked
        # on held-out responsiveness (paper: validation-selected
        # effects generalize, rho = 0.945).
        assert sel["ploss"] <= 0.03
        assert sel["gain"] >= fx["gain"] + 0.10
        assert sel["gain_held"] >= fx["gain_held"] + 0.08
        # Frontier gain is monotone in the budget: a looser budget
        # never selects a worse operating point.
        assert fr[0.01]["gain"] <= fr[0.03]["gain"] <= fr[0.05]["gain"]

        # Control: the gain comes from the LEARNED direction, not
        # from perturbing activations at the same strength with a
        # generic direction (mean of 8 random directions).
        assert sel["gain"] >= r["random_gain_mean"] + 0.35

        # (2) Joint behavior-capability survival rises with the
        # budget (paper: 37.5% at 1% -> 72.7% at 3%). Demonstrated
        # per-seed rise of +0.25 to +0.33 from 1% to 5%, strictly
        # increasing at each step; asserted strict with a floor.
        s1, s3, s5 = (r["survival"][b] for b in BUDGETS)
        assert s1 < s3 < s5
        assert s5 - s1 >= 0.15

        # (3) THE WARNING: the 3%-selected point passes every
        # protected probe within budget (asserted above) while its
        # UNTESTED capability loss blows through that same budget
        # -- demonstrated at 4-7x the budget (0.12-0.22 absolute);
        # asserted at 2.5x. Passing the protected probes does not
        # certify the untested capability.
        assert sel["uloss"] >= 2.5 * 0.03
        assert sel["uloss"] >= sel["ploss"] + 0.07
