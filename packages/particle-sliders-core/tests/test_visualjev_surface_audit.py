"""CPU gate for the surface-manipulation claims of Zeng
et al., "VisualJevBench: A Benchmark and Black-Box Attacks
for Visual Typed Decision Models" (arXiv:2611.25435): a
typed decision model (Visual Jev, jev-1.14.0) can be
manipulated through substance-preserving changes to the
input surface, using only typed-output (decision) feedback.
Image-specific attacks flip decisions at up to 69.6% ASR;
a single UNIVERSAL manipulation, optimized once, transfers
to unseen images and tasks at 33.15% average ASR; geometric
transformations transfer cross-scenario at 40.09%.
Provenance: cycle 9 of the Particle Sliders paper watch.

Adaptation (modality is visual, mechanism is ours): our
curation stage reads text candidates, not images, but the
mechanism is modality-independent -- a typed scorer's
selections can be moved by surface features of the
candidate that do not change its substance, and one shared
surface pattern can move the WHOLE pool at once. The
universal case is the dangerous one for curation: it biases
selection systematically, not noisily. This test is the
external-manipulation counterpart to #219's
internal-consistency audit (bundling flips): before
trusting a pool, measure selection stability under
substance-preserving surface perturbations, per-candidate
and pool-wide.

Toy operationalization: candidates carry a true quality
feature q (the substance) and a surface feature s that does
not change true quality. In every pool s is generated
correlated with q (s = rho * q + sqrt(1 - rho^2) * eps,
rho = 0.80) -- the realistic precondition: in real pools
surface and quality DO correlate (polished candidates are
often better). The typed scorer is a ridge regression of
true quality on (q, s), fitted on a 4000-candidate training
pool; decisions are the typed accept/reject readout
(score > 0). Ridge shrinkage spreads weight across the
correlated pair, so the fitted scorer carries a spurious
surface weight b > 0 even though s is causally irrelevant
to quality given q. Two modeling choices are load-bearing
and stated plainly: (1) the fit is regularized -- an
unregularized OLS fit on (q, s) against q would drive b to
~0, and it is precisely the regularized / finite-capacity
fits used in practice that convert correlation into
spurious weight; (2) the attacker's surface budget is
bounded, |s| <= 2 (two natural standard deviations) -- an
unbounded "surface" change would stop being surface.

Arms (attacks are black-box: they query only the typed
decision, never the score; the search is a 25-point grid
over the allowed surface range):
  (a) clean selections (baseline decisions);
  (b) per-item optimized surface flip: for each candidate,
      search the grid for a surface value that flips its
      decision, q held fixed (substance-preserving);
      ASR_specific = fraction of decisions flippable;
  (c) universal: ONE surface value s*, chosen by grid
      search on a separate 1000-candidate dev pool to
      maximize flips there, then applied unchanged to all
      2000 candidates of an unseen eval pool (transfer to
      unseen candidates) and to a cross-scenario pool
      (quality distribution shifted / rescaled, same
      surface-generation process);
  (d) randomization control: each candidate's s redrawn
      from its natural marginal -- the no-attack
      surface-change rate;
  (e) surface-blind control: a scorer refit on q alone;
      the same surface changes must flip ~nothing.

Measured on pop-os (particlegan 0.10.1), 3 seeds -- fitted
weights first (a on q, b on s), then flip rates:
  weights:    a 0.465-0.472, b 0.249-0.253
  specific:   0.697 / 0.714 / 0.722
  universal (dev-chosen s* = +2.00 every seed):
              dev 0.368 / 0.395 / 0.358,
              eval 0.352 / 0.351 / 0.359,
              cross-scenario 0.343 / 0.352 / 0.352
  randomize:  0.164 / 0.162 / 0.153
  blind:      0.000 on every seed

The paper's structure reproduces: per-item attacks are the
strongest (toy 0.70-0.72 at the +-2 surface budget; paper
up to 69.6%), one shared surface value optimized once
still flips about a third of an unseen pool's selections
(toy 0.35; paper 33.15%) and transfers to a new scenario
essentially undiminished (toy 0.34-0.35), both far above
the no-attack randomization rate (0.15-0.16), and a scorer
that cannot see the surface is exactly immune (0 flips).
Assertions below encode exactly these demonstrated
margins, no more.

What is NOT asserted: Jev-1.14's exact ASR numbers (the
toy scorer is a fitted stand-in; the claim tested is the
transfer structure -- specific > universal >> randomization
-- not any model's absolute vulnerability); anything about
image manipulations themselves; and robustness of an
UNREGULARIZED scorer (see the load-bearing choices above).
Pure numpy -- no GAN machinery, because the claim is about
selection stability under surface change, not training
dynamics. Skips cleanly when the installed particlegan does
not provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test in
CI).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
    # (any exception type: an ImportError from a failed inner
    # import, or a heavier raise -- both must skip, not error)
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


RHO = 0.80           # surface-quality correlation in every pool
LAM = 0.7            # ridge strength (per-sample scale)
N_TRAIN = 4000       # scorer fitting pool
N_DEV = 1000         # universal-attack optimization pool
N_EVAL = 2000        # unseen eval pool (and cross-scenario pool)
S_MAX = 2.0          # attacker surface budget: |s| <= 2 (2 sigma)
GRID = np.linspace(-S_MAX, S_MAX, 25)  # black-box search grid
TAU = 0.0            # typed decision threshold on the score


def _gen(rng, n):
    """One pool: quality q and surface s correlated at RHO."""
    q = rng.standard_normal(n)
    s = RHO * q + np.sqrt(1.0 - RHO ** 2) * rng.standard_normal(n)
    return q, s


def _fit(seed):
    """Ridge scorer on (q, s) plus the surface-blind refit.

    Returns (a, b, a_blind): score = a * q + b * s for the
    full scorer; the blind scorer's score is a_blind * q.
    """
    rng = np.random.default_rng(1000 + seed)
    q, s = _gen(rng, N_TRAIN)
    y = q + 0.1 * rng.standard_normal(N_TRAIN)
    x = np.stack([q, s], 1)
    gram = x.T @ x / N_TRAIN + LAM * np.eye(2)
    a, b = np.linalg.solve(gram, x.T @ y / N_TRAIN)
    a_blind = (q @ y / N_TRAIN) / (q @ q / N_TRAIN + LAM)
    return float(a), float(b), float(a_blind)


def run_seed(seed):
    """All arms for one seed; attacks see decisions only."""
    a, b, a_blind = _fit(seed)

    def decide(q, s, aa, bb):
        return (aa * q + bb * s) > TAU

    rng = np.random.default_rng(2000 + seed)
    q_dev, s_dev = _gen(rng, N_DEV)
    q_ev, s_ev = _gen(rng, N_EVAL)
    clean_dev = decide(q_dev, s_dev, a, b)
    clean_ev = decide(q_ev, s_ev, a, b)

    # (c) universal: one surface value for the whole pool,
    # chosen on the dev pool by decision feedback only.
    s_star, dev_best = None, -1.0
    for cand in GRID:
        rate = float(np.mean(
            decide(q_dev, np.full(N_DEV, cand), a, b) != clean_dev))
        if rate > dev_best:
            dev_best, s_star = rate, float(cand)
    uni_eval = float(np.mean(
        decide(q_ev, np.full(N_EVAL, s_star), a, b) != clean_ev))

    # Cross-scenario transfer: quality distribution shifted
    # and rescaled; surface generated by the same process
    # relative to the scenario's standardized quality.
    rng_x = np.random.default_rng(4000 + seed)
    q_x = 0.4 + 0.8 * rng_x.standard_normal(N_EVAL)
    s_x = (RHO * (q_x - 0.4) / 0.8
           + np.sqrt(1.0 - RHO ** 2) * rng_x.standard_normal(N_EVAL))
    uni_xfer = float(np.mean(
        decide(q_x, np.full(N_EVAL, s_star), a, b)
        != decide(q_x, s_x, a, b)))

    # (b) per-item optimized surface flip (decision feedback
    # only): a candidate counts as flipped if ANY grid value
    # changes its decision, q held fixed throughout.
    flipped = np.zeros(N_EVAL, dtype=bool)
    for cand in GRID:
        flipped |= decide(q_ev, np.full(N_EVAL, cand), a, b) != clean_ev
    asr_specific = float(np.mean(flipped))

    # (d) randomization control: surface redrawn per
    # candidate from its natural marginal, no optimization.
    rng_r = np.random.default_rng(3000 + seed)
    rand_rate = float(np.mean(
        decide(q_ev, rng_r.standard_normal(N_EVAL), a, b) != clean_ev))

    # (e) surface-blind control: the blind scorer's decisions
    # under the strongest surface changes (any grid value).
    clean_blind = decide(q_ev, s_ev, a_blind, 0.0)
    blind_rate = 0.0
    for cand in GRID:
        blind_rate = max(blind_rate, float(np.mean(
            decide(q_ev, np.full(N_EVAL, cand), a_blind, 0.0)
            != clean_blind)))

    return dict(a=a, b=b, a_blind=a_blind, s_star=s_star,
                uni_dev=dev_best, uni_eval=uni_eval,
                uni_xfer=uni_xfer, specific=asr_specific,
                rand=rand_rate, blind=blind_rate)


def test_scorer_surface_weight_and_specific_attack():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The regularized fit on the correlated pool picks up
        # a substantial spurious surface weight, while true
        # quality remains the dominant term (demonstrated
        # a 0.465-0.472, b 0.249-0.253).
        assert r["b"] > 0.15, (
            f"seed {seed}: fitted surface weight "
            f"{r['b']:+.3f}")
        assert r["a"] > r["b"], (
            f"seed {seed}: quality weight {r['a']:+.3f} not "
            f"above surface weight {r['b']:+.3f}")
        # Per-item black-box surface search flips most
        # decisions (demonstrated 0.697-0.722; the paper's
        # image-specific attacks reach up to 69.6%).
        assert r["specific"] >= 0.60, (
            f"seed {seed}: per-item ASR {r['specific']:.3f}")
        assert r["specific"] > r["uni_eval"] + 0.25, (
            f"seed {seed}: per-item ASR {r['specific']:.3f} "
            f"vs universal {r['uni_eval']:.3f}")


def test_universal_manipulation_transfers():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # One surface value, optimized once on the dev pool,
        # flips about a third of an UNSEEN pool's selections
        # (demonstrated eval 0.351-0.359, dev 0.358-0.395;
        # the paper's universal ASR is 33.15%).
        assert r["uni_eval"] >= 0.28, (
            f"seed {seed}: universal ASR on unseen pool "
            f"{r['uni_eval']:.3f}")
        assert abs(r["uni_dev"] - r["uni_eval"]) <= 0.08, (
            f"seed {seed}: universal dev {r['uni_dev']:.3f} "
            f"vs eval {r['uni_eval']:.3f} -- no transfer")
        # The same single value transfers cross-scenario
        # essentially undiminished (demonstrated
        # 0.343-0.352, within 0.01 of the eval rate).
        assert r["uni_xfer"] >= 0.28, (
            f"seed {seed}: universal cross-scenario ASR "
            f"{r['uni_xfer']:.3f}")
        assert r["uni_xfer"] >= r["uni_eval"] - 0.05, (
            f"seed {seed}: cross-scenario {r['uni_xfer']:.3f} "
            f"vs eval {r['uni_eval']:.3f}")


def test_controls_bound_the_effect():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Unoptimized surface change flips only the
        # no-attack base rate (demonstrated 0.153-0.164),
        # far below the optimized universal manipulation
        # (demonstrated gap 0.187-0.206).
        assert r["rand"] <= 0.22, (
            f"seed {seed}: randomization flip rate "
            f"{r['rand']:.3f}")
        assert r["uni_eval"] > r["rand"] + 0.12, (
            f"seed {seed}: universal {r['uni_eval']:.3f} vs "
            f"randomization {r['rand']:.3f}")
        # A scorer that cannot see the surface is immune to
        # the same changes (demonstrated exactly 0 flips).
        assert r["blind"] <= 0.005, (
            f"seed {seed}: surface-blind flip rate "
            f"{r['blind']:.3f}")
