"""CPU toy for the ADAPT pool-size-aware abstention claim
(arXiv:2611.25754): retrieval-based routers typically select a
candidate even when no suitable service exists; ADAPT adds an
explicit abstention option, defines routing confidence as the
SEPARATION between the highest relevance score and the candidate
score distribution, and shows that the confidence distribution
changes SYSTEMATICALLY with candidate-pool size K -- so a FIXED
abstention threshold becomes increasingly miscalibrated as
services are added, while a K-aware threshold (adapted as the
pool grows, no classifier retraining) restores an effective
accuracy-cost trade-off.

Paper: "ADAPT: Adaptive Decision and Abstention with
Pool-Size-Aware Thresholds for Service Routing" (Srun, Adnan,
Shaukat & Kuhn). The abstract reports NO headline numbers --
stated plainly. The claim tested here is the geometric /
statistical one the abstract does make: fixed-threshold
miscalibration grows with K, and K-aware thresholding holds the
operating point.

Toy operationalization -- an open-set routing world. Each query
either has a true matching service in the pool or has NONE (the
abstention case). Candidate relevance scores: the matched
candidate is drawn from a signal distribution N(d, 1), d = 3;
every non-matched candidate from the noise distribution N(0, 1).
Pool size K is swept over {5, 10, 20, 50, 100, 200}. Routing
confidence is the paper's separation definition,

    C = top1 score - mean of the remaining K-1 scores.

(The top1 - top2 margin is computed from the same score matrices
as a reported variant; the paper's definition is primary.)

Arms -- the SCORER (score distributions) is identical for every
arm; only the abstention threshold moves:
  (a) FIXED: one threshold tau_fixed calibrated at K0 = 5 to a
      target false-accept rate (FAR) of alpha = 0.05 on
      no-match calibration queries, then used at every K.
  (b) K-AWARE: a threshold schedule tau(K), the (1 - alpha)
      quantile of the no-match confidence distribution estimated
      on held-out calibration queries AT THAT K. No scorer
      retraining -- the schedule is the whole mechanism.
  (c) Controls: a SHUFFLED schedule (the tau(K) values assigned
      to the wrong pool sizes, reversed) -- any varying threshold
      is not enough, the schedule must track K in the right
      direction; and an ORACLE per-K threshold (the quantile
      computed on the evaluation queries themselves) as an
      upper-bound reference the calibration-based schedule
      should nearly match.

Metrics per K, >= 3 seeds: FAR on no-match queries (fraction
wrongly accepted instead of abstained) and TAR on matched
queries (fraction correctly accepted) for each arm.

Measured on pop-os (particlegan 0.10.1), 3 seeds:
  tau(K) rises monotonically 2.575-2.611 at K = 5 to
  3.479-3.484 at K = 200 (tau_fixed is the K = 5 value).
  FAR_fixed: 0.050-0.055 at K = 5, 0.213-0.240 at K = 50,
  exploding to 0.604-0.635 at K = 200 (the miscalibration).
  FAR_aware: 0.046-0.057 at EVERY K (the held operating
  point), within 0.007 of the oracle arm's 0.050.
  FAR_shuffled: 0.004-0.005 at K = 5 (over-abstains) and
  0.604-0.635 at K = 200 (inherits the fixed arm's failure)
  -- the control fails at both ends.
  TAR_aware falls honestly with K (0.643-0.656 at K = 5 to
  0.344-0.346 at K = 200): at a held FAR, larger pools are
  genuinely harder. TAR_fixed RISES (to 0.864-0.877 at
  K = 200) -- but only by accepting the no-match flood; that
  apparent gain IS the miscalibration, not performance.
  Margin variant (top1 - top2): the K-dependence REVERSES --
  fixed-threshold FAR collapses 0.046-0.054 -> 0.002-0.003
  and TAR collapses 0.571-0.590 -> 0.070-0.082
  (over-abstention), while the K-aware schedule holds margin
  FAR at 0.046-0.060 at every K. Fixed thresholds
  miscalibrate under either confidence definition; the
  direction of the error depends on the definition.

Assertions below encode exactly these margins, no more.

What is NOT asserted: any accuracy-cost number against
generative LLM routing or on the paper's specialist datasets
(the abstract reports none and a toy cannot express them). The
claim tested is the threshold geometry: the no-match confidence
distribution drifts with K (an extreme-value effect -- the max
of K noise draws grows ~ sqrt(2 log K) while the matched
separation stays ~ d), a fixed threshold cannot hold an
operating point under that drift, and a per-K calibrated
schedule can.

Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family).
Pure CPU toy -- no GAN machinery, because the claim is about
the statistics of a selection threshold, not training dynamics.
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except ImportError:  # particlegan present but its import raises
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


POOL_SIZES = (5, 10, 20, 50, 100, 200)
K0 = 5              # pool size the fixed threshold is calibrated at
ALPHA = 0.05        # target false-accept rate on no-match queries
D_SIGNAL = 3.0      # matched candidate's mean relevance
N_CAL = 20000       # held-out calibration queries per (seed, K)
N_EVAL = 4000       # evaluation queries per (seed, K, match type)
SEEDS = (0, 1, 2)


def _scores(rng, K, n, matched):
    """Relevance-score matrix (n, K). Column 0 is the matched
    candidate when `matched`; every other entry is noise."""
    scores = rng.normal(0.0, 1.0, (n, K))
    if matched:
        scores[:, 0] = rng.normal(D_SIGNAL, 1.0, n)
    return scores


def _confidences(scores):
    """(separation, margin) per query from one score matrix.

    separation = top1 - mean(rest)   -- the paper's definition
    margin     = top1 - top2         -- reported variant
    """
    top1 = scores.max(axis=1)
    mean_rest = (scores.sum(axis=1) - top1) / (scores.shape[1] - 1)
    ordered = np.sort(scores, axis=1)
    return top1 - mean_rest, ordered[:, -1] - ordered[:, -2]


def run_seed(seed):
    """All arms at every K for one seed. Calibration and
    evaluation use disjoint rng streams (held-out calibration);
    the oracle arm alone reads the evaluation set."""
    out = {}
    # Pass 1: calibration thresholds per K (no-match queries only).
    tau_sep, tau_marg = {}, {}
    for K in POOL_SIZES:
        rng = np.random.default_rng(100_000 + 1_000 * seed + K)
        sep, marg = _confidences(_scores(rng, K, N_CAL, matched=False))
        tau_sep[K] = float(np.quantile(sep, 1.0 - ALPHA))
        tau_marg[K] = float(np.quantile(marg, 1.0 - ALPHA))
    tau_fixed = tau_sep[K0]
    tau_marg_fixed = tau_marg[K0]
    # Shuffled control: the aware schedule's values, reversed --
    # tau_shuffled(K) is the threshold calibrated for the mirror K.
    mirror = {K: POOL_SIZES[-1 - i] for i, K in enumerate(POOL_SIZES)}
    # Pass 2: evaluation.
    for K in POOL_SIZES:
        rng = np.random.default_rng(200_000 + 1_000 * seed + K)
        sep_nm, marg_nm = _confidences(_scores(rng, K, N_EVAL, False))
        sep_m, marg_m = _confidences(_scores(rng, K, N_EVAL, True))
        oracle_sep = float(np.quantile(sep_nm, 1.0 - ALPHA))
        oracle_marg = float(np.quantile(marg_nm, 1.0 - ALPHA))
        out[K] = dict(
            tau_aware=tau_sep[K],
            far_fixed=float(np.mean(sep_nm >= tau_fixed)),
            far_aware=float(np.mean(sep_nm >= tau_sep[K])),
            far_oracle=float(np.mean(sep_nm >= oracle_sep)),
            far_shuffled=float(np.mean(sep_nm >= tau_sep[mirror[K]])),
            tar_fixed=float(np.mean(sep_m >= tau_fixed)),
            tar_aware=float(np.mean(sep_m >= tau_sep[K])),
            marg_far_fixed=float(np.mean(marg_nm >= tau_marg_fixed)),
            marg_far_aware=float(np.mean(marg_nm >= tau_marg[K])),
            marg_tar_fixed=float(np.mean(marg_m >= tau_marg_fixed)),
            marg_tar_aware=float(np.mean(marg_m >= tau_marg[K])),
        )
    return out


def test_fixed_threshold_miscalibrates_with_pool_size():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # Calibrated as intended at K0: FAR sits at the target.
        assert abs(r[K0]["far_fixed"] - ALPHA) < 0.025, (
            f"seed {seed}: FAR_fixed(K0) {r[K0]['far_fixed']:.3f}")
        # The aware schedule itself must move UP with K (the
        # no-match separation distribution drifts up).
        taus = [r[K]["tau_aware"] for K in POOL_SIZES]
        assert all(b > a for a, b in zip(taus, taus[1:])), (
            f"seed {seed}: tau(K) not increasing: "
            f"{[round(t, 3) for t in taus]}")
        # The fixed arm's false-accept rate grows materially and
        # monotonically-ish with K: the miscalibration.
        assert r[50]["far_fixed"] > r[K0]["far_fixed"] + 0.08, (
            f"seed {seed}: FAR_fixed(50) {r[50]['far_fixed']:.3f} "
            f"vs FAR_fixed(5) {r[K0]['far_fixed']:.3f}")
        assert r[200]["far_fixed"] > 0.40, (
            f"seed {seed}: FAR_fixed(200) {r[200]['far_fixed']:.3f}")
        assert r[200]["far_fixed"] > 5.0 * r[K0]["far_fixed"], (
            f"seed {seed}: FAR_fixed growth "
            f"{r[K0]['far_fixed']:.3f} -> {r[200]['far_fixed']:.3f}")


def test_k_aware_threshold_holds_operating_point():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        for K in POOL_SIZES:
            # Held flat at the target at EVERY pool size.
            assert abs(r[K]["far_aware"] - ALPHA) < 0.02, (
                f"seed {seed} K={K}: FAR_aware {r[K]['far_aware']:.3f}")
            # Calibration-based schedule ~ oracle upper bound.
            assert abs(r[K]["far_aware"] - r[K]["far_oracle"]) < 0.02, (
                f"seed {seed} K={K}: FAR_aware {r[K]['far_aware']:.3f} "
                f"vs oracle {r[K]['far_oracle']:.3f}")
        # The shuffled control fails at both ends: a varying
        # threshold that does not track K is not the mechanism.
        assert r[K0]["far_shuffled"] < 0.02, (
            f"seed {seed}: FAR_shuffled(5) {r[K0]['far_shuffled']:.3f} "
            "-- should over-abstain under the K=200 threshold")
        assert r[200]["far_shuffled"] > 0.40, (
            f"seed {seed}: FAR_shuffled(200) "
            f"{r[200]['far_shuffled']:.3f} -- inherits the K0 "
            "threshold's failure")


def test_matched_cost_is_bounded_and_the_fixed_gain_is_fake():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # At K0 the arms coincide: sane matched coverage.
        assert 0.55 < r[K0]["tar_aware"] < 0.75, (
            f"seed {seed}: TAR_aware(5) {r[K0]['tar_aware']:.3f}")
        # Holding the operating point costs matched coverage as
        # pools grow -- honestly, but bounded: the router still
        # routes a useful fraction at K = 200.
        assert r[200]["tar_aware"] > 0.25, (
            f"seed {seed}: TAR_aware(200) {r[200]['tar_aware']:.3f}")
        assert r[200]["tar_aware"] < r[K0]["tar_aware"], (
            f"seed {seed}: TAR_aware should fall with K: "
            f"{r[K0]['tar_aware']:.3f} -> {r[200]['tar_aware']:.3f}")
        # The fixed arm's apparent matched-coverage GAIN at large
        # K is bought entirely with the no-match flood: it beats
        # the aware arm by a wide margin precisely where its FAR
        # has exploded.
        assert r[200]["tar_fixed"] > r[200]["tar_aware"] + 0.30, (
            f"seed {seed}: TAR_fixed(200) {r[200]['tar_fixed']:.3f} "
            f"vs TAR_aware(200) {r[200]['tar_aware']:.3f}")


def test_margin_variant_miscalibrates_in_opposite_direction():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # Under top1 - top2 the extremes crowd as K grows, so the
        # SAME fixed-threshold policy now over-abstains: FAR and
        # TAR both collapse instead of exploding.
        assert abs(r[K0]["marg_far_fixed"] - ALPHA) < 0.025, (
            f"seed {seed}: margin FAR_fixed(5) "
            f"{r[K0]['marg_far_fixed']:.3f}")
        assert r[200]["marg_far_fixed"] < 0.015, (
            f"seed {seed}: margin FAR_fixed(200) "
            f"{r[200]['marg_far_fixed']:.3f}")
        assert r[200]["marg_tar_fixed"] < 0.15, (
            f"seed {seed}: margin TAR_fixed(200) "
            f"{r[200]['marg_tar_fixed']:.3f}")
        assert r[200]["marg_tar_fixed"] < r[K0]["marg_tar_fixed"] - 0.30, (
            f"seed {seed}: margin TAR_fixed collapse "
            f"{r[K0]['marg_tar_fixed']:.3f} -> "
            f"{r[200]['marg_tar_fixed']:.3f}")
        # K-aware thresholding holds the operating point under
        # this confidence definition too.
        for K in POOL_SIZES:
            assert abs(r[K]["marg_far_aware"] - ALPHA) < 0.02, (
                f"seed {seed} K={K}: margin FAR_aware "
                f"{r[K]['marg_far_aware']:.3f}")
