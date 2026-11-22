"""CPU gate for the update-consistency trait-recovery claim
of Egan, Gilligan-Lee, O'Riordan & Zhu, "Defence against
the dark arts of distillation: Unsupervised discovery of
subliminal learning" (arXiv:2610.13536). The paper's
estimator (STIFLE: Shared Trait Identification From Learned
Ensembles) rests on one asymmetry: a subliminally
transferred trait is expressed CONSISTENTLY across diverse
teacher-generated datasets, while task-specific components
vary -- so a trait direction can be recovered from the
parameter updates alone, with no trait labels, and
projecting that direction out of an update suppresses the
trait with almost no task-performance cost (validated by
the paper on Cloud et al.'s subliminal-learning
benchmarks, where the recovered traits are animal
preferences). Provenance: cycle 17 of the Particle Sliders
paper watch.

Toy operationalization: the smallest linear world in which
the asymmetry exists by construction. d = 48. K = 8 data
slices; slice k has an independent random task direction
t_k (norm 2.0); ALL slices share one trait direction tau
(unit) with loading beta = 1.2. Per-slice targets

    y = X . (t_k + beta * tau) + noise,

n = 240 rows/slice, noise sigma = 0.5, X standard normal.
The per-slice student is a ridge fit (lambda = 1) from the
zero base, so the per-slice induced shift is just the
fitted weight vector Delta_k. The estimator under test is
the cross-slice MEAN of {Delta_k}, normalized -- the
consistent component. (Centering before looking for a
shared component would remove exactly the thing being
estimated; a centered-PCA control arm below demonstrates
that null explicitly.)

Arms and controls, all paired per seed (trait and clean
arms of a seed share every draw; only beta differs):
  (a) recovery: cos(mean_k Delta_k, tau) vs the best
      single-slice cos, and vs a K-sweep over the first
      1 / 2 / 4 / 8 slices;
  (b) centered-PCA control: first principal component of
      the CENTERED slice shifts -- must NOT recover tau;
  (c) clean control: beta = 0 -- the estimator must find
      nothing (cos at chance, ~1/sqrt(48) = 0.14);
  (d) suppression: a SEPARATELY trained full-pool update
      Delta_full (one ridge fit over all slices pooled,
      fresh data and fresh noise -- never the mean vector
      itself, projecting which out of itself would be a
      tautology). Suppress by removing the recovered
      direction: Delta_full - (Delta_full . u) u with
      u the normalized cross-slice mean. Readouts: trait
      component Delta . tau before/after, and task cost as
      mean per-slice held-out MSE against TASK-ONLY targets
      (y = X . t_k, fresh X), before/after;
  (e) random-projection control: the same projection with
      a random unit direction in place of u -- the trait
      readout must survive it.

Measured on pop-os (particlegan 0.10.1), 5 seeds -- trait
arm first (recovery cos / best single-slice cos / K-sweep
1,2,4,8 / centered-PC1 |cos|), then suppression (trait
readout before -> after / after random projection; task
MSE before -> suppressed / random-projection), then the
clean arm (recovery cos; pooled trait readout before ->
after):
  seed 0: rec 0.890, best 0.626, K 0.39/0.71/0.77/0.89,
          PC 0.164; trait 1.13 -> +0.001 / rand 1.08;
          MSE 4.80 -> 4.06 / 4.71; clean rec -0.024,
          clean trait -0.065 -> -0.051
  seed 1: rec 0.855, best 0.615, K 0.47/0.66/0.68/0.86,
          PC 0.024; trait 1.10 -> +0.004 / rand 0.99;
          MSE 5.13 -> 4.19 / 5.02; clean rec -0.153,
          clean trait -0.100 -> +0.002
  seed 2: rec 0.829, best 0.588, K 0.57/0.68/0.77/0.83,
          PC 0.187; trait 1.09 -> +0.024 / rand 1.09;
          MSE 4.99 -> 4.32 / 4.97; clean rec -0.169,
          clean trait -0.106 -> +0.011
  seed 3: rec 0.891, best 0.675, K 0.55/0.67/0.85/0.89,
          PC 0.015; trait 1.33 -> +0.023 / rand 1.21;
          MSE 5.30 -> 4.19 / 5.09; clean rec +0.078,
          clean trait +0.135 -> +0.085
  seed 4: rec 0.874, best 0.685, K 0.43/0.55/0.73/0.87,
          PC 0.057; trait 1.22 -> +0.013 / rand 1.18;
          MSE 4.82 -> 3.99 / 4.79; clean rec -0.032,
          clean trait +0.022 -> +0.043

The arbitration, stated plainly: the recovery half of the
paper's claim reproduces exactly as the mechanism says it
should. The cross-slice mean recovers tau at cos
0.83-0.89 on every seed, beats the best single slice by
0.19-0.28, and improves monotonically in K on every seed
(the 5-seed mean sweep is 0.48 / 0.65 / 0.76 / 0.87 for
K = 1 / 2 / 4 / 8). The clean arm sits at chance
(|cos| <= 0.17, against the 0.14 chance level), and the
centered-PCA control is a null (|cos| <= 0.19): the trait
lives in the shared mean, not in cross-slice variance.
Suppression removes 98-100% of the pooled update's trait
readout (|after| <= 0.024 against before 1.09-1.33),
while a random projection of the same form retains
90-100% of it. On task cost the toy is stronger than the
paper's "almost no cost": held-out task-only MSE IMPROVES
under suppression (ratio 0.79-0.87), because in this world
the trait component of the pooled update is pure error
against task-only targets, and removing it outweighs
removing the slice-mean task component that the recovered
direction also carries. Assertions below encode exactly
these margins, no more.

What is NOT asserted: anything about real LM weight
updates, where slice updates are not ridge fits, tasks are
not isotropic random directions, and the trait is not a
single unit vector with a fixed loading on every slice;
the paper's amplification / interpretation use of the
recovered direction (only recovery and suppression are
tested); and suppression safety in the clean arm's task
readout -- with beta = 0 the "recovered" direction is
mostly the finite-K slice-mean task direction, and
projecting it out of the pooled update does raise clean
task MSE (~10-20%, toward the zero-update level): the
estimator cannot distinguish a shared trait from a shared
task-mean component, which in this toy is a finite-K
averaging artifact (norm ~2/sqrt(K)) and in a real audit
would be a genuine confound to rule out first. Two
modeling gifts, stated plainly: the trait loading beta is
identical on every slice (the paper's consistency premise,
assumed here rather than tested), and each slice's task is
exactly as strong as every other's. Pure numpy -- no GAN
machinery, because the claim is about update geometry, not
training dynamics. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family; PR #211 is what enables
this test in CI).
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


D = 48               # weight-space dimension
K = 8                # data slices per seed
N = 240              # rows per slice
SIGMA = 0.5          # target noise
LAM = 1.0            # ridge penalty
BETA = 1.2           # shared trait loading on every slice
TASK_NORM = 2.0      # per-slice task direction norm
SEED_BASE = 7000     # RNG stream base (see iteration notes)
SEEDS = (0, 1, 2, 3, 4)
K_SWEEP = (1, 2, 4, 8)


def _ridge(x, y):
    return np.linalg.solve(x.T @ x + LAM * np.eye(D), x.T @ y)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_seed(seed, beta):
    """One paired arm (trait if beta = BETA, clean if 0).

    Draw order is fixed -- tau, task directions, per-slice
    data, pooled data, random-projection direction,
    held-out data -- and identical for both arms of a seed,
    so trait and clean differ only through beta.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    tau = rng.standard_normal(D)
    tau /= np.linalg.norm(tau)
    tasks = []
    for _ in range(K):
        t = rng.standard_normal(D)
        t /= np.linalg.norm(t)
        tasks.append(TASK_NORM * t)
    tasks = np.array(tasks)

    deltas = []
    for k in range(K):
        x = rng.standard_normal((N, D))
        y = x @ (tasks[k] + beta * tau) + rng.normal(0, SIGMA, N)
        deltas.append(_ridge(x, y))
    deltas = np.array(deltas)

    mean_shift = deltas.mean(axis=0)
    u = mean_shift / np.linalg.norm(mean_shift)
    k_sweep = {kk: _cos(deltas[:kk].mean(axis=0), tau)
               for kk in K_SWEEP}
    centered = deltas - mean_shift
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    # Separately trained full-pool update: fresh data and
    # fresh noise, one ridge fit over all slices pooled.
    xs, ys = [], []
    for k in range(K):
        x = rng.standard_normal((N, D))
        y = x @ (tasks[k] + beta * tau) + rng.normal(0, SIGMA, N)
        xs.append(x)
        ys.append(y)
    d_full = _ridge(np.vstack(xs), np.concatenate(ys))

    suppressed = d_full - (d_full @ u) * u
    r = rng.standard_normal(D)
    r /= np.linalg.norm(r)
    rand_suppressed = d_full - (d_full @ r) * r

    # Held-out task readout: fresh inputs, task-only targets.
    xh = [rng.standard_normal((N, D)) for _ in range(K)]
    yh = [xh[k] @ tasks[k] for k in range(K)]

    def task_mse(w):
        return float(np.mean([np.mean((xh[k] @ w - yh[k]) ** 2)
                               for k in range(K)]))

    return {
        "rec": _cos(mean_shift, tau),
        "best": max(_cos(deltas[k], tau) for k in range(K)),
        "k_sweep": k_sweep,
        "pc1": abs(_cos(vt[0], tau)),
        "trait_before": float(d_full @ tau),
        "trait_after": float(suppressed @ tau),
        "trait_rand": float(rand_suppressed @ tau),
        "mse_before": task_mse(d_full),
        "mse_supp": task_mse(suppressed),
        "mse_rand": task_mse(rand_suppressed),
    }


_CACHE = {}


def _arm(seed, beta):
    key = (seed, beta)
    if key not in _CACHE:
        _CACHE[key] = run_seed(seed, beta)
    return _CACHE[key]


def test_cross_slice_mean_recovers_trait():
    _empf_recipe()
    for seed in SEEDS:
        arm = _arm(seed, BETA)
        clean = _arm(seed, 0.0)
        # Recovery is strong on every seed (demonstrated
        # 0.83-0.89; chance is ~0.14).
        assert arm["rec"] >= 0.78, (
            f"seed {seed}: cross-slice mean recovery cos "
            f"{arm['rec']:.3f}")
        # ... and beats the best single slice by a clear
        # margin (demonstrated 0.19-0.28): consistency
        # across slices, not any one slice, carries it.
        assert arm["rec"] >= arm["best"] + 0.12, (
            f"seed {seed}: recovery {arm['rec']:.3f} vs "
            f"best single slice {arm['best']:.3f}")
        # Clean control: no trait, no recovery (demonstrated
        # |cos| <= 0.17), and the paired gap is wide
        # (demonstrated 0.81-1.01).
        assert abs(clean["rec"]) <= 0.25, (
            f"seed {seed}: clean recovery cos "
            f"{clean['rec']:+.3f}")
        assert arm["rec"] - clean["rec"] >= 0.55, (
            f"seed {seed}: trait {arm['rec']:.3f} vs clean "
            f"{clean['rec']:+.3f}")


def test_recovery_grows_with_slice_count():
    _empf_recipe()
    means = {kk: [] for kk in K_SWEEP}
    for seed in SEEDS:
        arm = _arm(seed, BETA)
        ks = arm["k_sweep"]
        for kk in K_SWEEP:
            means[kk].append(ks[kk])
        # Endpoints separate widely on every seed
        # (demonstrated K=8 minus K=1 of 0.26-0.50, and
        # K=4 minus K=1 of 0.20-0.38).
        assert ks[8] >= ks[1] + 0.18, (
            f"seed {seed}: K sweep {ks}")
        assert ks[4] >= ks[1] + 0.12, (
            f"seed {seed}: K sweep {ks}")
        assert ks[8] > ks[4], (
            f"seed {seed}: K sweep {ks}")
    # The 5-seed mean sweep rises at every step
    # (demonstrated 0.48 / 0.65 / 0.76 / 0.87).
    avg = [float(np.mean(means[kk])) for kk in K_SWEEP]
    assert avg[0] < avg[1] < avg[2] < avg[3], (
        f"mean K sweep {avg}")


def test_centered_pca_misses_the_trait():
    _empf_recipe()
    for seed in SEEDS:
        arm = _arm(seed, BETA)
        # Centering removes the shared component by
        # construction: the leading centered PC is a null
        # (demonstrated |cos| 0.02-0.19), far below the
        # mean estimator on the same shifts (demonstrated
        # gaps 0.64-0.88).
        assert arm["pc1"] <= 0.30, (
            f"seed {seed}: centered PC1 |cos| {arm['pc1']:.3f}")
        assert arm["rec"] - arm["pc1"] >= 0.45, (
            f"seed {seed}: mean {arm['rec']:.3f} vs "
            f"centered PC1 {arm['pc1']:.3f}")


def test_projection_suppresses_trait_at_no_task_cost():
    _empf_recipe()
    for seed in SEEDS:
        arm = _arm(seed, BETA)
        # The pooled update carries a substantial trait
        # readout (demonstrated 1.09-1.33, against the
        # loading beta = 1.2).
        assert arm["trait_before"] >= 0.90, (
            f"seed {seed}: pooled trait readout "
            f"{arm['trait_before']:.3f}")
        # Projecting out the recovered direction removes
        # >= 95% of it (demonstrated |after| <= 0.024,
        # i.e. 98-100% removed).
        assert abs(arm["trait_after"]) <= 0.05 * arm["trait_before"], (
            f"seed {seed}: trait readout after suppression "
            f"{arm['trait_after']:+.3f} vs before "
            f"{arm['trait_before']:.3f}")
        # A random projection of the same form leaves the
        # trait readout standing (demonstrated retention
        # 0.90-1.00).
        assert arm["trait_rand"] >= 0.80 * arm["trait_before"], (
            f"seed {seed}: trait readout after random "
            f"projection {arm['trait_rand']:.3f} vs before "
            f"{arm['trait_before']:.3f}")
        # Task cost: none in this toy -- held-out task-only
        # MSE improves under suppression (demonstrated
        # ratios 0.79-0.87 of the unsuppressed MSE, and
        # below the random-projection MSE on every seed).
        assert arm["mse_supp"] <= 0.95 * arm["mse_before"], (
            f"seed {seed}: task MSE suppressed "
            f"{arm['mse_supp']:.3f} vs before "
            f"{arm['mse_before']:.3f}")
        assert arm["mse_supp"] <= arm["mse_rand"] - 0.30, (
            f"seed {seed}: task MSE suppressed "
            f"{arm['mse_supp']:.3f} vs random projection "
            f"{arm['mse_rand']:.3f}")


def test_clean_pool_finds_and_removes_nothing():
    _empf_recipe()
    for seed in SEEDS:
        clean = _arm(seed, 0.0)
        # With beta = 0 the pooled update's trait readout
        # is already near zero (demonstrated |before|
        # <= 0.14) and suppression moves it by almost
        # nothing in absolute terms (demonstrated |change|
        # <= 0.12, |after| <= 0.09).
        assert abs(clean["trait_before"]) <= 0.25, (
            f"seed {seed}: clean pooled trait readout "
            f"{clean['trait_before']:+.3f}")
        assert abs(clean["trait_after"]) <= 0.15, (
            f"seed {seed}: clean trait readout after "
            f"suppression {clean['trait_after']:+.3f}")
        assert abs(clean["trait_after"]
                   - clean["trait_before"]) <= 0.20, (
            f"seed {seed}: clean suppression moved the "
            f"trait readout from {clean['trait_before']:+.3f} "
            f"to {clean['trait_after']:+.3f}")
