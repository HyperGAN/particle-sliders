"""CPU gate for the trait-space monitoring claim of
Nghiem, Ho, Wiegreffe & Daume III, "Trait-space
Monitoring for Emergent Misalignment During Supervised
Finetuning" (arXiv:2606.07631). The paper's claim has
three parts. (1) Drift MAGNITUDE does not separate
dangerous from benign fine-tuning runs: its matched
benign--dangerous control shows substantial
representational drift under benign fine-tuning too, so
a monitor that gates on how far a run has moved cannot
tell the two apart. (2) A FIXED coordinate system of
seven alignment-relevant directions does separate them:
the 7D drift PROFILE in that basis distinguishes
dangerous from benign runs (the paper's monitors attain
AUROC 0.989, FNR 1.8%, FPR 2.0% across 468 checkpoints).
(3) Drift in trait space has a dominant axis: one
direction explains 78.6% of the variance of trait-space
drift and is stable across datasets, extraction methods
and capacity. Provenance: cycle 18 of the Particle
Sliders paper watch.

Toy operationalization: the smallest linear drift world
in which the three parts can be decided separately.
d = 64. The monitor basis U is the first 7 columns of a
random orthogonal matrix (the fixed coordinate system,
drawn once per seed). The dangerous profile is the fixed
7-vector p proportional to [1.0, 0.6, 0.5, 0.3, 0.2,
0.1, 0.05] (normalized); the dangerous drift direction
is U.p. Each run is a 12-checkpoint trajectory: at every
step the position advances by (total / 12) along the
run's direction plus isotropic per-step noise
(sigma = 0.05, the upper end of the design range, so the
noise is as hostile to the profile claims as the design
allows). Dangerous runs (6 per seed): direction =
normalize(U.p + 0.15 * random unit vector) -- the trait
direction with a fixed tilt -- and total drift norm
uniform in [2.5, 3.5]. Benign runs (6 per seed):
direction uniform on the sphere, total drift norm
uniform in [2.5, 4.5]. Benign drift is as large as or
LARGER than dangerous drift by construction; that is
the paper's matched-control finding, stated plainly:
any magnitude readout starts out confounded. Both arms
share the same trajectory noise process, so the arms
differ only in direction and in the total-norm range.

Readouts on the final displacement Delta (and, for the
dominant-axis analysis, on every checkpoint >= 4):
magnitude ||Delta||; profile vector z = U^T Delta;
profile score = cos(z, p). Metrics across the 6 + 6 runs
of a seed: AUROC of magnitude and AUROC of profile score
(dangerous = positive class, Mann--Whitney form). The
dominant-axis analysis pools the dangerous runs' profile
vectors z over checkpoints >= 4 and runs (54 vectors per
seed) and takes PC1 of the centered pool: its share of
variance, its alignment |cos(PC1, p)|, and a split-half
stability |cos(PC1_even, PC1_odd)| over even- vs
odd-indexed runs.

Arms and controls:
  (a) magnitude readout (the failed monitor): AUROC of
      ||Delta|| must stay at or below 0.75 on every seed
      -- it cannot separate, and on most seeds it ranks
      benign runs ABOVE dangerous ones, because benign
      totals are drawn from the wider, higher range;
  (b) profile readout (the paper's monitor): AUROC of
      cos(U^T Delta, p) must reach 0.95 on every seed,
      and beat the magnitude AUROC by a wide margin on
      the same runs;
  (c) wrong-basis control: the same cosine score
      computed in a disjoint 7D basis (columns 7--13 of
      the same orthogonal matrix, with the same p read
      as coordinates there) must NOT separate -- the
      separation lives in the specific fixed basis, not
      in any 7D projection;
  (d) dominant axis: PC1 of the dangerous profile pool
      must carry a clear majority of the variance, align
      with p, and reproduce across split halves.

Measured on pop-os (particlegan 0.10.1), 5 seeds
(magnitude AUROC / profile AUROC / wrong-basis AUROC /
PC1 share / |cos(PC1, p)| / split-half |cos|):
  seed 0: mag 0.361, prof 1.000, wrong 0.472,
          share 0.805, align 0.997, stab 0.977
  seed 1: mag 0.333, prof 1.000, wrong 0.250,
          share 0.828, align 0.996, stab 0.990
  seed 2: mag 0.028, prof 1.000, wrong 0.472,
          share 0.803, align 0.997, stab 0.990
  seed 3: mag 0.278, prof 1.000, wrong 0.500,
          share 0.782, align 0.998, stab 0.978
  seed 4: mag 0.111, prof 1.000, wrong 0.750,
          share 0.877, align 0.998, stab 0.975

The arbitration, stated plainly: all three parts of
the claim reproduce at toy scale. Magnitude does not
merely fail to separate -- it inverts: AUROC 0.03-0.36,
below chance on every seed, because benign totals are
drawn as large as or larger than dangerous ones (the
paper's matched-control situation). The profile score
in the fixed basis separates perfectly (AUROC 1.000 on
every seed; per-seed margin over magnitude 0.64-0.97),
while the identical score in a disjoint basis sits at
chance (0.25-0.75 across seeds, mean ~0.49 -- wide
scatter is expected at 6-vs-6 runs, and no seed comes
near the true basis's 1.000). The dangerous profile
pool has a dominant axis: PC1 share 0.78-0.88 (an
isotropic 7D pool would give ~0.14), aligned with the
planted profile at |cos| 0.996-0.998 and stable across
split halves at 0.975-0.990. The toy share lands near
the paper's 78.6% but that value is not asserted --
only dominance, alignment and stability are.
Assertions below encode exactly these margins, no more.

What is NOT asserted: the paper's exact numbers. The
toy does not reproduce AUROC 0.989 / FNR 1.8% / FPR 2.0%
as values, and it does NOT assert the paper's 78.6%
dominant-axis share -- the toy-scale share is what it
is, reported above. Nothing here is asserted about real
LM activations, where the trait basis must be extracted
rather than drawn, dangerous drift is not a tilted copy
of one fixed profile, benign drift is not isotropic,
and checkpoints are not a linear random walk. Two
modeling gifts, stated plainly: the dangerous profile p
is known exactly and is the same on every dangerous run
(the paper must estimate its directions and its
dangerous runs vary), and the benign arm's isotropy is
what makes its profile scores concentrate near chance --
a benign run family concentrated near the trait subspace
would be a harder control. Pure numpy -- no GAN
machinery, because the claim is about drift geometry,
not training dynamics. Skips cleanly when the installed
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


D = 64               # drift-space dimension
TRAIT_DIM = 7        # monitor basis size (the paper's seven)
N_CKPT = 12          # checkpoints per run
RUNS = 6             # runs per arm per seed
SIGMA = 0.05         # per-step isotropic noise
TILT = 0.15          # dangerous-direction tilt off U.p
CKPT_MIN = 4         # first checkpoint in the axis pool
SEED_BASE = 31000    # RNG stream base (see iteration notes)
SEEDS = (0, 1, 2, 3, 4)

P_RAW = np.array([1.0, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05])
PROFILE = P_RAW / np.linalg.norm(P_RAW)


def _auroc(pos, neg):
    """Mann--Whitney AUROC: P(pos > neg) + 0.5 * P(pos == neg)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    wins = sum(float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
               for p in pos)
    return wins / (len(pos) * len(neg))


def _pc1(rows):
    centered = rows - rows.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def run_seed(seed):
    """One seed: monitor basis, 6 dangerous + 6 benign runs.

    Draw order is fixed -- basis matrix, then per run
    (dangerous first): tilt / direction, total norm,
    per-step noise -- so every readout of a seed comes
    from the same draws.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    a = rng.standard_normal((D, D))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))  # deterministic column signs
    basis = q[:, :TRAIT_DIM]             # the monitor basis U
    wrong = q[:, TRAIT_DIM:2 * TRAIT_DIM]  # disjoint control basis
    trait_dir = basis @ PROFILE

    def make_run(dangerous):
        if dangerous:
            g = rng.standard_normal(D)
            g /= np.linalg.norm(g)
            direction = trait_dir + TILT * g
            direction /= np.linalg.norm(direction)
            total = rng.uniform(2.5, 3.5)
        else:
            direction = rng.standard_normal(D)
            direction /= np.linalg.norm(direction)
            total = rng.uniform(2.5, 4.5)
        pos = np.zeros(D)
        traj = []
        for _ in range(N_CKPT):
            pos = pos + (total / N_CKPT) * direction \
                + rng.normal(0, SIGMA, D)
            traj.append(pos.copy())
        return traj

    dangerous = [make_run(True) for _ in range(RUNS)]
    benign = [make_run(False) for _ in range(RUNS)]

    def profile_score(delta, basis_):
        z = basis_.T @ delta
        return float(z @ PROFILE / np.linalg.norm(z))

    mag_d = [float(np.linalg.norm(t[-1])) for t in dangerous]
    mag_b = [float(np.linalg.norm(t[-1])) for t in benign]
    sco_d = [profile_score(t[-1], basis) for t in dangerous]
    sco_b = [profile_score(t[-1], basis) for t in benign]
    wrg_d = [profile_score(t[-1], wrong) for t in dangerous]
    wrg_b = [profile_score(t[-1], wrong) for t in benign]

    # Dominant axis: dangerous profile vectors at
    # checkpoints >= CKPT_MIN, pooled over runs.
    pool = np.array([basis.T @ t[s] for t in dangerous
                     for s in range(CKPT_MIN - 1, N_CKPT)])
    centered = pool - pool.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    var = sv ** 2
    pc1 = vt[0]
    even = np.array([basis.T @ dangerous[i][s]
                     for i in range(0, RUNS, 2)
                     for s in range(CKPT_MIN - 1, N_CKPT)])
    odd = np.array([basis.T @ dangerous[i][s]
                    for i in range(1, RUNS, 2)
                    for s in range(CKPT_MIN - 1, N_CKPT)])

    return {
        "mag_auroc": _auroc(mag_d, mag_b),
        "prof_auroc": _auroc(sco_d, sco_b),
        "wrong_auroc": _auroc(wrg_d, wrg_b),
        "share": float(var[0] / var.sum()),
        "align": abs(float(pc1 @ PROFILE)),
        "stab": abs(float(_pc1(even) @ _pc1(odd))),
    }


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_magnitude_does_not_separate():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        # The failed monitor: magnitude AUROC stays at or
        # below 0.75 on every seed (demonstrated
        # 0.03-0.36, below chance throughout) -- benign
        # runs drift as far as or farther than dangerous
        # ones, by construction.
        assert res["mag_auroc"] <= 0.75, (
            f"seed {seed}: magnitude AUROC "
            f"{res['mag_auroc']:.3f}")


def test_profile_score_separates_where_magnitude_fails():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        # The paper's monitor: profile AUROC at ceiling on
        # every seed (demonstrated 1.000 throughout); the
        # assertion keeps a 0.95 floor.
        assert res["prof_auroc"] >= 0.95, (
            f"seed {seed}: profile AUROC "
            f"{res['prof_auroc']:.3f}")
        # Same runs, same displacements: the profile beats
        # magnitude by a wide margin (demonstrated
        # 0.64-0.97).
        assert res["prof_auroc"] - res["mag_auroc"] >= 0.25, (
            f"seed {seed}: profile {res['prof_auroc']:.3f} "
            f"vs magnitude {res['mag_auroc']:.3f}")
        # Wrong-basis control: the identical cosine score
        # in a disjoint basis does not separate
        # (demonstrated 0.25-0.75, chance-level scatter
        # at 6-vs-6 runs).
        assert res["wrong_auroc"] <= 0.80, (
            f"seed {seed}: wrong-basis AUROC "
            f"{res['wrong_auroc']:.3f}")


def test_trait_space_drift_has_a_stable_dominant_axis():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        # PC1 carries a clear majority of the dangerous
        # profile pool's variance (demonstrated
        # 0.78-0.88; an isotropic 7D pool would give
        # ~1/7 = 0.14). The paper's 78.6% is NOT asserted.
        assert res["share"] >= 0.60, (
            f"seed {seed}: PC1 share {res['share']:.3f}")
        # ... and PC1 is the dangerous profile direction
        # (demonstrated 0.996-0.998).
        assert res["align"] >= 0.90, (
            f"seed {seed}: |cos(PC1, p)| {res['align']:.3f}")
        # ... and the axis reproduces across split halves
        # of the runs (demonstrated 0.975-0.990).
        assert res["stab"] >= 0.85, (
            f"seed {seed}: split-half |cos| "
            f"{res['stab']:.3f}")
