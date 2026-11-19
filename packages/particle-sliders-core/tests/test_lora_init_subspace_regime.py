"""CPU gate for the LoRA init-subspace regime claim (arXiv:2611.23428):
when labeled data are scarce relative to representation dimension, a
LoRA adapter's initial input-subspace DIRECTION -- not its init norm --
decides how much it gains; the advantage vanishes as data grow, and a
pre-training alignment coefficient rho tracks the realized gain.

Paper: "When Does Subspace Direction Matter for LoRA? Regime Analysis
of the Magnitude Principle in Few-Shot Adaptation" (Subedi, Shen &
Zhao). It refines the recent view that spectral LoRA initializers help
through update MAGNITUDE alone. Its claims: (1) an adapter's initial
input-subspace direction matters most when labeled data are scarce
relative to representation dimension AND the init subspace captures
supervised task structure -- in that regime the adapter's row span
remains sensitive to its starting point; (2) a first-step analysis
motivates an alignment coefficient rho (init subspace vs task) that
closely tracks realized gains across classification tasks, usable as a
low-cost pre-training diagnostic; (3) PivotLoRA -- init from activation
PCA at prediction-relevant positions plus one output-preserving
orthogonal pivot early in training -- achieves the best average rank
among 13 methods on few-shot BERT-base classification and improves over
vanilla LoRA in 29 of 30 settings DESPITE starting from a smaller
input-factor init norm; (4) gains diminish with more labeled data,
consistent with the regime account; extensions to 7B/8B Llama decoders
and GSM8K/StrategyQA. (Numbers in this paragraph are the paper's,
reported from its abstract; the toy below tests the regime law, not
the BERT numbers, and does not implement the orthogonal pivot.)

Toy operationalization: linear regression in R^96 with a spiked
activation covariance and a KNOWN low-dimensional task subspace, the
paper's analysis setting in miniature. Inputs x ~ N(0, Sigma), Sigma
with eigenvalues 3.0 on the 6 task directions T, 5.0 on 4 high-variance
distractor directions (variance the task does not read), 0.3 on the
remaining 86. The target is y = w* . x + noise (noise std 0.3, signal
variance 2.0) with w* supported exactly on T. The adapter is a LoRA
pair DeltaW = B A, rank r = 6, B initialized to zero (the standard
LoRA start, so the adapter output is zero at init and A's row span is
the adapter's initial input subspace). Both factors train by full-batch
gradient descent (4000 steps, lr 0.03, exact sufficient statistics
G = X^T X / n, c = X^T y / n, so the step count does not scale with n).

Why the regime appears here, mechanically: the gradient w.r.t. A is a
sum of outer products with the TRAINING inputs, so the component of
row-span(A) orthogonal to the training-data span never moves. When
n << d the data span covers a small fraction of the space and the init
subspace persists; when n >= d the data span is everything and the
init is fully overwritten. That is the toy form of the paper's "row
span remains sensitive to its starting point" in the scarce regime.

Init arms (A's row span; s0 = 0.5 is the matched init scale):
  A1 task-aligned -- rows = an orthonormal basis of T, norm s0. The
     PivotLoRA-style arm. Simplification, stated honestly: the paper
     builds this subspace from activation PCA at prediction-relevant
     positions; the toy uses the oracle task subspace T itself as the
     stand-in for "the subspace that procedure recovers when it
     captures supervised task structure" (the paper's stated success
     condition). rho(A1) = 1 by construction.
  A2 random -- rows = a random orthonormal 6-frame, norm s0 (vanilla
     LoRA-style init). rho ~ 0.05-0.07.
  A3 magnitude control -- rows = a random orthonormal 6-frame forced
     ORTHOGONAL to T (rho = 0), at TWICE the init norm (2 * s0). If
     magnitude were the mechanism, A3 should win at small n; the paper
     says direction is.
  A4 raw activation PCA -- rows = the top-6 principal components of a
     4000-point UNLABELED activation pool, norm s0. PCA captures
     variance, not task: the top components mix the 4 distractors
     (eigenvalue 5) with 2 of the task directions, rho ~ 0.34. This
     arm separates "activation geometry" from "task alignment" -- the
     paper's condition is PCA at prediction-RELEVANT positions, not
     raw PCA.

rho (computed BEFORE training, per arm per seed): the fraction of the
init row span's energy inside T, ||P_T A^T||_F^2 / ||A||_F^2 -- the
toy form of the paper's alignment coefficient (principal-angle energy
between row-span(A_init) and the task subspace).

Metrics: held-out MSE on 4000 fresh points per arm per n, n swept over
{16, 64, 256, 1024}; gain = null MSE (predicting 0) minus arm MSE;
Spearman rank correlation between rho and gain across all 12
(arm, seed) points at the scarcest n.

Measured on pop-os (particlegan 0.10.1): held-out MSE at n = 16 --
A1 0.267-0.494 vs A2 0.573-0.805 (aligned better by 0.16-0.42 in every
seed) vs A3 0.915-1.226 (aligned better by 0.57-0.84 despite A3's 2x
norm; A3 is also WORSE than the matched-norm random A2 by 0.34-0.42,
so the extra magnitude actively hurts when the direction is wrong) vs
A4 0.490-0.769 (raw PCA beats random in only 1 of 3 seeds and never
matches task alignment). At n = 64 the A1-over-A2 advantage shrinks to
0.02-0.05; at n = 256 and n = 1024 it is 0.000 in every seed -- all
four arms converge to the same held-out MSE (0.129-0.157 at n = 256,
0.097-0.100 at n = 1024). Pooled rho-vs-gain Spearman at n = 16 is
0.79 (asserted >= 0.7). Assertions
below encode exactly these margins, no more. One honest wrinkle: in
seed 2 the raw-PCA arm (rho 0.34) finishes slightly BEHIND the random
arm (rho 0.07) -- rho tracks the gain ordering pooled across arms and
seeds, not in every single cell, because A4's distractor rows carry
5x-variance features that also slow B's fit. Skips cleanly when the
installed particlegan does not provide the EMPF recipe. Pure
numpy/torch linear algebra -- no GAN machinery, because the claim is
about adapter init geometry, not training a generator.
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


D = 96            # representation dimension
K = 6             # task-subspace dimension
R = 6             # adapter rank
NOISE = 0.3       # label noise std
STEPS = 4000      # full-batch GD steps (sufficient statistics: n-free)
LR = 0.03
NS = (16, 64, 256, 1024)
N_TEST = 4000
N_UNLAB = 4000    # unlabeled activations for the raw-PCA arm
S0 = 0.5          # matched init scale for A1 / A2 / A4
SEEDS = (0, 1, 2)


def _world(rng):
    """Spiked covariance, task subspace T, target w* supported on T."""
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    eig = np.full(D, 0.3)
    eig[:K] = 3.0            # task directions
    eig[K:K + 4] = 5.0       # high-variance distractors
    task = q_basis[:, :K]
    coeffs = rng.standard_normal(K)
    coeffs *= np.sqrt(2.0 / np.sum(eig[:K] * coeffs ** 2))  # signal var 2
    w_star = task @ coeffs
    chol = q_basis * np.sqrt(eig)
    return task, w_star, chol


def _sample(rng, chol, w_star, n):
    x = rng.standard_normal((n, D)) @ chol.T
    y = x @ w_star + NOISE * rng.standard_normal(n)
    return x, y


def _rho(a_mat, task):
    """Fraction of row-span(A) energy inside the task subspace T."""
    return float(np.sum((a_mat @ task) ** 2) / np.sum(a_mat ** 2))


def _train(gram, cross, a0):
    """Full-batch GD on both LoRA factors from B = 0.

    grad w.r.t. A is B outer (G w_eff - c): every update direction lies
    in the training-data span, so A's component outside that span is
    frozen at init -- the mechanism under test.
    """
    a = a0.copy()
    b = np.zeros(R)
    for _ in range(STEPS):
        g = gram @ (a.T @ b) - cross
        grad_b = 2.0 * a @ g
        grad_a = 2.0 * np.outer(b, g)
        b = b - LR * grad_b
        a = a - LR * grad_a
    return a, b


def _mse(x, y, a, b):
    return float(np.mean((x @ a.T @ b - y) ** 2))


def _spearman(xs, ys):
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        srt = np.asarray(v)[order]
        i = 0
        while i < len(srt):
            j = i
            while j + 1 < len(srt) and srt[j + 1] == srt[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


_CACHE = {}


def run_seed(seed):
    if seed in _CACHE:
        return _CACHE[seed]
    rng = np.random.default_rng(7000 + seed)
    task, w_star, chol = _world(rng)
    x_te, y_te = _sample(rng, chol, w_star, N_TEST)
    x_un, _ = _sample(rng, chol, w_star, N_UNLAB)
    cov = np.cov(x_un.T)
    _, evecs = np.linalg.eigh(cov)
    pca_rows = evecs[:, ::-1][:, :R].T

    def rand_frame():
        m, _ = np.linalg.qr(rng.standard_normal((D, R)))
        return m.T

    def anti_frame():
        m = rng.standard_normal((D, R))
        m = m - task @ (task.T @ m)
        mm, _ = np.linalg.qr(m)
        return mm.T

    arms = {
        "aligned": (task.T.copy(), S0),
        "random": (rand_frame(), S0),
        "big_misaligned": (anti_frame(), 2 * S0),
        "raw_pca": (pca_rows.copy(), S0),
    }
    rhos = {name: _rho(rows, task) for name, (rows, _) in arms.items()}

    x_all, y_all = _sample(rng, chol, w_star, max(NS))
    null = float(np.mean(y_te ** 2))
    table = {}
    for n in NS:
        x, y = x_all[:n], y_all[:n]
        gram = x.T @ x / n
        cross = x.T @ y / n
        table[n] = {}
        for name, (rows, scale) in arms.items():
            a, b = _train(gram, cross, scale * rows)
            table[n][name] = _mse(x_te, y_te, a, b)
    out = dict(table=table, rhos=rhos, null=null)
    _CACHE[seed] = out
    return out


def test_aligned_init_wins_only_when_data_are_scarce():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        t = r["table"]
        # Scarce regime (n = 16 << d = 96): direction decides.
        assert t[16]["aligned"] < t[16]["random"] - 0.1, (
            f"seed {seed}: n=16 aligned {t[16]['aligned']:.3f} vs "
            f"random {t[16]['random']:.3f}")
        assert t[16]["aligned"] < t[16]["raw_pca"] - 0.05, (
            f"seed {seed}: n=16 aligned {t[16]['aligned']:.3f} vs "
            f"raw PCA {t[16]['raw_pca']:.3f} -- task alignment should "
            f"beat variance alignment when data are scarce")
        # The advantage is a regime curve: already small at n = 64.
        adv16 = t[16]["random"] - t[16]["aligned"]
        adv64 = t[64]["random"] - t[64]["aligned"]
        assert adv64 < adv16, (
            f"seed {seed}: advantage should shrink from n=16 "
            f"({adv16:.3f}) to n=64 ({adv64:.3f})")
        # Abundant regime (n = 1024 > d): init fully overwritten, all
        # arms converge to the same held-out loss.
        for name in ("random", "big_misaligned", "raw_pca"):
            assert abs(t[1024]["aligned"] - t[1024][name]) < 0.02, (
                f"seed {seed}: n=1024 aligned {t[1024]['aligned']:.3f} "
                f"vs {name} {t[1024][name]:.3f} -- the scarce-data "
                f"advantage should be gone")
        assert adv16 > 0.1, (
            f"seed {seed}: n=16 advantage {adv16:.3f} should be the "
            f"headline margin, not a rounding difference")


def test_rho_predicts_gain_and_magnitude_does_not():
    _empf_recipe()
    rhos, gains = [], []
    for seed in SEEDS:
        r = run_seed(seed)
        t, rho, null = r["table"], r["rhos"], r["null"]
        # rho is what the construction says it is (pre-training check).
        assert rho["aligned"] > 0.99, (
            f"seed {seed}: aligned rho {rho['aligned']:.3f}")
        assert rho["big_misaligned"] < 0.01, (
            f"seed {seed}: anti-aligned rho {rho['big_misaligned']:.3f}")
        assert 0.2 < rho["raw_pca"] < 0.5, (
            f"seed {seed}: raw-PCA rho {rho['raw_pca']:.3f} -- PCA "
            f"should capture some, not most, of the task subspace")
        assert rho["random"] < 0.15, (
            f"seed {seed}: random rho {rho['random']:.3f}")
        # Magnitude control: at 2x the init norm and rho = 0, A3 must
        # NOT beat the matched-norm random init at scarce n -- if
        # magnitude were the mechanism it would. It is worse.
        assert t[16]["big_misaligned"] > t[16]["random"] + 0.2, (
            f"seed {seed}: n=16 big-misaligned "
            f"{t[16]['big_misaligned']:.3f} vs random "
            f"{t[16]['random']:.3f} -- double norm, wrong direction")
        assert t[16]["aligned"] < t[16]["big_misaligned"] - 0.4, (
            f"seed {seed}: n=16 aligned {t[16]['aligned']:.3f} vs "
            f"big-misaligned {t[16]['big_misaligned']:.3f}")
        for name in rho:
            rhos.append(rho[name])
            gains.append(null - t[16][name])
    # The diagnostic claim: rho, computed before any training, tracks
    # the realized gain ordering across arms and seeds at scarce n.
    corr = _spearman(rhos, gains)
    assert corr >= 0.7, (
        f"rho-vs-gain Spearman at n=16 is {corr:.3f} over "
        f"{len(rhos)} (arm, seed) points; rho should track the gain "
        f"ordering as the paper's diagnostic claims")
