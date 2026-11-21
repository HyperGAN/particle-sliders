"""CPU gate for the probe-vs-steer claim (arXiv:2611.19499): the
direction that probes best is not the direction that steers best --
probing favors the covariance-adjusted discriminant, additive steering
at fixed side effect favors the plain mean difference.

Paper: "Do Probing and Steering Favor the Same Directions Within and
Across Language Models?" (Sadeghi & Becker). It builds a one-parameter
family interpolating between a covariance-adjusted discriminant (the
probe-optimal end) and the plain mean difference, and finds, in every
model tested, that the best probe direction is not the best steering
direction: probing favors the discriminant end, steering favors
directions closer to the mean difference. The separation held in 59 of
60 layer-model-concept combinations. (Its secondary finding -- an
intermediate direction sometimes steering 27-56% stronger for
sentiment -- is flagged by the paper itself as unstable and is NOT
asserted here; neither is the abstention application, steering toward
"unknown" raising abstention 18-44 points on uncertain errors.)

Toy operationalization: synthetic class-conditional states in R^20
with KNOWN latent structure and anisotropic, correlated covariance, so
the discriminant Sigma^-1 Delta_mu and the mean difference Delta_mu
genuinely differ (measured cos 0.29-0.30 across seeds). Two orthonormal
ground-truth axes do the work:

  p -- the causal concept axis: the class means differ along it by 1.0
       and the ground-truth behavior readout is projection on p.
       Natural variation along p is LARGE (variance 4.0).
  q -- a predictive-but-noncausal axis: the class means differ along
       it by only EPS_Q = 0.3, it does not enter the behavior readout
       at all, but natural variation along q is TINY (variance 0.01).

A probe can exploit q: a 0.3 mean shift against 0.1 noise is a huge
signal-to-noise ratio, so whitening (Sigma^-1) rotates the discriminant
almost entirely onto q (measured cos(discriminant, q) ~ 1.00). A
steering vector cannot: moving a state along q changes the behavior
readout by ~nothing, however cleanly q separates the classes. The mean
difference, dominated by the large raw shift along p, points mostly at
the causal axis (measured cos(mean difference, p) 0.94-0.95). This is
the paper's mechanism in miniature: whitening rewards low-variance
predictive features; behavior change rewards the axis the behavior
actually reads.

Family: d(t) ~ (1 - t) * discriminant_hat + t * mean_difference_hat,
both ends unit-normalized before interpolating, t on a 0.0-1.0 grid in
steps of 0.1, d(t) re-normalized.

Metrics per seed, on held-out states:
  probe:    AUC of the projection h . d(t) against the class label
            (Mann-Whitney rank AUC).
  steering: apply LAM * d(t) to held-out states; behavior gain is the
            mean shift of the ground-truth readout h . p; side effect
            is the RMS displacement orthogonal to p (everything the
            step moves that is not the behavior axis -- nuisance-axis
            and unrelated-readout movement included); efficiency is
            gain / side effect, i.e. behavior change per unit
            off-target movement, the toy form of the paper's
            fixed-side-effect steering evaluation.
  control:  one fixed random unit direction per seed, scored on both
            metrics (must sit near null on both).

Transfer diagnostic (the paper's "across models" half, in miniature):
a second world sharing p, q and Delta_mu but with isotropic variance
(q is no longer low-variance there). The discriminant harvested in the
home world collapses as a probe in the second world, while the mean
difference does not -- whitening is fitted to one world's noise
geometry, the raw mean shift is not.

Measured on pop-os (particlegan 0.10.1): probe AUC at the discriminant
end 0.983-0.985 vs 0.630-0.681 at the mean-difference end; steering
efficiency 0.008-0.009 at the discriminant end vs 2.68-3.00 at the
mean-difference end; random control AUC 0.54-0.57, efficiency
0.20-0.33; both metrics move monotonically along the family in every
seed (probe argmax t = 0.0, steering argmax t = 1.0); transfer drops
the discriminant's probe AUC by 0.39-0.41 while the mean difference's
AUC rises (0.63-0.68 home, 0.75-0.78 transferred). Assertions below encode exactly these margins, no
more. Skips cleanly when the installed particlegan does not provide
the EMPF recipe. Pure numpy/torch state geometry -- no GAN machinery,
because the claim is about harvested directions, not training.
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


D = 20            # state dimension
N = 4000          # states per seed (half train / half held-out)
EPS_Q = 0.3       # class-mean shift along the noncausal axis q
VAR_P = 4.0       # natural variance along the causal axis p
VAR_Q = 0.01      # natural variance along q (the probe's free lunch)
LAM = 1.0         # steering step length (unit directions, so |step| = LAM)
TS = np.linspace(0.0, 1.0, 11)


def _unit(x):
    return x / np.linalg.norm(x)


def _cos(x, y):
    return float(_unit(x) @ _unit(y))


def _auc(scores, labels):
    """Mann-Whitney rank AUC, average ranks for ties."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    srt = scores[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def _world(rng, variances):
    """States h = y * Delta/2 + noise, noise covariance Q diag(var) Q^T.

    Returns states, labels, and the ground-truth axes p, q and Delta.
    """
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    p, q = q_basis[:, 0], q_basis[:, 1]
    delta = p + EPS_Q * q
    chol = q_basis * np.sqrt(variances)          # Sigma^{1/2}
    y = rng.choice([-1, 1], size=N)
    h = (y[:, None] * (delta / 2)[None, :]
         + rng.standard_normal((N, D)) @ chol.T)
    return h, y, p, q, delta, q_basis, chol


def run_seed(seed):
    rng = np.random.default_rng(9000 + seed)
    variances = np.ones(D)
    variances[0], variances[1] = VAR_P, VAR_Q
    h, y, p, q, delta, q_basis, _ = _world(rng, variances)
    h_tr, h_te = h[:N // 2], h[N // 2:]
    y_tr, y_te = y[:N // 2], y[N // 2:]

    # Harvest both ends of the family from the train split only.
    mu_pos = h_tr[y_tr == 1].mean(0)
    mu_neg = h_tr[y_tr == -1].mean(0)
    d_mu = mu_pos - mu_neg
    pooled = np.cov(np.vstack([h_tr[y_tr == 1] - mu_pos,
                               h_tr[y_tr == -1] - mu_neg]).T)
    discriminant = np.linalg.solve(pooled + 1e-6 * np.eye(D), d_mu)
    disc_hat, dmu_hat = _unit(discriminant), _unit(d_mu)

    def probe(direction):
        return _auc(h_te @ direction, y_te)

    def steer(direction):
        step = LAM * direction
        gain = float(((h_te + step) @ p).mean() - (h_te @ p).mean())
        off = step - (step @ p) * p
        side = float(np.sqrt((off ** 2).sum()))
        return gain, side, gain / side if side > 0 else float("inf")

    rows = []
    for t in TS:
        d = _unit((1 - t) * disc_hat + t * dmu_hat)
        gain, side, eff = steer(d)
        rows.append(dict(t=float(t), auc=probe(d), gain=gain,
                         side=side, eff=eff))
    rand = _unit(rng.standard_normal(D))
    r_gain, _, r_eff = steer(rand)

    # Transfer: same axes and Delta, isotropic noise in the second world.
    rng_b = np.random.default_rng(31000 + seed)
    y_b = rng_b.choice([-1, 1], size=N)
    h_b = (y_b[:, None] * (delta / 2)[None, :]
           + rng_b.standard_normal((N, D)) @ q_basis.T)
    transfer = dict(disc_auc=_auc(h_b @ disc_hat, y_b),
                    dmu_auc=_auc(h_b @ dmu_hat, y_b))

    return dict(
        rows=rows,
        cos_ends=_cos(disc_hat, dmu_hat),
        cos_disc_q=_cos(disc_hat, q),
        cos_dmu_p=_cos(dmu_hat, p),
        rand_auc=probe(rand), rand_gain=r_gain, rand_eff=r_eff,
        transfer=transfer,
    )


def test_probe_and_steer_favor_different_ends():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        rows = r["rows"]
        disc_row, dmu_row = rows[0], rows[-1]
        t_probe = rows[int(np.argmax([x["auc"] for x in rows]))]["t"]
        t_steer = rows[int(np.argmax([x["eff"] for x in rows]))]["t"]

        # Trap guard: whitening must genuinely rotate the direction --
        # if the two ends nearly coincide the toy tests nothing.
        assert r["cos_ends"] < 0.5, (
            f"seed {seed}: cos(discriminant, mean diff) "
            f"{r['cos_ends']:.3f} -- ends too close to separate")
        assert r["cos_disc_q"] > 0.8 and r["cos_dmu_p"] > 0.8, (
            f"seed {seed}: discriminant should sit on q "
            f"({r['cos_disc_q']:.3f}) and mean diff on p "
            f"({r['cos_dmu_p']:.3f})")

        # Probing favors the discriminant end (paper: probe optimum at
        # the discriminant in 59/60 combinations).
        assert t_probe <= 0.1, (
            f"seed {seed}: probe argmax at t={t_probe:.1f}, "
            f"AUCs {[round(x['auc'], 3) for x in rows]}")
        assert disc_row["auc"] > 0.9, (
            f"seed {seed}: discriminant probe AUC {disc_row['auc']:.3f}")
        assert disc_row["auc"] > dmu_row["auc"] + 0.2, (
            f"seed {seed}: probe AUC discriminant {disc_row['auc']:.3f} "
            f"vs mean diff {dmu_row['auc']:.3f}")

        # Steering at fixed side effect favors the mean-difference end.
        assert t_steer >= 0.9, (
            f"seed {seed}: steering argmax at t={t_steer:.1f}, "
            f"effs {[round(x['eff'], 3) for x in rows]}")
        assert dmu_row["eff"] > 2.0, (
            f"seed {seed}: mean-diff steering efficiency "
            f"{dmu_row['eff']:.3f}")
        assert disc_row["eff"] < 0.1, (
            f"seed {seed}: discriminant steering efficiency "
            f"{disc_row['eff']:.3f} -- the probe's direction should "
            f"barely move the behavior readout")
        assert dmu_row["eff"] > disc_row["eff"] + 1.5, (
            f"seed {seed}: steering efficiency mean diff "
            f"{dmu_row['eff']:.3f} vs discriminant {disc_row['eff']:.3f}")
        assert dmu_row["gain"] > 0.8 and disc_row["gain"] < 0.1, (
            f"seed {seed}: steering gain mean diff {dmu_row['gain']:.3f} "
            f"vs discriminant {disc_row['gain']:.3f} at equal step length")

        # Random control: near null on both metrics, beaten clearly.
        assert r["rand_auc"] < 0.65, (
            f"seed {seed}: random-direction probe AUC {r['rand_auc']:.3f}")
        assert r["rand_eff"] < 0.6, (
            f"seed {seed}: random-direction steering efficiency "
            f"{r['rand_eff']:.3f}")
        assert dmu_row["eff"] > r["rand_eff"] + 1.5, (
            f"seed {seed}: mean-diff efficiency {dmu_row['eff']:.3f} vs "
            f"random {r['rand_eff']:.3f}")


def test_mean_difference_transfers_discriminant_does_not():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        home_disc = r["rows"][0]["auc"]
        home_dmu = r["rows"][-1]["auc"]
        # Whitening is fitted to the home world's noise geometry: in
        # the isotropic second world the harvested discriminant loses
        # most of its probe advantage, while the raw mean difference
        # keeps (here, gains) its accuracy.
        assert home_disc - r["transfer"]["disc_auc"] > 0.2, (
            f"seed {seed}: discriminant probe AUC home {home_disc:.3f} "
            f"-> transfer {r['transfer']['disc_auc']:.3f}")
        assert r["transfer"]["dmu_auc"] >= home_dmu - 0.05, (
            f"seed {seed}: mean-diff probe AUC home {home_dmu:.3f} "
            f"-> transfer {r['transfer']['dmu_auc']:.3f}")
