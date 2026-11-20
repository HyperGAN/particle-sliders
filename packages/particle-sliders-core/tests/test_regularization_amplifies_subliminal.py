"""CPU gate for the regularization-amplifies claim (arXiv:2611.10397):
subliminal learning is typically ABSENT under full fine-tuning, but
low-rank adaptation or explicit penalties can INDUCE strong transfer
of a hidden trait through semantically unrelated carrier data -- and
adding more carrier data (an implicit regularizer) strengthens the
covert transfer further.

Paper: "Regularization Amplifies Subliminal Learning" (Feng,
Springer, Ghosal, Zhong & Raghunathan). The authors systematically
study whether constraining adaptation suppresses covert influence
(subliminal learning, phantom transfer, logit-linear selection) and
find the opposite: regularization amplifies the transfer of hidden
biases. Subliminal learning is typically absent under full
fine-tuning, but low-rank adaptation or explicit penalties can induce
strong transfer; phantom transfer and logit-linear selection show
similar trends; adding more data, an implicit form of regularization,
also strengthens covert influence. Their mechanism hypothesis --
simplicity biases induced by regularization may facilitate covert
influence, and the optimal regularization level may depend on task
complexity -- is a hypothesis, NOT established, and is not asserted
here (see the Frobenius-penalty null below for why the distinction
matters in this toy).

Toy operationalization: a linear student world where the trait is
measurable exactly (cf. the SVD toy of PR #240 and the Fisher toy of
PR #241). The student is a linear map whose update dW (M x D, M = 36,
D = 32) starts at zero; the teacher's map differs from the student's
init by a residual R with KNOWN singular structure:

  R = R_base + GAMMA * u a^T

  trait   GAMMA = 3.0, a single rank-1 component: output direction u,
          input direction a. It is the LARGEST single component of R
          (the simplest explanation of the teacher's behavior) but a
          minority of R's total energy.
  base    R_base: 30 components, singular values
          [2.4, 2.1, 1.8, 1.5, 1.2, 1.0, then 0.55 x 24] on random
          orthonormal axes (u, a excluded) -- the carrier task's real,
          higher-rank content.

Carrier data: x ~ N(0, I_D) (semantically unrelated to the trait: the
trait readout directions are never singled out in the inputs),
targets y = R x + eta, eta per-example teacher idiosyncrasy
(sigma = 1.2) that no student can generalize. Every arm trains on the
SAME carrier set per seed with matched full-batch gradient budget:

  full        dW unconstrained, plain GD (500 steps, lr 0.2)
  lora2/lora6 dW = B A factorized, rank 2 / rank 6 (1000 steps,
              lr 0.1, A ~ 0.1 * N(0, I), B = 0)
  nuc         full dW + nuclear-norm penalty (proximal GD, singular
              values soft-thresholded by lr * 1.0 per step) -- an
              explicit penalty that biases toward low rank
  fro         full dW + Frobenius weight decay toward init (0.02) --
              an explicit penalty that does NOT bias toward low rank
  clean       teacher WITHOUT the trait (R_base only), trained with
              the strongest-constraint arm (lora2)

Metrics per seed (held-out carriers: 2048 fresh inputs):
  cos     cos(dW, u a^T): the trait's share of the installed update
          (the toy form of the EAS alignment of PR #240 -- the
          quantity a behavioral preference readout sees, since the
          trait must compete with everything else the update does)
  gain    u^T dW a: the trait component's installed strength
          (GAMMA = 3.0 is full recovery)
  R^2     held-out carrier fit of dW against noiseless targets

Measured on pop-os (particlegan 0.10.1), seeds 0-2:
  cos:  full 0.46-0.48, lora2 0.75-0.76, lora6 0.56-0.58,
        nuc 0.56-0.60, fro 0.46-0.48 (== full), clean |cos| <= 0.001
  gain: full 2.93-3.13, lora2 2.90-3.05, lora6 2.90-3.12,
        nuc 2.39-2.61, clean |gain| <= 0.003
  R^2:  full 0.77-0.79, lora2 0.40-0.41, lora6 0.68, nuc 0.75-0.76
  data sweep at fixed rank 2 (carriers 64 -> 256 -> 1024):
        cos 0.53-0.61 -> 0.70-0.78 -> 0.77-0.78 (endpoint-ordered on
        every seed, +0.17 to +0.25; seed means strictly increasing),
        gain 2.23-2.45 -> 2.65-3.05 -> 2.96-3.01
  budget guard: the full arm at 4x steps is unchanged (cos within
        0.01, gain still ~3.0) -- the full-rank deficit is the
        converged solution's composition, not a training-budget
        artifact.

What this toy does and does NOT reproduce (the re-scope): the
constraint-opens-the-channel ORDERING reproduces with clear margins,
the clean-teacher control installs nothing in any arm, and more data
at fixed rank strengthens the installed trait. But the full-rank arm
recovers the trait COMPONENT at full gain (~3.0) -- in a linear world
the trait is not absent from the full-rank weights; it is DILUTED,
one component among thirty, so its share of the update (cos 0.46-0.48)
is barely half the rank-2 arm's (0.75-0.76). The paper's "absent
under full fine-tuning" is a behavioral statement (trait preference
not expressed); the toy's cos is the behavioral-facing quantity, and
on it the effect holds, at reduced strength. The penalty result is
also narrower than the paper's: only the simplicity-biasing
(nuclear-norm) penalty opens the channel; isotropic Frobenius decay
toward init provably cannot change cos (it rescales the whole update)
and empirically does not (within 0.002 of full on every seed) --
asserted below as a documented null, not hidden. Assertions encode
exactly the demonstrated margins, no more. Skips cleanly when the
installed particlegan does not provide the EMPF recipe. Pure
numpy/torch linear algebra -- no GAN machinery, because the claim is
about how constraints shape a distilled update, not about GAN
training.
"""
import functools

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


D = 32             # input dimension
M = 36             # output dimension
GAMMA = 3.0        # trait singular value (largest single component)
BETAS = np.array([2.4, 2.1, 1.8, 1.5, 1.2, 1.0] + [0.55] * 24)
SIGMA = 1.2        # per-example teacher idiosyncrasy noise
N_CARRIER = 256    # carrier examples for the main arms
STEPS_FULL = 500   # full / penalty arms: GD steps at LR_FULL
LR_FULL = 0.2
STEPS_LORA = 1000  # factorized arms: GD steps at LR_LORA
LR_LORA = 0.1
WD_FRO = 0.02      # Frobenius weight-decay strength (toward init)
LAM_NUC = 1.0      # nuclear-norm penalty strength
SWEEP_NS = (64, 256, 1024)
SEEDS = (0, 1, 2)


def _world(rng, gamma=GAMMA):
    """Teacher residual R = R_base + gamma * u a^T on random axes."""
    U, _ = np.linalg.qr(rng.standard_normal((M, M)))
    V, _ = np.linalg.qr(rng.standard_normal((D, D)))
    u, a = U[:, 0], V[:, 0]
    r_base = (U[:, 1:len(BETAS) + 1] * BETAS) @ V[:, 1:len(BETAS) + 1].T
    return r_base + gamma * np.outer(u, a), u, a, r_base


def _carriers(rng, residual, n):
    x = rng.standard_normal((n, D))
    y = x @ residual.T + SIGMA * rng.standard_normal((n, M))
    return (torch.tensor(x, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64))


def _gd_full(x, y, steps=STEPS_FULL, wd=0.0):
    w = torch.zeros(M, D, dtype=torch.float64)
    n = len(x)
    for _ in range(steps):
        w -= LR_FULL * (-(2.0 / n) * (y - x @ w.T).T @ x + 2 * wd * w)
    return w.numpy()


def _gd_lora(x, y, rank, rng, steps=STEPS_LORA):
    n = len(x)
    a = torch.tensor(rng.standard_normal((rank, D)) * 0.1,
                     dtype=torch.float64)
    b = torch.zeros(M, rank, dtype=torch.float64)
    for _ in range(steps):
        e = y - x @ (b @ a).T
        g_b = -(2.0 / n) * e.T @ x @ a.T
        g_a = -(2.0 / n) * b.T @ e.T @ x
        b -= LR_LORA * g_b
        a -= LR_LORA * g_a
    return (b @ a).numpy()


def _gd_nuc(x, y, lam=LAM_NUC):
    w = torch.zeros(M, D, dtype=torch.float64)
    n = len(x)
    for _ in range(STEPS_FULL):
        w -= LR_FULL * (-(2.0 / n) * (y - x @ w.T).T @ x)
        u_, s, vt = torch.linalg.svd(w, full_matrices=False)
        w = (u_ * torch.clamp(s - LR_FULL * lam, min=0.0)) @ vt
    return w.numpy()


def _metrics(w, u, a, x_te, y_te):
    cos = float(np.sum(w * np.outer(u, a)) / np.linalg.norm(w))
    gain = float(u @ w @ a)
    r2 = 1.0 - (float(np.mean((x_te @ w.T - y_te) ** 2))
                / float(np.mean(y_te ** 2)))
    return dict(cos=cos, gain=gain, r2=r2)


@functools.lru_cache(maxsize=None)
def run_seed(seed):
    rng = np.random.default_rng(5000 + seed)
    residual, u, a, r_base = _world(rng)
    x_te = rng.standard_normal((2048, D))
    y_te = x_te @ residual.T
    y_te_clean = x_te @ r_base.T

    x, y = _carriers(np.random.default_rng(6000 + seed), residual,
                     N_CARRIER)
    arms = {
        "full": _gd_full(x, y),
        "full_long": _gd_full(x, y, steps=4 * STEPS_FULL),
        "lora2": _gd_lora(x, y, 2, np.random.default_rng(900 + seed)),
        "lora6": _gd_lora(x, y, 6, np.random.default_rng(901 + seed)),
        "nuc": _gd_nuc(x, y),
        "fro": _gd_full(x, y, wd=WD_FRO),
    }
    out = {k: _metrics(w, u, a, x_te, y_te) for k, w in arms.items()}

    # Clean-teacher control under the strongest-constraint arm.
    x_c, y_c = _carriers(np.random.default_rng(7000 + seed), r_base,
                         N_CARRIER)
    w_clean = _gd_lora(x_c, y_c, 2, np.random.default_rng(902 + seed))
    out["clean"] = _metrics(w_clean, u, a, x_te, y_te_clean)

    # Data-size sweep at fixed rank 2.
    sweep = {}
    for n in SWEEP_NS:
        x_s, y_s = _carriers(np.random.default_rng(11000 + seed),
                             residual, n)
        w_s = _gd_lora(x_s, y_s, 2, np.random.default_rng(903 + seed))
        sweep[n] = _metrics(w_s, u, a, x_te, y_te)
    out["sweep"] = sweep
    return out


def test_constraint_opens_the_covert_channel():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        full, lora2, lora6 = r["full"], r["lora2"], r["lora6"]
        nuc, fro, clean = r["nuc"], r["fro"], r["clean"]

        # Clean-teacher control: no trait in the teacher, no trait in
        # the student, under the strongest constraint.
        assert abs(clean["cos"]) <= 0.05, (
            f"seed {seed}: clean control cos {clean['cos']:.3f}")
        assert abs(clean["gain"]) <= 0.1, (
            f"seed {seed}: clean control gain {clean['gain']:.3f}")
        assert clean["r2"] >= 0.3, (
            f"seed {seed}: clean control carrier R^2 {clean['r2']:.3f}"
            " -- the control arm itself failed to train")

        # Carrier fit: every trait arm genuinely trains on the
        # carriers (the trait gap is not a training failure).
        assert full["r2"] >= 0.7, (
            f"seed {seed}: full carrier R^2 {full['r2']:.3f}")
        assert lora6["r2"] >= 0.6, (
            f"seed {seed}: lora6 carrier R^2 {lora6['r2']:.3f}")
        assert lora2["r2"] >= 0.3, (
            f"seed {seed}: lora2 carrier R^2 {lora2['r2']:.3f}")
        assert nuc["r2"] >= 0.65, (
            f"seed {seed}: nuc carrier R^2 {nuc['r2']:.3f}")

        # The trait component IS recovered by the constrained arms
        # (gain near GAMMA) -- and, the documented re-scope, by the
        # full arm too: what full-rank training fails to produce is
        # trait DOMINANCE of the update, not the component itself.
        assert lora2["gain"] >= 2.4, (
            f"seed {seed}: lora2 trait gain {lora2['gain']:.3f}")
        assert full["gain"] >= 2.4, (
            f"seed {seed}: full trait gain {full['gain']:.3f} -- if "
            "this fails the toy's character has changed; the claim "
            "tested is dilution, not component absence")

        # Headline: the constraint opens the channel. Full-rank's
        # installed trait is weak (cos <= 0.55); rank and the
        # simplicity-biasing penalty install it clearly above full.
        assert full["cos"] <= 0.55, (
            f"seed {seed}: full-rank installed trait cos "
            f"{full['cos']:.3f} -- should stay diluted")
        assert lora2["cos"] >= 0.65, (
            f"seed {seed}: lora2 installed trait cos {lora2['cos']:.3f}")
        assert lora2["cos"] >= full["cos"] + 0.20, (
            f"seed {seed}: lora2 cos {lora2['cos']:.3f} vs full "
            f"{full['cos']:.3f}")
        assert lora6["cos"] >= full["cos"] + 0.05, (
            f"seed {seed}: lora6 cos {lora6['cos']:.3f} vs full "
            f"{full['cos']:.3f}")
        assert nuc["cos"] >= full["cos"] + 0.05, (
            f"seed {seed}: nuclear-penalty cos {nuc['cos']:.3f} vs "
            f"full {full['cos']:.3f}")
        assert nuc["gain"] >= 2.0, (
            f"seed {seed}: nuclear-penalty trait gain {nuc['gain']:.3f}")

        # Documented null: isotropic Frobenius decay toward init
        # rescales the update without rotating it toward the trait --
        # not every explicit penalty opens the channel.
        assert abs(fro["cos"] - full["cos"]) <= 0.05, (
            f"seed {seed}: frobenius cos {fro['cos']:.3f} vs full "
            f"{full['cos']:.3f} -- the documented null moved")


def test_more_data_strengthens_and_budget_does_not_explain():
    _empf_recipe()
    cos_means = []
    for seed in SEEDS:
        r = run_seed(seed)
        sweep = r["sweep"]
        # More carrier data at fixed low rank strengthens the
        # installed trait (endpoint-ordered on every seed).
        assert sweep[1024]["cos"] >= sweep[64]["cos"] + 0.10, (
            f"seed {seed}: sweep cos n=64 {sweep[64]['cos']:.3f} -> "
            f"n=1024 {sweep[1024]['cos']:.3f}")
        assert sweep[1024]["gain"] >= sweep[64]["gain"] + 0.3, (
            f"seed {seed}: sweep gain n=64 {sweep[64]['gain']:.3f} -> "
            f"n=1024 {sweep[1024]['gain']:.3f}")
        cos_means.append([sweep[n]["cos"] for n in SWEEP_NS])

        # Budget guard: 4x training does not rescue the full-rank
        # arm -- its deficit is the converged update's composition.
        assert abs(r["full_long"]["cos"] - r["full"]["cos"]) <= 0.05, (
            f"seed {seed}: full cos at 4x budget "
            f"{r['full_long']['cos']:.3f} vs {r['full']['cos']:.3f}")
        assert r["full_long"]["cos"] <= 0.55, (
            f"seed {seed}: full cos at 4x budget "
            f"{r['full_long']['cos']:.3f}")
    means = np.mean(cos_means, axis=0)
    assert means[0] < means[1] < means[2], (
        f"sweep seed-mean cos not increasing: "
        f"{[round(float(m), 3) for m in means]}")
