"""CPU gate for the CurvDelta claim (arXiv:2611.24654): an update
constrained to the low-curvature subspace of a preservation loss
retains a substantial fraction of a target task's gain at a tiny
fraction of the preservation damage of an unconstrained update --
and it is the LOW-curvature subspace specifically that does this.

Paper: "CurvDelta: Low-Curvature Delta Modeling For Surgical
Learning and Unlearning" (Xiao, Liang, Chen, Wang, Chen & Lu).
CurvDelta moves capability preservation into the parameterization
of the update: around the pre-adaptation model, the local geometry
of a preservation loss defines a low-curvature space in which
parameter updates leave that loss nearly invariant; the space is
constructed scalably with K-FAC and a factorized projection, and
CurvLoRA is the low-rank variant in the same geometry. The paper
evaluates on mathematical reasoning, code generation, and knowledge
unlearning (TOFU/MUSE), reporting a favorable acquisition-retention
trade-off and, on unlearning, retention close to the from-scratch
retraining reference. The abstract reports NO headline numbers, so
this test asserts no magnitudes from the paper -- it tests the
geometric mechanism the trade-off rests on.

Toy operationalization: both losses are the local quadratics the
paper's own framing assumes around the pre-adaptation model w = 0.

  preservation loss:  0.5 * w^T H w,  H = Q diag(lam) Q^T with a
      known spread spectrum -- N_SHARP = 8 sharp directions
      (lam ~ U(150, 250)) and 56 flat ones (lam ~ U(0.2, 1.0)) in
      R^64, eigenbasis Q known to the test (the toy stand-in for
      the paper's K-FAC curvature estimate).
  target loss:  0.5 * ||w - w*||^2 for a target optimum w* of
      norm 3. The target Hessian is the identity BY DESIGN: the
      only anisotropy in the problem is the preservation geometry
      under test, so no arm can win or lose through target-side
      curvature alignment. w* is drawn isotropically, so in
      expectation every rank-K subspace holds the same share of
      the target -- the arms differ only in the curvature of the
      subspace they may move in.

Training is projected gradient descent (40 steps, eta = 0.4, which
converges each arm to its subspace optimum), the toy form of the
paper's gradient training under a parameterized constraint -- not
a one-shot closed form. Per seed, metrics are averaged over 4
independent target draws sharing one H (a single draw leaves the
random control's damage needlessly noisy: how much sharp energy a
random subspace happens to capture varies draw to draw).

Arms (all rank/projector structure fixed per seed):
  full    -- unconstrained (identity projector).
  low     -- gradient projected onto the bottom-K eigenspace of H
             (K = 24 flattest directions; the CurvDelta arm, and
             the analogue of CurvLoRA's rank constraint inside the
             preservation geometry).
  random  -- gradient projected onto a fixed random K-dimensional
             subspace (matched rank: any rank constraint alone
             should NOT reproduce the effect).
  sharp   -- gradient projected onto the top-K eigenspace (the
             wrong end of the spectrum: should damage most per
             unit gain).
  matched -- diagnostic, not trained: the full arm's endpoint
             rescaled to exactly the low arm's preservation damage;
             its gain is the frontier control (what unconstrained
             training buys at the same damage budget).

Metrics per seed (averaged over the 4 target draws):
  gain    = reduction in target loss from w = 0,
  damage  = increase in preservation loss (= final preservation
            loss, since w = 0 is its minimum),
  gain / damage.

Measured on pop-os (particlegan 0.10.1): full arm gain 4.500,
damage 101.5-142.0 on every seed. Low arm: gain 1.62-1.84
(36-41% of full) at damage 0.63-0.74 (0.4-0.7% of full),
gain/damage 2.40-2.90. Random arm: gain/damage 0.024-0.055,
damage 26.1-87.2 (wide range: how much sharp energy a random
subspace captures varies by seed even after averaging draws).
Sharp arm: gain/damage 0.013-0.019 -- worst on every seed --
with damage 100.1-140.8, i.e. 98.6-99.1% of the unconstrained
arm's damage despite reaching only a 24-dimensional subspace.
Matched-damage unconstrained gain 0.59-0.76 vs the low arm's
1.62-1.84 at the same damage. Assertions below encode roughly
half the thinnest observed margins, no more. (One re-scope from
prototyping: raw sharp-vs-random DAMAGE does not separate
cleanly -- ratio 1.15 on the thinnest seed -- so the sharp arm's
badness is asserted per unit gain and against the full arm,
where it does separate, not as a raw-damage ordering against
the random control.) Skips cleanly when the installed
particlegan does not provide the EMPF recipe. Pure numpy/torch
geometry -- no GAN machinery, because the claim is about update
geometry, not training dynamics of a GAN.
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


D = 64            # parameter dimension
K = 24            # rank of every projected arm's subspace
N_SHARP = 8       # number of sharp (large-eigenvalue) directions
R = 3.0           # norm of the target optimum w*
DRAWS = 4         # target draws per seed, metrics averaged
STEPS = 40        # projected gradient steps per arm per draw
ETA = 0.4         # step size (target Hessian is I: contraction 0.6/step)


def run_seed(seed):
    rng = np.random.default_rng(12000 + seed)
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    lam = np.empty(D)
    lam[:N_SHARP] = rng.uniform(150.0, 250.0, N_SHARP)
    lam[N_SHARP:] = rng.uniform(0.2, 1.0, D - N_SHARP)
    order = np.argsort(-lam)                      # descending: sharp first
    lam, q_basis = lam[order], q_basis[:, order]

    def projector(cols):
        return cols @ cols.T

    rand_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    arms = {
        "full": np.eye(D),
        "low": projector(q_basis[:, D - K:]),
        "random": projector(rand_basis[:, :K]),
        "sharp": projector(q_basis[:, :K]),
    }

    totals = {name: [0.0, 0.0] for name in arms}  # [gain, damage]
    matched_gain = 0.0
    for _ in range(DRAWS):
        w_star = rng.standard_normal(D)
        w_star *= R / np.linalg.norm(w_star)
        init_loss = 0.5 * float(w_star @ w_star)
        finals = {}
        for name, proj in arms.items():
            w = np.zeros(D)
            for _ in range(STEPS):
                w = w - ETA * (proj @ (w - w_star))
            finals[name] = w
            gain = init_loss - 0.5 * float((w - w_star) @ (w - w_star))
            eig = q_basis.T @ w
            damage = 0.5 * float(lam @ (eig ** 2))
            totals[name][0] += gain
            totals[name][1] += damage
        # Frontier diagnostic: rescale the full endpoint to the low
        # arm's damage for this draw and score its gain.
        eig_full = q_basis.T @ finals["full"]
        eig_low = q_basis.T @ finals["low"]
        dmg_full = 0.5 * float(lam @ (eig_full ** 2))
        dmg_low = 0.5 * float(lam @ (eig_low ** 2))
        alpha = np.sqrt(dmg_low / dmg_full)
        resid = alpha * finals["full"] - w_star
        matched_gain += init_loss - 0.5 * float(resid @ resid)

    out = {}
    for name, (gain, damage) in totals.items():
        gain, damage = gain / DRAWS, damage / DRAWS
        out[name] = dict(gain=gain, damage=damage,
                         gd=gain / damage if damage > 0 else float("inf"))
    out["matched_gain"] = matched_gain / DRAWS
    out["init_loss"] = 0.5 * R * R
    out["sharp_mean"] = float(lam[:N_SHARP].mean())
    out["low_mean"] = float(lam[D - K:].mean())
    return out


def test_low_curvature_update_preserves_at_matched_rank():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        full, low = r["full"], r["low"]
        rand, sharp = r["random"], r["sharp"]

        # Trap guard: the spectrum must genuinely spread -- if the
        # "sharp" and "flat" ends nearly coincide, the toy's
        # subspaces are interchangeable and the test shows nothing.
        assert r["sharp_mean"] > 100.0 * r["low_mean"], (
            f"seed {seed}: mean sharp eigenvalue {r['sharp_mean']:.1f} "
            f"vs mean bottom-K eigenvalue {r['low_mean']:.3f}")

        # Convergence sanity: the unconstrained arm reaches the
        # target optimum, so projected arms' shortfalls are the
        # constraint's doing, not an unconverged optimizer's.
        assert full["gain"] > 0.98 * r["init_loss"], (
            f"seed {seed}: full-arm gain {full['gain']:.3f} of "
            f"initial target loss {r['init_loss']:.3f}")

        # The CurvDelta claim: the low-curvature arm keeps a
        # substantial share of the gain at a tiny share of the
        # damage (measured: 36-41% of gain, 0.4-0.7% of damage).
        assert low["gain"] >= 0.30 * full["gain"], (
            f"seed {seed}: low-arm gain {low['gain']:.3f} vs full "
            f"{full['gain']:.3f}")
        assert low["damage"] <= 0.02 * full["damage"], (
            f"seed {seed}: low-arm damage {low['damage']:.3f} vs "
            f"full {full['damage']:.3f}")

        # It must be the low-curvature subspace specifically: at
        # matched rank the random subspace is far worse per unit
        # gain, and the sharp subspace is worst of all (measured
        # gain/damage: low 2.40-2.90, random 0.024-0.055, sharp
        # 0.013-0.019; random/sharp ratio thinnest at 1.50).
        assert low["gd"] >= 20.0 * rand["gd"], (
            f"seed {seed}: gain/damage low {low['gd']:.3f} vs "
            f"random {rand['gd']:.3f}")
        assert low["gd"] >= 40.0 * sharp["gd"], (
            f"seed {seed}: gain/damage low {low['gd']:.3f} vs "
            f"sharp {sharp['gd']:.3f}")
        assert rand["gd"] >= 1.3 * sharp["gd"], (
            f"seed {seed}: gain/damage random {rand['gd']:.3f} vs "
            f"sharp {sharp['gd']:.3f} -- the wrong end of the "
            f"spectrum should be the worst place to update")
        # The sharp arm is not merely inefficient: confined to a
        # 24-of-64 subspace it still inflicts 98.6-99.1% of the
        # unconstrained arm's preservation damage.
        assert sharp["damage"] >= 0.9 * full["damage"], (
            f"seed {seed}: damage sharp {sharp['damage']:.3f} vs "
            f"full {full['damage']:.3f}")


def test_low_curvature_beats_unconstrained_at_matched_damage():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Frontier form of the claim: rescaling the unconstrained
        # update down to the low arm's damage budget buys far less
        # gain than constraining the update's direction does
        # (measured: matched gain 0.59-0.76 vs low-arm gain
        # 1.62-1.84 at identical damage, ratio 2.2-3.1).
        assert r["low"]["gain"] >= 1.7 * r["matched_gain"], (
            f"seed {seed}: low-arm gain {r['low']['gain']:.3f} vs "
            f"damage-matched unconstrained gain "
            f"{r['matched_gain']:.3f}")
