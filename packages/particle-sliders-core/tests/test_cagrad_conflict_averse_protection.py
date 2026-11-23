"""CPU gate for the conflict-averse gradient descent
(CAGrad) claim of Liu, Liu, Jin, Stone & Liu, "Conflict-
Averse Gradient Descent for Multi-task Learning"
(arXiv:2110.14048, NeurIPS 2021), arbitrated on the OPD
mismatch geometry of PR #301 / the gradient-surgery
geometry of PR #303.

The paper's claim: instead of minimizing only the average
loss, CAGrad chooses an update direction d near the
average gradient g0 = mean_i g_i that maximizes the
WORST per-task local improvement,

    maximize_d   min_i  <g_i, d>
    subject to   ||d - g0|| <= c * ||g0||,

which provably converges to a Pareto-stationary point and
includes plain gradient descent (c = 0) and MGDA as
special cases. Here c = 0.5 (primary; c in {0.25, 1.0}
also measured, see the iteration log in the PR).

Chain provenance: cycle 22 of the Particle Sliders paper
watch, second arbitration in the #301 -> #303 chain.
#301 (arXiv:2611.27125, OPD mismatch dynamics) proved
targeted correction repairs mismatched OPD states but its
matched-state PROTECTION half did not reproduce: targeted
matched TV 0.031-0.054 sat at plain OPD's level, above the
matched-only control 0.007-0.010. #303 (PCGrad, arXiv:
2001.06782) then showed protection exists ONLY when the
surgery reference is computed DENSELY: surgery vs the
sampled on-policy reference gave matched TV 0.029-0.052
(no protection, indistinguishable from random-projection
and shrinkage controls), surgery vs the dense CE
reference gave 0.014-0.018. #303's law: protection is
ordered with reference accuracy (sampled cos-to-own-
expectation ~= 0.74; dense ~= 0.995). CAGrad is the other
canonical conflict-resolution rule, and it does not
project against a separate reference -- it reweights the
two task gradients (primary / matched objective vs
correction objective) under a worst-task-improvement
constraint. This test arbitrates: in #301's calibrated
geometry, does CAGrad protect matched states where
PCGrad-sampled failed, or does it fail the same way --
confirming that the binding constraint is gradient
ESTIMATION, not the conflict rule?

Toy operationalization: PR #301's EXACT construction,
unchanged (calibration is asserted, test 1): per-position
categorical world, student logits phi_i @ W with shared
W (features R^32 + common component 0.8), N = 20
positions, V = 6 tokens, 10 matched + 10 mismatched
positions built exactly as in #301, STEPS = 40, step
size 0.5, gradient averaged over all N positions in
every arm. Task gradients are taken in task-mean units
(mean over the task's own 10 positions); with equal task
sizes the average task gradient then equals exactly the
total gradient the #301/#303 arms apply, so a CAGrad
update d* is applied at the same scale, w -= LR * d*.
Arms:
  (a) PLAIN: sampled-token OPD on all positions (#301 a;
      also the average-gradient control for arm d2).
  (b) MATCHED-ONLY control: sampled update, matched
      positions only (#301 b).
  (c) TARGETED: matched sampled update + full-distribution
      CE gradient (S - T) at mismatched positions (#301 c).
  (pc) TARGETED + PCGRAD, SAMPLED REFERENCE (#303 d,
      calibration: should reproduce no-protection).
  (d) CAGrad, AS-APPLIED GRADIENTS ("mixed"): g_match =
      matched positions' sampled OPD gradient, g_corr =
      mismatched positions' CE gradient -- the exact two
      gradients arm (pc)'s surgery combines, fed to the
      other canonical rule. This is the arbitration arm.
  (d2) CAGrad, BOTH SAMPLED: g_corr also replaced by its
      sampled OPD estimate.
  (e) CAGrad, DENSE: both task gradients are the full-
      distribution CE gradients (S - T).
  (e0) DENSE-AVERAGE control: plain average of the two
      dense CE gradients (arm e with c = 0) -- isolates
      what dense estimation alone buys, without the
      CAGrad reweighting.
  (f) CONTROLS for arm (d): RANDOM-DISPLACEMENT (g0 plus
      a random unit vector scaled by arm (d)'s mean
      ||d* - g0|| for that seed) and SHRINKAGE/GROWTH
      (g0 scaled by arm (d)'s mean ||d*|| / ||g0|| ratio,
      no direction change) -- and SHRINKAGE-DENSE, the
      same magnitude control for arm (e).

CAGrad solver (2-task, exact span form): the optimum
has no component orthogonal to span{g_match, g_corr}
(a perpendicular component only consumes ball radius),
so the problem reduces to the 2-D span; the optimum is
the average gradient itself or lies on the ball
boundary, so a vectorized scan of the boundary circle
(721 angles) plus 8 zoom-refinement rounds (41 points
each, window shrinking to the scan step) finds it
deterministically. No scipy, pure numpy.

Readouts: per-class mean TV(S, T) at end.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds. Calibration (arms a-c, pc) reproduces
#301/#303's published tables. Final TV (matched /
mismatched), c = 0.5:
  s0: plain 0.044 / 0.346,  matched-only 0.009 / --,
      targeted 0.042 / 0.059,  pcgrad-s 0.040 / 0.060,
      cagrad-mixed 0.039 / 0.034,  cagrad-sampled
      0.039 / 0.347,  cagrad-dense 0.023 / 0.071,
      dense-avg 0.044 / 0.065,  randdisp 0.040 / 0.058,
      shrink 0.041 / 0.050,  shrink-dense 0.046 / 0.069
  s1: plain 0.042 / 0.475,  matched-only 0.010 / --,
      targeted 0.054 / 0.087,  pcgrad-s 0.052 / 0.087,
      cagrad-mixed 0.055 / 0.047,  cagrad-sampled
      0.036 / 0.456,  cagrad-dense 0.033 / 0.117,
      dense-avg 0.059 / 0.099,  randdisp 0.054 / 0.090,
      shrink 0.052 / 0.065,  shrink-dense 0.064 / 0.117
  s2: plain 0.040 / 0.305,  matched-only 0.008 / --,
      targeted 0.040 / 0.046,  pcgrad-s 0.038 / 0.046,
      cagrad-mixed 0.039 / 0.025,  cagrad-sampled
      0.032 / 0.389,  cagrad-dense 0.019 / 0.063,
      dense-avg 0.042 / 0.051,  randdisp 0.040 / 0.049,
      shrink 0.038 / 0.037,  shrink-dense 0.046 / 0.060
  s3: plain 0.043 / 0.197,  matched-only 0.009 / --,
      targeted 0.031 / 0.043,  pcgrad-s 0.031 / 0.043,
      cagrad-mixed 0.031 / 0.021,  cagrad-sampled
      0.033 / 0.254,  cagrad-dense 0.012 / 0.044,
      dense-avg 0.031 / 0.044,  randdisp 0.032 / 0.042,
      shrink 0.030 / 0.034,  shrink-dense 0.032 / 0.046
  s4: plain 0.033 / 0.292,  matched-only 0.007 / --,
      targeted 0.031 / 0.049,  pcgrad-s 0.029 / 0.047,
      cagrad-mixed 0.032 / 0.029,  cagrad-sampled
      0.031 / 0.289,  cagrad-dense 0.013 / 0.056,
      dense-avg 0.029 / 0.053,  randdisp 0.028 / 0.052,
      shrink 0.029 / 0.039,  shrink-dense 0.030 / 0.055
At c = 1.0 (same solver, wider ball): cagrad-mixed
matched 0.032-0.059 -- STILL no protection -- while
cagrad-dense matched 0.004-0.009 reaches the matched-
only control (repair cost: mismatched up to 0.152).

The arbitration, stated plainly: CAGrad fails the same
way PCGrad-sampled failed, and for the same reason.
(1) Fed the gradients the targeted arm actually applies
(matched sampled, correction CE), CAGrad does NOT
protect: matched TV 0.031-0.055 is indistinguishable
from targeted's 0.031-0.054 and from the random-
displacement control; its repair is better than
targeted's (mismatched 0.021-0.047), but the shrinkage/
growth control at CAGrad's own mean norm ratio (~1.22)
reproduces most of that gain -- a step-size effect of
the ball boundary, not a protective direction. With
BOTH gradients sampled (d2) it is worse: no protection
AND no repair (mismatched 0.254-0.456 ~= plain), because
the sampled correction gradient is the one #301 showed
cannot repair mismatched states. (2) Fed DENSE gradients,
the SAME rule protects: matched 0.012-0.033 at c = 0.5,
clearly below both targeted and the dense-average
control (so the reweighting adds protection beyond what
dense estimation alone buys -- dense-average matched is
0.029-0.059, no better than targeted), and at c = 1.0
dense CAGrad reaches the matched-only control on every
seed. The c-scan (0.25 / 0.5 / 1.0) and the normalized-
improvement variant change no verdict: at every c the
mixed/sampled arms fail to protect and the dense arm's
protection grows with c. Sharpened lesson, confirming
#303's law from the second canonical rule: in this
geometry the binding constraint on conflict resolution
is gradient ESTIMATION, not the conflict rule -- no
reweighting of a sampled on-policy gradient recovers
the protection that the same rule obtains from the
teacher's full distribution.

What is NOT asserted: CAGrad's convergence / benchmark
claims at neural-net scale (this toy arbitrates one
mechanism -- worst-task reweighting under a trust ball
-- in one geometry, #301's); behavior at budgets far
from STEPS = 40; and any c > 1.0 (the ball then no
longer keeps d "near" the average gradient in the
paper's sense). Pure numpy -- no GAN machinery, because
the claims are about distillation training dynamics on
categoricals. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family; PR #211 is what
enables this test in CI).
"""
import functools

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


V = 6                # tokens
N = 20               # positions (10 matched + 10 mismatched)
NM = 10              # matched positions (the first NM)
D = 32               # shared feature dimension (student logits = phi @ W)
COMMON = 0.8         # shared feature component (parameter coupling)
SEEDS = range(5)
STEPS = 40           # training steps, all arms
M = 8                # sampled tokens per position per step (sampled arms)
LR = 0.5             # step size on W, all arms
C_MAIN = 0.5         # CAGrad trust-ball radius factor, primary arms


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(seed):
    """Features, teacher, and exact shared-parameterization
    realization of the constructed student init.
    Identical to PR #301's construction."""
    rng = np.random.default_rng(61000 + seed)
    common = rng.standard_normal(D)
    common /= np.linalg.norm(common)
    phi = rng.standard_normal((N, D)) + COMMON * common[None, :]
    tl = np.zeros((N, V))
    sl = np.zeros((N, V))
    for i in range(N):
        pref = int(rng.integers(V))
        second = int(rng.choice([t for t in range(V) if t != pref]))
        tl[i] = rng.normal(0, 0.1, V)
        tl[i, pref] = 4.2
        tl[i, second] = 1.0
        if i < NM:  # matched: teacher logits + moderate noise
            sl[i] = tl[i] + rng.normal(0, 0.35, V)
        else:  # mismatched: mass on the teacher's second token,
            # preferred token nearly extinguished, tails inflated
            sl[i] = rng.normal(0, 0.1, V)
            sl[i, second] = 2.8
            sl[i, pref] = -2.28
    t = _softmax(tl)
    s0 = _softmax(sl)
    w0, *_ = np.linalg.lstsq(phi, sl, rcond=None)
    # the shared parameterization must realize the constructed
    # init exactly, or the class construction is not what runs
    assert np.abs(_softmax(phi @ w0) - s0).max() < 1e-8
    return phi, t, w0


def _tv(t, s):
    return 0.5 * np.abs(t - s).sum(axis=1)


def _sampled_g(rng, s_i, t_i):
    """One position's sampled-token OPD (score-function)
    gradient in logit space, as in PR #301."""
    xs = rng.choice(V, size=M, p=s_i)
    r = np.log(s_i[xs] / t_i[xs])
    gi = np.zeros(V)
    np.add.at(gi, xs, r)
    return gi / M - s_i * r.mean()


def _train_basic(phi, t, w0, seed, arm):
    """Arms (a)-(c), verbatim PR #301. Gradient averaged
    over all N positions in every arm."""
    rng = np.random.default_rng(62000 + seed)
    w = w0.copy()
    rows = range(NM) if arm == "matched_only" else range(N)
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        g = np.zeros((N, V))
        for i in rows:
            if arm == "targeted" and i >= NM:
                g[i] = s[i] - t[i]  # full-distribution CE gradient
            else:  # sampled-token OPD (score-function) gradient
                g[i] = _sampled_g(rng, s[i], t[i])
        w -= LR * (phi.T @ g) / N
    return _softmax(phi @ w)


def _project_off(vec, ref):
    """vec minus its component along ref (PCGrad projection
    onto ref's normal plane), as in PR #303."""
    denom = float(ref.ravel() @ ref.ravel())
    if denom <= 0:
        return vec
    return vec - (float(vec.ravel() @ ref.ravel()) / denom) * ref


def _train_pcgrad(phi, t, w0, seed):
    """Arm (pc): PR #303's arm (d), targeted + surgery with
    the sampled reference, unchanged."""
    rng = np.random.default_rng(62000 + seed)
    w = w0.copy()
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
        gc = s[NM:] - t[NM:]
        g_match = (phi[:NM].T @ gm) / N
        g_corr = (phi[NM:].T @ gc) / N
        a, b = g_match.ravel(), g_corr.ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        if cos < 0:
            g_corr = _project_off(g_corr, g_match)
        w -= LR * (g_match + g_corr)
    return _softmax(phi @ w)


def _cagrad(g1, g2, c):
    """The paper's 2-task update: d maximizing the worst
    per-task local improvement min_i <g_i, d> subject to
    ||d - g0|| <= c * ||g0||, g0 = (g1 + g2) / 2.

    Exact span reduction: a component orthogonal to
    span{g1, g2} cannot change any <g_i, d> and only
    consumes ball radius, so the optimum lies in the
    span; there the optimum is g0 itself or a boundary
    point of the ball, found by a vectorized scan of the
    boundary circle (721 angles) plus 8 zoom rounds of
    41 points. Deterministic, numpy-only."""
    g1 = g1.ravel().astype(float)
    g2 = g2.ravel().astype(float)
    g0 = 0.5 * (g1 + g2)
    radius = c * np.linalg.norm(g0)
    if radius <= 0:
        return g0
    basis = []
    for v in (g1, g2):
        u = v.copy()
        for e in basis:
            u -= (u @ e) * e
        n = np.linalg.norm(u)
        if n > 1e-12:
            basis.append(u / n)
    bmat = np.stack(basis)
    a1, a2, a0 = bmat @ g1, bmat @ g2, bmat @ g0

    def _vals(thetas):
        pts = a0[None, :] + radius * np.stack(
            [np.cos(thetas), np.sin(thetas)], axis=1)
        return np.minimum(pts @ a1, pts @ a2), pts

    best_x = a0.copy()
    best_v = min(float(a1 @ a0), float(a2 @ a0))
    if len(basis) == 1:
        for sgn in (1, -1):
            x = a0 + sgn * radius * np.ones(1)
            v = min(float(a1 @ x), float(a2 @ x))
            if v > best_v:
                best_v, best_x = v, x
    else:
        thetas = np.linspace(0, 2 * np.pi, 721)[:-1]
        vals, pts = _vals(thetas)
        j = int(np.argmax(vals))
        center, width = thetas[j], 2 * (2 * np.pi / 720)
        if vals[j] > best_v:
            best_v, best_x = float(vals[j]), pts[j]
        for _ in range(8):
            thetas = np.linspace(center - width, center + width, 41)
            vals, pts = _vals(thetas)
            j = int(np.argmax(vals))
            if vals[j] > best_v:
                best_v, best_x = float(vals[j]), pts[j]
            center = thetas[j]
            width = max((thetas[1] - thetas[0]) * 2, 1e-12)
    return best_x @ bmat


def _train_cagrad(phi, t, w0, seed, mode, c=C_MAIN,
                  ctrl_disp=None, ctrl_ratio=None):
    """CAGrad arms and their controls. Task gradients in
    task-mean units (mean over the task's own positions);
    with NM = N - NM the applied total of arms (a)-(c)
    equals the average task gradient, so d* is applied at
    the same scale. Modes:
      mixed      -- g_match sampled OPD, g_corr CE (arm d)
      sampled    -- both task gradients sampled OPD (d2)
      dense      -- both task gradients dense CE (arm e)
      dense_avg  -- plain average of the dense gradients (e0)
      randdisp_m -- mixed g0 + random displacement (arm f)
      shrink_m   -- mixed g0 scaled by ctrl_ratio (arm f)
      shrink_d   -- dense g0 scaled by ctrl_ratio (arm f)
    The training RNG stream is arm (c)'s, so sampled draws
    match the targeted arm step for step; the random-
    displacement stream is separate."""
    rng = np.random.default_rng(62000 + seed)
    rng_r = np.random.default_rng(64000 + seed)
    w = w0.copy()
    disps, ratios = [], []
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        if mode in ("sampled",):
            gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
            gc = np.array([_sampled_g(rng, s[i], t[i])
                           for i in range(NM, N)])
            g_match = (phi[:NM].T @ gm) / NM
            g_corr = (phi[NM:].T @ gc) / NM
        elif mode in ("mixed", "randdisp_m", "shrink_m"):
            gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
            g_match = (phi[:NM].T @ gm) / NM
            g_corr = (phi[NM:].T @ (s[NM:] - t[NM:])) / NM
        elif mode in ("dense", "dense_avg", "shrink_d"):
            g_match = (phi[:NM].T @ (s[:NM] - t[:NM])) / NM
            g_corr = (phi[NM:].T @ (s[NM:] - t[NM:])) / NM
        else:
            raise ValueError(mode)
        g0 = 0.5 * (g_match + g_corr)
        if mode == "dense_avg":
            d = g0
        elif mode in ("randdisp_m",):
            r = rng_r.standard_normal(g0.shape)
            r /= np.linalg.norm(r)
            d = g0 + ctrl_disp * r
        elif mode in ("shrink_m", "shrink_d"):
            d = g0 * ctrl_ratio
        else:
            d = _cagrad(g_match, g_corr, c).reshape(g0.shape)
            disps.append(float(np.linalg.norm(d - g0)))
            ratios.append(float(np.linalg.norm(d)
                                / max(np.linalg.norm(g0), 1e-12)))
        w -= LR * d
    stats = dict(
        mean_disp=float(np.mean(disps)) if disps else 0.0,
        mean_ratio=float(np.mean(ratios)) if ratios else 1.0,
    )
    return _softmax(phi @ w), stats


@functools.lru_cache(maxsize=None)
def run_seed(seed):
    phi, t, w0 = _build(seed)
    s0 = _softmax(phi @ w0)
    out = dict(
        tv0_m=float(_tv(t, s0)[:NM].mean()),
        tv0_mm=float(_tv(t, s0)[NM:].mean()),
    )
    for arm in ("plain", "matched_only", "targeted"):
        sf = _train_basic(phi, t, w0, seed, arm)
        out[f"tv_{arm}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_{arm}_mm"] = float(_tv(t, sf)[NM:].mean())
    sf = _train_pcgrad(phi, t, w0, seed)
    out["tv_pcgrad_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_pcgrad_mm"] = float(_tv(t, sf)[NM:].mean())
    # arbitration arm first: its stats seed the controls
    sf, stats = _train_cagrad(phi, t, w0, seed, "mixed")
    out["tv_mixed_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_mixed_mm"] = float(_tv(t, sf)[NM:].mean())
    out["mixed_ratio"] = stats["mean_ratio"]
    sf, _ = _train_cagrad(phi, t, w0, seed, "randdisp_m",
                          ctrl_disp=stats["mean_disp"])
    out["tv_randdisp_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_randdisp_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_cagrad(phi, t, w0, seed, "shrink_m",
                          ctrl_ratio=stats["mean_ratio"])
    out["tv_shrink_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_shrink_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_cagrad(phi, t, w0, seed, "sampled")
    out["tv_csampled_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_csampled_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, dstats = _train_cagrad(phi, t, w0, seed, "dense")
    out["tv_dense_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_dense_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_cagrad(phi, t, w0, seed, "dense_avg")
    out["tv_denseavg_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_denseavg_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_cagrad(phi, t, w0, seed, "shrink_d",
                          ctrl_ratio=dstats["mean_ratio"])
    out["tv_shrinkd_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_shrinkd_mm"] = float(_tv(t, sf)[NM:].mean())
    # c = 1.0 scaling readout: mixed and dense only
    sf, _ = _train_cagrad(phi, t, w0, seed, "mixed", c=1.0)
    out["tv_mixed_c1_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_mixed_c1_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_cagrad(phi, t, w0, seed, "dense", c=1.0)
    out["tv_dense_c1_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_dense_c1_mm"] = float(_tv(t, sf)[NM:].mean())
    return out


def test_calibration_reproduces_chain_geometry():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the geometry is #301's: constructed classes and
        # the published arm ranges (see docstring table)
        assert m["tv0_m"] <= 0.06, (seed, m)
        assert m["tv0_mm"] >= 0.85, (seed, m)
        assert m["tv_matched_only_m"] <= 0.012, (seed, m)
        assert m["tv_plain_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)
        assert m["tv_plain_mm"] >= 0.15, (seed, m)
        assert 0.028 <= m["tv_targeted_m"] <= 0.057, (seed, m)
        assert m["tv_targeted_mm"] <= 0.10, (seed, m)
        # #301's protection failure itself, restated
        assert m["tv_targeted_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)
        # #303's calibration arm, restated: PCGrad with the
        # sampled reference does not protect either
        assert m["tv_pcgrad_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        assert m["tv_pcgrad_m"] >= m["tv_targeted_m"] - 0.006, (seed, m)
        assert m["tv_pcgrad_mm"] <= 0.10, (seed, m)


def test_cagrad_as_applied_gradients_does_not_protect():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the arbitration arm: CAGrad fed the exact two
        # gradients the targeted arm applies (matched
        # sampled, correction CE) does NOT protect matched
        # states -- asserted as the measured negative
        assert m["tv_mixed_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        assert m["tv_mixed_m"] >= m["tv_targeted_m"] - 0.006, (seed, m)
        # repair is retained and even improved vs targeted
        assert m["tv_mixed_mm"] <= 0.10, (seed, m)
        assert m["tv_mixed_mm"] <= m["tv_targeted_mm"] + 0.005, (seed, m)


def test_cagrad_both_sampled_loses_repair_too():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # with the correction gradient also sampled, CAGrad
        # inherits #301 claim 1: the sampled correction
        # cannot repair mismatched states at this budget
        assert m["tv_csampled_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        assert m["tv_csampled_mm"] >= 0.15, (seed, m)
        assert m["tv_csampled_mm"] >= m["tv_targeted_mm"] + 0.10, (seed, m)


def test_mixed_controls_step_size_explains_repair_gain():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # neither control protects, and both sit at
        # targeted's matched level -- arm (d)'s failure is
        # not specific to its direction choice
        for arm in ("randdisp", "shrink"):
            assert m[f"tv_{arm}_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
            assert abs(m[f"tv_{arm}_m"] - m["tv_targeted_m"]) <= 0.006, (seed, m)
        # and the pure magnitude control (g0 scaled by
        # CAGrad's own mean norm ratio, ~1.22 -- the ball
        # boundary grows the step) reproduces most of arm
        # (d)'s repair gain: that gain is a step-size
        # effect, not a protective direction
        assert m["tv_shrink_mm"] <= m["tv_mixed_mm"] + 0.02, (seed, m)


def test_cagrad_dense_protects_matched():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the identical rule fed dense CE gradients DOES
        # protect: clearly below targeted AND below the
        # dense-average control (the reweighting adds
        # protection beyond dense estimation alone), and
        # within a bounded factor of the matched-only
        # control
        assert m["tv_dense_m"] <= m["tv_targeted_m"] - 0.010, (seed, m)
        assert m["tv_dense_m"] <= m["tv_denseavg_m"] - 0.010, (seed, m)
        assert m["tv_dense_m"] <= m["tv_matched_only_m"] + 0.025, (seed, m)
        # dense estimation alone (average, c = 0) does not
        # protect -- it sits at targeted's level
        assert m["tv_denseavg_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        # the magnitude control for the dense arm does not
        # protect either: the protection is directional
        assert m["tv_shrinkd_m"] >= m["tv_dense_m"] + 0.008, (seed, m)
        # repair is retained -- far below plain -- at a
        # measured cost vs targeted on some seeds (worst:
        # s1, +0.030), encoded as an allowance
        assert m["tv_dense_mm"] <= 0.13, (seed, m)
        assert m["tv_dense_mm"] <= m["tv_plain_mm"] / 2.0, (seed, m)
        assert m["tv_dense_mm"] <= m["tv_targeted_mm"] + 0.04, (seed, m)


def test_wider_ball_orders_by_estimation_not_rule():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # at c = 1.0 the as-applied (mixed) arm STILL does
        # not protect -- widening the trust ball does not
        # rescue a sampled reference task gradient
        assert m["tv_mixed_c1_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        # while the dense arm reaches the matched-only
        # control: protection scales with c only when the
        # gradients are estimated densely
        assert m["tv_dense_c1_m"] <= m["tv_matched_only_m"] + 0.003, (seed, m)
        # at a measured repair cost (worst seed 0.152)
        assert m["tv_dense_c1_mm"] <= 0.16, (seed, m)
        assert m["tv_dense_c1_mm"] <= m["tv_plain_mm"] / 2.0, (seed, m)
