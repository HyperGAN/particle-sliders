"""CPU gate for the dense-reference refresh pricing probe:
what does the DENSE reference of PR #303 cost when it is
not recomputed every step? Anchors: McCandlish et al.,
"An Empirical Model of Large-Batch Training"
(arXiv:1812.06162) for the estimation framing, and Yu et
al., "Gradient Surgery for Multi-Task Learning" (PCGrad,
arXiv:2001.06782) for the surgery itself.

Chain provenance: cycle 24 of the Particle Sliders paper
watch, a direct follow-up to PR #301 (OPD mismatch
dynamics), PR #303 (PCGrad gradient surgery), PR #306
(CAGrad) and PR #310 (gradient-estimation protection).
#301: targeted correction repairs mismatched states
(TV 0.043-0.087 vs plain 0.197-0.475) but does NOT
protect matched states (0.031-0.054 vs the matched-only
control 0.007-0.010). #303: PCGrad surgery supplies the
protection ONLY against a densely computed reference
(matched TV 0.014-0.018); a sampled reference gives no
protection (0.029-0.052, at the random-projection /
shrinkage controls). #306: CAGrad splits identically --
estimation, not the conflict rule, binds. #310: averaging
a sampled reference buys protection monotonically but
saturates BELOW dense (an estimand ceiling: the sampled
gradient's expectation drifts to cos ~= 0.76-0.90 vs the
dense CE gradient along the trajectory), and an EMA
reference (beta = 0.9) sits in the no-protection band
(matched 0.030-0.048) through staleness.

This probe prices the dense rule's COST side, the axis
#310 left open: PCGrad surgery against the dense CE
reference REFRESHED every k steps and FROZEN between
refreshes, k in {1, 5, 10, 20, 40} with STEPS = 40, so
k = 1 IS #303's dense arm and k = 40 is an init-only
reference. At each refresh step the reference is
recomputed as the dense matched full-distribution CE
gradient at the CURRENT student; between refreshes the
frozen vector is reused verbatim. Per-step
cos(frozen ref, current dense gradient) is tracked to
expose the staleness curve directly.

Toy operationalization: PR #301's EXACT construction,
unchanged (calibration is asserted, test 1):
per-position categorical world, student logits phi_i @ W
with shared W (features R^32 + common component 0.8),
N = 20 positions, V = 6 tokens, 10 matched + 10
mismatched positions, STEPS = 40, step size 0.5, M = 8
sampled tokens per position per step. Every surgery arm
applies the SAME updates as #303's targeted arm (matched
sampled OPD update + mismatched CE correction); arms
differ ONLY in the reference the correction is projected
against:

  (a) PLAIN / (b) MATCHED-ONLY / (c) TARGETED: #301's
      arms, verbatim (calibration).
  (d) SURGERY, SAMPLED REFERENCE (n = 1): #303's sampled
      arm, the no-protection endpoint.
  (e) SURGERY, DENSE REFERENCE: #303's arm (g) -- the
      matched full-distribution CE gradient (S - T),
      recomputed every step; identical to refresh k = 1
      (asserted bit for bit).
  (f) SURGERY, EMA REFERENCE (beta = 0.9): #310's arm --
      the free estimator, a different staleness structure
      (an average of past SAMPLED gradients) than a frozen
      DENSE vector; kept as the staleness contrast.
  (g) SURGERY, FROZEN DENSE REFERENCE, k in {5, 10, 20,
      40}: the probe arms.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds. Calibration (arms a-e) reproduces #301's /
#303's / #310's published tables (the two band edges
that differ from the rounded published figures are the
dense arm on seed 0, 0.0185 vs the rounded 0.018, and
the EMA arm on seed 3, 0.0298 vs the rounded 0.030;
assertions use the anchor files' relative-margin form).
Final TV, matched (mismatched in parentheses):
  seed  k=1     k=5     k=10    k=20    k=40
  s0    .0185   .0295   .0339   .0381   .0417
        (.073)  (.068)  (.064)  (.066)  (.059)
  s1    .0154   .0325   .0412   .0427   .0545
        (.140)  (.123)  (.110)  (.103)  (.087)
  s2    .0142   .0243   .0300   .0351   .0396
        (.075)  (.059)  (.054)  (.051)  (.046)
  s3    .0175   .0247   .0279   .0281   .0310
        (.044)  (.044)  (.044)  (.045)  (.043)
  s4    .0143   .0239   .0276   .0267   .0312
        (.053)  (.051)  (.049)  (.051)  (.049)
  seed-mean matched: .0160 / .0270 / .0321 / .0341 /
      .0396 for k = 1 / 5 / 10 / 20 / 40
  sampled (n=1): .040/.052/.038/.031/.029 matched
  EMA (beta=.9): .038/.048/.036/.030/.032 matched
  matched-only control: .009/.010/.008/.009/.007
Trajectory-mean cos(frozen ref, current dense gradient):
  seed  k=5     k=10    k=20    k=40
  s0    .882    .763    .471    .076
  s1    .833    .635    .315    -.406
  s2    .865    .699    .482    -.101
  s3    .904    .734    .440    -.109
  s4    .826    .572    .314    -.197
  final-10-step mean cos at k=40 is NEGATIVE on every
  seed (-.21 / -.57 / -.32 / -.41 / -.27): the init
  reference ends the run anti-correlated with the dense
  gradient it is meant to stand in for.

The verdict, stated plainly: PROVEN on all four claims,
at the margins asserted below. (i) Matched-state
protection decays with k: the seed-mean matched TV is
strictly increasing at every step of the k ladder
(.0160 -> .0270 -> .0321 -> .0341 -> .0396), and per
seed 19 of the 20 ladder steps are non-decreasing (the
single exception is seed 4, k=10 -> k=20, a 0.0009 dip,
inside the seed-mean margin) -- assertions therefore
encode the seed-mean ladder plus per-seed endpoint
margins, not a per-seed monotone fiction. (ii) The
largest k whose matched TV stays within +0.005 of the
dense (k=1) arm is k = 1 ONLY: already at k = 5 every
seed sits 0.007-0.017 above its dense arm. A weaker,
partial form of protection -- beating the sampled
no-protection arm -- survives through k = 20 on every
seed and is gone at k = 40. (iii) The init-only arm
(k=40) lands exactly where the staleness curve says it
must: at the sampled arm's level, 0.000-0.002 ABOVE it
on every seed (matched 0.031-0.055), i.e. inside the
no-protection band and marginally worse than the
reference that at least tracks the current student
noisily. (iv) Repair is retained at every k:
mismatched TV stays at 0.043-0.140 (vs plain's
0.197-0.475) throughout, and the seed-mean repair
figure slightly IMPROVES with k (.077 -> .057) -- a
stale reference interferes less with the correction's
own job; the cost of staleness is paid entirely on the
protection side.

The EMA-vs-frozen contrast, both bands reported: the
two staleness structures are NOT interchangeable. A
frozen DENSE reference refreshed every 5 steps (matched
0.024-0.033) beats the EMA of sampled gradients
(0.030-0.048) on every seed by 0.005-0.015, because the
frozen vector is at least the RIGHT estimand, merely
old; the EMA is a stale average of the WRONG estimand
(#310's ceiling applies to it at every age). By k = 20
the frozen arm has decayed to the EMA's level
(seed-means 0.034 vs 0.037), and by k = 40 it has fallen
through it to the sampled band. The price curve for our
pipeline, in one line: dense-reference protection does
not amortize -- refreshing 8x less often (k=5 of 40
steps) already forfeits most of the dense band's gain,
so a protection mechanism wired into training must pay
for the dense reference essentially every step, or be
budgeted as a partial (sampled-beating, dense-missing)
mechanism at k <= 20 and as no protection at all at
k = 40.

Iteration log (10-iteration loop protocol):
  round 1: measurement run on pop-os (all arms, 5
    seeds) -- calibration reproduces the published
    tables; the refresh ladder, staleness curve and
    repair retention measured as tabulated above.
  round 2: assertions encoded at exactly the measured
    margins -- seed-mean monotone ladder (smallest
    step +0.002) instead of per-seed monotonicity
    (seed 4's 0.0009 dip), calibration bands in the
    anchor files' relative-margin form (dense seed 0
    0.0185 and EMA seed 3 0.0298 sit just outside the
    rounded published bands). Full suite green on
    pop-os; ship.

What is NOT asserted: refresh behavior at step counts
far from STEPS = 40 or step sizes other than 0.5;
protection at k values between the measured ladder
points (the curve is sampled, not interpolated); EMA
behavior at other beta; and any claim about the dense
reference's own compute cost beyond the step-count
ratio k itself. Pure numpy -- no GAN machinery, because
the claims are about distillation training dynamics on
categoricals. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family; PR #211 is what
enables this test in CI).
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


V = 6                # tokens
N = 20               # positions (10 matched + 10 mismatched)
NM = 10              # matched positions (the first NM)
D = 32               # shared feature dimension (student logits = phi @ W)
COMMON = 0.8         # shared feature component (parameter coupling)
SEEDS = range(5)
STEPS = 40           # training steps, all arms
M = 8                # sampled tokens per position per step (sampled arms)
LR = 0.5             # step size on W, all arms
EMA_BETA = 0.9       # EMA reference decay (effective window ~10 steps)
KS = (1, 5, 10, 20, 40)  # dense-reference refresh intervals


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(seed):
    """Features, teacher, and exact shared-parameterization
    realization of the constructed student init.
    Identical to PR #301's / #303's construction."""
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


def _expected_g(s_i, t_i):
    """Analytic expectation of _sampled_g under x ~ s_i."""
    r = np.log(s_i / t_i)
    return s_i * r - s_i * float((s_i * r).sum())


def _w_grad(phi, g_logits):
    return (phi[:NM].T @ g_logits) / N


def _project_off(vec, ref):
    """vec minus its component along ref (PCGrad projection
    onto ref's normal plane). Caller checks the conflict
    sign; this is the pure projection."""
    denom = float(ref.ravel() @ ref.ravel())
    if denom <= 0:
        return vec
    return vec - (float(vec.ravel() @ ref.ravel()) / denom) * ref


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(a.ravel() @ b.ravel() / (na * nb))


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


def _train_surgery(phi, t, w0, seed, mode):
    """Control surgery arms (d)-(f). The applied matched
    update is always the first sampled draw, taken from
    the same RNG stream arm (c) uses, so trajectories
    differ ONLY in the reference the correction is
    projected against. Returns (final student, stats)."""
    rng = np.random.default_rng(62000 + seed)
    w = w0.copy()
    ema = None
    cos_exp_list = []
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
        g_app = _w_grad(phi, gm)
        g_corr = (phi[NM:].T @ (s[NM:] - t[NM:])) / N
        g_exp = _w_grad(phi, np.array(
            [_expected_g(s[i], t[i]) for i in range(NM)]))
        if mode == "avg1":
            ref = g_app
        elif mode == "dense":
            ref = _w_grad(phi, s[:NM] - t[:NM])
        elif mode == "ema":
            ema = (g_app.copy() if ema is None
                   else EMA_BETA * ema + (1 - EMA_BETA) * g_app)
            ref = ema
        else:
            raise ValueError(mode)
        cos_exp_list.append(_cos(ref, g_exp))
        g_use = g_corr
        if float(g_corr.ravel() @ ref.ravel()) < 0:
            g_use = _project_off(g_corr, ref)
        w -= LR * (g_app + g_use)
    stats = dict(traj_cos_exp=float(np.mean(cos_exp_list)))
    return _softmax(phi @ w), stats


def _train_refresh(phi, t, w0, seed, k):
    """Probe arms (g): PCGrad surgery against the DENSE CE
    reference refreshed every k steps and frozen between
    refreshes. At a refresh step (step % k == 0) the
    reference is recomputed as the dense matched
    full-distribution CE gradient at the CURRENT student;
    between refreshes the frozen vector is reused
    verbatim. k = 1 is #303's dense arm; k = 40 (STEPS =
    40) is an init-only reference. Same applied-update RNG
    stream as every other surgery arm. Tracks per-step
    cos(frozen ref, current dense gradient)."""
    rng = np.random.default_rng(62000 + seed)
    w = w0.copy()
    frozen = None
    cos_dense_list = []
    for step in range(STEPS):
        s = _softmax(phi @ w)
        gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
        g_app = _w_grad(phi, gm)
        g_corr = (phi[NM:].T @ (s[NM:] - t[NM:])) / N
        g_dense_now = _w_grad(phi, s[:NM] - t[:NM])
        if step % k == 0:
            frozen = g_dense_now.copy()
        ref = frozen
        cos_dense_list.append(_cos(ref, g_dense_now))
        g_use = g_corr
        if float(g_corr.ravel() @ ref.ravel()) < 0:
            g_use = _project_off(g_corr, ref)
        w -= LR * (g_app + g_use)
    arr = np.array(cos_dense_list)
    stats = dict(
        traj_cos_dense=float(arr.mean()),
        cos_min=float(arr.min()),
        cos_first=float(arr[:10].mean()),
        cos_last=float(arr[-10:].mean()),
    )
    return _softmax(phi @ w), stats


_CACHE = {}


def run_seed(seed):
    if seed in _CACHE:
        return _CACHE[seed]
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
    for mode in ("avg1", "dense", "ema"):
        sf, stats = _train_surgery(phi, t, w0, seed, mode)
        out[f"tv_{mode}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_{mode}_mm"] = float(_tv(t, sf)[NM:].mean())
        out[f"traj_cos_{mode}"] = stats["traj_cos_exp"]
    for k in KS:
        sf, stats = _train_refresh(phi, t, w0, seed, k)
        out[f"tv_ref{k}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_ref{k}_mm"] = float(_tv(t, sf)[NM:].mean())
        out[f"traj_cos_ref{k}"] = stats["traj_cos_dense"]
        out[f"cos_min_ref{k}"] = stats["cos_min"]
        out[f"cos_first_ref{k}"] = stats["cos_first"]
        out[f"cos_last_ref{k}"] = stats["cos_last"]
    _CACHE[seed] = out
    return out


def test_calibration_reproduces_prior_tables():
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
        # #303's endpoints, in the anchor's relative-margin
        # form (the rounded published dense band 0.014-0.018
        # reads 0.0185 unrounded on seed 0)
        assert 0.028 <= m["tv_avg1_m"] <= 0.053, (seed, m)
        assert m["tv_avg1_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        assert m["tv_dense_m"] <= 2.3 * m["tv_matched_only_m"], (seed, m)
        assert m["tv_dense_m"] <= m["tv_targeted_m"] - 0.010, (seed, m)
        # #310's EMA band (0.0298 unrounded on seed 3)
        assert 0.029 <= m["tv_ema_m"] <= 0.048, (seed, m)
        assert m["tv_ema_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        # k = 1 refresh IS the dense arm, bit for bit: same
        # applied-update stream, reference recomputed at
        # every step from the same current student
        assert abs(m["tv_ref1_m"] - m["tv_dense_m"]) < 1e-12, (seed, m)
        assert abs(m["tv_ref1_mm"] - m["tv_dense_mm"]) < 1e-12, (seed, m)
        assert m["traj_cos_ref1"] == 1.0, (seed, m)


def test_staleness_cos_decays_with_refresh_interval():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the frozen reference's alignment with the CURRENT
        # dense gradient collapses as the refresh interval
        # grows, on every seed, at wide measured margins
        assert m["traj_cos_ref5"] >= m["traj_cos_ref10"] + 0.08, (seed, m)
        assert m["traj_cos_ref10"] >= m["traj_cos_ref20"] + 0.15, (seed, m)
        assert m["traj_cos_ref20"] >= m["traj_cos_ref40"] + 0.30, (seed, m)
        # the init-only reference is, on trajectory average,
        # essentially orthogonal to the dense gradient ...
        assert m["traj_cos_ref40"] <= 0.10, (seed, m)
        # ... and ANTI-correlated with it over the final
        # 10 steps on every seed
        assert m["cos_last_ref40"] < 0.0, (seed, m)


def test_protection_decays_with_refresh_interval():
    _empf_recipe()
    means = {k: float(np.mean([run_seed(s)[f"tv_ref{k}_m"]
                               for s in SEEDS])) for k in KS}
    # claim (i), seed-mean form: matched TV strictly
    # increases at every step of the k ladder (smallest
    # measured step +0.002, k=10 -> k=20)
    for lo, hi in zip(KS, KS[1:]):
        assert means[hi] >= means[lo] + 0.001, means
    for seed in SEEDS:
        m = run_seed(seed)
        # per-seed endpoint margins (the ladder's interior
        # has one 0.0009 dip, seed 4 k=10 -> k=20, which is
        # reported in the docstring, not asserted away)
        assert m["tv_ref5_m"] >= m["tv_ref1_m"] + 0.006, (seed, m)
        assert m["tv_ref40_m"] >= m["tv_ref5_m"] + 0.005, (seed, m)
        assert m["tv_ref40_m"] >= m["tv_ref1_m"] + 0.012, (seed, m)


def test_largest_protective_refresh_is_k1_only():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # claim (ii), dense-band form: within a stated
        # margin of +0.005 over the dense (k=1) arm, NO
        # k >= 5 qualifies on any seed -- already k = 5
        # sits 0.007-0.017 above dense (asserted at 0.006)
        assert m["tv_ref5_m"] >= m["tv_dense_m"] + 0.006, (seed, m)
        # claim (ii), partial-protection form: beating the
        # sampled no-protection arm survives through
        # k = 20 on every seed (smallest measured margin
        # 0.0017, seed 4 at k = 10; asserted at 0.001) ...
        for k in (5, 10, 20):
            assert m[f"tv_ref{k}_m"] <= m["tv_avg1_m"] - 0.001, \
                (seed, k, m)
        # ... and is gone at k = 40 (see the init-only test)


def test_init_only_reference_lands_at_sampled_level():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # claim (iii), as measured: the init-only arm lands
        # AT the sampled arm, 0.000-0.002 above it on every
        # seed -- inside the no-protection band, marginally
        # worse than a reference that at least tracks the
        # current student noisily
        assert m["tv_ref40_m"] >= m["tv_avg1_m"] - 0.001, (seed, m)
        assert m["tv_ref40_m"] <= m["tv_avg1_m"] + 0.003, (seed, m)
        assert m["tv_ref40_m"] >= m["tv_matched_only_m"] + 0.018, \
            (seed, m)


def test_repair_retained_at_every_refresh_interval():
    _empf_recipe()
    means_mm = {k: float(np.mean([run_seed(s)[f"tv_ref{k}_mm"]
                                  for s in SEEDS])) for k in KS}
    # claim (iv): the seed-mean repair figure does not
    # decay with k -- it slightly improves (.077 -> .057)
    for lo, hi in zip(KS, KS[1:]):
        assert means_mm[hi] <= means_mm[lo], means_mm
    for seed in SEEDS:
        m = run_seed(seed)
        for k in KS:
            # far below plain, in the targeted/dense cost
            # band, at every refresh interval
            assert m[f"tv_ref{k}_mm"] <= 0.15, (seed, k, m)
            assert m[f"tv_ref{k}_mm"] <= m["tv_plain_mm"] / 2.5, \
                (seed, k, m)


def test_frozen_dense_and_ema_are_different_staleness():
    _empf_recipe()
    mean_ema = float(np.mean([run_seed(s)["tv_ema_m"] for s in SEEDS]))
    mean_ref20 = float(np.mean([run_seed(s)["tv_ref20_m"]
                                for s in SEEDS]))
    for seed in SEEDS:
        m = run_seed(seed)
        # a frozen DENSE reference at k = 5 beats the EMA
        # of sampled gradients on every seed (measured
        # margins 0.005-0.015, asserted at 0.004): the
        # frozen vector is the right estimand, merely old,
        # while the EMA is stale AND carries #310's
        # estimand ceiling at every age
        assert m["tv_ref5_m"] <= m["tv_ema_m"] - 0.004, (seed, m)
    # by k = 20 the frozen arm has decayed to the EMA's
    # level (seed-means 0.034 vs 0.037), and by k = 40 it
    # has fallen through it to the sampled band (the
    # init-only test above)
    assert mean_ref20 <= mean_ema, (mean_ref20, mean_ema)
