"""CPU gate for the sparse-unbiased-reference claim of
"Distillation with Errors: Sparse Unbiased Backpropagation
for Knowledge Distillation" (arXiv:2611.27131), arbitrated
on the protection geometry of PRs #303 / #310.

The paper's claims (from its abstract, the only results
attributed to it here): full-vocabulary KD requires costly
output-head backpropagation; DWE exploits the zero-sum
structure of logit gradients to construct sparse PAIRED
gradient estimates whose EXPECTATION matches the
full-vocabulary gradient; the estimator is proven
unbiased; at the same sampling budget it reduces
hidden-state and output-weight gradient MSE by 63.5% and
57.9% vs importance sampling; up to 1.95x output-head
speedup in off-policy KD and 2.20x in on-policy
distillation; downstream performance is maintained.

Chain provenance: cycle 26 of the Particle Sliders paper
watch, a direct follow-up to PR #303 (gradient surgery,
branch paper-test/gradient-surgery-opd) and PR #310
(gradient-estimation protection, branch
paper-test/gradient-estimation-protection). #303: PCGrad
surgery protects matched states ONLY against a densely
computed reference (matched TV 0.014-0.018); against the
sampled on-policy reference (cos ~= 0.74 to its own
expectation) it is indistinguishable from controls
(0.029-0.052). #310: averaging sampled references
converges to the sampled gradient's EXPECTATION, which
drifts to cos ~= 0.76-0.90 vs the dense CE gradient along
the trajectory -- an ESTIMAND gap, not a variance gap;
EMA references do not protect; the standing rule is that
a protection reference must be computed DENSELY. This
probe arbitrates that rule: DWE's sparse estimator is
unbiased for the DENSE gradient itself (it samples the
VOCABULARY axis, not the rollout/state axis #310 priced).
If an unbiased sparse reference protects at a fraction
of the dense cost, #310's rule refines to
"unbiased-or-dense"; if it fails, density itself is
load-bearing.

Toy operationalization: #310's protection toy, adapted
only where the question forces it. Calibration (test 1)
reruns #301/#303/#310's EXACT V = 6 construction and
asserts their published bands before any new arm runs.
The probe geometry keeps #301's per-position construction
(N = 20 positions, 10 matched + 10 mismatched, features
R^32 + common component 0.8, STEPS = 40, step size 0.5,
M = 8 sampled tokens) but widens the vocabulary to
V = 48 so sparse-vs-dense is a real distinction; the
non-preferred tail logits sit at -2.0 so the collective
tail mass does not swamp the per-token geometry #301
built (teacher top token ~= 0.88, as at V = 6). Every
surgery arm applies the SAME updates as #303's targeted
arm (matched sampled OPD update + mismatched CE
correction, PCGrad-projected against a reference when
they conflict); arms differ ONLY in the reference:

  (a) PLAIN / (b) MATCHED-ONLY / (c) TARGETED: #301's
      arms (calibration).
  (d) SURGERY, SAMPLED reference: #303's failing arm --
      the applied sampled matched gradient itself.
  (e) SURGERY, DENSE reference: #303's protecting arm --
      the matched full-distribution CE gradient (S - T).
  (f) SURGERY, SPARSE-UNBIASED (DWE) reference at budgets
      k in {8, 16, 24} of V = 48 vocabulary contributions
      per matched position (fractions 1/6, 1/3, 1/2). Per
      position, with dense logit gradient g = S - T
      (zero-sum: sum_v g_v = 0 exactly):
        * HEAD: the k/2 entries of largest |g_v| are
          computed exactly. (In KD the forward pass
          already produces S and T densely; what DWE
          sparsifies is the BACKWARD pass through the
          output head, so choosing the head from forward
          quantities costs no extra backward work.)
        * The tail sum S_R = -sum_head g is then known
          EXACTLY from the zero-sum structure, so every
          tail entry starts from the exact tail mean
          S_R / |R|.
        * TAIL: P = k/4 independent PAIRS (a, b) drawn
          uniformly from the tail contribute
          (|R|/2) (g_a - g_b) (e_a - e_b) / P. Each pair
          term has expectation g_v - S_R/|R| at entry v
          (the zero-sum mean is what makes the paired
          difference an unbiased estimate of the
          mean-shifted gradient), so
          E[estimate] = g at EVERY entry: the estimator
          is unbiased for the dense gradient itself, not
          for a sampled surrogate. Budget accounting is
          exact: k/2 head + 2 * (k/4) paired = k exact
          vocabulary contributions.
  (g) SURGERY, IMPORTANCE-SAMPLING reference at k = 16:
      the paper's named control -- k draws per position
      from q = (S + T)/2 with the standard 1/(k q)
      reweighting, no head, no pairing, no zero-sum
      correction. Its MSE is also measured head-to-head
      against arm (f) at every budget.

Readouts: per-class mean TV(S, T) at end; the reference
estimators' W-space MSE and Monte Carlo bias against the
dense gradient at init (400 draws); and the mean /
first-10-step / last-10-step cos(reference, dense
gradient) along each surgery trajectory.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds. Calibration at V = 6 reproduces #303/#310's
bands (test 1). Probe geometry (V = 48) final TV,
matched (mismatched in parentheses):
  seed  matched-only  dense     sampled   DWE-8     DWE-16
  s0    .014          .020      .035      .030      .022
        (.877)        (.032)    (.030)    (.031)    (.031)
  s1    .016          .022      .033      .028      .026
        (.877)        (.032)    (.033)    (.032)    (.033)
  s2    .016          .021      .028      .024      .022
        (.877)        (.030)    (.029)    (.030)    (.029)
  s3    .017          .021      .033      .028      .025
        (.877)        (.038)    (.036)    (.036)    (.037)
  s4    .015          .023      .040      .030      .027
        (.876)        (.043)    (.041)    (.041)    (.041)
  DWE-24 matched: .023/.022/.021/.022/.023 -- within
  +0.0033 of dense on every seed. IS-16 matched:
  .034/.032/.026/.032/.037 -- DWE-16 beats it on every
  seed by 0.004-0.012. Mismatched repair is retained by
  every surgery arm (0.029-0.043 vs plain's 0.126-0.311).
Estimator fidelity at init (W-space, 400 draws): DWE
Monte Carlo bias <= 0.0001 in max-entry absolute error
at every budget (dense-gradient norm 0.033-0.054);
W-space MSE vs the importance-sampling control at
matched budget: DWE/IS = 0.12-0.15 at k = 8, 0.06-0.10
at k = 16, 0.04-0.06 at k = 24 -- an 85-96% MSE
reduction, the paper's direction (63.5% / 57.9%) and
beyond it in this geometry. Trajectory cos(reference,
DENSE gradient), mean over steps: sampled 0.23-0.27,
IS-16 0.42-0.59, DWE-8 0.64-0.77, DWE-16 0.80-0.88,
DWE-24 0.89-0.93 -- and it does NOT drift down: the
last-10-step cos at k = 24 is 0.88-0.95 (first-10:
0.92-0.96), where #310's sampled family lost its
estimand entirely (its expectation sits at cos
0.76-0.90 to dense and a single draw far below).

The verdict, stated plainly: PROVEN, in the sharpened
form the numbers support. (1) Estimand fidelity: the
sparse paired estimator is unbiased for the dense
gradient in measurement (bias at Monte Carlo noise
floor) and its trajectory cos to the dense gradient
stays at 0.80-0.93 for budgets >= 1/3 of the vocabulary
with no downward drift -- the failure mode #310
diagnosed (convergence to the WRONG estimand) does not
occur, because the sampling is over the vocabulary
axis against a zero-sum identity, not over rollouts.
(2) Protection: surgery against the sparse-unbiased
reference at k = 24 (half the vocabulary) lands within
+0.0033 of the dense band on every seed and 0.007-0.017
below the sampled arm; at k = 16 (one third) it is
within +0.0043 of dense and beats the matched-budget
importance-sampling reference on every seed.
(3) Price curve: seed-mean matched TV falls
monotonically in budget (0.0279 -> 0.0241 -> 0.0223 for
k = 8 -> 16 -> 24, dense 0.0211, sampled 0.0338); the
cheapest budget that protects (within +0.006 of dense
AND >= 0.004 below sampled, every seed) is k = 16 --
one third of the dense reference's vocabulary cost.
#310's rule therefore REFINES: a protection reference
need not be dense, but it must be UNBIASED FOR the
dense gradient; a biased-but-averaged or
sampled-estimand reference fails at any budget (#310),
and an unbiased one protects at a third of the cost.

What is NOT asserted: the paper's speedup factors
(1.95x / 2.20x -- wall-clock systems claims this CPU
toy does not price; the budget fractions are the toy's
cost proxy); the paper's exact MSE percentages (the
toy asserts its own measured 85-96% reduction, not
63.5% / 57.9%); DWE's construction in detail beyond
the abstract (the head + zero-sum tail-mean + paired
difference operationalization above is this probe's
faithful reading of "sparse paired estimates whose
expectation matches the full-vocabulary gradient");
protection at budgets below 1/6 or geometries far from
#301's; and any claim at neural-net scale. Pure numpy
-- no GAN machinery, because the claims are about
distillation training dynamics on categoricals. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test
family; PR #211 is what enables this test in CI).
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


N = 20               # positions (10 matched + 10 mismatched)
NM = 10              # matched positions (the first NM)
D = 32               # shared feature dimension (student logits = phi @ W)
COMMON = 0.8         # shared feature component (parameter coupling)
SEEDS = range(5)
STEPS = 40           # training steps, all arms
M = 8                # sampled tokens per position per step (sampled arms)
LR = 0.5             # step size on W, all arms
V_CAL = 6            # calibration vocabulary (#301's exact construction)
V_PROBE = 48         # probe vocabulary (sparse-vs-dense is real)
TAIL_PROBE = -2.0    # probe tail logit offset (keeps #301's per-token scale)
BUDGETS = (8, 16, 24)  # DWE budgets, of V_PROBE contributions per position
EST_TRIALS = 400     # Monte Carlo draws for MSE / bias at init


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(seed, v, tail):
    """Features, teacher, and exact shared-parameterization
    realization of the constructed student init. With
    v = V_CAL and tail = 0.0 this is PR #301's / #303's /
    #310's construction bit for bit (same RNG stream)."""
    rng = np.random.default_rng(61000 + seed)
    common = rng.standard_normal(D)
    common /= np.linalg.norm(common)
    phi = rng.standard_normal((N, D)) + COMMON * common[None, :]
    tl = np.zeros((N, v))
    sl = np.zeros((N, v))
    for i in range(N):
        pref = int(rng.integers(v))
        second = int(rng.choice([t for t in range(v) if t != pref]))
        tl[i] = tail + rng.normal(0, 0.1, v)
        tl[i, pref] = 4.2
        tl[i, second] = 1.0
        if i < NM:  # matched: teacher logits + moderate noise
            sl[i] = tl[i] + rng.normal(0, 0.35, v)
        else:  # mismatched: mass on the teacher's second token,
            # preferred token nearly extinguished, tails inflated
            sl[i] = tail + rng.normal(0, 0.1, v)
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
    v = len(s_i)
    xs = rng.choice(v, size=M, p=s_i)
    r = np.log(s_i[xs] / t_i[xs])
    gi = np.zeros(v)
    np.add.at(gi, xs, r)
    return gi / M - s_i * r.mean()


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


def _dwe_ref(rng, phi, s, t, k):
    """Sparse UNBIASED (DWE) reference: W-space estimate of
    the matched dense CE gradient using exactly k exact
    vocabulary contributions per matched position --
    k/2 head entries exact, the tail's sum recovered
    exactly from the zero-sum structure of g = S - T, and
    k/4 uniform tail PAIRS contributing paired-difference
    corrections. E[estimate] = dense gradient (see the
    module docstring for the unbiasedness argument)."""
    gd = s[:NM] - t[:NM]
    v = gd.shape[1]
    head = k // 2
    pairs = k // 4
    est = np.zeros_like(gd)
    for i in range(NM):
        g = gd[i]
        h_idx = np.argsort(-np.abs(g))[:head]
        mask = np.ones(v, dtype=bool)
        mask[h_idx] = False
        r_idx = np.where(mask)[0]
        m = len(r_idx)
        est[i, h_idx] = g[h_idx]
        # zero-sum: the tail sums to minus the head sum,
        # so the tail mean is exact, not estimated
        est[i, r_idx] = -g[h_idx].sum() / m
        for _ in range(pairs):
            ai, bi = rng.choice(m, size=2, replace=False)
            a, b = r_idx[ai], r_idx[bi]
            d = (m / 2.0) * (g[a] - g[b]) / pairs
            est[i, a] += d
            est[i, b] -= d
    return _w_grad(phi, est)


def _is_ref(rng, phi, s, t, k):
    """Importance-sampling reference at the same budget:
    k draws per matched position from q = (S + T)/2 with
    1/(k q) reweighting. No head, no pairing, no zero-sum
    correction -- the paper's named control."""
    gd = s[:NM] - t[:NM]
    v = gd.shape[1]
    q = 0.5 * (s[:NM] + t[:NM])
    q = q / q.sum(axis=1, keepdims=True)
    est = np.zeros_like(gd)
    for i in range(NM):
        xs = rng.choice(v, size=k, p=q[i])
        np.add.at(est[i], xs, gd[i][xs] / (k * q[i][xs]))
    return _w_grad(phi, est)


def _train_basic(phi, t, w0, seed, arm):
    """Arms (a)-(c), verbatim PR #301. Gradient averaged
    over all N positions in every arm."""
    rng = np.random.default_rng(62000 + seed)
    w = w0.copy()
    rows = range(NM) if arm == "matched_only" else range(N)
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        g = np.zeros((N, t.shape[1]))
        for i in rows:
            if arm == "targeted" and i >= NM:
                g[i] = s[i] - t[i]  # full-distribution CE gradient
            else:  # sampled-token OPD (score-function) gradient
                g[i] = _sampled_g(rng, s[i], t[i])
        w -= LR * (phi.T @ g) / N
    return _softmax(phi @ w)


def _train_surgery(phi, t, w0, seed, mode, k=None):
    """Surgery arms (d)-(g). The applied matched update is
    always the first sampled draw, taken from the same RNG
    stream arm (c) uses, so trajectories differ ONLY in the
    reference the correction is projected against. Sparse
    reference draws come from a separate stream (64000)
    and never touch the applied updates. Returns (final
    student, stats) with cos(reference, DENSE gradient)
    over the trajectory (mean / first 10 / last 10 steps)."""
    rng = np.random.default_rng(62000 + seed)
    rng_x = np.random.default_rng(64000 + seed)
    w = w0.copy()
    cos_list = []
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
        g_app = _w_grad(phi, gm)
        g_corr = (phi[NM:].T @ (s[NM:] - t[NM:])) / N
        g_dense = _w_grad(phi, s[:NM] - t[:NM])
        if mode == "sampled":
            ref = g_app
        elif mode == "dense":
            ref = g_dense
        elif mode == "dwe":
            ref = _dwe_ref(rng_x, phi, s, t, k)
        elif mode == "is":
            ref = _is_ref(rng_x, phi, s, t, k)
        else:
            raise ValueError(mode)
        cos_list.append(_cos(ref, g_dense))
        g_use = g_corr
        if float(g_corr.ravel() @ ref.ravel()) < 0:
            g_use = _project_off(g_corr, ref)
        w -= LR * (g_app + g_use)
    stats = dict(
        cos_dense=float(np.mean(cos_list)),
        cos_first=float(np.mean(cos_list[:10])),
        cos_last=float(np.mean(cos_list[-10:])),
    )
    return _softmax(phi @ w), stats


def _measure_estimators(phi, t, w0, seed):
    """W-space MSE and Monte Carlo bias of the sparse
    references against the DENSE gradient at init, from
    EST_TRIALS independent draws per (budget, estimator)."""
    s = _softmax(phi @ w0)
    g_dense = _w_grad(phi, s[:NM] - t[:NM]).ravel()
    rng = np.random.default_rng(77000 + seed)
    out = dict(gw_norm=float(np.linalg.norm(g_dense)))
    for k in BUDGETS:
        for name, fn in (("dwe", _dwe_ref), ("is", _is_ref)):
            draws = np.array([fn(rng, phi, s, t, k).ravel()
                              for _ in range(EST_TRIALS)])
            out[f"mse_{name}{k}"] = float(
                ((draws - g_dense) ** 2).sum(axis=1).mean())
            out[f"bias_{name}{k}"] = float(
                np.abs(draws.mean(axis=0) - g_dense).max())
    return out


_CACHE = {}


def run_seed(seed, v, tail):
    key = (seed, v)
    if key in _CACHE:
        return _CACHE[key]
    phi, t, w0 = _build(seed, v, tail)
    s0 = _softmax(phi @ w0)
    out = dict(
        tv0_m=float(_tv(t, s0)[:NM].mean()),
        tv0_mm=float(_tv(t, s0)[NM:].mean()),
    )
    for arm in ("plain", "matched_only", "targeted"):
        sf = _train_basic(phi, t, w0, seed, arm)
        out[f"tv_{arm}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_{arm}_mm"] = float(_tv(t, sf)[NM:].mean())
    for mode in ("sampled", "dense"):
        sf, stats = _train_surgery(phi, t, w0, seed, mode)
        out[f"tv_{mode}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_{mode}_mm"] = float(_tv(t, sf)[NM:].mean())
        out[f"cos_{mode}"] = stats["cos_dense"]
    if v == V_PROBE:
        for k in BUDGETS:
            sf, stats = _train_surgery(phi, t, w0, seed, "dwe", k)
            out[f"tv_dwe{k}_m"] = float(_tv(t, sf)[:NM].mean())
            out[f"tv_dwe{k}_mm"] = float(_tv(t, sf)[NM:].mean())
            out[f"cos_dwe{k}"] = stats["cos_dense"]
            out[f"cosf_dwe{k}"] = stats["cos_first"]
            out[f"cosl_dwe{k}"] = stats["cos_last"]
        sf, stats = _train_surgery(phi, t, w0, seed, "is", 16)
        out["tv_is16_m"] = float(_tv(t, sf)[:NM].mean())
        out["tv_is16_mm"] = float(_tv(t, sf)[NM:].mean())
        out["cos_is16"] = stats["cos_dense"]
        out.update(_measure_estimators(phi, t, w0, seed))
    _CACHE[key] = out
    return out


def _cal(seed):
    return run_seed(seed, V_CAL, 0.0)


def _probe(seed):
    return run_seed(seed, V_PROBE, TAIL_PROBE)


def test_calibration_reproduces_opd_mismatch_geometry():
    _empf_recipe()
    for seed in SEEDS:
        m = _cal(seed)
        # the geometry is #301's: constructed classes and
        # the published arm ranges (see #310's docstring)
        assert m["tv0_m"] <= 0.06, (seed, m)
        assert m["tv0_mm"] >= 0.85, (seed, m)
        assert m["tv_matched_only_m"] <= 0.012, (seed, m)
        assert m["tv_plain_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)
        assert m["tv_plain_mm"] >= 0.15, (seed, m)
        assert 0.028 <= m["tv_targeted_m"] <= 0.057, (seed, m)
        assert m["tv_targeted_mm"] <= 0.10, (seed, m)
        # the endpoints the probe refines: #303's sampled
        # surgery does not protect, its dense surgery does
        assert m["tv_sampled_m"] >= m["tv_matched_only_m"] + 0.018, \
            (seed, m)
        assert m["tv_dense_m"] <= 2.3 * m["tv_matched_only_m"], (seed, m)
        assert m["tv_dense_m"] <= m["tv_targeted_m"] - 0.010, (seed, m)


def test_probe_geometry_sits_on_the_family_scale():
    _empf_recipe()
    for seed in SEEDS:
        m = _probe(seed)
        # widening V to 48 with the tail offset preserves
        # #301's per-token geometry and #303's ordering:
        # dense protects, sampled does not
        assert m["tv0_m"] <= 0.06, (seed, m)
        assert m["tv0_mm"] >= 0.85, (seed, m)
        assert m["tv_matched_only_m"] <= 0.020, (seed, m)
        assert m["tv_dense_m"] <= 0.024, (seed, m)
        assert m["tv_sampled_m"] >= m["tv_dense_m"] + 0.005, (seed, m)
        assert 0.025 <= m["tv_targeted_m"] <= 0.045, (seed, m)
        assert m["tv_plain_mm"] >= 0.12, (seed, m)


def test_sparse_reference_is_unbiased_for_the_dense_gradient():
    _empf_recipe()
    for seed in SEEDS:
        m = _probe(seed)
        for k in BUDGETS:
            # the Monte Carlo mean of the sparse estimate
            # sits on the dense gradient: max-entry bias
            # at the noise floor of 400 draws, and never
            # above the (also unbiased) IS control's --
            # the estimand is the DENSE gradient itself,
            # unlike #310's sampled family, whose limit is
            # a different vector (cos 0.76-0.90 to dense)
            assert m[f"bias_dwe{k}"] <= 0.001, (seed, k, m)
            assert m[f"bias_dwe{k}"] <= m[f"bias_is{k}"] + 0.0001, \
                (seed, k, m)


def test_sparse_mse_beats_importance_sampling_at_matched_budget():
    _empf_recipe()
    for seed in SEEDS:
        m = _probe(seed)
        # W-space (output-weight) gradient MSE, DWE vs the
        # importance-sampling control at the SAME budget:
        # the paper's direction (63.5% / 57.9% reductions),
        # asserted at this toy's measured margins
        assert m["mse_dwe8"] <= 0.35 * m["mse_is8"], (seed, m)
        assert m["mse_dwe16"] <= 0.25 * m["mse_is16"], (seed, m)
        assert m["mse_dwe24"] <= 0.25 * m["mse_is24"], (seed, m)


def test_estimand_fidelity_holds_along_the_trajectory():
    _empf_recipe()
    for seed in SEEDS:
        m = _probe(seed)
        # cos(sparse reference, DENSE gradient) stays high
        # along the whole trajectory at budgets >= 1/3 --
        # it does NOT drift down the way #310's sampled
        # estimand does (sampled reference: 0.23-0.27 here)
        assert m["cos_dwe16"] >= 0.75, (seed, m)
        assert m["cos_dwe24"] >= 0.85, (seed, m)
        assert m["cosl_dwe16"] >= 0.72, (seed, m)
        assert m["cosl_dwe24"] >= 0.85, (seed, m)
        assert m["cosl_dwe24"] >= m["cosf_dwe24"] - 0.08, (seed, m)
        # and it dominates every biased / sampled control
        # by a wide margin at the same steps
        assert m["cos_dwe16"] >= m["cos_is16"] + 0.20, (seed, m)
        assert m["cos_dwe24"] >= m["cos_sampled"] + 0.50, (seed, m)


def test_sparse_unbiased_reference_protects_matched_states():
    _empf_recipe()
    for seed in SEEDS:
        m = _probe(seed)
        # at half the vocabulary cost the sparse reference
        # lands in the dense band; at one third it is
        # within a hair of it -- and clearly below the
        # sampled arm and the matched-budget IS arm
        assert m["tv_dwe24_m"] <= m["tv_dense_m"] + 0.005, (seed, m)
        assert m["tv_dwe16_m"] <= m["tv_dense_m"] + 0.006, (seed, m)
        assert m["tv_dwe24_m"] <= m["tv_sampled_m"] - 0.005, (seed, m)
        assert m["tv_dwe16_m"] <= m["tv_sampled_m"] - 0.004, (seed, m)
        assert m["tv_dwe16_m"] <= m["tv_is16_m"] - 0.002, (seed, m)
        # mismatched repair is retained at every budget:
        # far below plain, in the targeted/dense cost band
        for k in BUDGETS:
            assert m[f"tv_dwe{k}_mm"] <= 0.05, (seed, k, m)
            assert m[f"tv_dwe{k}_mm"] <= m["tv_plain_mm"] / 2.5, \
                (seed, k, m)


def test_price_curve_and_cheapest_protecting_budget():
    _empf_recipe()
    means = {k: float(np.mean([_probe(s)[f"tv_dwe{k}_m"]
                               for s in SEEDS])) for k in BUDGETS}
    dense_mean = float(np.mean([_probe(s)["tv_dense_m"] for s in SEEDS]))
    # protection is monotone in the sparse budget on the
    # seed mean, converging to (not past) the dense band
    assert means[16] <= means[8] - 0.002, means
    assert means[24] <= means[16] - 0.001, means
    assert means[24] >= dense_mean - 0.001, means
    for seed in SEEDS:
        m = _probe(seed)
        assert m["tv_dwe24_m"] <= m["tv_dwe8_m"] - 0.001, (seed, m)
    # the cheapest protecting budget is k = 16 (one third
    # of the vocabulary): k = 8's seed mean still sits
    # clearly above the dense band, while k = 16 meets
    # the protection criterion of the previous test on
    # every seed (asserted there)
    assert means[8] >= dense_mean + 0.005, means
