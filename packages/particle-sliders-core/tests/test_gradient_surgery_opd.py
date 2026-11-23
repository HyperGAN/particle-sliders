"""CPU gate for the gradient-surgery (PCGrad) claim of Yu,
Kumar, Gupta, Levine, Hausman & Finn, "Gradient Surgery
for Multi-Task Learning" (arXiv:2001.06782, NeurIPS 2020),
arbitrated on the OPD mismatch geometry of PR #301.

The paper's claim: when two task gradients conflict
(negative cosine similarity), projecting each task's
gradient onto the normal plane of the conflicting task's
gradient removes the detrimental interference component,
so multi-task training recovers single-task learning
quality on each task while keeping the other task's gains.
The precondition is measurable: conflicting gradients
(cos < 0) must actually occur.

Chain provenance: cycle 21 of the Particle Sliders paper
watch, and a direct follow-up to PR #301 (arXiv:2611.27125,
OPD mismatch dynamics). #301 proved targeted correction
repairs mismatched states but its matched-state PROTECTION
half did not reproduce: targeted matched TV 0.031-0.054
sat at plain OPD's level, above the matched-only control
0.007-0.010 in every geometry tried. #301's closing
lesson: "targeted correction by gradient REWEIGHTING alone
does not protect matched states when the correction shares
parameters with them; protection would need the correction
to be localized or isolated, not just concentrated."
PCGrad surgery is the canonical localization-in-gradient-
space mechanism. This test arbitrates whether surgery
supplies the protection #301 lacked.

Toy operationalization: PR #301's EXACT construction,
unchanged (calibration is asserted, test 1): per-position
categorical world, student logits phi_i @ W with shared
W (features R^32 + common component 0.8), N = 20 positions,
V = 6 tokens, 10 matched + 10 mismatched positions built
exactly as in #301, STEPS = 40, step size 0.5, gradient
averaged over all N positions in every arm. Arms differ
ONLY in how the mismatched-correction gradient is combined
with the matched gradient:
  (a) PLAIN: sampled-token OPD on all positions (#301 a).
  (b) MATCHED-ONLY control: sampled update, matched
      positions only (#301 b).
  (c) TARGETED: matched sampled update + full-distribution
      CE gradient (S - T) at mismatched positions (#301 c).
  (d) TARGETED + SURGERY (the paper's mechanism, applied
      asymmetrically -- the correction is the guest,
      matched learning is what is protected): per step,
      g_match = matched positions' sampled OPD gradient
      w.r.t. W (the gradient arm (c) actually applies) and
      g_corr = mismatched CE gradient w.r.t. W; if
      cos(g_corr, g_match) < 0, g_corr is replaced by its
      projection onto the normal plane of g_match.
  (e) RANDOM-PROJECTION control: same conditional
      projection, but onto the normal plane of a random
      unit vector (fresh per step, separate seeded stream)
      -- controls for "any projection helps by shrinking
      or redirecting the correction".
  (f) SHRINKAGE control: g_corr scaled by the mean norm
      ratio arm (d)'s projection produces on that seed,
      no direction change -- isolates direction vs
      magnitude.
  (g) TARGETED + SURGERY, DENSE REFERENCE: identical to
      (d) except the protected direction g_match is the
      matched positions' full-distribution CE gradient
      (S - T) w.r.t. W instead of its sampled estimate.
      The applied matched update is still the sampled OPD
      update of arm (c); only the surgery REFERENCE
      changes. In distillation the teacher's full
      distribution at matched states is already computed,
      so this reference is available at no extra sampling
      cost -- the variant asks whether surgery's failure
      or success turns on reference-gradient accuracy.

Readouts: per-class mean TV(S, T) at end; per-step
cos(g_corr, g_match) along arm (d)'s trajectory (how
often the paper's conflict precondition occurs).

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds. Calibration (arms a-c) reproduces #301's
published table exactly. Final TV (matched / mismatched):
  s0: plain 0.044 / 0.346,  matched-only 0.009 / --,
      targeted 0.042 / 0.059,  surgery 0.040 / 0.060,
      random 0.041 / 0.060,  shrink 0.041 / 0.062,
      surgery-dense-ref 0.018 / 0.073
  s1: plain 0.042 / 0.475,  matched-only 0.010 / --,
      targeted 0.054 / 0.087,  surgery 0.052 / 0.087,
      random 0.054 / 0.088,  shrink 0.054 / 0.091,
      surgery-dense-ref 0.015 / 0.140
  s2: plain 0.040 / 0.305,  matched-only 0.008 / --,
      targeted 0.040 / 0.046,  surgery 0.038 / 0.046,
      random 0.040 / 0.046,  shrink 0.039 / 0.047,
      surgery-dense-ref 0.014 / 0.075
  s3: plain 0.043 / 0.197,  matched-only 0.009 / --,
      targeted 0.031 / 0.043,  surgery 0.031 / 0.043,
      random 0.030 / 0.044,  shrink 0.031 / 0.043,
      surgery-dense-ref 0.017 / 0.044
  s4: plain 0.033 / 0.292,  matched-only 0.007 / --,
      targeted 0.031 / 0.049,  surgery 0.029 / 0.047,
      random 0.031 / 0.049,  shrink 0.032 / 0.049,
      surgery-dense-ref 0.014 / 0.053
Conflict readout (arm d trajectory): cos(g_corr, g_match)
< 0 on 90-100% of steps per seed; mean cos -0.11 to -0.22.
The paper's precondition holds nearly every step -- and
is weak: the anti-aligned component is only |cos| ~=
0.11-0.22 of the correction gradient.

The arbitration, stated plainly: the PCGrad claim
reproduces only in PART, and the part that fails is the
one as literally specified. (1) With the reference taken
as the matched gradient OPD actually applies -- the
sampled score-function estimate -- surgery does NOT
protect: matched TV 0.029-0.052 is indistinguishable
from targeted's 0.031-0.054 and from both controls
(random projection, pure shrinkage), i.e. the projection
removes a component, but the component it removes is
aimed by a noisy direction (cos between the sampled
matched gradient and its own expectation is ~0.74 at
init), and the leak that displaces matched states is
not concentrated along it. (2) With the SAME projection
formula and the SAME applied updates, but the reference
computed exactly (dense CE gradient, arm g), surgery
DOES protect: matched TV 0.014-0.018, within ~2x of the
matched-only control on every seed and 0.014-0.039 below
targeted, while mismatched repair is mostly retained
(0.044-0.140 vs plain's 0.197-0.475; the cost lands on
mismatched TV, worst on s1: 0.140 vs targeted's 0.087).
Variants tried in the loop (full log in the PR):
symmetric PCGrad -- no protection either, so the failure
is not the asymmetry; per-position PCGrad (each matched
position a separate task) -- no better; reference = the
analytic expectation of the sampled gradient --
intermediate protection (matched 0.017-0.024), ordering
exactly with reference accuracy (sampled < expected <
dense; the expected sampled gradient is nearly parallel
to the dense CE gradient, cos ~0.995); and a non-PCGrad
diagnostic that nulls the correction's induced change
on matched LOGITS (output space, not gradient space)
protects fully (matched 0.007-0.010 = control) at
higher repair cost -- the leak is localizable in
principle, and gradient surgery approximates that
localization only as well as its reference gradient is
estimated. The sharpened lesson for our pipeline:
gradient surgery is not a drop-in protection for
on-policy distillation corrections; IF surgery is used,
its protected direction must be computed densely from
the teacher distribution, never from the sampled
on-policy gradient it is meant to protect.

What is NOT asserted: PCGrad's multi-task benchmark
claims at neural-net scale (this toy arbitrates one
mechanism -- projection against a reference gradient --
in one geometry, #301's); any claim at budgets far from
STEPS = 40; and arm (g)'s repair cost beyond the
measured seeds (s1's 0.140 is asserted as an allowed
cost, not bounded tightly). Pure numpy -- no GAN
machinery, because the claims are about distillation
training dynamics on categoricals. Skips cleanly when
the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI).
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
P_TOP = 0.9          # teacher top-p threshold (calibration readout)


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


def _topp_mass(t, s):
    """Student mass on the teacher's top-p set, per position."""
    out = np.zeros(N)
    order = np.argsort(-t, axis=1)
    for i in range(N):
        cum = 0.0
        for j in order[i]:
            cum += t[i, j]
            out[i] += s[i, j]
            if cum >= P_TOP:
                break
    return out


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
    onto ref's normal plane). Caller checks the conflict
    sign; this is the pure projection."""
    denom = float(ref.ravel() @ ref.ravel())
    if denom <= 0:
        return vec
    return vec - (float(vec.ravel() @ ref.ravel()) / denom) * ref


def _train_combined(phi, t, w0, seed, arm, shrink_factor=None):
    """Arms (d)-(g): matched sampled update and mismatched
    CE correction computed as separate W-gradients, then
    combined per arm. Returns (final student, stats).
    The training RNG stream is identical to arm (c)'s, so
    the matched sampled gradients are the same draws as in
    targeted; the random-projection stream is separate so
    it cannot perturb them."""
    rng = np.random.default_rng(62000 + seed)
    rng_r = np.random.default_rng(63000 + seed)
    w = w0.copy()
    cos_list, ratios = [], []
    for _ in range(STEPS):
        s = _softmax(phi @ w)
        gm = np.array([_sampled_g(rng, s[i], t[i]) for i in range(NM)])
        gc = s[NM:] - t[NM:]
        g_match = (phi[:NM].T @ gm) / N
        g_corr = (phi[NM:].T @ gc) / N
        a, b = g_match.ravel(), g_corr.ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        cos_list.append(cos)
        g_corr_use = g_corr
        if arm == "surgery":
            if cos < 0:
                g_corr_use = _project_off(g_corr, g_match)
        elif arm == "surgery_dense":
            g_dense = (phi[:NM].T @ (s[:NM] - t[:NM])) / N
            ad, nd = g_dense.ravel(), np.linalg.norm(g_dense.ravel())
            cos_d = float(ad @ b / (nd * nb)) if nd > 0 and nb > 0 else 0.0
            if cos_d < 0:
                g_corr_use = _project_off(g_corr, g_dense)
        elif arm == "random":
            r = rng_r.standard_normal(g_match.shape)
            r /= np.linalg.norm(r)
            if float(g_corr.ravel() @ r.ravel()) < 0:
                g_corr_use = _project_off(g_corr, r)
        elif arm == "shrink":
            g_corr_use = g_corr * shrink_factor
        if nb > 0:
            ratios.append(float(np.linalg.norm(g_corr_use) / nb))
        w -= LR * (g_match + g_corr_use)
    stats = dict(
        conflict_frac=float(np.mean([c < 0 for c in cos_list])),
        mean_cos=float(np.mean(cos_list)),
        mean_ratio=float(np.mean(ratios)) if ratios else 1.0,
    )
    return _softmax(phi @ w), stats


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
    sf, stats = _train_combined(phi, t, w0, seed, "surgery")
    out["tv_surgery_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_surgery_mm"] = float(_tv(t, sf)[NM:].mean())
    out.update(stats)
    sf, _ = _train_combined(phi, t, w0, seed, "random")
    out["tv_random_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_random_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_combined(phi, t, w0, seed, "shrink",
                            shrink_factor=stats["mean_ratio"])
    out["tv_shrink_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_shrink_mm"] = float(_tv(t, sf)[NM:].mean())
    sf, _ = _train_combined(phi, t, w0, seed, "surgery_dense")
    out["tv_dense_m"] = float(_tv(t, sf)[:NM].mean())
    out["tv_dense_mm"] = float(_tv(t, sf)[NM:].mean())
    return out


def test_calibration_reproduces_opd_mismatch_geometry():
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
        # #301's protection failure itself, restated: the
        # targeted correction does not protect matched states
        assert m["tv_targeted_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)


def test_gradient_conflict_precondition_holds():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the paper's precondition is not the missing piece:
        # the correction gradient conflicts with the matched
        # gradient on nearly every step of every seed
        assert m["conflict_frac"] >= 0.80, (seed, m)
        assert m["mean_cos"] < 0, (seed, m)


def test_sampled_reference_surgery_does_not_protect():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # repair is retained under surgery (the projection
        # does not break the correction's own job)
        assert m["tv_surgery_mm"] <= 0.10, (seed, m)
        assert m["tv_surgery_mm"] <= m["tv_plain_mm"] / 2.5, (seed, m)
        # but protection does NOT appear with the sampled
        # reference: matched states stay at targeted/plain
        # levels, well above the matched-only control --
        # asserted as the measured negative result
        assert m["tv_surgery_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
        assert m["tv_surgery_m"] >= m["tv_targeted_m"] - 0.006, (seed, m)


def test_projection_controls_do_not_protect():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # neither control protects, and both sit at
        # targeted's level -- so arm (d)'s failure is not
        # "projection per se fails"; it is that the sampled
        # reference aims the projection no better than a
        # random direction or a pure shrinkage does
        for arm in ("random", "shrink"):
            assert m[f"tv_{arm}_m"] >= m["tv_matched_only_m"] + 0.018, (seed, m)
            assert abs(m[f"tv_{arm}_m"] - m["tv_targeted_m"]) <= 0.006, (seed, m)


def test_dense_reference_surgery_protects_matched():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the identical projection, aimed by the exact
        # (dense CE) matched gradient, protects: within
        # ~2x of the matched-only control and clearly
        # below targeted on every seed
        assert m["tv_dense_m"] <= 2.3 * m["tv_matched_only_m"], (seed, m)
        assert m["tv_dense_m"] <= m["tv_targeted_m"] - 0.010, (seed, m)
        # repair is mostly retained -- far below plain --
        # at a measured cost vs targeted on some seeds
        # (worst: s1, +0.053), encoded as an allowance
        assert m["tv_dense_mm"] <= 0.15, (seed, m)
        assert m["tv_dense_mm"] <= m["tv_plain_mm"] / 2.5, (seed, m)
        assert m["tv_dense_mm"] <= m["tv_targeted_mm"] + 0.06, (seed, m)
