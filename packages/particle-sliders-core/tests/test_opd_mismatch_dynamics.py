"""CPU gate for the on-policy distillation (OPD) learning-
dynamics claims of Zhang, Liao, Zeng, Wang, Li, Zhao
et al., "When Students Fail to Learn from Teachers:
Learning Dynamics of On-Policy Distillation"
(arXiv:2611.27125). The paper traces why OPD fails even
with a capable teacher to three mechanisms:

  (1) There are states where sampled-token OPD updates
      fail to correct teacher-student mismatch. The
      mismatched states are characterized by the student's
      mass on the teacher's top-p set: low mass on
      teacher-preferred tokens marks them.
  (2) Learning at mismatched states is difficult, and
      their gradients INTERFERE with learning at other
      tokens -- mismatched-state updates are not merely
      wasted, they damage learning elsewhere.
  (3) Continued training on mismatched states suppresses
      the student's current modes and leads sampling
      toward states where the teacher is unreliable (the
      dynamics behind spurious agreement); the findings
      motivate TARGETED correction of mismatched states.

(The abstract reports no headline numbers -- stated
plainly; the claims tested here are the mechanism
structure, not LM magnitudes.) Provenance: cycle 20 of
the Particle Sliders paper watch. This is the mechanism
layer under PR #281's excess/deficit measurement split:
#281 showed agreement is gameable and excess/deficit are
the gate variables; this paper adds a state classifier
computable during rollouts, the interference warning for
allocation rules (#223/#227), and the correction rule.

Toy operationalization: a per-position categorical world
in the #281 shape, with ONE load-bearing change from
#281: the student is NOT per-position free logits.
Student logits at position i are phi_i @ W, with phi_i
a fixed random feature vector in R^32 (plus a shared
common component) and W a single shared 32 x V matrix.
Interference (claim 2) is only measurable if positions
share parameters or an update pathway; with independent
per-position logits, mismatched updates could not damage
matched states by construction. Teacher and student-init
logits are constructed per position (below) and realized
EXACTLY through the shared parameterization by least
squares (D = 32 > N = 20 positions, phi full row rank;
the test asserts the realization to 1e-8).

Positions: N = 20, V = 6 tokens. Every teacher is peaked:
preferred token prob ~0.909, second ~0.037, four tails
~0.0136 each, so the teacher's top-p set at p = 0.9 is
{pref} alone. The two constructed state classes:
  MATCHED (10 positions): student init = teacher logits
    + Gaussian noise (sigma 0.35); init TV ~0.02-0.04,
    student mass on the teacher top-p set ~0.89.
  MISMATCHED (10 positions): student init puts ~0.80
    mass on the teacher's SECOND token, ~0.005 on the
    teacher's preferred token, ~0.049 on each tail;
    init TV ~0.90, top-p mass ~0.005. This is the
    slow-correction regime, and the construction went
    through two refuted variants to find it (iteration
    log in the PR): if the student's excess mass sits on
    a token the teacher rates ~0, the log-ratio and the
    KL are both large and the expected sampled gradient
    DOES raise the deficit token quickly; correction is
    slow exactly when the deficit token is nearly
    extinguished (its sampled-gradient rise rate is
    proportional to its own mass) while the excess token
    is teacher-plausible, so suppressing it spills mass
    onto the tails instead of the preferred token --
    the paper's mode-suppression dynamic, in miniature.

Arms (identical init, identical training RNG seed,
identical step count STEPS = 40 and step size; gradient
averaged over all N positions in every arm, so a matched
position's direct step size is the same in every arm --
the arms differ ONLY in which positions contribute and
by which gradient):
  (a) PLAIN: sampled-token OPD on all positions. Per
      step and position, M = 8 tokens are sampled from
      the student and the score-function OPD gradient
      mean_s[ log(S(x)/T(x)) (e_x - S) ] is applied --
      the on-policy form of #281 phenomenon 3.
  (b) MATCHED-ONLY control: the same sampled update,
      but only matched positions contribute. Isolates
      interference: matched positions receive the same
      direct signal as in (a), plus nothing else.
  (c) TARGETED: matched positions get the sampled
      update as in (a); mismatched positions get the
      full-distribution cross-entropy gradient (S - T)
      instead -- correction concentrated on the states
      the classifier flags, using the teacher's whole
      distribution rather than sampled tokens.

Readouts: per-class mean TV(S, T); per-class mean
student mass on the teacher's top-p (p = 0.9) set --
the classifier score, at init and after training.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds -- init TV matched / mismatched, init
top-p mass matched / mismatched:
  s0: 0.039 / 0.904,  0.886 / 0.005
  s1: 0.032 / 0.904,  0.894 / 0.005
  s2: 0.028 / 0.903,  0.901 / 0.005
  s3: 0.040 / 0.902,  0.889 / 0.005
  s4: 0.019 / 0.902,  0.905 / 0.005
Final TV (matched / mismatched) per arm:
  s0: plain 0.044 / 0.346,  matched-only 0.009 / --,
      targeted 0.042 / 0.059
  s1: plain 0.042 / 0.475,  matched-only 0.010 / --,
      targeted 0.054 / 0.087
  s2: plain 0.040 / 0.305,  matched-only 0.008 / --,
      targeted 0.040 / 0.046
  s3: plain 0.043 / 0.197,  matched-only 0.009 / --,
      targeted 0.031 / 0.043
  s4: plain 0.033 / 0.292,  matched-only 0.007 / --,
      targeted 0.031 / 0.049
Final top-p mass on mismatched states: plain
0.565 / 0.438 / 0.610 / 0.717 / 0.627 (s0..s4);
targeted 0.870 / 0.832 / 0.874 / 0.877 / 0.876.

The arbitration, stated plainly: claims (1), (2) and
the classifier half of (3) reproduce at this budget.
(1) The identical sampled update that takes matched
states to TV 0.007-0.010 (matched-only arm) leaves
mismatched states at TV 0.197-0.475 under plain OPD --
a 22-53x endpoint gap at the same budget, step size
and parameterization; their top-p mass recovers only
to 0.44-0.72 of the ~0.90 matched level. This is a
timescale claim, as in #281: at much longer budgets
the sampled arm also escapes, and at much higher step
sizes both arms destabilize. (2) Matched states under
plain OPD end at TV 0.033-0.044 vs 0.007-0.010 under
the matched-only control (gap 0.026-0.035, 3.7-4.9x):
the only difference between the arms is the presence
of mismatched-state gradients, so the foregone matched
correction is interference through the shared W, not a
step-size artifact (normalization is over all N in
every arm -- verified in iteration; an earlier draft
normalized per-arm and the artifact was real).
(4, classifier) Top-p mass separates the constructed
classes perfectly at init on every seed (AUROC 1.0;
matched >= 0.886, mismatched <= 0.005 per class mean,
and the per-position ranges do not overlap).

Claim (3) reproduces only in PART. The repair half
holds: targeted correction takes mismatched states to
TV 0.043-0.087 (4.6-6.6x below plain) and their top-p
mass to 0.83-0.88, near the matched states' level.
The protection half does NOT reproduce in this toy:
matched states under targeted correction end at TV
0.031-0.054, statistically indistinguishable from
plain OPD's 0.033-0.044 and 0.022-0.044 ABOVE the
matched-only control -- the correction's gradients
leak through the same shared trunk and displace
matched states about as much as plain's noisy
mismatched gradients do, and matched recovery from
that displacement is slow (peaked-softmax Jacobian).
This negative held in every geometry tried (feature
dimension 24-64, common-component coupling 0-1.2,
mismatched share 6/24-10/20, flat vs peaked matched
teachers, budgets 15-150 steps; full log in the PR).
The sharpened lesson for our pipeline: targeted
correction by gradient REWEIGHTING alone does not
protect matched states when the correction shares
parameters with them; protection would need the
correction to be localized or isolated, not just
concentrated. Assertions below encode exactly the
demonstrated margins, including the negative one --
the protection failure is asserted as a measured
result (targeted matched TV >= control + 0.02), so a
future variant that DOES protect matched states will
fail this test loudly and force a re-read.

What is NOT asserted: the paper's LLM-scale effects
(excessive length, repetition, benchmark degradation);
spurious agreement as a sampling-trajectory phenomenon
(here it appears only as its local shadow: plain OPD's
suppressed excess mass lands on teacher-tail tokens,
leaving top-p mass at 0.44-0.72 while raw token
agreement at visited states looks far healthier); and
any claim at budgets far from STEPS = 40 -- the
separations are budget-relative, like #281's deficit
block. Pure numpy -- no GAN machinery, because the
claims are about distillation training dynamics on
categoricals. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the
house guard for this paper-test family; PR #211 is
what enables this test in CI).
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
P_TOP = 0.9          # teacher top-p threshold for the classifier


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(seed):
    """Features, teacher, and exact shared-parameterization
    realization of the constructed student init."""
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


def _train(phi, t, w0, seed, arm):
    """One arm. Gradient averaged over all N positions in
    every arm; arms differ only in which positions
    contribute and by which gradient."""
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
                xs = rng.choice(V, size=M, p=s[i])
                r = np.log(s[i, xs] / t[i, xs])
                gi = np.zeros(V)
                np.add.at(gi, xs, r)
                g[i] = gi / M - s[i] * r.mean()
        w -= LR * (phi.T @ g) / N
    return _softmax(phi @ w)


def _auroc(scores_hi, scores_lo):
    """P(a random hi score exceeds a random lo score)."""
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0
               for a in scores_hi for b in scores_lo)
    return wins / (len(scores_hi) * len(scores_lo))


def run_seed(seed):
    phi, t, w0 = _build(seed)
    s0 = _softmax(phi @ w0)
    out = dict(
        tv0_m=float(_tv(t, s0)[:NM].mean()),
        tv0_mm=float(_tv(t, s0)[NM:].mean()),
        topp0_m=_topp_mass(t, s0)[:NM],
        topp0_mm=_topp_mass(t, s0)[NM:],
    )
    for arm in ("plain", "matched_only", "targeted"):
        sf = _train(phi, t, w0, seed, arm)
        out[f"tv_{arm}_m"] = float(_tv(t, sf)[:NM].mean())
        out[f"tv_{arm}_mm"] = float(_tv(t, sf)[NM:].mean())
        out[f"topp_{arm}_mm"] = float(_topp_mass(t, sf)[NM:].mean())
    return out


def test_top_p_mass_classifier_separates_states():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # constructed classes are what the labels say
        assert m["tv0_m"] <= 0.06, (seed, m)
        assert m["tv0_mm"] >= 0.85, (seed, m)
        # the classifier: student mass on the teacher's
        # top-p set separates them perfectly at init
        assert m["topp0_m"].mean() >= 0.85, (seed, m)
        assert m["topp0_mm"].mean() <= 0.02, (seed, m)
        assert m["topp0_m"].min() >= 0.70, (seed, m)
        assert m["topp0_mm"].max() <= 0.05, (seed, m)
        assert _auroc(m["topp0_m"], m["topp0_mm"]) >= 0.99, (seed, m)


def test_plain_opd_corrects_mismatched_far_slower():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # the sampled update demonstrably CAN correct states
        # of the matched type at this budget (control arm)
        assert m["tv_matched_only_m"] <= 0.012, (seed, m)
        # ... but under plain OPD the mismatched states are
        # still far from the teacher at the same budget
        assert m["tv_plain_mm"] >= 0.15, (seed, m)
        assert m["tv_plain_mm"] >= 10 * m["tv_matched_only_m"], (seed, m)
        # and their top-p mass recovers only partway toward
        # the matched states' ~0.89 level
        assert m["topp_plain_mm"] <= 0.75, (seed, m)


def test_mismatched_gradients_interfere_with_matched():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # matched states under plain OPD (which also trains
        # mismatched states) end measurably worse than under
        # the matched-only control -- same direct signal,
        # same step size, same normalization; the only
        # difference is the mismatched gradients' presence
        assert m["tv_plain_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)
        assert m["tv_plain_m"] >= 3 * m["tv_matched_only_m"], (seed, m)


def test_targeted_repairs_mismatched_but_does_not_protect_matched():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        # repair half of the targeted-correction claim: holds
        assert m["tv_targeted_mm"] <= 0.10, (seed, m)
        assert m["tv_targeted_mm"] <= m["tv_plain_mm"] / 2.5, (seed, m)
        assert m["topp_targeted_mm"] >= 0.80, (seed, m)
        # protection half: does NOT hold in this toy -- the
        # correction leaks through the shared trunk and
        # matched states stay at plain-like levels, well
        # above the matched-only control. Asserted as the
        # measured negative result (see docstring).
        assert m["tv_targeted_m"] >= m["tv_matched_only_m"] + 0.02, (seed, m)
