"""CPU gate for the VOPD claim of "Verifiable On-Policy
Distillation" (arXiv:2611.30325, Shi et al.). Chain context:
PR #319's law (DWE) showed an estimator unbiased for the
DENSE gradient itself, sampling the vocabulary axis, holds
the dense estimand in the reference seat; this test checks
the same "right estimand, spread densely" shape in the
REWARD seat.

The paper's claim, from its abstract and Sections 3-4:
RLVR supervision is sparse -- a verifier reward is
propagated only through sampled trajectories and gives no
signal for alternative actions. OPD gives dense teacher
feedback over the full vocabulary but is not grounded in
verified outcomes. VOPD couples them: under the
implicit-reward view of OPD, d(s,a) = log p_T(a|s) -
log p_ref(a|s), teacher supervision and verifier feedback
share a reward space. For each prompt group, VOPD
standardizes the verifier rewards V_i and the cumulative
teacher implicit rewards D_i into advantages A^V and A^D,
projects the verifier advantage onto the teacher direction
(the least-squares coefficient is the within-group
correlation c_x = (1/n) sum_i A^V_i A^D_i), rescales to
sigma_D * A-hat^V = c_x (D_i - D-bar), and
Rao-Blackwellizes the resulting sampled-action estimator
over the full vocabulary: the dense objective is
c_x * sum_a p_theta(a|s) d(s,a), i.e. the loss
KL(p_theta || p_T) - lambda * c_x * sum_a p_theta(a|s)
d(s,a), equivalently reverse-KL distillation toward the
adaptively extrapolated target q(a|s) proportional to
p_T^(1 + lambda c_x) p_ref^(-lambda c_x). Verifier-grounded
supervision thereby reaches actions that were NEVER
SAMPLED, with no additional verifier evaluations. When the
verifier rewards have zero within-group variance, c_x is
undefined and the paper falls back to a fixed extrapolation
coefficient of 0.25.

Question decided here, at toy scale: at a MATCHED verifier
budget (the verifier labels only a sparse set of sampled
state-action pairs), does projecting the sparse feedback
onto the teacher's implicit-reward direction supervise
never-sampled correct actions -- beating sampled-only
propagation -- and how close does it come to the
dense-verifier oracle? And the boundary: densification can
only move mass along the teacher direction, so does it
propagate verifier INFORMATION, or just teacher prior,
where teacher and verifier DISAGREE?

OPERATIONALIZATION, stated as an adaptation (cf. cycle
28's #322): the paper works on token sequences; this toy
is the single-step (contextual-bandit) reduction. A state
is the prompt, an action is the whole response, so the
cumulative teacher reward D_i of Eq. 6 is exactly
d(s, a_i) for the sampled action, and the paper's
immediate-token approximation (Eq. 13) is exact, not an
approximation. Per state, a group of N_GROUP = 12 actions
is sampled once from the student's initial policy (one
on-policy rollout batch, trained on for STEPS epochs --
the multi-epoch-on-one-batch regime). The verifier labels
ONLY the sampled pairs (V = 1 if the action is the state's
correct action, else 0): budget = S * N_GROUP labels per
arm, identical sampled groups and labels across arms
1-5. c_s is the paper's Eq. 9 correlation per state;
zero-variance groups take the paper's fallback coefficient
0.25. The projected arm's loss is the paper's Eq. 17:
KL(p || p_T) - coef_s * sum_a p(a|s) d(s,a) with
coef_s = LAM_V * c_s (or the fallback). The teacher's
implicit-reward direction d(s,a) = log p_T(a|s) -
log p_ref(a|s) with p_ref uniform is known EXACTLY in
the toy.

Toy world: A = 10 actions, S = 30 states per seed, 5
seeds (per-seed action identities are re-drawn, so the
seeds are 5 genuine worlds). Each state has a correct
action, a decoy and a habit action, all distinct.
Teacher: in AGREE states (15) p_T puts 0.55 on the
correct action; in DISAGREE states (15) p_T puts 0.55
on the decoy and only 0.05 on the correct action -- the
teacher is strong but wrong on half the world, by
construction, so the teacher-prior-vs-verifier-information
boundary is measured, not assumed. Student init puts
0.34 on the habit action and only 0.04 on the correct
action, so sampling is sparse/biased: the correct action
is never sampled in 15-21 of 30 states per seed, and --
structurally -- EVERY never-sampled state has an
all-incorrect group, i.e. zero verifier variance, i.e.
it sits in the paper's fallback regime. That fact is
itself one of the findings below.

Arms (matched verifier budget AND matched update count:
same init, same STEPS = 400 SGD steps at LR = 0.5, same
rollout batch for arms 1-5):
  (1) nofb      -- plain OPD, KL(p || p_T) only.
  (2) sampled   -- OPD + GRPO-style policy gradient on
                   the sampled pairs only (standardized
                   verifier advantages on log p(a_i|s)).
                   The naive loop: feedback never leaves
                   sampled actions.
  (3) projected -- VOPD as above (OPD loss minus the
                   projected, Rao-Blackwellized dense
                   term over the full action set).
  (4) oracle    -- OPD + dense TRUE-reward term: the
                   verifier labels EVERY action
                   (r = +1 correct, -0.25 else), spread
                   densely. Ceiling; budget-exempt and
                   labeled as such.
  (5) shuffled  -- projected, but d is replaced by a
                   per-state random permutation of d
                   (same multiset, action identity
                   destroyed) in both the c computation
                   and the dense spread. Must fail.
  (6) sampled2x -- sampled-only with TWICE the verifier
                   budget (24 pairs/state): the
                   budget-fairness rival -- what simply
                   buying more labels buys. Budget-UNFAIR
                   by construction and labeled as such;
                   its never-sampled readout is w.r.t.
                   the shared budget-B group, so it
                   directly samples many of those actions.

Readouts: student mass on the correct action, overall
and on the never-sampled subset (headline), split by
teacher AGREE / DISAGREE; TV distance to the
verifier-consistent one-hot target.

Measured on pop-os (particlegan 0.10.1), 5 seeds, mean
student mass on the correct action:
  arm        all    never   never-agree  never-dis  agree-all  dis-all   TV
  nofb       0.300   0.292      0.550       0.050      0.550     0.050   0.700
  sampled    0.425   0.292      0.550       0.050      0.742     0.109   0.575
  projected  0.430   0.349      0.684       0.034      0.812     0.047   0.570
  oracle     0.836   0.831      0.981       0.691      0.981     0.691   0.164
  shuffled   0.330   0.291      0.550       0.050      0.578     0.082   0.670
  sampled2x  0.486   0.488      0.884       0.117      0.856     0.116   0.514
Never-sampled recovery share, (projected - sampled) /
(oracle - sampled): pooled 0.105; teacher-AGREE subset
0.31; teacher-DISAGREE subset negative (projected is
BELOW sampled-only there). c_s means: +0.88..+0.90 in
agree states, -0.11..-0.19 in disagree states (per-seed
ranges, all five seeds negative after the round-5
group-size fix). Shuffled matches sampled-only on the
never-sampled readout (0.291 vs 0.292): the gain is the
teacher direction's identity, not densification per se.

Verdicts encoded below -- PARTIAL:
  (a) PROVEN at toy scale, on teacher-agree states:
      projected densification supervises never-sampled
      correct actions, +0.134 mass over sampled-only
      (0.684 vs 0.550), recovering 31% of the dense
      oracle's gain there, positive on every seed, with
      the shuffled-direction control flat. Pooled over
      all never-sampled states the margin is +0.057
      (10.5% of the oracle gain).
  (b) The mechanism on never-sampled states is the
      paper's ZERO-VARIANCE FALLBACK, not the
      correlation projection: every never-sampled state
      has an all-incorrect group by construction, so c_s
      is undefined there and what densifies is
      verifier-GATED teacher extrapolation (fixed 0.25),
      not verifier information about that state. The
      projection proper earns its margin on states WITH
      verifier variance (agree-all 0.812 vs sampled
      0.742).
  (c) NOT REPRODUCED where the teacher disagrees with
      the verifier: in disagree states the projected
      arm LOSES to sampled-only, overall (0.047 vs
      0.109) and on never-sampled states (0.034 vs
      0.050, where the fallback extrapolates toward the
      decoy). The estimated c_s is correctly negative
      but small (-0.11..-0.19): reversing along d
      spreads the freed mass over ALL low-d actions,
      while sampled-only PG pushes the specific verified
      action. Densification propagates verifier
      information only as far as the teacher direction
      can represent it -- exactly the paper's own
      "teacher-representable component" caveat, here
      measured as a sign flip in who wins.
  (d) Budget-fairness: at TWICE the verifier budget,
      sampled-only beats projected-at-B on the
      never-sampled readout (0.488 vs 0.349) -- mostly
      by sampling the "never-sampled" actions itself.
      Densification is worth ~a tenth of the oracle
      gain for free; it is not worth more than labels.

Iteration log (one refinement per round):
  1. baseline design (group 8, init correct mass 0.07,
     oracle weight 1): projected never 0.361 vs sampled
     0.301 vs oracle 0.484; recovery 33% against a weak
     ceiling; disagree c_s sign noisy across seeds.
  2. sparser sampling (init correct mass 0.07 -> 0.04):
     never-sampled states 17-21 -> 22-23 of 30; margins
     hold (projected 0.336 vs sampled 0.282).
  3. oracle weight 1 -> 3 (a real ceiling): oracle
     never 0.459 -> 0.826; pooled recovery share drops
     to 0.099 -- the round-1 share was a weak-ceiling
     artifact.
  4. verifier weight LAM_V 2 -> 4: almost nothing moves
     (agree-all 0.777 -> 0.781) -- verifier weight is
     not the bottleneck; reverted.
  5. group size 8 -> 12 (matched budget rise, all
     arms): disagree c_s becomes consistently negative
     on all 5 seeds (-0.11..-0.19); the boundary in
     (c) is estimation-noise-free. Final design.
  6. convergence check (STEPS 400 -> 800 in the
     prototype): readouts identical to 2 decimals;
     STEPS = 400 is at equilibrium.
  7. torch port, first pop-os run: 2 of 8 tests fail --
     a mean-over-states in the ported loss shrank every
     (independent) state row's gradient by 1/S, an
     effective LR of 0.017 instead of 0.5. Port bug,
     not a design signal.
  8. per-state losses summed in the port: all 8 tests
     pass on pop-os and every readout reproduces the
     numpy prototype table to 3 decimals (per-seed
     never-agree gain exactly +0.134 on all 5 seeds --
     those states sit at the fallback equilibrium,
     which is seed-independent given the teacher).

What is NOT asserted: the paper's benchmark numbers,
anything about sequence-level VOPD (where one prompt
group's c_x is shared across many prefixes -- a coupling
this bandit reduction removes by construction), and
anything about real LM teachers, whose implicit-reward
directions are not known exactly and whose disagreement
with a verifier is structured, not a swapped decoy.
Pure torch categorical geometry -- no GAN machinery,
because the claim is about the feedback objective, not
training a GAN. Skips cleanly when the installed
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


A = 10               # actions per state
S_AGREE = 15         # states where the teacher's top action is correct
S_DIS = 15           # states where the teacher's top action is a decoy
S = S_AGREE + S_DIS
N_GROUP = 12         # sampled actions per state (verifier budget / state)
STEPS = 400          # SGD steps on the fixed rollout batch, every arm
LR = 0.5
LAM_V = 2.0          # verifier weight on the projected dense term
LAM_PG = 1.0         # sampled-only policy-gradient weight
LAM_O = 3.0          # oracle dense true-reward weight
FALLBACK = 0.25      # the paper's zero-variance fallback coefficient
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("nofb", "sampled", "projected", "oracle", "shuffled", "sampled2x")


def _build(seed):
    """One world + one shared rollout batch (arms 1-5 identical)."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(S, dtype=int)
    decoy = np.zeros(S, dtype=int)
    habit = np.zeros(S, dtype=int)
    p_t = np.zeros((S, A))
    p_init = np.zeros((S, A))
    for s in range(S):
        perm = rng.permutation(A)
        correct[s], decoy[s], habit[s] = perm[0], perm[1], perm[2]
        rest = [a for a in range(A)
                if a not in (correct[s], decoy[s], habit[s])]
        if s < S_AGREE:
            p_t[s, correct[s]] = 0.55
            p_t[s, decoy[s]] = 0.10
            p_t[s, habit[s]] = 0.08
        else:
            p_t[s, decoy[s]] = 0.55
            p_t[s, correct[s]] = 0.05
            p_t[s, habit[s]] = 0.08
        rem = 1.0 - p_t[s].sum()
        for a in rest:
            p_t[s, a] = rem / len(rest)
        p_init[s, habit[s]] = 0.34
        p_init[s, correct[s]] = 0.04
        p_init[s, decoy[s]] = 0.07
        rem = 1.0 - p_init[s].sum()
        for a in rest:
            p_init[s, a] = rem / len(rest)
    group = np.stack([rng.choice(A, size=N_GROUP, p=p_init[s])
                      for s in range(S)])
    group2 = np.stack([rng.choice(A, size=2 * N_GROUP, p=p_init[s])
                       for s in range(S)])
    d_shuf = np.stack([np.log(p_t[s])[rng.permutation(A)]
                       - np.log(1.0 / A) for s in range(S)])
    return correct, p_t, p_init, group, group2, d_shuf


def _proj_coeff(ver, dvals):
    """The paper's Eq. 9 correlation per state; NaN if undefined."""
    c = np.full(S, np.nan)
    for s in range(S):
        sv, sd = ver[s].std(), dvals[s].std()
        if sv > 1e-9 and sd > 1e-9:
            c[s] = np.mean(((ver[s] - ver[s].mean()) / sv)
                           * ((dvals[s] - dvals[s].mean()) / sd))
    return c


def _advantages(ver):
    out = np.zeros_like(ver)
    for s in range(S):
        if ver[s].std() > 1e-9:
            out[s] = (ver[s] - ver[s].mean()) / ver[s].std()
    return out


def _run_arm(arm, seed):
    correct, p_t, p_init, group, group2, d_shuf_np = _build(seed)
    log_p_t = torch.tensor(np.log(p_t), dtype=torch.float64)
    d = log_p_t - float(np.log(1.0 / A))          # implicit reward, exact
    d_shuf = torch.tensor(d_shuf_np, dtype=torch.float64)
    grp = torch.tensor(group)
    grp2 = torch.tensor(group2)
    ver = (group == correct[:, None]).astype(np.float64)
    ver2 = (group2 == correct[:, None]).astype(np.float64)
    d_at = d.numpy()[np.arange(S)[:, None], group]
    d_shuf_at = d_shuf_np[np.arange(S)[:, None], group]
    c_true = _proj_coeff(ver, d_at)
    c_shuf = _proj_coeff(ver, d_shuf_at)
    coef = torch.tensor(
        np.where(np.isnan(c_true), FALLBACK, LAM_V * c_true),
        dtype=torch.float64)
    coef_shuf = torch.tensor(
        np.where(np.isnan(c_shuf), FALLBACK, LAM_V * c_shuf),
        dtype=torch.float64)
    adv = torch.tensor(_advantages(ver), dtype=torch.float64)
    adv2 = torch.tensor(_advantages(ver2), dtype=torch.float64)
    reward = torch.tensor(
        np.where(np.arange(A)[None, :] == correct[:, None], 1.0, -0.25),
        dtype=torch.float64)

    theta = torch.tensor(np.log(p_init), dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.SGD([theta], lr=LR)
    rows = torch.arange(S)
    for _ in range(STEPS):
        opt.zero_grad()
        logp = torch.log_softmax(theta, dim=1)
        p = logp.exp()
        # Per-state losses SUMMED (theta rows are independent;
        # a mean over states would shrink every row's gradient
        # by 1/S and silently change the step size).
        loss = (p * (logp - log_p_t)).sum(dim=1).sum()   # OPD: KL(p||p_T)
        if arm == "sampled":
            loss = loss - LAM_PG * (adv * logp[rows[:, None], grp]).mean(dim=1).sum()
        elif arm == "sampled2x":
            loss = loss - LAM_PG * (adv2 * logp[rows[:, None], grp2]).mean(dim=1).sum()
        elif arm == "projected":
            loss = loss - (coef * (p * d).sum(dim=1)).sum()
        elif arm == "shuffled":
            loss = loss - (coef_shuf * (p * d_shuf).sum(dim=1)).sum()
        elif arm == "oracle":
            loss = loss - LAM_O * (p * reward).sum(dim=1).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.log_softmax(theta, dim=1).exp().numpy()
    never = np.array([correct[s] not in group[s] for s in range(S)])
    agree = np.arange(S) < S_AGREE
    pc = p[np.arange(S), correct]
    tv = 0.5 * np.abs(p - np.eye(A)[correct]).sum(axis=1)
    out = {
        "pc_all": float(pc.mean()),
        "pc_never": float(pc[never].mean()),
        "pc_never_agree": float(pc[never & agree].mean()),
        "pc_never_dis": float(pc[never & ~agree].mean()),
        "pc_agree": float(pc[agree].mean()),
        "pc_dis": float(pc[~agree].mean()),
        "tv": float(tv.mean()),
        "n_never": int(never.sum()),
        "frac_fallback_never": float(np.isnan(c_true[never]).mean()),
        "c_agree": float(np.nanmean(c_true[agree])),
        "c_dis": float(np.nanmean(c_true[~agree])),
    }
    return out


@pytest.fixture(scope="module")
def results():
    _empf_recipe()
    return {arm: [_run_arm(arm, s) for s in SEEDS] for arm in ARMS}


def _mean(results, arm, key):
    return float(np.mean([r[key] for r in results[arm]]))


def test_never_sampled_states_are_all_in_fallback_regime(results):
    # Structural fact the verdict rests on: a never-sampled
    # correct action means an all-incorrect group, hence zero
    # verifier variance, hence the paper's fallback -- on
    # every seed, for every never-sampled state.
    for arm in ("nofb",):
        for r in results[arm]:
            assert r["n_never"] >= 10
            assert r["frac_fallback_never"] == 1.0


def test_projection_coefficient_signs(results):
    # c_s is estimated from the shared batch (arm-independent);
    # agree states: strongly positive on every seed; disagree
    # states: negative on every seed (round-5 group size).
    for r in results["nofb"]:
        assert r["c_agree"] > 0.7
        assert r["c_dis"] < 0.0


def test_projected_supervises_never_sampled_agree_states(results):
    # Headline (claim a): projected beats sampled-only on
    # never-sampled correct actions where the teacher agrees
    # with the verifier -- demonstrated +0.134, asserted +0.08.
    for s in SEEDS:
        gain = (results["projected"][s]["pc_never_agree"]
                - results["sampled"][s]["pc_never_agree"])
        assert gain > 0.05, (s, gain)
    gain = (_mean(results, "projected", "pc_never_agree")
            - _mean(results, "sampled", "pc_never_agree"))
    assert gain > 0.08
    # Recovery share of the oracle's gain there: demonstrated
    # 0.31, asserted >= 0.20.
    share = ((_mean(results, "projected", "pc_never_agree")
              - _mean(results, "sampled", "pc_never_agree"))
             / (_mean(results, "oracle", "pc_never_agree")
                - _mean(results, "sampled", "pc_never_agree")))
    assert share > 0.20


def test_projected_beats_sampled_pooled_never_sampled(results):
    # Pooled over agree + disagree never-sampled states the
    # margin survives but shrinks: demonstrated +0.057,
    # asserted +0.03; pooled recovery share demonstrated
    # 0.105, asserted >= 0.05.
    gain = (_mean(results, "projected", "pc_never")
            - _mean(results, "sampled", "pc_never"))
    assert gain > 0.03
    share = (gain / (_mean(results, "oracle", "pc_never")
                     - _mean(results, "sampled", "pc_never")))
    assert share > 0.05


def test_shuffled_direction_control_is_flat(results):
    # Densifying along a permuted direction buys nothing on
    # never-sampled actions: demonstrated 0.291 vs sampled
    # 0.292 and projected 0.349.
    shuf = _mean(results, "shuffled", "pc_never")
    samp = _mean(results, "sampled", "pc_never")
    proj = _mean(results, "projected", "pc_never")
    assert abs(shuf - samp) < 0.02
    assert proj - shuf > 0.03


def test_disagree_boundary_projected_loses_to_sampled(results):
    # Claim (c), the measured boundary: where the teacher
    # disagrees with the verifier, densification loses to
    # sampled-only propagation -- overall (demonstrated
    # 0.047 vs 0.109) and on never-sampled states, where
    # the fallback extrapolates toward the decoy
    # (demonstrated 0.034 vs 0.050).
    assert (_mean(results, "projected", "pc_dis")
            < _mean(results, "sampled", "pc_dis") - 0.03)
    assert (_mean(results, "projected", "pc_never_dis")
            <= _mean(results, "sampled", "pc_never_dis"))


def test_oracle_is_the_ceiling(results):
    oracle = _mean(results, "oracle", "pc_never")
    for arm in ARMS:
        if arm != "oracle":
            assert oracle > _mean(results, arm, "pc_never")


def test_budget_rival_two_x_labels_beats_projection(results):
    # Claim (d), asserted honestly against the paper's
    # framing: doubling the verifier budget and spending
    # it on more sampled pairs beats densification at the
    # matched budget on the never-sampled readout
    # (demonstrated 0.488 vs 0.349) -- largely by sampling
    # the "never-sampled" actions itself.
    assert (_mean(results, "sampled2x", "pc_never")
            > _mean(results, "projected", "pc_never") + 0.05)
