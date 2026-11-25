"""CPU gate for the graded-verifier follow-up to PR #326
(VOPD -- Verifiable On-Policy Distillation, arXiv:2611.30325,
Shi et al.). This is a CHAIN test: it reuses #326's toy
construction verbatim and changes exactly one thing -- the
verifier's feedback channel, binary -> graded.

What #326 found (cycle 29, measured there): projecting
sparse BINARY verifier feedback onto the teacher's
implicit-reward direction (Rao-Blackwellized over the
vocabulary) raises never-sampled correct-action mass
(sampled-only 0.292 -> projected 0.349 pooled; dense
oracle 0.831; pooled recovery share 10.5% of the oracle
gain). Split: teacher-AGREE never-sampled 0.550 -> 0.684
(+0.134 on every seed); teacher-DISAGREE never-sampled
0.050 -> 0.034 (the projection LOSES -- it extrapolates
toward the decoy). Structural finding: with a binary
verifier + sparse sampling, every never-sampled state has
an all-incorrect sampled group BY CONSTRUCTION, so the
per-group coefficient c_s (paper Eq. 9) is undefined
there -- the headline never-sampled gain runs on the
paper's zero-variance FALLBACK (fixed 0.25, verifier-
gated teacher extrapolation), not on verifier information
about that state; the projection proper earns its margin
only on states WITH verifier variance. #326's own named
follow-up: a GRADED verifier is what would change that
picture. Standing scoping from cycle 29: #326's import is
gated on the per-group coefficient, with the fallback
caveat in binary-verifier regimes.

Follow-up question decided here: replace the binary
verifier with a GRADED one (partial credit). Group
variance now exists on states where the binary verifier
gave an all-incorrect group. (i) Does the Rao-Blackwellized
projection then earn its never-sampled margin through the
COEFFICIENT PROPER rather than the fallback -- i.e. how
much of the graded gain survives with the FALLBACK
DISABLED (the decisive ablation: fallback-off under
binary = sampled-only on the never-sampled subset, exact
in #326's analysis and re-measured here)? (ii) Does the
teacher-DISAGREE harm (0.050 -> 0.034 under binary) shrink,
persist, or grow when the verifier can express graded
disagreement?

GRADED OPERATIONALIZATION (this test's design choice,
stated plainly because the result turns on it): the
verifier returns a score in [0, 1] per sampled pair, at
the SAME number of verifier evaluations as the binary
arm (matched label budget = matched evaluations; the
graded channel is richer per evaluation, which is the
point of the comparison). Scores: correct action 1.0;
the DECOY 0.6; one designated near action 0.3; everything
else 0.0. Rationale: partial credit in real verifiers
(unit tests passed, rubric points) accrues to answers
that are substantively close to correct. In #326's world
the decoy IS the close-but-wrong action -- it is the
action the teacher finds plausible enough to put 0.55 on
in disagree states -- so it is the canonical near-miss,
and one further action (perm[3], teacher-neutral) is
weakly close. The habit action (a spurious student bias)
and the remaining actions are simply wrong: 0.0.
A SENSITIVITY variant (graded2) moves ALL partial credit
off the decoy onto teacher-neutral rest actions
(near perm[3] 0.6, mid perm[4] 0.3, decoy 0.0): it exists
to measure how much of the headline is the PLACEMENT of
partial credit rather than gradedness per se. It is a
first-class readout below, not a footnote.

Toy world: IDENTICAL to #326 (same _build, same rng
consumption, so the binary arms reproduce #326's table
exactly and serve as the replication anchor). A = 10
actions, S = 30 states/seed (15 teacher-AGREE, 15
teacher-DISAGREE), 5 seeds, group of N_GROUP = 12 sampled
actions/state from the student init (correct mass 0.04),
one on-policy batch trained for STEPS = 400 SGD steps at
LR = 0.5, all arms identical. Teacher implicit-reward
direction d(s,a) = log p_T(a|s) - log p_ref(a|s), p_ref
uniform, known exactly. Projected arms use the paper's
Eq. 17 loss KL(p||p_T) - coef_s * sum_a p(a|s) d(s,a)
with coef_s = LAM_V * c_s when c_s is defined, else the
paper's fallback 0.25 (fallback arms) or 0.0 (fallback-
DISABLED arms: the dense term is simply off for that
state).

Arms (matched verifier evaluations + matched updates):
  bin_sampled    -- #326 sampled-only (binary PG). Anchor.
  bin_proj_fb    -- #326 projected (binary, fallback on).
                    Anchor; must reproduce #326.
  bin_proj_nofb  -- binary projected, fallback DISABLED
                    (diagnostic: isolates how much of
                    #326's headline was the fallback).
  gr_sampled     -- sampled-only PG, GRADED advantages.
  gr_proj_fb     -- graded projected, fallback on.
  gr_proj_nofb   -- graded projected, FALLBACK DISABLED
                    (the decisive arm: coefficient proper
                    only).
  gr2_proj_fb    -- sensitivity placement, fallback on.
  gr2_proj_nofb  -- sensitivity placement, fallback off.
  oracle         -- dense verifier ceiling (#326's arm,
                    budget-exempt, labeled).

Readouts per seed: student mass on the correct action,
overall and on the never-sampled subset (correct action
absent from the shared 12-group), split teacher-agree /
teacher-disagree; the fraction of states where c_s is
DEFINED, per verifier regime, overall and on the never-
sampled subset; fallback-attributable share of the
never-sampled gain, (proj_fb - proj_nofb) /
(proj_fb - sampled) within each regime.

Measured on pop-os (particlegan 0.10.1), 5 seeds, mean
student mass on the correct action (numpy prototype and
torch port agree; see iteration log):
  arm           all    never  never-agree  never-dis
  bin_sampled   0.425   0.292      0.550       0.050
  bin_proj_fb   0.430   0.349      0.684       0.034
  bin_proj_nofb 0.396   0.292      0.550       0.050
  gr_sampled    0.434   0.342      0.703       0.004
  gr_proj_fb    0.444   0.385      0.765       0.030
  gr_proj_nofb  0.441   0.380      0.753       0.031
  gr2_proj_fb   0.315   0.165      0.255       0.081
  gr2_proj_nofb 0.311   0.158      0.239       0.082
  oracle        0.836   0.831      0.981       0.691
Coefficient DEFINED fraction, never-sampled subset:
binary 0.00 (every seed, #326's structural fact) ->
graded 0.91 (per-seed 0.84-0.95); overall 0.41 -> 0.95.
c_s on never-sampled states under graded: agree +0.32,
disagree +0.44 (both positive -- the decoy's partial
credit aligns with the teacher's decoy ranking in both
regimes). Under graded2 the same coefficients are
-0.36 / -0.31 (sign flip, measured).

Verdicts encoded below -- PARTIAL, and placement-bound:
  (a) PROVEN (primary graded design): the mechanism
      CHANGES from extrapolation to information. The
      coefficient is defined on 91% of never-sampled
      states (was 0%), and with the fallback DISABLED
      the graded projection keeps its never-sampled
      margin: pooled 0.380 vs graded-sampled 0.342
      (fallback-attributable share of the graded gain
      only 0.12 pooled, 0.19 on agree states) -- where
      binary fallback-off collapses to sampled-only
      exactly (0.292 = 0.292). Graded projection also
      beats binary projection pooled (0.385 vs 0.349)
      and on agree states (0.765 vs 0.684; recovery of
      the oracle's agree gain 50% vs #326's 31%).
  (b) The graded channel helps EVERY carrier, not just
      the projection: graded sampled-only alone reaches
      0.703 on agree never-sampled states (binary
      sampled: 0.550) -- nonzero advantages on
      never-sampled-correct groups move the sampled
      logits, and the OPD term refills toward the
      teacher, whose top bucket is the correct action.
      The projection proper adds +0.050 (agree) /
      +0.038 (pooled) on top of that, fallback-off.
  (c) The DISAGREE harm PERSISTS under graded, and its
      carrier changes: gr_proj_nofb never-disagree is
      0.031 vs binary sampled 0.050 (binary projected:
      0.034) -- because c_s there is now genuinely
      POSITIVE (+0.44): the graded verifier itself
      endorses the decoy at 0.6, so densification
      toward the teacher direction is verifier-faithful
      and still exact-correct-harmful. Graded sampled-
      only is worst of all there (0.004): partial credit
      for the decoy is partial credit for the wrong
      answer, and every graded carrier spends it.
  (d) PLACEMENT IS LOAD-BEARING (sensitivity): move
      partial credit off the decoy onto teacher-neutral
      actions and the coefficient on never-sampled
      states flips NEGATIVE in both regimes (-0.36 /
      -0.31 -- the only teacher-direction variation in
      an all-incorrect group is decoy/habit, both now
      scored 0 against graded-positive low-d rest
      actions). The fallback-off graded2 projection
      then DESTROYS agree never-sampled mass (0.239 vs
      0.550 sampled) while slightly helping disagree
      (0.082 vs 0.050, reversal away from the decoy).
      "Graded verifier" is not one intervention: WHICH
      wrong actions earn partial credit decides the
      sign of the coefficient on exactly the states
      densification exists to reach.
  (e) Partial-credit MAGNITUDE is not the driver: halving
      every partial credit (0.6->0.3, 0.3->0.15) leaves
      every readout unchanged to 3 decimals -- on
      never-sampled groups the rescaling is pure, and
      both c_s (a correlation) and standardized PG
      advantages are scale-invariant. Ordering, not
      dose.

Iteration log (one refinement per round):
  1. Baseline (primary graded design, #326 construction
     verbatim): binary anchor reproduces #326's table to
     3 decimals; graded defined-fraction on never-sampled
     0.00 -> 0.91; fallback-off retains the graded margin
     (0.380 vs 0.385); disagree harm persists (0.031);
     sensitivity placement flips c_s negative and
     agree-never collapses (0.239). Claims decided in
     shape; per-seed agree gain for gr_proj_nofb over
     gr_sampled is mean +0.050 but ranges -0.059..+0.155
     (assertions below are mean-based for graded arms,
     unlike #326's seed-constant fallback equilibrium).
  2. Convergence check (STEPS 400 -> 800, prototype):
     every readout identical to 3 decimals; 400 is at
     equilibrium for all nine arms.
  3. Partial-credit dose probe (all partial credits
     halved): readouts unchanged to 3 decimals -- the
     scale-invariance in verdict (e), measured, not
     assumed. (A first attempt at this round via a
     careless sed also rewrote the habit init mass and
     was discarded; the clean rerun is what is cited.)
  4. Torch port in #326's exact form (per-state losses
     summed, float64, SGD), pop-os run: 8/8 pass in 12.2s
     and the readout table reproduces the numpy prototype
     EXACTLY, every cell to 3 decimals (anchor included).
     Clean skip verified against the broken particlegan
     stub (module-level skip, no errors). Claims decided
     at round 1 in shape; rounds 2-4 were validity checks
     (convergence, dose, port parity), so the loop stops
     at 4 of 10.

What is NOT asserted: the paper's benchmark numbers,
sequence-level VOPD, real LM teachers/verifiers, and any
claim that graded feedback helps in general -- verdict
(d) is precisely the boundary on that. The graded2
variant is a placement sensitivity, not a proposed
verifier design. Pure torch categorical geometry -- no
GAN machinery, because the claim is about the feedback
objective. Skips cleanly when the installed particlegan
does not provide the EMPF recipe (house guard for this
paper-test family; PR #211 enables this test in CI).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
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
# arm -> (verifier regime, carrier mode)
ARM_SPEC = {
    "bin_sampled": ("binary", "pg"),
    "bin_proj_fb": ("binary", "proj_fb"),
    "bin_proj_nofb": ("binary", "proj_nofb"),
    "gr_sampled": ("graded", "pg"),
    "gr_proj_fb": ("graded", "proj_fb"),
    "gr_proj_nofb": ("graded", "proj_nofb"),
    "gr2_proj_fb": ("graded2", "proj_fb"),
    "gr2_proj_nofb": ("graded2", "proj_nofb"),
    "oracle": ("binary", "oracle"),
}
ARMS = tuple(ARM_SPEC)


def _build(seed):
    """One world + one shared rollout batch. IDENTICAL rng
    consumption to #326's _build (near/mid are extra indices
    into the same per-state permutation -- no new draws), so
    the binary arms reproduce #326 exactly."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(S, dtype=int)
    decoy = np.zeros(S, dtype=int)
    habit = np.zeros(S, dtype=int)
    near = np.zeros(S, dtype=int)
    mid = np.zeros(S, dtype=int)
    p_t = np.zeros((S, A))
    p_init = np.zeros((S, A))
    for s in range(S):
        perm = rng.permutation(A)
        correct[s], decoy[s], habit[s] = perm[0], perm[1], perm[2]
        near[s], mid[s] = perm[3], perm[4]
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
    return correct, decoy, habit, near, mid, p_t, p_init, group, group2, d_shuf


def _score_table(kind, correct, decoy, near, mid):
    """Full (S, A) verifier score table for a regime."""
    V = np.zeros((S, A))
    for s in range(S):
        V[s, correct[s]] = 1.0
        if kind == "graded":
            V[s, decoy[s]] = 0.6
            V[s, near[s]] = 0.3
        elif kind == "graded2":
            V[s, near[s]] = 0.6
            V[s, mid[s]] = 0.3
    return V


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
    kind, mode = ARM_SPEC[arm]
    (correct, decoy, habit, near, mid, p_t, p_init,
     group, group2, d_shuf_np) = _build(seed)
    log_p_t = torch.tensor(np.log(p_t), dtype=torch.float64)
    d = log_p_t - float(np.log(1.0 / A))          # implicit reward, exact
    grp = torch.tensor(group)
    Vtab = _score_table(kind, correct, decoy, near, mid)
    ver = Vtab[np.arange(S)[:, None], group]
    d_at = d.numpy()[np.arange(S)[:, None], group]
    c = _proj_coeff(ver, d_at)
    if mode == "proj_fb":
        coef_np = np.where(np.isnan(c), FALLBACK, LAM_V * c)
    elif mode == "proj_nofb":
        coef_np = np.where(np.isnan(c), 0.0, LAM_V * c)
    else:
        coef_np = np.zeros(S)
    coef = torch.tensor(coef_np, dtype=torch.float64)
    adv = torch.tensor(_advantages(ver), dtype=torch.float64)
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
        # Per-state losses SUMMED (#326 round-7 lesson: a mean
        # over states shrinks every independent row's gradient
        # by 1/S and silently changes the step size).
        loss = (p * (logp - log_p_t)).sum(dim=1).sum()   # OPD: KL(p||p_T)
        if mode == "pg":
            loss = loss - LAM_PG * (adv * logp[rows[:, None], grp]).mean(dim=1).sum()
        elif mode in ("proj_fb", "proj_nofb"):
            loss = loss - (coef * (p * d).sum(dim=1)).sum()
        elif mode == "oracle":
            loss = loss - LAM_O * (p * reward).sum(dim=1).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.log_softmax(theta, dim=1).exp().numpy()
    never = np.array([correct[s] not in group[s] for s in range(S)])
    agree = np.arange(S) < S_AGREE

    def _cmean(mask):
        vals = c[mask]
        return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")

    pc = p[np.arange(S), correct]
    out = {
        "pc_all": float(pc.mean()),
        "pc_never": float(pc[never].mean()),
        "pc_never_agree": float(pc[never & agree].mean()),
        "pc_never_dis": float(pc[never & ~agree].mean()),
        "pc_agree": float(pc[agree].mean()),
        "pc_dis": float(pc[~agree].mean()),
        "n_never": int(never.sum()),
        "frac_def": float((~np.isnan(c)).mean()),
        "frac_def_never": float((~np.isnan(c[never])).mean()),
        "c_never_agree": _cmean(never & agree),
        "c_never_dis": _cmean(never & ~agree),
    }
    return out


@pytest.fixture(scope="module")
def results():
    _empf_recipe()
    return {arm: [_run_arm(arm, s) for s in SEEDS] for arm in ARMS}


def _mean(results, arm, key):
    return float(np.mean([r[key] for r in results[arm]]))


def test_binary_anchor_reproduces_326(results):
    # The binary arms are #326's construction unchanged:
    # its headline margins must reappear (demonstrated:
    # pooled never 0.349 vs 0.292; never-agree +0.134;
    # never-disagree 0.034 vs 0.050), and the structural
    # fact must hold: c_s is defined on ZERO never-sampled
    # states under the binary verifier, every seed.
    for r in results["bin_sampled"]:
        assert r["frac_def_never"] == 0.0
        assert r["n_never"] >= 10
    gain = (_mean(results, "bin_proj_fb", "pc_never")
            - _mean(results, "bin_sampled", "pc_never"))
    assert gain > 0.03
    gain_agree = (_mean(results, "bin_proj_fb", "pc_never_agree")
                  - _mean(results, "bin_sampled", "pc_never_agree"))
    assert gain_agree > 0.08
    assert (_mean(results, "bin_proj_fb", "pc_never_dis")
            < _mean(results, "bin_sampled", "pc_never_dis"))


def test_graded_coefficient_defined_on_never_sampled(results):
    # The graded channel creates within-group variance on
    # states where the binary verifier's group was all-
    # incorrect: defined fraction on the never-sampled
    # subset 0.00 -> demonstrated 0.91 (per-seed 0.84-0.95);
    # overall 0.41 -> 0.95.
    for r in results["gr_sampled"]:
        assert r["frac_def_never"] > 0.70
    assert _mean(results, "gr_sampled", "frac_def_never") > 0.80
    assert _mean(results, "gr_sampled", "frac_def") > 0.90
    assert (_mean(results, "gr_sampled", "frac_def")
            > _mean(results, "bin_sampled", "frac_def") + 0.40)


def test_fallback_disabled_graded_retains_gain(results):
    # THE decisive ablation. Under binary, fallback-off
    # collapses to sampled-only on the never-sampled
    # subset EXACTLY (demonstrated 0.292 = 0.292): #326's
    # headline there was all fallback. Under graded, the
    # fallback-off arm keeps its margin over graded
    # sampled-only (demonstrated pooled +0.038, agree
    # +0.050): the coefficient proper now earns it.
    bin_gap = abs(_mean(results, "bin_proj_nofb", "pc_never")
                  - _mean(results, "bin_sampled", "pc_never"))
    assert bin_gap < 0.005
    gain = (_mean(results, "gr_proj_nofb", "pc_never")
            - _mean(results, "gr_sampled", "pc_never"))
    assert gain > 0.02
    gain_agree = (_mean(results, "gr_proj_nofb", "pc_never_agree")
                  - _mean(results, "gr_sampled", "pc_never_agree"))
    assert gain_agree > 0.03
    # Fallback-attributable share of the graded gain:
    # demonstrated 0.12 pooled -- asserted < 0.35, against
    # the binary regime's 1.00 by construction above.
    share = ((_mean(results, "gr_proj_fb", "pc_never")
              - _mean(results, "gr_proj_nofb", "pc_never"))
             / (_mean(results, "gr_proj_fb", "pc_never")
                - _mean(results, "gr_sampled", "pc_never")))
    assert share < 0.35


def test_graded_projection_beats_binary_projection(results):
    # Graded + projection + fallback vs the #326 arm:
    # demonstrated pooled never 0.385 vs 0.349, never-
    # agree 0.765 vs 0.684 (oracle-gain recovery on agree
    # states 50% vs #326's 31%).
    assert (_mean(results, "gr_proj_fb", "pc_never")
            > _mean(results, "bin_proj_fb", "pc_never") + 0.02)
    assert (_mean(results, "gr_proj_fb", "pc_never_agree")
            > _mean(results, "bin_proj_fb", "pc_never_agree") + 0.04)


def test_graded_sampled_carrier_also_gains_on_agree(results):
    # Verdict (b): the graded channel helps even the naive
    # carrier on agree never-sampled states (demonstrated
    # 0.703 vs binary sampled 0.550) -- nonzero advantages
    # on all-incorrect-binary groups move sampled logits
    # and the OPD refill favours the teacher's top bucket.
    assert (_mean(results, "gr_sampled", "pc_never_agree")
            > _mean(results, "bin_sampled", "pc_never_agree") + 0.08)


def test_disagree_harm_persists_under_graded(results):
    # Verdict (c): the harm does NOT shrink. The graded
    # coefficient on never-sampled disagree states is
    # genuinely positive (demonstrated +0.44 -- the
    # verifier endorses the decoy at 0.6 and the teacher
    # ranks it top), the fallback-off projection still
    # loses exact-correct mass there (0.031 vs binary
    # sampled 0.050), and graded sampled-only is worst
    # of all (0.004).
    assert _mean(results, "gr_sampled", "c_never_dis") > 0.30
    assert (_mean(results, "gr_proj_nofb", "pc_never_dis")
            < _mean(results, "bin_sampled", "pc_never_dis") - 0.01)
    assert _mean(results, "gr_sampled", "pc_never_dis") < 0.02


def test_partial_credit_placement_flips_coefficient(results):
    # Verdict (d), the sensitivity: partial credit placed
    # on teacher-neutral rest actions (decoy scored 0)
    # flips c_s on never-sampled states NEGATIVE in both
    # regimes (demonstrated -0.36 agree / -0.31 disagree),
    # and the fallback-off projection then destroys agree
    # never-sampled mass (0.239 vs 0.550) while slightly
    # helping disagree (0.082 vs 0.050). Placement, not
    # gradedness, sets the sign.
    assert _mean(results, "gr2_proj_nofb", "c_never_agree") < -0.20
    assert _mean(results, "gr2_proj_nofb", "c_never_dis") < -0.20
    assert (_mean(results, "gr2_proj_nofb", "pc_never_agree")
            < _mean(results, "bin_sampled", "pc_never_agree") - 0.15)
    assert (_mean(results, "gr2_proj_nofb", "pc_never_dis")
            > _mean(results, "bin_sampled", "pc_never_dis") + 0.01)


def test_oracle_is_the_ceiling(results):
    oracle = _mean(results, "oracle", "pc_never")
    for arm in ARMS:
        if arm != "oracle":
            assert oracle > _mean(results, arm, "pc_never")
