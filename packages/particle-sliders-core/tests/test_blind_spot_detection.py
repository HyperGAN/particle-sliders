"""CPU gate for the CHAIN claim of the Particle Sliders paper
watch, cycles 32 x 27: when the trait installed in a student
IS a judge blind spot's category refusal, which audit arm can
SEE it? Chain of two proven anchors:

  * PR #320 (Blind Spots Travel, arXiv:2611.25191, cycle 27):
    a judge's CATEGORY blind spot transfers through curation
    and installs category refusal in the student (toy: >0.90
    held-out refusal in the blind category, nothing elsewhere,
    nothing from a random judge); per-item knowledge does not
    transfer. The loss is invisible to accuracy among answered
    questions, which stays at the answer-key arm's level.
  * PR #340 (Refusal Is Not Scored, arXiv:2611.33690,
    cycle 32): an audit that DISCARDS the refusal category
    before scoring is blind to refusal disparity; passing
    refusals to the scorer as clean SHRINKS the measured
    disparity (and sign-flips it in the zero-scored-bias
    world); only refusal-rate-as-outcome recovers the planted
    disparity (ratio ~2.0x at planted rates 0.10 vs 0.20).

Neither anchor tested the pair. #320 installed the refusal
but audited it with the refusal rate itself; #340 planted the
refusal disparity directly, with no installation step. The
chain question: run #320's installation, then read the
student with #340's three audit arms -- does the audit see
what curation installed?

Toy operationalization. Stage 1 (installation, #320's
construction verbatim): 4 categories x 15 training items;
the student factorises into per-item knowledge
d_i ~ N(1.4, 0.45), a per-item bias delta_i trained only on
that item, a small shared answer-side drift s_cat, and a
per-CATEGORY answer/abstain gate u_cat (init 1.2). 35
episodes of judge rewards move the gate. Students:
  (a) BLIND-curated: answer key everywhere except category
      0, where a TRUE answer is rejected with prob 11/12.
  (b) SIGHTED-curated: the pure answer key (control student).
  (c) RANDOM-curated: coin-flip rewards (shuffled-labels
      control -- no category belief, no installation).
  (d) SIGHTED-2: a second answer-key student on an
      independent reward stream (zero-disparity control).

Stage 2 (audit, #340's apparatus): for each student and
each category, draw N=8,000 audit responses (#340's cell
discipline): refused with the student's installed gate
probability 1 - sigmoid(u_cat); answered items draw scored
bias ~ N(0.30, 0.10) clipped to [0, 1] -- the SAME mean for
both students, which is #320's demonstrated hidden-loss
fact (answered accuracy matches between blind and key
arms), taken as the scored-content input. The audit's
comparison axis is student identity: disparities are
blind-curated MINUS sighted-curated, per category. Arms:
  (1) FILTERED: discard refusals, score the remainder.
  (2) UNFILTERED: refusals passed to the scorer, scored
      clean (0.0), per #340's lexicon-scorer finding.
  (3) REFUSAL-RATE-as-outcome: refusal rate per student /
      category as a first-class outcome. Reported as a
      DIFFERENCE, not #340's ratio: the sighted student's
      installed refusal is ~0.000, so the ratio form
      degenerates (division by a sampling-noise denominator);
      the difference is the recovery statistic that survives
      the chain.

Predictions (from the anchors' demonstrated margins):
the blind student installs refusal ~0.91 in category 0;
FILTERED reports ~0 there -- it certifies the installed
student clean on exactly the blind category; UNFILTERED
goes strongly NEGATIVE (~-0.27): the installed refusals,
scored clean, make the blind-curated student look LESS
biased than the sighted one -- #340's zero-bias sign-flip,
amplified because the installed refusal (0.91) dwarfs
#340's planted one (0.20); REFUSAL-RATE recovers the
installed disparity (audit diff within sampling noise of
the analytic gate diff). Controls: category 1 (non-blind)
shows no disparity in any arm; the random-curated student
shows at most diffuse gate drift (<= ~0.48), far below the
installed 0.91; the sighted-vs-sighted audit shows ~0 in
every arm (no false positives).

Modeling gifts, stated plainly: the factorised gate is
#320's stated gift (gate and knowledge are separate
parameters, as the paper's blocked-abstention result
licenses); the scored-bias distribution among answered
items is planted equal across students (that equality IS
#320's measured hidden-loss result, not an assumption
smuggled in); the scorer's clean-0 for refusal text is
#340's empirical lexicon finding taken as given. The claim
tested is the pipeline arithmetic of the chain: curation
installs in the refusal coordinate, and only an audit that
reads the refusal coordinate can see it.

CODEBASE LANDING: scripts/check_slider_gate.py's check()
and the gate's audit design. Rule form: a gate audit that
would detect blind-spot installation MUST carry a
per-category refusal-rate readout (in our seat: the
UNKNOWN rate -- UNKNOWN is our refusal class, the Tiny Aya
seat) as a first-class outcome, differenced against a
sighted/control reference; a filtered score audit certifies
nothing about category coverage, and an unfiltered score
audit is worse than silent -- it actively flatters the
installed student. Beside #320's rule (probe category
coverage) and #340's rule (never filter refusals before
scoring): the audit must COMPARE refusal rates across
students or arms, because a single student's score profile
cannot reveal its own blind spot.

Measured on pop-os (particlegan 0.10.1), 5 seeds -- see the
PR body for the full table; assertions below encode the
demonstrated margins with modest headroom. Pure numpy toy
on CPU, single-threaded, seeded. Skips cleanly when the
installed particlegan does not provide the EMPF recipe
(the house guard for this paper-test family, PR #273's
guarded-import pattern).
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


# ------------------------------------------------------- stage 1: install
# #320's construction, verbatim dynamics.
CATS = 4             # question categories; category 0 is the blind one
NTR = 15             # training items per category
EPISODES = 35        # passes over the training items
U0 = 1.2             # init category-gate logit, P(answer) ~ 0.77
LR_U = 0.03          # gate step per unit reward
LR_S = 0.003         # shared answer-side drift step
LR_D = 0.10          # per-item (delta) step -- never reaches held-out
CONS_K = 32          # samples for the (unused-here) consensus stream
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("key", "blind", "random", "key2")
BLIND = 0
CONTROL_CAT = 1

# --------------------------------------------------------- stage 2: audit
# #340's apparatus: n per cell, scored bias among answered.
N_AUDIT = 8000
BIAS_MEAN = 0.30     # identical across students (#320 hidden loss)
SCORE_SD = 0.10


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _knowledge(seed):
    """Per-seed knowledge streams, identical across arms (#320)."""
    rng = np.random.default_rng(1000 + seed)
    d_tr = rng.normal(1.4, 0.45, size=(CATS, NTR))
    d_he = rng.normal(1.4, 0.45, size=(CATS, NTR))
    cons = np.zeros((CATS, NTR), dtype=int)
    for c in range(CATS):
        for i in range(NTR):
            draws = rng.binomial(CONS_K, _sigmoid(d_tr[c, i]))
            cons[c, i] = 1 if draws >= CONS_K // 2 else 0
    return d_tr, d_he, cons


def _train(seed, arm):
    """One #320 training run. Returns the installed per-category
    refusal probabilities 1 - sigmoid(u_cat). 'key2' is a second
    answer-key student on an independent reward stream."""
    d_tr, _, cons = _knowledge(seed)
    rng = np.random.default_rng(2000 + seed * 10 + ARMS.index(arm))
    u = np.full(CATS, U0)
    s_cat = np.zeros(CATS)
    delta = np.zeros((CATS, NTR))
    judge = "key" if arm == "key2" else arm
    for _ in range(EPISODES):
        for c in range(CATS):
            for i in range(NTR):
                if rng.random() > _sigmoid(u[c]):
                    continue  # abstains: reward 0, no update
                q = _sigmoid(d_tr[c, i] - delta[c, i] + s_cat[c])
                is_true = rng.random() < q
                if judge == "key":
                    r = 1 if is_true else -1
                elif judge == "blind":
                    if c == BLIND and is_true:
                        # rejects 11 of 12 correct answers
                        r = 1 if rng.random() < 1 / 12 else -1
                    else:
                        r = 1 if is_true else -1
                else:  # random: coin flip, no category belief
                    r = 1 if rng.random() < 0.5 else -1
                u[c] += LR_U * r
                s_cat[c] += LR_S * r * (1 if is_true else -1)
                delta[c, i] += LR_D * r * (-1 if is_true else 1)
                delta[c, i] = np.clip(delta[c, i], -2.0, 2.0)
    return 1.0 - _sigmoid(u)


def _audit_cell(rng, refusal_p):
    """#340's cell: refused mask + bias scores (refusals score 0)."""
    refused = rng.random(N_AUDIT) < refusal_p
    scores = np.clip(rng.normal(BIAS_MEAN, SCORE_SD, N_AUDIT), 0.0, 1.0)
    scores = np.where(refused, 0.0, scores)
    return refused, scores


def _audit_pair(seed, refusal_test, refusal_ref, cat):
    """#340's three arms, test student minus reference student,
    on one category."""
    rng = np.random.default_rng(5000 + seed)
    ra, sa = _audit_cell(rng, refusal_ref[cat])
    rb, sb = _audit_cell(rng, refusal_test[cat])
    return {
        # Arm 1: filtered -- score answered items only.
        "filtered": float(sb[~rb].mean() - sa[~ra].mean()),
        # Arm 2: unfiltered -- refusals scored clean (0.0, planted).
        "unfiltered": float(sb.mean() - sa.mean()),
        # Arm 3: refusal rate as a first-class outcome.
        "refusal_ref": float(ra.mean()),
        "refusal_test": float(rb.mean()),
        "refusal_diff": float(rb.mean() - ra.mean()),
        # The analytic installed disparity (gate probabilities).
        "installed_diff": float(refusal_test[cat] - refusal_ref[cat]),
    }


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        gates = {a: _train(seed, a) for a in ARMS}
        _CACHE[seed] = {
            "gates": gates,
            "target": _audit_pair(seed, gates["blind"], gates["key"],
                                  BLIND),
            "control": _audit_pair(seed, gates["blind"], gates["key"],
                                   CONTROL_CAT),
            "shuffled": _audit_pair(seed, gates["random"], gates["key"],
                                    BLIND),
            "zero": _audit_pair(seed, gates["key2"], gates["key"],
                                BLIND),
        }
    return _CACHE[seed]


# ------------------------------------------------------------------ tests
def test_filtered_certifies_clean_unfiltered_flatters():
    _empf_recipe()
    for seed in SEEDS:
        out = run_seed(seed)
        tgt = out["target"]
        # Installation landed (#320): the blind-curated student's
        # gate refuses the blind category (demonstrated analytic
        # 0.907..0.923; audit-sampled diff 0.903..0.921).
        assert out["gates"]["blind"][BLIND] >= 0.80, (
            f"seed {seed}: installed refusal "
            f"{out['gates']['blind'][BLIND]:.3f}")
        assert tgt["refusal_diff"] >= 0.80, (
            f"seed {seed}: audit refusal diff {tgt['refusal_diff']:+.3f}")
        # Arm 1 (FILTERED) certifies the student clean on exactly
        # the blind category (demonstrated -0.005..+0.003) while
        # the installed disparity stands in the discarded pile.
        assert abs(tgt["filtered"]) <= 0.02, (
            f"seed {seed}: filtered {tgt['filtered']:+.3f} "
            f"vs installed diff {tgt['installed_diff']:+.3f}")
        # Arm 2 (UNFILTERED) sign-flips: clean-scored refusals
        # make the blind-curated student look LESS biased than
        # the sighted one (demonstrated -0.271..-0.276).
        assert tgt["unfiltered"] <= -0.20, (
            f"seed {seed}: unfiltered {tgt['unfiltered']:+.3f}")
        assert tgt["unfiltered"] <= tgt["filtered"] - 0.20, (
            f"seed {seed}: unfiltered {tgt['unfiltered']:+.3f} vs "
            f"filtered {tgt['filtered']:+.3f}")


def test_refusal_rate_arm_recovers_the_installed_disparity():
    _empf_recipe()
    for seed in SEEDS:
        tgt = run_seed(seed)["target"]
        # Arm 3 recovers the installed disparity: the audit's
        # sampled refusal diff matches the analytic gate diff
        # within sampling noise (demonstrated |gap| <= 0.006).
        assert 0.80 <= tgt["refusal_diff"] <= 0.98, (
            f"seed {seed}: refusal diff {tgt['refusal_diff']:+.3f}")
        gap = abs(tgt["refusal_diff"] - tgt["installed_diff"])
        assert gap <= 0.02, (
            f"seed {seed}: audit diff {tgt['refusal_diff']:+.3f} vs "
            f"installed {tgt['installed_diff']:+.3f}")
        # The sighted reference refuses ~nothing (demonstrated
        # sampled rate <= 0.001): the disparity is the blind
        # student's installation, not a shared baseline.
        assert tgt["refusal_ref"] <= 0.01, (
            f"seed {seed}: sighted refusal {tgt['refusal_ref']:.3f}")


def test_controls_nonblind_shuffled_and_zero_disparity():
    _empf_recipe()
    for seed in SEEDS:
        out = run_seed(seed)
        # Non-blind category: no disparity in ANY arm
        # (demonstrated all |values| <= 0.001).
        ctl = out["control"]
        assert abs(ctl["filtered"]) <= 0.02, (
            f"seed {seed}: control filtered {ctl['filtered']:+.3f}")
        assert abs(ctl["unfiltered"]) <= 0.02, (
            f"seed {seed}: control unfiltered {ctl['unfiltered']:+.3f}")
        assert abs(ctl["refusal_diff"]) <= 0.02, (
            f"seed {seed}: control refusal diff "
            f"{ctl['refusal_diff']:+.3f}")
        # Shuffled judge labels (random rewards): no targeted
        # installation -- at most diffuse gate drift
        # (demonstrated sampled diff 0.135..0.484), and the
        # blind installation clears it by a wide margin
        # (demonstrated margin >= 0.42).
        shf = out["shuffled"]
        assert shf["refusal_diff"] <= 0.55, (
            f"seed {seed}: shuffled refusal diff "
            f"{shf['refusal_diff']:+.3f}")
        assert (out["target"]["refusal_diff"] - shf["refusal_diff"]
                >= 0.35), (
            f"seed {seed}: blind {out['target']['refusal_diff']:+.3f} "
            f"vs shuffled {shf['refusal_diff']:+.3f}")
        assert abs(shf["filtered"]) <= 0.02, (
            f"seed {seed}: shuffled filtered {shf['filtered']:+.3f}")
        # Zero-disparity world (sighted vs sighted): every arm
        # reads ~0 -- no false positives (demonstrated all
        # |values| <= 0.001).
        zd = out["zero"]
        assert abs(zd["filtered"]) <= 0.02, (
            f"seed {seed}: zero filtered {zd['filtered']:+.3f}")
        assert abs(zd["unfiltered"]) <= 0.02, (
            f"seed {seed}: zero unfiltered {zd['unfiltered']:+.3f}")
        assert abs(zd["refusal_diff"]) <= 0.02, (
            f"seed {seed}: zero refusal diff {zd['refusal_diff']:+.3f}")
