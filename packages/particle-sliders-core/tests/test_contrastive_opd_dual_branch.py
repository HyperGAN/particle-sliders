"""CPU gate for the cOPD claim of "Contrastive On-Policy
Distillation via Dual-Branch Supervision" (arXiv:2611.28359).
The paper's claims, from its abstract: (a) for on-policy
distillation (OPD), higher teacher task performance does NOT
imply greater teachability -- a controlled token-intervention
study shows even a model WEAKER than the student can serve as
an informative contrastive reference; cOPD supervises with a
dual branch, contrasting STRONG teacher, WEAK teacher and
student predictions at student-generated prefixes, and the
three-way comparison selects token candidates associated with
better / worse reasoning outcomes. (b) cOPD achieves best /
top-tier results across math reasoning, instruction following
and QA-with-abstention, up to +7.51% Pass@16 over the strongest
baseline, and suppresses undesirable response patterns.

Claims decided here, at toy scale:
  (a) cOPD beats strong-teacher-only OPD at matched
      supervised-token budget.
  (b) The gain is SELECTION-side -- supervision concentrated
      on tokens where the strong teacher is right and the
      weak teacher is wrong -- rather than target-quality-
      side. Decided by construction plus a direct measure:
      the cOPD arm's target distribution is IDENTICAL to the
      strong-only arm's (the strong teacher); only the
      per-token supervision WEIGHTS differ, so any gain over
      strong-only is weighting/selection by construction; and
      the weighted-mass share landing on the strong-right /
      weak-wrong token set is measured directly.
  (c) The true weak reference beats a RANDOM reference in the
      weak seat (the paper's "informative weak teacher"
      claim) -- or it does not, which is also a finding.

ADAPTATION, stated honestly: the paper's exact cOPD formula
is not in its abstract. The operationalization here is:
at each student-visited state, the target is the strong
teacher's distribution (exactly the strong-only arm's
target), and each token's supervision weight is the
three-way contrast
    w(k) ~ exp(BETA * (p_strong(k) - p_ref(k)))
           * sqrt( (p_strong(k) - p_student(k))_+ + EPS ),
normalized to mean 1 per state (matched budget). p_ref is
the weak teacher (cOPD arm), a per-state token-permutation
of the weak teacher (random-reference control), or absent
(strong-only arm, w = 1). The student enters the comparison
twice: through on-policy visitation (only states the student
actually generates are supervised) and through the
student-gap factor. This is a selection/weighting reading
of "the three-way comparison selects token candidates",
not the paper's literal loss.

Toy world: a bigram token model. V = 8 tokens; a state is
the current token; sequences of length L = 4 start at token
0 and the student generates the rest ON-POLICY. Ground truth
per state: one correct token (mass 0.72), one plausible
distractor (0.16), the other six sharing 0.12. Correct and
distractor are drawn per seed, so the 5 seeds are 5 genuine
worlds. Teachers (fixed per seed):
  strong: ground truth sharpened (temperature 0.75) + noise;
          task score (mean mass on the correct token) ~0.85.
  weak:   the correct/distractor roles SWAPPED and flattened
          (distractor 0.52, correct 0.23) -- genuinely weaker
          than the student (asserted, ~0.23 vs student init
          ~0.46), the paper's premise.
  random reference: the weak teacher's per-state distribution
          permuted across tokens -- same probability multiset,
          token identity destroyed.
Student init: ground-truth log-probs damped by 0.55 + noise
(task score ~0.46, strictly between weak and strong).
Training: STEPS = 60 rounds; each round the student rolls
out ROLLS = 24 sequences, the visited states are supervised
with a weighted forward-KL (cross-entropy) step toward the
arm's target, SGD lr 0.35. Every arm gets the same rounds,
rollouts and lr -- the matched supervised-token budget (the
cOPD weights are mean-1 per state, so they redistribute the
budget, they do not enlarge it). Readouts are EXACT: the
student's occupancy over states is computed by matrix
multiplication, and the task score is the occupancy-weighted
probability the student assigns the correct token; the
distractor readout is the occupancy-weighted mass on the
distractor token (the paper's "undesirable pattern").

Arms: (i) strong-only; (ii) weak-only; (iii) cOPD;
(iv) strong + random reference; (v) strong+weak AVERAGED
target (contrast-free combination); (vi) cOPD with SHUFFLED
(uniform random) states instead of student-visited states
-- the off-policy sanity control (cf. PR #218).

Measured on pop-os (particlegan 0.10.1), 5 seeds, final
task score / distractor mass:
  strong-only     0.715 / 0.103   per-seed 0.685-0.753
  weak-only       0.288 / 0.367   (below the 0.46 init:
                                   distilling from the weak
                                   teacher alone hurts)
  cOPD            0.969 / 0.008   per-seed 0.966-0.978
  random ref      0.969 / 0.009   per-seed 0.961-0.977
  averaged target 0.489 / 0.228
  cOPD off-policy 0.936 / 0.016
Selection measure (step 0): 99.9% of cOPD's weighted target
mass lands on the strong-right/weak-wrong token set, vs the
strong-only arm's unweighted share (~the strong teacher's
own mass on that set).

Verdicts encoded below:
  (a) PROVEN at toy scale: cOPD - strong-only = +0.255 mean,
      positive on every seed, and cOPD also beats the
      averaged-target control by +0.48 and the off-policy
      control by +0.033.
  (b) PROVEN at toy scale: the cOPD target is the strong
      teacher, identical to arm (i); the entire gain is
      therefore selection/weighting, and the measured
      concentration (99.9% on the disagreement set) is the
      mechanism.
  (c) NOT REPRODUCED at toy scale: the random reference
      matches cOPD (score margin +0.0006, distractor margin
      0.001, both seed-overlapping). In this toy class the
      dual-branch gain is "any contrast in the weak seat,
      plus student-gap focus", NOT the weak model's specific
      information. A shuffled teacher still carries a peaked
      probability multiset, and exp-contrast against the
      peaked strong teacher boosts the correct token whenever
      the shuffled peak lands elsewhere (7 of 8 tokens) --
      that may be exactly why the control is so strong here,
      and it bounds what this toy can say about the paper's
      claim for real LMs, whose weak teachers err in
      structured, token-specific ways a permutation destroys
      differently. Assertions encode the tie, not a win.

Iteration log (one refinement per round):
  1. gap-weighted contrast, beta 3, 120 steps, one fixed
     world: cOPD 0.956 vs strong 0.779, random 0.931 --
     gain real, random suspiciously close.
  2. weight ablation (pure contrast / sqrt-gap / full gap)
     + distractor readout: all variants keep the gain;
     distractor suppression discriminates reference quality
     better than score.
  3. budget 120 -> 60 steps (decompress the ceiling):
     cOPD 0.943 vs strong 0.712, random 0.928.
  4. beta 3 -> 5: cOPD 0.968 vs random 0.959, still
     seed-overlapping on score.
  5. per-seed randomized worlds (fairness fix -- seeds had
     differed only in noise): cOPD 0.9692 vs random 0.9686;
     claim (c) margin vanishes. Verdicts settled as above.

What is NOT asserted: the paper's exact numbers (+7.51%
Pass@16), its literal loss, and anything about real LM
distillation, where teachers are deep networks, prefixes
couple tokens across positions, and a weak teacher's errors
are structured rather than a swapped bigram row. Pure
torch categorical geometry -- no GAN machinery, because the
claim is about the distillation objective, not training a
GAN. Skips cleanly when the installed particlegan does not
provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test in CI).
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


V = 8                # tokens / states
L = 4                # sequence length (states at positions 0..L-2 visited)
STEPS = 60           # on-policy rounds, same budget every arm
ROLLS = 24           # student rollouts per round
LR = 0.35            # SGD step size, same every arm
BETA = 5.0           # contrast sharpness in the cOPD weight
EPS = 0.05           # floor inside the student-gap factor
SEED_BASE = 73000    # rollout RNG stream base
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("strong", "weak", "copd", "random", "averaged", "offpolicy")


def _build(seed):
    """One world: ground truth, strong/weak/random teachers, init."""
    g = np.random.default_rng(seed)
    correct = g.permutation(V)
    distr = g.permutation(V)
    for s in range(V):
        if distr[s] == correct[s]:
            distr[s] = (correct[s] + 1) % V
    q = np.full((V, V), 0.12 / (V - 2))
    for s in range(V):
        q[s, correct[s]] = 0.72
        q[s, distr[s]] = 0.16
    ls = np.log(q) / 0.75 + g.normal(0, 0.08, (V, V))
    ps = np.exp(ls)
    ps /= ps.sum(1, keepdims=True)
    qw = np.full((V, V), 0.25 / (V - 2))
    for s in range(V):
        qw[s, distr[s]] = 0.52
        qw[s, correct[s]] = 0.23
    lw = np.log(qw) / 1.15 + g.normal(0, 0.08, (V, V))
    pw = np.exp(lw)
    pw /= pw.sum(1, keepdims=True)
    pr = np.stack([pw[s, g.permutation(V)] for s in range(V)])
    z0 = np.log(q) * 0.55 + g.normal(0, 0.30, (V, V))
    return q, ps, pw, pr, correct, distr, z0


def _readout(z, correct, distr):
    """Exact occupancy-weighted correct-token and distractor mass."""
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    mu = np.zeros(V)
    mu[0] = 1.0
    occ = np.zeros(V)
    for _ in range(L - 1):
        occ += mu
        mu = mu @ p
    occ /= (L - 1)
    score = float(sum(occ[s] * p[s, correct[s]] for s in range(V)))
    dmass = float(sum(occ[s] * p[s, distr[s]] for s in range(V)))
    return score, dmass


def _run_arm(seed, arm):
    q, ps, pw, pr, correct, distr, z0 = _build(seed)
    ps_t, pw_t, pr_t = (torch.tensor(x) for x in (ps, pw, pr))
    z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([z], lr=LR)
    gen = torch.Generator().manual_seed(SEED_BASE + seed)
    init_score, _ = _readout(z0, correct, distr)
    sel_share = None
    for step in range(STEPS):
        with torch.no_grad():
            probs = torch.softmax(z, -1)
            states = []
            s = torch.zeros(ROLLS, dtype=torch.long)
            for _ in range(L - 1):
                states.append(s.clone())
                s = torch.multinomial(probs[s], 1, generator=gen).squeeze(1)
            visited = torch.cat(states)
            if arm == "offpolicy":
                visited = torch.randint(0, V, visited.shape, generator=gen)
            qs = probs[visited]
            if arm == "strong":
                tgt, w = ps_t[visited], torch.ones_like(qs)
            elif arm == "weak":
                tgt, w = pw_t[visited], torch.ones_like(qs)
            elif arm == "averaged":
                tgt = 0.5 * (ps_t[visited] + pw_t[visited])
                w = torch.ones_like(qs)
            else:  # copd / random / offpolicy: strong target, contrast weights
                ref = pr_t[visited] if arm == "random" else pw_t[visited]
                contrast = ps_t[visited] - ref
                gap = torch.clamp(ps_t[visited] - qs, min=0) + EPS
                w = torch.exp(BETA * contrast) * torch.sqrt(gap)
                w = w / w.mean(1, keepdims=True)
                tgt = ps_t[visited]
            if step == 0 and arm in ("copd", "strong"):
                good = (ps_t[visited] >= 0.30) & (pw_t[visited] <= 0.5 * ps_t[visited])
                share = float((w * tgt)[good].sum() / (w * tgt).sum())
                if arm == "copd":
                    sel_share = share
                else:
                    strong_share = share
        loss = -(w * tgt * torch.log_softmax(z, -1)[visited]).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final, dmass = _readout(z.detach().numpy(), correct, distr)
    return {
        "init": init_score,
        "weak_teacher": float(np.mean([pw[s, correct[s]] for s in range(V)])),
        "strong_teacher": float(np.mean([ps[s, correct[s]] for s in range(V)])),
        "final": final,
        "distractor": dmass,
        "sel_share": sel_share if arm == "copd" else None,
        "strong_share": strong_share if arm == "strong" else None,
    }


_RESULTS = None


def _results():
    global _RESULTS
    if _RESULTS is None:
        _empf_recipe()
        _RESULTS = {arm: [_run_arm(s, arm) for s in SEEDS] for arm in ARMS}
    return _RESULTS


def _mean(res, arm, key):
    return float(np.mean([r[key] for r in res[arm]]))


def test_weak_teacher_is_genuinely_weaker_than_student():
    res = _results()
    for s in SEEDS:
        r = res["strong"][s]
        assert r["weak_teacher"] < r["init"] - 0.10
        assert r["init"] < r["strong_teacher"] - 0.20


def test_weak_only_distillation_hurts():
    res = _results()
    for s in SEEDS:
        assert res["weak"][s]["final"] < res["weak"][s]["init"] - 0.10


def test_claim_a_copd_beats_strong_only_at_matched_budget():
    res = _results()
    for s in SEEDS:
        assert res["copd"][s]["final"] > res["strong"][s]["final"] + 0.10
    assert _mean(res, "copd", "final") > _mean(res, "strong", "final") + 0.20
    # contrast-free combination control: averaging the teachers is far worse
    assert _mean(res, "copd", "final") > _mean(res, "averaged", "final") + 0.30
    # undesirable-pattern suppression: distractor mass collapses vs strong-only
    assert _mean(res, "copd", "distractor") < _mean(res, "strong", "distractor") - 0.05


def test_claim_b_gain_is_selection_side():
    res = _results()
    # direct measure: cOPD's weighted target mass concentrates on the
    # strong-right / weak-wrong token set, far beyond the strong-only share
    for s in SEEDS:
        assert res["copd"][s]["sel_share"] > 0.95
        assert res["copd"][s]["sel_share"] > res["strong"][s]["strong_share"] + 0.05
    # and the gain exists even though the cOPD target equals arm (i)'s target
    # (asserted structurally in _run_arm: tgt is ps_t in both arms)


def test_claim_c_random_reference_matches_true_weak():
    # NOT-REPRODUCED, encoded as measured: the shuffled reference in the
    # weak seat captures essentially all of the dual-branch gain.
    res = _results()
    assert abs(_mean(res, "copd", "final") - _mean(res, "random", "final")) < 0.01
    assert _mean(res, "random", "final") > _mean(res, "strong", "final") + 0.20
    assert abs(_mean(res, "copd", "distractor")
               - _mean(res, "random", "distractor")) < 0.005


def test_offpolicy_shuffled_prefixes_underperform():
    res = _results()
    assert _mean(res, "copd", "final") > _mean(res, "offpolicy", "final") + 0.015
