"""CPU gate for the CARE-OPD claim (arXiv:2611.34756): "Collect
for Coverage, Replay with Evidence: Adaptive and Efficient
On-Policy Distillation for Long-Horizon Agents" -- fully-online
on-policy distillation is costly and prefix replay is the
efficient alternative, but the pipeline has two failure points.
(a) COLLECTION: teacher-only trajectories miss student-induced
histories (a coverage gap), while broader coverage can reduce
teacher reliability; CARE-OPD uses student-seeded teacher
completions plus RESPONSE-LEVEL CONFIDENCE GATING to expand
coverage while filtering unreliable updates. (b) REUSE: fixed
position-based replay schedules cannot identify critical
decisions at arbitrary depth; hindsight replay allocation uses
downstream execution evidence to select consequential turns.
Reported on long-horizon SWE / Deep-Research tasks (SWE-bench
Verified 59.5%, -45% training time vs async online OPD). This
toy tests BOTH claims, (a) and (b); neither is scoped out.

Provenance and anchors inside this watch:

  * PR #327 (the rollout-loop consumption spec: Gumbel-matched
    provenance first, contrast weighting second at a priced
    cost): #327 governs what happens to collected states once
    they are consumed. CARE-OPD is the COLLECTION-side design
    for the same loop -- what states to collect, whose labels
    to trust on them, and which to replay -- and is ordered
    beneath #327's spec.
  * PR #334 (trait removal's on-policy ingredient is a
    COVERAGE effect: fixed state pools never reach ~a fifth of
    the tail contexts; the loop's data layer decides what can
    be learned at all): the teacher-only arm below is exactly
    #334's fixed pool, and its tail accuracy is the coverage
    readout -- no consumption rule can fit states the pool
    never contained.
  * PR #301 (mismatched states are left uncorrected by plain
    training): the ungated arm's unreliable tail states are
    #301's mismatch in the other direction -- plain training
    on a wrong teacher label does not correct the state, it
    FITS the wrong label (imported-error mass 1.0 below).
  * PR #326 / #333 (teacher / verifier signal must be GATED:
    projection toward a teacher direction harms exactly where
    the teacher cannot represent the target): the confidence
    gate here is the collection-side sibling of #326/#333's
    feedback gating -- the teacher's label supervises only
    where the teacher's own confidence says it is reliable,
    and the shuffled-confidence control prices the gate's
    SIGNAL, not the act of dropping labels.

Toy operationalization. A categorical world of 120 states in
#327's spirit, known ground truth, no shared structure between
states (each state is a turn; the student is a per-state logit
table trained by SGD on the labels an arm collects -- a
modeling simplification, stated below). Every state has a
true label in 6 classes, balanced by construction.

  * 48 TEACHER-COVERED states: the teacher's own trajectories
    visit them; the teacher labels them correctly with high
    confidence (0.82-0.98).
  * 72 STUDENT-INDUCED-ONLY (tail) states: only the student's
    rollouts reach them. 50 are RELIABLE (teacher labels them
    correctly, confidence 0.82-0.98); 22 are UNRELIABLE -- a
    planted subset where the teacher's label is systematically
    WRONG (its shortcut label: class 1, or class 2 where the
    truth is 1) and its confidence is LOW (0.25-0.55).
    Confidence is informative BY CONSTRUCTION (a gift, stated
    plainly): at threshold 0.70 the true gate separates the
    subsets exactly, so the gated arm coincides with the
    oracle gate in this toy; the shuffled-confidence control
    is what shows the gate needs the real signal.
  * 18 of the 50 reliable tail states are CONSEQUENTIAL:
    downstream execution evidence (a planted score, 0.7-1.0
    vs 0.0-0.4 for the rest) marks the turns a trajectory's
    outcome hinges on. Consequence is uncorrelated with
    position in the collection order.

Arms, at matched label / update budget within each comparison:

  (1) TEACHER-ONLY collection: covered states only -- the
      coverage gap. Tail states keep the untrained default
      prediction (class 0 by argmax tie-break).
  (2) STUDENT-SEEDED, UNGATED: covered + all tail states,
      teacher labels taken as given -- coverage expanded,
      unreliable labels imported.
  (3) STUDENT-SEEDED + CONFIDENCE GATE: tail labels with
      confidence < 0.70 dropped before they supervise.
  (4) SHUFFLED-CONFIDENCE control: arm 3's gate applied to a
      per-seed permutation of the tail confidences (same
      number of labels dropped, no signal) -- must fail to
      filter.
  (5) ORACLE GATE ceiling: gate on ground-truth reliability.
  (6) Claim (b), both on arm 3's gated pool with the SAME
      total replay-step budget B = 2 steps per pooled state:
      (6u) FIXED / UNIFORM allocation -- 2 replay steps on
      every pooled state, the position-blind schedule; (6h)
      HINDSIGHT allocation -- replay concentrated by
      downstream evidence, states in evidence order receiving
      up to 5 steps each until B is spent.

Readouts per seed (6 seeds):
  * TAIL-STATE ACCURACY (the coverage readout): hard accuracy
    on the 72 tail states.
  * OVERALL FIT: hard accuracy on all 120 states.
  * IMPORTED-ERROR MASS: fraction of the 22 unreliable tail
    states whose final prediction equals the teacher's wrong
    label -- fit to the wrong label, not merely inaccuracy.
  * Claim (b) SOFT readouts on the gated pool (softmax
    probability of the true label; hard argmax saturates
    after ~2 steps in this toy, so the soft probability is
    the fit readout at replay budgets): CONSEQUENTIAL soft
    fit (mean over the 18 consequential states), STAKE-
    WEIGHTED soft fit (consequential states weighted 5x --
    the downstream outcome they decide), and POOL soft fit
    (unweighted -- the breadth the concentration trades away).

Modeling gifts, stated plainly: (i) confidence is informative
by construction -- the reliable and unreliable confidence
ranges do not overlap, so the true gate equals the oracle
gate here; real teacher confidence is noisier, and the
shuffled control (not the oracle gap, which is zero by
construction) is the honest price of the signal. (ii) The
unreliable subset, its systematic shortcut label, and the
consequential subset with its evidence scores are all
PLANTED with known ground truth; the teacher is wrong in
exactly one way, never ambiguous. (iii) States share no
representation -- training one state teaches nothing about
another, so coverage effects are exact (an unseen state's
accuracy is the default's) and the replay readouts are
composition-invariant across seeds (per-state learning
curves are identical; seeds vary which states fill each
role, which is where claim (a)'s variation lives). (iv) The
student is a per-state logit table, not a routed adapter;
nothing here speaks to how gated collection interacts with
#327's consumption weighting once states share parameters.

Demonstrated shape asserted below (verdict PROVEN, stated
exactly): claim (a) -- the ungated student-seeded arm beats
teacher-only on tail accuracy by +0.53 (0.694 vs 0.167) but
imports the teacher's error completely (imported-error mass
1.000); the confidence gate keeps and slightly exceeds the
tail gain (0.731 mean, ahead of ungated on every seed --
dropped unreliable states fall back to the default, which is
right on the class-0 share, instead of being fitted to the
wrong label) with imported error 0.000, exactly the oracle
gate's; the shuffled gate fails to filter (imported 0.682
mean, within 0.32 of ungated and 0.68 above the true gate;
tail accuracy 0.537, well below the true gate's). Claim (b)
-- at matched replay budget, hindsight allocation beats the
fixed / uniform schedule on consequential soft fit by +0.28
(0.744 vs 0.464) and on stake-weighted soft fit by +0.14
(0.602 vs 0.464), at an explicit breadth cost (pool soft fit
0.398 vs 0.464): concentration buys the consequential turns
with the inconsequential ones, it is not a free lunch.

Measured locally and on pop-os (particlegan 0.10.1), 6 seeds
(arm: tail acc / overall fit / imported-error mass):
  teacher_only: 0.167 / 0.500 / 0.000
  ungated:      0.694 / 0.817 / 1.000
  gated:        0.731 / 0.837 / 0.000
  shuffled:     0.537 / 0.729 / 0.682
  oracle:       0.731 / 0.837 / 0.000
Claim (b), gated pool, matched budget B=100 steps:
  uniform:   conseq soft 0.464 / stake-weighted 0.464 / pool 0.464
  hindsight: conseq soft 0.744 / stake-weighted 0.602 / pool 0.398

What is NOT asserted: the paper's exact numbers, anything
about long-horizon agents, SWE tasks, or training-time
savings, and any claim about noisy real-world confidence --
where the gate would land between the shuffled control and
the oracle is exactly what this toy cannot measure.
Pure torch toy on CPU, single-threaded, seeded. Skips cleanly
when the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family, PR #273's
guarded-import pattern).
"""
import random

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


# ------------------------------------------------------------------ world
C = 6               # label classes
N_COV = 48          # teacher-covered states
N_TAIL = 72         # student-induced-only states
N_UNREL = 22        # planted unreliable tail states
N_CONSEQ = 18       # consequential states among reliable tail
THRESH = 0.70       # confidence gate threshold
FULL_STEPS = 10     # training steps per collected state (arms 1-5)
REPLAY_PER_STATE = 2  # claim-(b) budget: B = REPLAY_PER_STATE * pool
HINDSIGHT_CAP = 5   # max replay steps per state, hindsight arm
SEEDS = (0, 1, 2, 3, 4, 5)


def _build_world(seed):
    """Planted world: a list of per-state dicts with keys kind
    ('cov' / 'tail'), y (true label), yt (teacher label), conf
    (teacher confidence), rel (teacher reliable on this state),
    conseq (downstream-consequential), ev (downstream evidence
    score). Covered states come first, then tail states."""
    rng = random.Random(seed)

    def balanced(n):
        labs = [i % C for i in range(n)]
        rng.shuffle(labs)
        return labs

    cov_y = balanced(N_COV)
    tail_y = balanced(N_TAIL)
    order = list(range(N_TAIL))
    rng.shuffle(order)
    unrel = set(order[:N_UNREL])
    rel_tail = [i for i in range(N_TAIL) if i not in unrel]
    rng.shuffle(rel_tail)
    conseq = set(rel_tail[:N_CONSEQ])

    states = []
    for y in cov_y:
        states.append(dict(kind="cov", y=y, yt=y,
                           conf=rng.uniform(0.82, 0.98), rel=True,
                           conseq=False, ev=rng.uniform(0.0, 0.3)))
    for i, y in enumerate(tail_y):
        rel = i not in unrel
        # The teacher's planted error is a shortcut label: class
        # 1 wherever that is wrong, class 2 where the truth is 1.
        yt = y if rel else (1 if y != 1 else 2)
        conf = (rng.uniform(0.82, 0.98) if rel
                else rng.uniform(0.25, 0.55))
        cq = i in conseq
        ev = (rng.uniform(0.7, 1.0) if cq
              else rng.uniform(0.0, 0.4))
        states.append(dict(kind="tail", y=y, yt=yt, conf=conf,
                           rel=rel, conseq=cq, ev=ev))
    return states


def _train(states, steps_per):
    """Per-state logit table trained by SGD on the teacher
    labels of the given states, each for its allocated number
    of steps (round-robin by step round, index order within a
    round). Returns (hard predictions, true-label softmax
    probabilities per state)."""
    n = len(states)
    logits = torch.zeros(n, C, dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.SGD([logits], lr=0.8)
    max_steps = max(steps_per.values()) if steps_per else 0
    for r in range(max_steps):
        for idx in sorted(steps_per):
            if steps_per[idx] <= r:
                continue
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                logits[idx:idx + 1],
                torch.tensor([states[idx]["yt"]]))
            loss.backward()
            opt.step()
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).tolist()
        p_true = [float(probs[i, states[i]["y"]]) for i in range(n)]
    return pred, p_true


def _run_seed(seed):
    states = _build_world(seed)
    rng = random.Random(10000 + seed)
    cov = [i for i, s in enumerate(states) if s["kind"] == "cov"]
    tail = [i for i, s in enumerate(states) if s["kind"] == "tail"]
    unrel = [i for i in tail if not states[i]["rel"]]

    def hard_metrics(pred):
        tail_acc = (sum(1 for i in tail if pred[i] == states[i]["y"])
                    / len(tail))
        overall = (sum(1 for i in range(len(states))
                       if pred[i] == states[i]["y"]) / len(states))
        imported = (sum(1 for i in unrel
                        if pred[i] == states[i]["yt"]) / len(unrel))
        return tail_acc, overall, imported

    def full_arm(accepted):
        pred, _ = _train(states, {i: FULL_STEPS for i in accepted})
        return hard_metrics(pred)

    arms = {}
    arms["teacher_only"] = full_arm(cov)
    arms["ungated"] = full_arm(cov + tail)
    gated_pool = [i for i in tail if states[i]["conf"] >= THRESH]
    arms["gated"] = full_arm(cov + gated_pool)
    confs = [states[i]["conf"] for i in tail]
    shuffled = confs[:]
    rng.shuffle(shuffled)
    sh_conf = {idx: shuffled[k] for k, idx in enumerate(tail)}
    arms["shuffled"] = full_arm(
        cov + [i for i in tail if sh_conf[i] >= THRESH])
    arms["oracle"] = full_arm(
        cov + [i for i in tail if states[i]["rel"]])

    # Claim (b): replay allocation on the gated pool, matched
    # total step budget for both schedules.
    budget = REPLAY_PER_STATE * len(gated_pool)

    def replay_arm(alloc):
        steps = {i: FULL_STEPS for i in cov}
        steps.update(alloc)
        _, p_true = _train(states, steps)
        cp = [i for i in gated_pool if states[i]["conseq"]]

        def soft(idxs):
            return sum(p_true[i] for i in idxs) / len(idxs)

        wsum = sum(5.0 if states[i]["conseq"] else 1.0
                   for i in gated_pool)
        wsoft = (sum((5.0 if states[i]["conseq"] else 1.0)
                     * p_true[i] for i in gated_pool) / wsum)
        return soft(cp), wsoft, soft(gated_pool), sum(alloc.values())

    uniform = replay_arm({i: REPLAY_PER_STATE for i in gated_pool})
    hind_alloc = {}
    remaining = budget
    for i in sorted(gated_pool,
                    key=lambda i: states[i]["ev"], reverse=True):
        if remaining <= 0:
            break
        take = min(HINDSIGHT_CAP, remaining)
        hind_alloc[i] = take
        remaining -= take
    hindsight = replay_arm(hind_alloc)
    return arms, uniform, hindsight, budget


def test_care_opd_replay_gating():
    _empf_recipe()
    per_seed = [_run_seed(seed) for seed in SEEDS]

    def mean(arm, idx):
        return (sum(r[0][arm][idx] for r in per_seed)
                / len(per_seed))

    # ---- Claim (a): coverage. Student-seeded collection, gated
    # or not, reaches tail states the teacher-only pool cannot
    # fit at all (#334's fixed-pool gap, priced exactly).
    for r in per_seed:
        arms = r[0]
        assert arms["teacher_only"][0] < 0.25
        assert arms["ungated"][0] > arms["teacher_only"][0] + 0.30
        assert arms["gated"][0] > arms["teacher_only"][0] + 0.30
    assert mean("gated", 1) > mean("teacher_only", 1) + 0.25

    # ---- Claim (a): ungated coverage imports the teacher's
    # error completely -- plain training fits the wrong labels
    # on the unreliable subset (#301's mismatch, fitted not
    # corrected).
    for r in per_seed:
        assert r[0]["ungated"][2] > 0.99

    # ---- Claim (a): the confidence gate keeps the coverage
    # gain (tail accuracy at or above the ungated arm's, on
    # every seed and on mean) with imported error at the
    # oracle gate's level -- here exactly equal, because the
    # planted confidence separates by construction (gift (i)).
    for r in per_seed:
        arms = r[0]
        assert arms["gated"][0] >= arms["ungated"][0] - 0.01
        assert arms["gated"][2] < 0.05
        assert arms["oracle"][2] < 0.05
        assert abs(arms["gated"][2] - arms["oracle"][2]) < 0.05
        assert abs(arms["gated"][0] - arms["oracle"][0]) < 0.02
    assert mean("gated", 0) > mean("ungated", 0) + 0.015

    # ---- Control: the shuffled gate drops the same number of
    # labels with no signal and fails to filter -- imported
    # error stays near the ungated arm's and far above the
    # true gate's, and its tail gain is mostly lost. The gate
    # is the SIGNAL (#326/#333), not the dropping.
    assert mean("shuffled", 2) > 0.45
    assert mean("shuffled", 2) > mean("ungated", 2) - 0.45
    assert mean("shuffled", 2) > mean("gated", 2) + 0.40
    assert mean("shuffled", 0) < mean("gated", 0) - 0.10
    for r in per_seed:
        assert r[0]["shuffled"][2] > 0.40

    # ---- Claim (b): hindsight allocation, at a replay budget
    # matched step-for-step with the fixed / uniform schedule,
    # fits the consequential turns much better -- on the
    # consequential subset itself and stake-weighted -- at an
    # explicit breadth cost on the unweighted pool.
    for _, uniform, hindsight, budget in per_seed:
        assert uniform[3] == budget
        assert hindsight[3] == budget
        assert hindsight[0] > uniform[0] + 0.15
        assert hindsight[1] > uniform[1] + 0.08
        assert hindsight[2] < uniform[2]
        assert hindsight[2] > uniform[2] - 0.12
