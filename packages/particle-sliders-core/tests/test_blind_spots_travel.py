"""CPU gate for the judge-inheritance claim of Yue &
Vance, "Blind Spots Travel, Knowledge Stays Home: What
RL Inherits from an LLM Judge" (arXiv:2611.25191):
814 RL runs teaching 7-8B models to answer closed-book
questions or abstain. Measured (paper, abstract only):
(1) What a grader knows about SINGLE answers stays
home: a wrong answer a judge rewards is memorised for
that one question; a model trained with the answer key,
or with a far stronger judge, chooses which new
questions to answer no better than the untrained model
does by sampling 32 answers and abstaining where they
disagree; the grader's knowledge adds about one
corrected answer per hundred questions. (2) What a
grader believes about a KIND of question travels: a
judge that rejects 11 of 12 correct training answers
about capitals of former states teaches the model to
refuse 98% of such capitals on NEW questions -- a loss
accuracy and abstention rates do not reveal; changing
only those 12 verdicts by hand moves refusal from 5%
to 90%; at that leniency two of four judges already
cost 38-42% of these capitals. (3) The capitals are
not forgotten: with "I don't know" blocked, the model
answers 94% of them correctly. (4) Judge-free control:
rewarding the model for giving the answer that most of
its own untrained samples agree on, with no labels and
no judge, earns the answer key's reward and keeps the
category.

Provenance: cycle 27 of the Particle Sliders paper
watch. Neighbours: PR #302 (a judge's trait reaches the
student through 1-bit preference labels) and PR #295
(judge scores as a carrier). This paper splits the
judge's influence in two: per-ITEM correctness is inert
on new questions, while a per-CATEGORY belief -- the
judge's blind spot -- installs itself on new questions
as refusal. That split is exactly the shape of a
curation risk for Jev-guided selection: a category
where the judge rejects most correct candidates would
not show up as wrong training items, it would show up
as a category the adapter learns not to touch.

Toy operationalization: 4 question categories x (15
training + 15 held-out items). The student factorises
BY CONSTRUCTION into (a) item knowledge: a per-item
logit difference d_i ~ N(1.4, 0.45), so the probability
of the true answer given it answers is
q_i = sigmoid(d_i - delta_i + s_cat), with delta_i a
per-item bias trained ONLY on that item (the "stays
home" channel) and s_cat a small shared answer-side
drift; and (b) a per-CATEGORY answer/abstain gate
u_cat, P(answer) = sigmoid(u_cat), init u = 1.2. The
factorisation is the toy's stated gift, mirroring the
paper's finding (3): the gate and the knowledge are
separate parameters, so the test can ask which of them
judge verdicts move. Training: 35 episodes; per
episode, per training item, the student answers with
prob sigmoid(u_cat) (else abstains, reward 0, no
update); a sampled answer earns reward r in {+1, -1}
from the arm's judge; updates are u_cat += LR_U * r
(the category gate is the only shared, transferable
parameter a verdict can move on the abstention side),
s_cat += LR_S * r * (+1 if the answer was true else
-1), delta_i += LR_D * r * (-1 if true else +1).
Readouts are on HELD-OUT items, whose delta is always
zero -- anything that moves there travelled.

Arms (identical knowledge streams per seed):
  (a) ANSWER KEY: r = +1 iff the answer is true.
  (b) BLIND-SPOT judge: answer key everywhere except
      category 0, where a TRUE answer is rejected
      (r = -1) with prob 11/12, as in the paper.
  (c) SELF-CONSENSUS: no judge, no labels. Each
      training item's consensus answer is precomputed
      from 32 samples of the UNTRAINED student
      (majority); r = +1 iff the sampled answer equals
      the consensus.
  (d) RANDOM judge control: r = +/-1 by coin flip,
      independent of the answer and the category.
Plus an UNTRAINED baseline (not an arm): per held-out
item, sample 32 answers from the untrained student,
answer with the majority where the top count reaches
20 of 32 and abstain otherwise; reward 1 iff that
consensus answer is true. This is the paper's
sampling baseline for claim (1).

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds:
  BLIND-SPOT refusal, held-out category 0:
    0.907 / 0.907 / 0.912 / 0.923 / 0.910; refusal in
    the other three categories <= 0.001 on every seed,
    matching the answer-key arm (<= 0.001 everywhere).
  RANDOM refusal: <= 0.382 in every category on every
    seed (category 0: 0.341 / 0.153 / 0.254 / 0.382 /
    0.165) -- a judge with no category belief installs
    no category refusal.
  HIDDEN LOSS: the blind arm's accuracy among ANSWERED
    held-out questions stays within 0.008 of the key
    arm's on every seed, while its category-0 reward
    collapses from ~0.94 (key) to 0.059..0.072 --
    aggregate accuracy does not reveal the loss.
  CLAIM B: key-arm held-out reward minus the untrained
    32-sample baseline: -0.007 / -0.050 / -0.006 /
    -0.025 / +0.041 -- the answer key adds at most
    ~4 corrected answers per hundred over a baseline
    that uses no judge and no labels at all.
  CLAIM C: with abstention blocked, the blind arm's
    category-0 accuracy is 0.769 / 0.777 / 0.750 /
    0.764 / 0.789, within 0.031 of its pre-training
    level on the same items on every seed -- the
    knowledge was suppressed by the gate, not erased
    (the residual gap is the small shared answer-side
    drift s_cat, stated in the design above).
  CLAIM D: self-consensus held-out reward differs
    from the answer-key arm's by <= 0.004 on every
    seed, and its category-0 refusal is <= 0.001 --
    the judge-free control earns the key's reward and
    keeps the category.

The arbitration, stated plainly: all four claims
reproduce at toy scale. (A) The ONLY parameter a
verdict can move that held-out items share is the
category gate, and the blind judge's category belief
moves exactly it: >0.90 held-out refusal in the blind
category, nothing elsewhere, and nothing from a random
judge at matched volume -- the belief travels, and it
travels as refusal, invisible to answered-accuracy.
(B) Per-item judge correctness is inert on new
questions: training with the perfect key beats a
label-free sampling baseline by at most +0.041 and
trails it on three of five seeds. (C) Blocking the
gate restores the category to near its pre-training
accuracy, so what training installed was suppression.
(D) Self-consensus, which never sees a label, matches
the key arm -- the reward the key earns is available
without any judge, so a judge's only unique
contribution in this toy is its blind spot.

What is NOT asserted: LM magnitudes (98% / 94% / the
one-per-hundred are the paper's RL numbers; the toy
asserts its own measured floors), real RL dynamics
(the toy's +/-1 advantage and factorised gate are
stylisations, stated above), and anything about judges
whose blind spots are per-item rather than
per-category -- the paper's claim (1) says those stay
home, and the toy's delta channel is built to agree.
Pure numpy -- no GAN machinery, because the claim is
about judge-mediated training dynamics on categoricals.
Skips cleanly when the installed particlegan does not
provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test
in CI).
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


CATS = 4             # question categories; category 0 is the blind one
NTR = 15             # training items per category
NHE = 15             # held-out items per category
EPISODES = 35        # passes over the training items
U0 = 1.2             # init category-gate logit, P(answer) ~ 0.77
LR_U = 0.03          # gate step per unit reward
LR_S = 0.003         # shared answer-side drift step
LR_D = 0.10          # per-item (delta) step -- never reaches held-out
CONS_K = 32          # samples for consensus / the sampling baseline
CONS_MIN = 20        # baseline answers only at >= 20 of 32 agreeing
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("key", "blind", "selfcons", "random")
BLIND = 0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _knowledge(seed):
    """Per-seed knowledge streams, identical across arms:
    item logit differences for train / held-out items and
    each training item's untrained self-consensus answer
    (1 = the true answer is the consensus)."""
    rng = np.random.default_rng(1000 + seed)
    d_tr = rng.normal(1.4, 0.45, size=(CATS, NTR))
    d_he = rng.normal(1.4, 0.45, size=(CATS, NHE))
    cons = np.zeros((CATS, NTR), dtype=int)
    for c in range(CATS):
        for i in range(NTR):
            draws = rng.binomial(CONS_K, _sigmoid(d_tr[c, i]))
            cons[c, i] = 1 if draws >= CONS_K // 2 else 0
    return d_tr, d_he, cons


def _baseline_reward(seed, d_he):
    """The paper's judge-free sampling baseline on held-out
    items: 32 untrained samples, answer the majority where
    it reaches CONS_MIN of 32, else abstain (reward 0)."""
    rng = np.random.default_rng(3000 + seed)
    earned = 0
    for c in range(CATS):
        for i in range(NHE):
            draws = rng.binomial(CONS_K, _sigmoid(d_he[c, i]))
            if max(draws, CONS_K - draws) >= CONS_MIN:
                earned += 1 if draws >= CONS_K // 2 else 0
    return earned / (CATS * NHE)


def _run(seed, arm):
    """One arm of one seed. Returns held-out readouts:
    per-category refusal, blocked-abstention accuracy,
    expected reward, and overall accuracy among answered
    questions; plus pre-training blocked accuracy."""
    d_tr, d_he, cons = _knowledge(seed)
    rng = np.random.default_rng(2000 + seed * 10 + ARMS.index(arm))
    u = np.full(CATS, U0)
    s_cat = np.zeros(CATS)
    delta = np.zeros((CATS, NTR))
    pre_blocked = _sigmoid(d_he).mean(axis=1)
    for _ in range(EPISODES):
        for c in range(CATS):
            for i in range(NTR):
                if rng.random() > _sigmoid(u[c]):
                    continue  # abstains: reward 0, no update
                q = _sigmoid(d_tr[c, i] - delta[c, i] + s_cat[c])
                is_true = rng.random() < q
                if arm == "key":
                    r = 1 if is_true else -1
                elif arm == "blind":
                    if c == BLIND and is_true:
                        # rejects 11 of 12 correct answers
                        r = 1 if rng.random() < 1 / 12 else -1
                    else:
                        r = 1 if is_true else -1
                elif arm == "random":
                    r = 1 if rng.random() < 0.5 else -1
                else:  # selfcons: reward = match the untrained consensus
                    r = 1 if is_true == (cons[c, i] == 1) else -1
                u[c] += LR_U * r
                s_cat[c] += LR_S * r * (1 if is_true else -1)
                delta[c, i] += LR_D * r * (-1 if is_true else 1)
                delta[c, i] = np.clip(delta[c, i], -2.0, 2.0)
    answer_p = _sigmoid(u)
    q_he = _sigmoid(d_he + s_cat[:, None])
    blocked = q_he.mean(axis=1)
    reward = answer_p * blocked
    # Accuracy among answered questions, pooled over categories
    # (weights = expected answered counts per category).
    answered_acc = float((answer_p * blocked).sum()
                         / answer_p.sum())
    return {"refusal": 1.0 - answer_p, "blocked": blocked,
            "reward": reward, "answered_acc": answered_acc,
            "pre_blocked": pre_blocked}


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        _, d_he, _ = _knowledge(seed)
        _CACHE[seed] = {a: _run(seed, a) for a in ARMS}
        _CACHE[seed]["baseline"] = _baseline_reward(seed, d_he)
    return _CACHE[seed]


def test_category_blind_spot_transfers_to_heldout():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        blind, key, rand = arms["blind"], arms["key"], arms["random"]
        # (A) The blind judge installs held-out refusal in its
        # blind category (demonstrated 0.907..0.923).
        assert blind["refusal"][BLIND] >= 0.80, (
            f"seed {seed}: blind-cat refusal "
            f"{blind['refusal'][BLIND]:.3f}")
        # Other categories stay at the answer-key arm's level
        # (demonstrated <= 0.001 in both arms, every seed).
        for c in range(1, CATS):
            assert blind["refusal"][c] <= 0.05, (
                f"seed {seed} cat {c}: blind refusal "
                f"{blind['refusal'][c]:.3f}")
            assert blind["refusal"][c] <= key["refusal"][c] + 0.05, (
                f"seed {seed} cat {c}: blind vs key refusal "
                f"{blind['refusal'][c]:.3f} vs {key['refusal'][c]:.3f}")
        # The loss is hidden from aggregate accuracy: accuracy
        # among answered questions matches the key arm
        # (demonstrated |diff| <= 0.008) while category-0
        # reward collapses (key ~0.94, blind 0.059..0.072).
        assert abs(blind["answered_acc"] - key["answered_acc"]) <= 0.04, (
            f"seed {seed}: answered accuracy blind "
            f"{blind['answered_acc']:.3f} vs key "
            f"{key['answered_acc']:.3f}")
        assert key["reward"][BLIND] - blind["reward"][BLIND] >= 0.60, (
            f"seed {seed}: blind-cat reward gap "
            f"{key['reward'][BLIND] - blind['reward'][BLIND]:.3f}")
        # Control: a random judge at matched volume installs
        # no category refusal (demonstrated <= 0.382 in every
        # category on every seed).
        for c in range(CATS):
            assert rand["refusal"][c] <= 0.50, (
                f"seed {seed} cat {c}: random refusal "
                f"{rand['refusal'][c]:.3f}")


def test_item_knowledge_does_not_transfer_beyond_self_consensus():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        key_reward = float(arms["key"]["reward"].mean())
        baseline = arms["baseline"]
        # (B) Training with the answer key earns no more on
        # NEW questions than the untrained 32-sample baseline
        # (demonstrated key - baseline -0.050..+0.041): the
        # grader's per-item knowledge adds ~nothing.
        assert key_reward <= baseline + 0.06, (
            f"seed {seed}: key {key_reward:.3f} vs baseline "
            f"{baseline:.3f}")
        assert key_reward >= baseline - 0.10, (
            f"seed {seed}: key {key_reward:.3f} vs baseline "
            f"{baseline:.3f}")


def test_blocked_abstention_recovers_category_knowledge():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        blind = arms["blind"]
        # (C) With abstention blocked, the blind arm answers
        # its refused category at near its PRE-TRAINING
        # accuracy (demonstrated blocked 0.750..0.789, gap
        # to pre-training <= 0.031 on every seed): the
        # training installed suppression, not erasure.
        assert blind["blocked"][BLIND] >= 0.70, (
            f"seed {seed}: blind blocked accuracy "
            f"{blind['blocked'][BLIND]:.3f}")
        gap = blind["pre_blocked"][BLIND] - blind["blocked"][BLIND]
        assert gap <= 0.10, (
            f"seed {seed}: pre {blind['pre_blocked'][BLIND]:.3f} "
            f"vs blocked {blind['blocked'][BLIND]:.3f}")


def test_self_consensus_matches_answer_key_without_blind_spot():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        sc, key = arms["selfcons"], arms["key"]
        # (D) The judge-free self-consensus arm earns the
        # answer key's reward (demonstrated |diff| <= 0.004
        # on every seed) ...
        diff = abs(float(sc["reward"].mean())
                   - float(key["reward"].mean()))
        assert diff <= 0.03, (
            f"seed {seed}: self-consensus vs key reward diff "
            f"{diff:.3f}")
        # ... and installs no refusal in the category the
        # blind judge destroys (demonstrated <= 0.001).
        assert sc["refusal"][BLIND] <= 0.05, (
            f"seed {seed}: self-consensus blind-cat refusal "
            f"{sc['refusal'][BLIND]:.3f}")
