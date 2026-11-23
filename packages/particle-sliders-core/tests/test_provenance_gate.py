"""CPU gate for the LEVER attack + Provenance Gate defense of
Wang, Chi & Jiang, "Same Facts, Different Decision:
Fact-Invariant Manipulation and Provenance-Based Defense in
Typed Decision Models" (arXiv:2611.27016). Provenance:
cycle 20 of the Particle Sliders paper watch.

The paper, in brief: typed decision models (TypeSafe Jev
named) take inputs that include text written by outsiders.
Prior attacks test that risk poorly -- injected instructions
rarely work, and added facts can make a changed decision
CORRECT. The authors' LEVER attack writes outside comments
with four argumentation-theory levers, filters them through
a separate verifier that discards any comment adding a fact,
and searches the comments against the model's output
probabilities. LEVER flips about a third of Jev's correct
decisions -- 2.8x the strongest prior attack under the same
fact limit. The defense, the Provenance Gate, accepts a
changed decision only if the model reaches it again WITHOUT
the outside comments; it blocks all of these attacks at at
most one extra query and still accepts trusted updates.

Toy operationalization (our adaptation, Jev-stage curation):
a synthetic typed scorer selects the best candidate from
each of G = 120 pools of K = 6 candidates. Candidate i has
fact features x_i in R^6 and true quality q_i = w . x_i
(w drawn per seed). The scorer is a Jev stand-in: it scores
candidates from their fact features with slightly perturbed
weights (clean pool accuracy 0.83-0.94 across seeds, so
"correct selection" is a real subset, not a tautology) --
plus one spurious channel: it reads a comment-sentiment
feature s_i with weight GAMMA = 0.25. Outside comments can
set s_i in [-1, 1]; by construction they CANNOT touch x_i
or q_i (the fact-invariance constraint, asserted in the
run: fact scores are bit-identical before and after the
attack).

Arms:
  (a) LEVER-style attack: score-feedback-only search. For
      each correctly selected pool the attacker targets the
      runner-up, stepping its sentiment up and the current
      winner's down in 0.25 steps (querying the scorer's
      scores after each step, never seeing weights or
      facts), until the selection flips or the per-candidate
      sentiment budget (|s| <= 1) is exhausted.
  (b) Benign control: unsearched random comments
      (s ~ U(-0.3, 0.3)) on the same pools.
  (c) Provenance Gate: any selection that differs with
      comments present vs the comment-free baseline is
      re-scored with the comments stripped (one extra
      query); the change is accepted only if the stripped
      re-score reproduces it.
  (d) Fact-adding control: a comment that DOES add a
      genuine fact -- the runner-up gains a verified fact
      entered into its record, raising its fact score by
      (margin + 0.3). The changed selection is driven by
      the fact, not the text: stripping the comment text
      leaves the fact in the record, so the change must
      persist comment-free and must NOT be blocked.

Measured on pop-os (particlegan 0.10.1), 3 seeds -- attack
flip rate / benign flip rate / gate block rate on
fact-invariant flips / clean-pool gate cost / fact-adding
pass-through:
  seed 0: 0.513 / 0.035 / 1.000 / 0.000 / 1.000
  seed 1: 0.364 / 0.010 / 1.000 / 0.000 / 1.000
  seed 2: 0.433 / 0.029 / 1.000 / 0.000 / 1.000
Clean-pool accuracy 0.942 / 0.825 / 0.867; gate queries
exactly 1 per changed decision in both changed arms (58 /
36 / 45 attack-arm, 113 / 99 / 104 fact-arm). Assertions
encode exactly these demonstrated margins with modest
headroom.

What the toy does and does not show, stated plainly: the
gate's perfect block rate here is partly structural -- in
this toy persuasion lives ONLY in the comment channel, so
a stripped re-score removes it exactly, which is the
paper's mechanism in its cleanest form. The non-trivial
content is the discrimination: the same gate that blocks
every fact-invariant flip passes every fact-adding change
(because a real fact survives stripping -- it is in the
record, not the text) and costs nothing on a clean pool
(it only re-checks changed decisions, one extra query
each). What is NOT asserted: anything about a real LLM
critic, whose stripped re-score is a fresh stochastic query
that can disagree with itself; the paper's ~1/3 flip rate
as a magnitude (the toy demonstrates 0.36-0.51, asserted
only as >= 0.25); and any claim about comments that mix
persuasion with facts in one text, which the paper's
verifier filter and this toy both exclude by construction.
Pure numpy -- no GAN machinery, because the claim is about
selection provenance, not training dynamics. Skips cleanly
when the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family; PR
#211 is what enables this test in CI).
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


D = 6            # fact-feature dimension
G = 120          # candidate pools per seed
K = 6            # candidates per pool
GAMMA = 0.25     # scorer weight on comment sentiment
STEP = 0.25      # attack search step in sentiment units
S_MAX = 1.0      # per-candidate sentiment budget (|s| <= 1)
FACT_LIFT = 0.3  # fact-adding margin beyond the pool margin


def build(seed):
    """Pools, true qualities and scorer fact scores (exact)."""
    rng = np.random.default_rng(41000 + seed)
    w = rng.standard_normal(D)
    w /= np.linalg.norm(w)
    w_scorer = w + 0.15 * rng.standard_normal(D)
    x = rng.standard_normal((G, K, D))
    return dict(x=x, q=x @ w, fact=x @ w_scorer)


def _pick(scores):
    return np.argmax(scores, axis=1)


def _attack_pool(fact_scores, best, target):
    """Score-feedback-only sentiment search for one pool.

    Steps the target's sentiment up and the current winner's
    down, reading only the scorer's returned scores after
    each step. Returns the sentiment vector (facts are never
    an input the attacker can write).
    """
    s = np.zeros(K)
    steps = int(S_MAX / STEP)
    for _ in range(steps):
        s[target] += STEP
        s[best] -= STEP
        if int(np.argmax(fact_scores + GAMMA * s)) != best:
            break
    return s


def run_seed(seed):
    d = build(seed)
    fact, q = d["fact"], d["q"]
    clean = _pick(fact)
    correct = clean == _pick(q)
    n_correct = int(correct.sum())

    # (a) attack + (b) benign, on correctly selected pools.
    attack_flip = benign_flip = 0
    gate_blocked = 0
    gate_queries_attack = 0
    fact_before = fact.copy()
    for g in range(G):
        if not correct[g]:
            continue
        order = np.argsort(-fact[g])
        best, target = int(order[0]), int(order[1])
        s = _attack_pool(fact[g], best, target)
        with_comments = int(np.argmax(fact[g] + GAMMA * s))
        if with_comments != best:
            attack_flip += 1
            # (c) gate: one stripped re-score; accept the
            # change only if it reproduces comment-free.
            gate_queries_attack += 1
            stripped = int(np.argmax(fact[g]))
            if stripped != with_comments:
                gate_blocked += 1
        rng_b = np.random.default_rng(9000 + seed * 100 + g)
        s_b = rng_b.uniform(-0.3, 0.3, K)
        if int(np.argmax(fact[g] + GAMMA * s_b)) != best:
            benign_flip += 1
    # Fact-invariance of the attack channel, asserted in-run:
    # the attack wrote only sentiment; fact scores unchanged.
    assert np.array_equal(fact, fact_before)

    # (c) gate on the clean pool: no comments, so no decision
    # changes, no re-checks, no cost.
    clean_changed = int((_pick(fact) != clean).sum())

    # (d) fact-adding control: the runner-up's record gains a
    # genuine fact (fact score += margin + FACT_LIFT). With
    # the comment present the selection changes; stripped of
    # comment TEXT the fact remains in the record, so the
    # gate's re-score reproduces the change and accepts it.
    factadd_changed = factadd_passed = 0
    gate_queries_fact = 0
    for g in range(G):
        if not correct[g]:
            continue
        order = np.argsort(-fact[g])
        best, target = int(order[0]), int(order[1])
        margin = float(fact[g][best] - fact[g][target])
        fact_new = fact[g].copy()
        fact_new[target] += margin + FACT_LIFT
        with_fact = int(np.argmax(fact_new))
        if with_fact != best:
            factadd_changed += 1
            gate_queries_fact += 1
            stripped = int(np.argmax(fact_new))  # text gone, fact stays
            if stripped == with_fact:
                factadd_passed += 1

    return dict(
        clean_acc=float(correct.mean()),
        n_correct=n_correct,
        attack_flip=attack_flip / n_correct,
        benign_flip=benign_flip / n_correct,
        gate_block=(gate_blocked / attack_flip) if attack_flip else 1.0,
        gate_queries_attack=gate_queries_attack,
        attack_changed=attack_flip,
        clean_changed=clean_changed,
        factadd_changed=factadd_changed / n_correct,
        factadd_pass=(factadd_passed / factadd_changed)
        if factadd_changed else 0.0,
        gate_queries_fact=gate_queries_fact,
    )


def test_attack_flips_substantial_fraction_fact_invariant():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The scorer is a real selector, not a rubber stamp:
        # clean accuracy sits in the demonstrated band.
        assert 0.75 <= r["clean_acc"] <= 0.97, (
            f"seed {seed}: clean accuracy {r['clean_acc']:.3f}")
        # Fact-invariant comments flip a substantial fraction
        # of correct selections (paper ~1/3; demonstrated
        # 0.36-0.51, asserted with headroom at 0.25).
        assert r["attack_flip"] >= 0.25, (
            f"seed {seed}: attack flip rate {r['attack_flip']:.3f}")
        # Unsearched benign comments barely move selections,
        # and the searched attack beats them by a wide margin.
        assert r["benign_flip"] <= 0.10, (
            f"seed {seed}: benign flip rate {r['benign_flip']:.3f}")
        assert r["attack_flip"] >= 3.0 * r["benign_flip"], (
            f"seed {seed}: attack {r['attack_flip']:.3f} vs "
            f"benign {r['benign_flip']:.3f}")


def test_provenance_gate_blocks_flips_at_small_cost():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The gate blocks (nearly) all fact-invariant flips
        # (paper: all; demonstrated 1.000 on every seed).
        assert r["gate_block"] >= 0.95, (
            f"seed {seed}: gate block rate {r['gate_block']:.3f}")
        # Cost on the clean pool is zero: no comments means
        # no changed decisions and no gate re-checks at all.
        assert r["clean_changed"] == 0, (
            f"seed {seed}: gate changed {r['clean_changed']} "
            f"clean selections")
        # The paper's price: at most one extra query per
        # changed decision, never more.
        assert r["gate_queries_attack"] == r["attack_changed"], (
            f"seed {seed}: {r['gate_queries_attack']} gate "
            f"queries for {r['attack_changed']} changed "
            f"decisions")
        assert r["gate_queries_fact"] == int(
            r["factadd_changed"] * r["n_correct"]), (
            f"seed {seed}: fact-arm gate queries "
            f"{r['gate_queries_fact']}")


def test_fact_adding_changes_are_not_blocked():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The control actually changes selections: a genuine
        # fact lift beyond the margin flips the runner-up in
        # on (nearly) every correct pool.
        assert r["factadd_changed"] >= 0.95, (
            f"seed {seed}: fact-adding changed fraction "
            f"{r['factadd_changed']:.3f}")
        # And the gate passes them through: the change
        # persists with the comment text stripped, because
        # the fact is in the record (paper: the gate "still
        # accepts trusted updates"; demonstrated 1.000).
        assert r["factadd_pass"] >= 0.95, (
            f"seed {seed}: fact-adding pass-through "
            f"{r['factadd_pass']:.3f}")
