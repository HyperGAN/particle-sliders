"""CPU gate for the self-hallucination memorization claim
of "LLMs Memorize Their Own Hallucinations During
Post-Training" (arXiv:2611.26863): with standard
post-training (outcome-reward RL + injected reasoning
behaviors), models memorize hallucinations they
themselves generate. Claims from the abstract: recall of
self-generated hallucinations that were REINFORCED during
training rises by 4.83% on average, vs just 0.17% for
self-hallucinations that received NO training signal;
models memorize even false facts they immediately
retract; memorization is strongest on the HARDEST
training questions; and rewarding a hallucination makes
the model less likely to abstain from it later
(abstention likelihood drops 7.23% on average). Six open
models, two families, two architectures.

Provenance: cycle 26 of the Particle Sliders paper
watch. This is the generator-side dual of PR #302
(test_preference_label_channel.py): there a biased
JUDGE's trait reached the student through 1-bit labels on
the student's own completions (judge-side transmission).
Here the trait is not external at all -- it is the
generator's OWN error mode, and the question is whether
selection/reward over its own generations memorizes that
error while mere exposure does not. PR #286's iterated
screened-distillation result (behavioral screen decays
~4x faster than the latent probe across generations) is
the lineage-side neighbour: traits persist through
selection pipelines. If selection at OUR curation gate
(check() in scripts/check_slider_gate.py) is a
memorization amplifier for the generator's own errors,
that is a curation-stage law, and the abstention readout
maps onto the UNKNOWN behaviour the Tiny Aya task shape
depends on.

Toy operationalization: 48 knowledge items, each with a
TRUE association and a planted FALSE association (a
plausible-but-wrong answer the student sometimes emits --
its self-hallucination) plus a generic OTHER answer. The
student is per-item logits over {true, false, other}
(false also carries a shared global template bias, the
mechanism by which a memorized false template can later
compete with abstention on unknown items). Difficulty:
24 EASY items (true logit init 1.3) and 24 HARD items
(true logit init -0.4); false logit init 0.25 for all.
Each round the student samples K = 6 completions per
item; a surface gate scores FLUENCY only (false template
mean 0.78 -- it is a fluent, well-worn pattern; true
completions 0.70 on easy items / 0.52 on hard items --
fluency of a true completion reflects how well the item
is known; other 0.35; noise sd 0.16) and a curation quota
keeps the TOP 2 per item as training material. The gate
never sees correctness. Training is one cross-entropy
gradient step (LR = 0.25) per kept completion; after each
round, true logits relax 30% toward their init (prior
knowledge reasserts itself -- weak priors reassert
weakly). 4 rounds. Readouts are probability readouts on
held-out probes (the items' answer distributions are
never trained on directly except through kept
completions): false-mode recall = mean P(false);
abstention = P(abstain) on unknown items whose options
are {guess via the shared false template, abstain,
generic}.

Arms (identical inits per seed; paired per-round RNG
streams):
  (a) REWARDED: train on the gate's top-2 per item,
      whatever type they are -- false completions that
      pass the gate are reinforced.
  (b) EXPOSURE-ONLY: the same probed items generate the
      same kind of pools and the gate scores them, but
      probed completions receive NO gradient; training
      continues at comparable volume on 24 disjoint
      FILLER items' gate-passing non-false completions.
      Mere generation/exposure, no signal on the
      hallucinations.
  (c) REWARDED-TRUE control: train on gate-passing TRUE
      completions (up to 2 per item) -- reward installs
      what it touches, so true recall should grow and
      false recall should not.
  (d) RANDOM-REWARD control: train on 2 uniformly random
      completions per item, content-independent.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds -- final-minus-init changes:
  REWARDED false recall: +0.216 / +0.235 / +0.234 /
    +0.252 / +0.224 (every seed, 4 rounds).
  EXPOSURE false recall: -0.103 / -0.102 / -0.101 /
    -0.100 / -0.097 (no growth on any seed; the small
    decline is shared-template-bias drift from the
    filler training, stated here, not hidden).
  RANDOM false recall: +0.006 / -0.013 / -0.041 /
    +0.034 / -0.019 (|change| <= 0.041).
  REWARDED-TRUE: true recall +0.126 / +0.118 / +0.118 /
    +0.115 / +0.133, false recall -0.155 / -0.146 /
    -0.144 / -0.141 / -0.162.
  DIFFICULTY (rewarded arm, hard minus easy false-recall
    growth): +0.104 / +0.078 / +0.086 / +0.068 / +0.084
    -- hard growth exceeds easy on every seed, mean
    concentration +0.084. The toy's concentration is
    real but modest, and its mechanism is visible: on
    easy items a fluent TRUE completion often takes one
    of the two quota slots and counter-trains; on hard
    items it rarely does, so the false mode trains
    unopposed.
  ABSTENTION (unknown items): rewarded arm -0.138 /
    -0.149 / -0.150 / -0.159 / -0.141 (abstention is
    suppressed); exposure arm +0.056 / +0.055 / +0.054 /
    +0.053 / +0.053 (no suppression).

The arbitration, stated plainly: the load-bearing claim
reproduces. (1) Rewarded >> exposure: reinforcing
gate-passing self-hallucinations grows their recall by
+0.22..+0.25 while the identical generations with no
training signal grow by at most +0.02 (they slightly
decay) -- a +0.32..+0.35 gap on every seed. Selection is the
amplifier; exposure is not. (2) The controls isolate
the mechanism: random reward at matched volume moves
false recall by at most 0.041, and rewarding TRUE
instead grows true recall (+0.12..+0.13) while false
recall falls -- the gate installs whatever it rewards,
so the false growth in arm (a) is caused by the false
completions the fluency gate keeps, not by training
volume or by training on own generations per se.
(3) Abstention: rewarded false-mode training suppresses
later abstention on unknown items by 0.14..0.16 via the
shared false-template bias, while exposure leaves
abstention unsuppressed (+0.06). (4) Difficulty: growth
concentrates on hard items on every seed, by
+0.07..+0.10 over easy items -- direction as in the
paper, magnitude modest in the toy.

What is NOT asserted: the paper's magnitudes (4.83% /
0.17% / -7.23% are LM post-training numbers; the toy
asserts its own measured floors), the retraction claim
(the toy has no retraction act -- not operationalized),
and anything about real RL pipelines, whose reward is
richer than a fluency quota. The gate's fluency
advantage for the false template is an assumption of
the toy (a hallucination that is NOT fluent would not
pass this gate, and the toy says nothing about it),
stated here rather than hidden. Pure numpy -- no GAN
machinery, because the claim is about selection-mediated
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


N = 48               # probed knowledge items (24 easy + 24 hard)
N_FILLER = 24        # disjoint filler items (exposure arm's training)
K = 6                # completions sampled per item per round
TOP = 2              # curation quota: gate keeps top-2 by fluency
ROUNDS = 4           # generate -> select -> train rounds
LR = 0.25            # cross-entropy step per kept completion
ANCHOR = 0.30        # per-round relaxation of true logits toward init
GBS = 0.045          # shared false-template bias share of the false gradient
THR = 0.62           # fluency pass threshold (filler / rewarded-true arms)
FLU = {"true_hard": 0.52, "true_easy": 0.70, "false": 0.78, "other": 0.35}
FLU_SD = 0.16
OTHER_LOGIT = -0.6
ABSTAIN_LOGIT = 0.9
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("rewarded", "exposure", "rewarded_true", "random")


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _probs(true_logit, false_logit, gbias):
    return _softmax(np.array([true_logit, false_logit + gbias, OTHER_LOGIT]))


def _abstain_prob(gbias):
    # Unknown item: guess via the shared false template vs abstain vs generic.
    return float(_softmax(np.array([gbias, ABSTAIN_LOGIT, -0.5]))[1])


def _flu_mean(comp, is_easy):
    if comp == 0:
        return FLU["true_easy"] if is_easy else FLU["true_hard"]
    if comp == 1:
        return FLU["false"]
    return FLU["other"]


def _run(seed, arm):
    """One arm of one seed. Returns histories (ROUNDS + 1 values,
    index 0 = init) for false recall (all / easy / hard), true
    recall, and the abstention rate on unknown items."""
    init_rng = np.random.default_rng(9000 + seed)
    is_easy = np.arange(N) < N // 2
    true_init = np.where(is_easy, 1.3, -0.4) + init_rng.normal(0, 0.1, N)
    false_init = 0.25 + init_rng.normal(0, 0.05, N)
    true_l = true_init.copy()
    false_l = false_init.copy()
    f_true = np.full(N_FILLER, 0.9)
    f_false = np.full(N_FILLER, 0.25)
    gbias = 0.0

    def false_recall(mask):
        return float(np.mean([_probs(true_l[i], false_l[i], gbias)[1]
                              for i in range(N) if mask[i]]))

    def true_recall():
        return float(np.mean([_probs(true_l[i], false_l[i], gbias)[0]
                              for i in range(N)]))

    all_mask = np.ones(N, dtype=bool)
    hist = {"false": [false_recall(all_mask)],
            "false_easy": [false_recall(is_easy)],
            "false_hard": [false_recall(~is_easy)],
            "true": [true_recall()],
            "abstain": [_abstain_prob(gbias)]}
    for r in range(ROUNDS):
        rng = np.random.default_rng(10000 + seed * 100 + r)
        pools = []
        for i in range(N):
            comps = rng.choice(3, size=K, p=_probs(true_l[i], false_l[i], gbias))
            flu = np.array([rng.normal(_flu_mean(c, is_easy[i]), FLU_SD)
                            for c in comps])
            pools.append((comps, flu))
        chosen = []  # (is_filler, item, type)
        if arm == "exposure":
            # Probed pools are generated and gate-scored above but
            # receive no gradient; training continues on filler
            # items' gate-passing non-false completions.
            for j in range(N_FILLER):
                comps = rng.choice(3, size=K, p=_probs(f_true[j], f_false[j], gbias))
                flu = np.array([rng.normal(_flu_mean(c, True), FLU_SD)
                                for c in comps])
                for jj in range(K):
                    if flu[jj] > THR and comps[jj] != 1:
                        chosen.append((True, j, int(comps[jj])))
        else:
            for i, (comps, flu) in enumerate(pools):
                if arm == "rewarded":
                    sel = [int(comps[j]) for j in np.argsort(flu)[-TOP:]]
                elif arm == "rewarded_true":
                    sel = [int(comps[j]) for j in range(K)
                           if flu[j] > THR and comps[j] == 0][:TOP]
                else:  # random: content-independent, matched volume
                    sel = [int(comps[j])
                           for j in rng.choice(K, size=TOP, replace=False)]
                for t in sel:
                    chosen.append((False, i, t))
        rng.shuffle(chosen)
        for is_filler, i, t in chosen:
            if is_filler:
                p = _probs(f_true[i], f_false[i], gbias)
                grad = -p.copy()
                grad[t] += 1.0
                if t == 0:
                    f_true[i] += LR * grad[0]
                f_false[i] += LR * 0.85 * grad[1]
            else:
                p = _probs(true_l[i], false_l[i], gbias)
                grad = -p.copy()
                grad[t] += 1.0
                if t == 0:
                    true_l[i] += LR * grad[0]
                false_l[i] += LR * 0.85 * grad[1]
            gbias += LR * GBS * grad[1]
        # Prior knowledge reasserts: true logits relax toward init.
        true_l += ANCHOR * (true_init - true_l)
        hist["false"].append(false_recall(all_mask))
        hist["false_easy"].append(false_recall(is_easy))
        hist["false_hard"].append(false_recall(~is_easy))
        hist["true"].append(true_recall())
        hist["abstain"].append(_abstain_prob(gbias))
    return hist


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = {a: _run(seed, a) for a in ARMS}
    return _CACHE[seed]


def _growth(hist, key):
    return hist[key][-1] - hist[key][0]


def test_rewarded_beats_exposure_on_false_recall():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        rew = _growth(arms["rewarded"], "false")
        exp = _growth(arms["exposure"], "false")
        # Reinforced self-hallucinations are memorized:
        # false recall grows >= +0.18 on every seed
        # (demonstrated +0.216..+0.252).
        assert rew >= 0.18, (
            f"seed {seed}: rewarded false-recall growth {rew:+.3f}")
        # The same generations with no training signal do
        # not grow (demonstrated -0.103..-0.097).
        assert exp <= 0.02, (
            f"seed {seed}: exposure false-recall growth {exp:+.3f}")
        # The load-bearing contrast: a >= 0.25 gap on
        # every seed (demonstrated +0.32..+0.35).
        assert rew - exp >= 0.25, (
            f"seed {seed}: rewarded-exposure gap {rew - exp:+.3f} "
            f"(rewarded {rew:+.3f}, exposure {exp:+.3f})")


def test_controls_isolate_the_gate():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        # Random reward at matched volume: no false-mode
        # growth beyond noise (demonstrated |change|
        # <= 0.041).
        rnd = _growth(arms["random"], "false")
        assert abs(rnd) <= 0.06, (
            f"seed {seed}: random false-recall growth {rnd:+.3f}")
        # Rewarding TRUE instead installs true: true
        # recall grows (demonstrated +0.115..+0.133)
        # while false recall falls (demonstrated
        # -0.162..-0.141) -- the gate installs what it
        # rewards.
        rt = arms["rewarded_true"]
        assert _growth(rt, "true") >= 0.09, (
            f"seed {seed}: rewarded-true true-recall growth "
            f"{_growth(rt, 'true'):+.3f}")
        assert _growth(rt, "false") <= -0.08, (
            f"seed {seed}: rewarded-true false-recall growth "
            f"{_growth(rt, 'false'):+.3f}")


def test_rewarded_hallucination_suppresses_abstention():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        rew = _growth(arms["rewarded"], "abstain")
        exp = _growth(arms["exposure"], "abstain")
        # Rewarded false-mode training suppresses later
        # abstention on unknown items (demonstrated
        # -0.159..-0.138).
        assert rew <= -0.10, (
            f"seed {seed}: rewarded abstention change {rew:+.3f}")
        # Exposure does not suppress it (demonstrated
        # +0.063..+0.066).
        assert exp >= 0.02, (
            f"seed {seed}: exposure abstention change {exp:+.3f}")
        assert exp - rew >= 0.15, (
            f"seed {seed}: abstention gap {exp - rew:+.3f}")


def test_memorization_concentrates_on_hard_items():
    _empf_recipe()
    gaps = []
    for seed in SEEDS:
        h = run_seed(seed)["rewarded"]
        gap = _growth(h, "false_hard") - _growth(h, "false_easy")
        gaps.append(gap)
        # Hard-item growth exceeds easy-item growth on
        # every seed (demonstrated +0.068..+0.104).
        assert gap >= 0.04, (
            f"seed {seed}: hard-minus-easy false-recall "
            f"growth {gap:+.3f} (hard {_growth(h, 'false_hard'):+.3f}, "
            f"easy {_growth(h, 'false_easy'):+.3f})")
    # Mean concentration (demonstrated +0.084).
    assert float(np.mean(gaps)) >= 0.06, (
        f"mean hard-minus-easy gap {float(np.mean(gaps)):+.3f}")
