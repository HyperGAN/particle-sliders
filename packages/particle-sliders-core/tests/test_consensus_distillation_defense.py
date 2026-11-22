"""CPU gate for the consensus-distillation defence claim of
Narang, Tajdini, Zhang, Morgenstern, "Defense via
Consensus: Mitigating Subliminal Learning through
Multi-Teacher Distillation" (MSCD), arXiv:2607.23394.
The paper makes two linked claims. (1) Pooling-style
combination does NOT veto a source-specific unwanted
behaviour: union training on pooled multi-teacher data and
KL-regularized averaging (LoRA merging in the paper)
retain the shared benefit but preserve the single-source
cost at a diluted, clearly nonzero rate -- the paper's
Lemma 1 shows arithmetic pooling keeps (1 + delta/2) of a
single teacher's probability increase and geometric
(KL) pooling keeps roughly its square root. (2)
Consensus-regenerated supervision -- regenerating targets
by cross-teacher agreement (the paper's minimum rule and
its base-relative minimum rule, section 4.3) -- drives a
single-source cost back to base while retaining the
shared benefit. The paper's measured headline: the
explicit-prefix MSCD student keeps the shared joke marker
on 94.1% of responses with 0/320 source prefixes, and the
subliminal MSCD student keeps 86.17% markers with only
0.55pp positive-excess animal preference, while union
training and LoRA merging keep substantially more cost.
Provenance: cycle 18 of the Particle Sliders paper watch.

Toy operationalization: the smallest categorical world in
which the asymmetry exists by construction. Vocab V = 24
over a single categorical draw. The base pi0 is the
softmax of random logits with the benefit-token and
cost-token logits pinned at -1.0. K = 4 teachers, each the
softmax of (base logits + shifts + per-teacher Gaussian
logit noise, sigma = 0.15): ALL teachers shift the
benefit token +2.2 logits (the shared benefit); teacher 0
ONLY also shifts the cost token +2.2 (the source-specific
cost); each teacher i shifts its own private quirk token
+1.5. The student is a free logit vector trained by SGD
(lr 0.15, 400 steps) on N = 20000 tokens sampled from
each arm's target distribution, starting from the base
logits. Readouts per arm are excesses over base of the
benefit-token probability, the cost-token probability,
and the mean quirk-token probability.

Arms, all paired per seed (every arm of a seed shares the
base, the teachers, and the sampling stream structure;
only the target distribution differs):
  single:    teacher 0 alone -- the cost installs fully;
             the reference the other arms are ratioed to.
  union:     the arithmetic mean of the 4 teacher
             distributions (pooled data) -- Lemma 1's
             arithmetic pooling arm.
  geo:       the normalized geometric mean of the
             teachers -- Lemma 1's KL-pooling arm.
  mscd_min:  the normalized per-token minimum across
             teachers (the paper's minimum rule).
  mscd_delta: the paper's base-relative minimum rule --
             per token, Delta_i = pi_i - pi0; if all
             Delta_i > 0 use pi0 + min Delta_i; if all
             Delta_i < 0 use pi0 + max Delta_i; otherwise
             pi0; clipped at 1e-9 and normalized.

Measured on pop-os (particlegan 0.10.1), 5 seeds --
per-seed cost excess per arm (single / union / geo /
mscd_min / mscd_delta), then benefit excess (union /
mscd_min / mscd_delta) and quirk excess (union /
mscd_min / mscd_delta):
  seed 0: cost +0.0574 / +0.0076 / +0.0026 / -0.0007 /
          +0.0003; benefit +0.0682 / +0.0714 / +0.0501;
          quirk +0.0176 / +0.0026 / +0.0003
  seed 1: cost +0.0252 / +0.0038 / +0.0015 / -0.0003 /
          +0.0004; benefit +0.0420 / +0.0309 / +0.0094;
          quirk +0.0181 / +0.0120 / +0.0003
  seed 2: cost +0.0693 / +0.0108 / +0.0036 / +0.0015 /
          +0.0005; benefit +0.0495 / +0.0566 / +0.0367;
          quirk +0.0198 / +0.0005 / +0.0006
  seed 3: cost +0.1133 / +0.0221 / +0.0041 / -0.0027 /
          +0.0001; benefit +0.1046 / +0.1103 / +0.0791;
          quirk +0.0140 / -0.0016 / +0.0013
  seed 4: cost +0.0615 / +0.0093 / +0.0022 / +0.0004 /
          +0.0015; benefit +0.0452 / +0.0555 / +0.0365;
          quirk +0.0244 / -0.0060 / +0.0031
Aggregates: cost single +0.0654, union +0.0107 (0.16x
single), geo +0.0028 (0.04x single), mscd_min -0.0004,
mscd_delta +0.0006; benefit union +0.0619, mscd_min
+0.0650 (1.05x union), mscd_delta +0.0424 (0.68x union);
quirk union +0.0188, mscd_min +0.0015 (0.08x union),
mscd_delta +0.0011 (0.06x union).

The arbitration, stated plainly: both halves of the
paper's claim reproduce. Pooling dilutes but does not
veto: union retains 0.13-0.20x of the single-teacher cost
excess on every seed, and geometric pooling retains less
(0.04x aggregate) but a clearly nonzero amount on every
seed -- neither combination rule is a defence. Both
consensus rules drive the cost back to base: aggregate
excess -0.0004 (minimum) and +0.0006 (base-relative),
at most 0.05x of union's for the delta rule, with every
per-seed value inside +/-0.003. The shared benefit
survives regeneration: the minimum rule matches or beats
union in 4 of 5 seeds (1.05x aggregate), and the
base-relative rule keeps 0.68x in aggregate, though its
per-seed retention varies (0.22-0.81x of union) because
it is bottlenecked by the least-boosting teacher's delta
on the benefit token. Private quirks -- single-source by
construction, like the cost -- are vetoed in aggregate
by both rules (<= 0.08x of union), though the minimum
rule's per-seed quirk suppression is uneven (seed 1:
0.66x of union), so the quirk assertions are aggregate
only. Assertions below encode exactly these margins,
no more.

What is NOT asserted: anything about real LM distillation,
where teachers disagree in context-dependent ways, the
shared benefit is not a single token every teacher shifts
identically, and consensus regeneration operates over
sequences rather than one categorical draw; the paper's
LoRA-merging arm (the toy's geo arm is its distribution-
level analogue only); and any claim that the minimum rule
is safe when a MAJORITY of teachers share the cost -- by
construction the rule only vetoes what at least one
teacher does not carry. Pure categorical SGD -- no GAN
machinery, because the claim is about how target
distributions combine, not training dynamics. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test
family; PR #211 is what enables this test in CI).
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


V = 24               # vocabulary size
K = 4                # teachers
BENEFIT = 0          # token every teacher boosts
COST = 1             # token only teacher 0 boosts
QUIRKS = (2, 3, 4, 5)  # teacher i's private quirk token
BENEFIT_SHIFT = 2.2  # shared benefit logit shift
COST_SHIFT = 2.2     # source-specific cost logit shift
QUIRK_SHIFT = 1.5    # private quirk logit shift
NOISE = 0.15         # per-teacher Gaussian logit noise
N_TOKENS = 20000     # sampled tokens per arm
STEPS = 400          # student SGD steps
LR = 0.15            # student SGD learning rate
SEED_BASE = 18000    # RNG stream base
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("single", "union", "geo", "mscd_min", "mscd_delta")


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _targets(base_logits, pi0, teachers):
    """The five arms' target distributions for one seed."""
    union = teachers.mean(axis=0)
    geo = np.exp(np.log(np.clip(teachers, 1e-12, None)).mean(axis=0))
    geo /= geo.sum()
    mscd_min = teachers.min(axis=0)
    mscd_min = mscd_min / mscd_min.sum()
    deltas = teachers - pi0
    delta_rule = pi0.copy()
    all_pos = (deltas > 0).all(axis=0)
    all_neg = (deltas < 0).all(axis=0)
    delta_rule[all_pos] = pi0[all_pos] + deltas[:, all_pos].min(axis=0)
    delta_rule[all_neg] = pi0[all_neg] + deltas[:, all_neg].max(axis=0)
    delta_rule = np.clip(delta_rule, 1e-9, None)
    delta_rule /= delta_rule.sum()
    return {
        "single": teachers[0],
        "union": union,
        "geo": geo,
        "mscd_min": mscd_min,
        "mscd_delta": delta_rule,
    }


def _train_student(base_logits, target, rng):
    """SGD a free logit vector on tokens sampled from target."""
    counts = rng.multinomial(N_TOKENS, target)
    counts_t = torch.tensor(counts, dtype=torch.float64)
    logits = torch.tensor(base_logits, dtype=torch.float64,
                          requires_grad=True)
    opt = torch.optim.SGD([logits], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        logp = torch.log_softmax(logits, dim=0)
        loss = -(counts_t * logp).sum() / N_TOKENS
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(logits, dim=0).numpy()


def run_seed(seed):
    """All five arms for one seed, paired on one draw.

    Draw order is fixed -- base logits, per-teacher noise,
    then per-arm token samples in ARMS order -- so arms
    differ only through their target distribution.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    base_logits = rng.standard_normal(V)
    base_logits[BENEFIT] = -1.0
    base_logits[COST] = -1.0
    pi0 = _softmax(base_logits)

    teachers = []
    for i in range(K):
        logits = base_logits.copy()
        logits[BENEFIT] += BENEFIT_SHIFT
        if i == 0:
            logits[COST] += COST_SHIFT
        logits[QUIRKS[i]] += QUIRK_SHIFT
        logits = logits + rng.normal(0, NOISE, V)
        teachers.append(_softmax(logits))
    teachers = np.array(teachers)

    targets = _targets(base_logits, pi0, teachers)
    out = {}
    for arm in ARMS:
        student = _train_student(base_logits, targets[arm], rng)
        out[arm] = {
            "benefit": float(student[BENEFIT] - pi0[BENEFIT]),
            "cost": float(student[COST] - pi0[COST]),
            "quirk": float(np.mean([student[q] - pi0[q]
                                    for q in QUIRKS])),
        }
    return out


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def _aggregate():
    agg = {arm: {"benefit": [], "cost": [], "quirk": []}
           for arm in ARMS}
    for seed in SEEDS:
        res = _seed(seed)
        for arm in ARMS:
            for key in agg[arm]:
                agg[arm][key].append(res[arm][key])
    return {arm: {key: float(np.mean(vals))
                  for key, vals in d.items()}
            for arm, d in agg.items()}


def test_pooling_dilutes_but_does_not_veto_cost():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        # Teacher 0 alone installs the cost fully.
        assert res["single"]["cost"] >= 0.010, (
            f"seed {seed}: single-teacher cost excess "
            f"{res['single']['cost']:+.4f}")
        # Union pooling retains a clearly nonzero share of
        # it on every seed: dilution, not veto.
        assert res["union"]["cost"] >= 0.002, (
            f"seed {seed}: union cost excess "
            f"{res['union']['cost']:+.4f}")
        assert res["union"]["cost"] >= 0.08 * res["single"]["cost"], (
            f"seed {seed}: union cost {res['union']['cost']:+.4f} "
            f"vs single {res['single']['cost']:+.4f}")
        # Geometric (KL) pooling also retains a nonzero
        # share -- Lemma 1's square-root retention is
        # weaker than arithmetic pooling here but no veto.
        assert res["geo"]["cost"] >= 0.001, (
            f"seed {seed}: geo cost excess "
            f"{res['geo']['cost']:+.4f}")
    agg = _aggregate()
    assert agg["union"]["cost"] >= 0.10 * agg["single"]["cost"], (
        f"aggregate union cost {agg['union']['cost']:+.4f} vs "
        f"single {agg['single']['cost']:+.4f}")
    assert agg["geo"]["cost"] >= 0.03 * agg["single"]["cost"], (
        f"aggregate geo cost {agg['geo']['cost']:+.4f} vs "
        f"single {agg['single']['cost']:+.4f}")


def test_consensus_rules_veto_single_source_cost():
    _empf_recipe()
    agg = _aggregate()
    for arm in ("mscd_min", "mscd_delta"):
        # Both consensus rules drive the single-source cost
        # back to base: near zero in absolute terms and a
        # small fraction of what union pooling retains.
        assert agg[arm]["cost"] <= 0.002, (
            f"aggregate {arm} cost excess {agg[arm]['cost']:+.4f}")
        assert agg[arm]["cost"] <= 0.30 * agg["union"]["cost"], (
            f"aggregate {arm} cost {agg[arm]['cost']:+.4f} vs "
            f"union {agg['union']['cost']:+.4f}")
    for seed in SEEDS:
        res = _seed(seed)
        for arm in ("mscd_min", "mscd_delta"):
            assert res[arm]["cost"] <= 0.003, (
                f"seed {seed}: {arm} cost excess "
                f"{res[arm]['cost']:+.4f}")
            assert res[arm]["cost"] <= 0.50 * res["union"]["cost"], (
                f"seed {seed}: {arm} cost {res[arm]['cost']:+.4f} "
                f"vs union {res['union']['cost']:+.4f}")


def test_consensus_rules_retain_shared_benefit():
    _empf_recipe()
    agg = _aggregate()
    # Union pooling itself retains the shared benefit --
    # the baseline the consensus rules are ratioed to.
    assert agg["union"]["benefit"] >= 0.010, (
        f"aggregate union benefit excess "
        f"{agg['union']['benefit']:+.4f}")
    # The minimum rule keeps essentially all of it.
    assert agg["mscd_min"]["benefit"] >= 0.80 * agg["union"]["benefit"], (
        f"aggregate mscd_min benefit {agg['mscd_min']['benefit']:+.4f} "
        f"vs union {agg['union']['benefit']:+.4f}")
    # The base-relative minimum rule keeps a clear
    # majority of it.
    assert agg["mscd_delta"]["benefit"] >= 0.50 * agg["union"]["benefit"], (
        f"aggregate mscd_delta benefit "
        f"{agg['mscd_delta']['benefit']:+.4f} vs union "
        f"{agg['union']['benefit']:+.4f}")
    for seed in SEEDS:
        res = _seed(seed)
        assert res["mscd_min"]["benefit"] >= 0.60 * res["union"]["benefit"], (
            f"seed {seed}: mscd_min benefit "
            f"{res['mscd_min']['benefit']:+.4f} vs union "
            f"{res['union']['benefit']:+.4f}")
        # Per-seed delta-rule retention varies more than
        # the minimum rule's (demonstrated ratios
        # 0.22-0.81): the floor is that it stays clearly
        # positive on every seed, with the strong claim
        # carried by the aggregate above.
        assert res["mscd_delta"]["benefit"] >= 0.15 * res["union"]["benefit"], (
            f"seed {seed}: mscd_delta benefit "
            f"{res['mscd_delta']['benefit']:+.4f} vs union "
            f"{res['union']['benefit']:+.4f}")


def test_consensus_rules_suppress_private_quirks():
    _empf_recipe()
    agg = _aggregate()
    # Union pooling carries each teacher's private quirk
    # at a diluted but nonzero rate.
    assert agg["union"]["quirk"] >= 0.001, (
        f"aggregate union quirk excess {agg['union']['quirk']:+.4f}")
    # A quirk, like the cost, is carried by only one
    # teacher, so both consensus rules veto it too.
    for arm in ("mscd_min", "mscd_delta"):
        assert agg[arm]["quirk"] <= 0.30 * agg["union"]["quirk"], (
            f"aggregate {arm} quirk {agg[arm]['quirk']:+.4f} vs "
            f"union {agg['union']['quirk']:+.4f}")
