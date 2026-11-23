"""CPU gate for the cluster-capped consensus repair probe:
a gauntlet of the natural repair for the correlation
boundary proven in PR #305 for Narang, Tajdini, Zhang,
Morgenstern, "Defense via Consensus: Mitigating Subliminal
Learning through Multi-Teacher Distillation" (MSCD),
arXiv:2607.23394.

The anchor chain. PR #285 (cycle 18) proved that at pool
composition, pooling merely DILUTES a cost carried by one
source (union cost excess 0.16x the single-source excess),
while cross-source consensus VETOES it (MSCD-min aggregate
cost excess -0.0004, MSCD-delta +0.0006) while retaining
the shared benefit at 1.05x (min) / 0.68x (delta) of
union. PR #305 (cycle 22) proved the boundary: consensus
can only veto what sources DISAGREE about, so veto power
1 - (consensus / union cost excess) collapses as the
cross-source cost correlation rho rises -- rho = 0:
1.06 (min) / 0.97 (delta); rho = 0.5: 0.45 / 0.68;
rho = 1: -0.15 / 0.22, where the fully correlated cost is
retained at 1.15x / 0.78x of union, exactly the retention
the genuinely shared benefit gets at the same rho.

Repair claim under test here: the disagreement structure
consensus needs is ESTIMABLE from the sources' own
behavior, so capping correlated clusters should restore
part of the lost veto. Procedure: (1) estimate each
source's induced shift on a fixed probe set -- here each
teacher's output shift vs base, s_i = log p_i - log pi0,
centered per teacher (the softmax normalization constant
is a per-teacher scalar, not a direction; centering
removes it), the estimation shape of the watch's #280
audit; (2) cluster sources by shift similarity --
best-merge agglomerative clustering with complete
linkage on the Euclidean distance between centered
shifts, merging while the complete-linkage distance is
below CLUSTER_DIST = 2.8 (justification: source pairs
differing only by their private quirks and the sigma =
0.15 logit noise measure at most 2.66 apart across every
(seed, rho) panel of this toy, while a pair additionally
differing by the full idiosyncratic cost loading at
rho = 0 measures at least 2.96 apart; 2.8 sits in that
gap); (3) cap each cluster's total weight in the
consensus aggregation: each cluster collapses to ONE
representative (the within-cluster mean distribution --
the cluster's single vote), and MSCD-min / MSCD-delta
run over cluster representatives instead of sources.

Decidable claims: (i) at rho = 0.5, true-cluster capping
recovers a measured share of the lost veto (veto power
strictly above plain consensus at the same rho) at
bounded benefit-retention cost; (ii) at rho = 1, capping
recovers NOTHING (all cost-carrying sources form one
cluster; the cap degenerates) -- the repair's boundary
is part of the claim; (iii) at rho = 0 calibration,
capping does not damage the intact veto. Controls at
every rho: plain consensus (PR #305's arms), union, and
a RANDOM-cluster cap -- a random partition of the
sources into clusters of the SAME sizes as the estimated
partition, capped identically. If random capping matches
true capping at rho = 0.5, any recovery is a
weight-shrinkage artifact, not correlation repair.

Toy: PR #305's pool-composition toy unchanged. Vocab
V = 24, single categorical draw. Base pi0 = softmax of
random logits with benefit- and cost-token logits pinned
at -1.0. K = 4 teachers = softmax(base logits + shifts
+ per-teacher Gaussian logit noise sigma = 0.15): ALL
teachers shift the benefit token +2.2 (shared benefit);
each teacher shifts its own private quirk token +1.5;
the cost shift on teacher i is rho * 2.2 + (1 - rho) *
2.2 * [i == 0], so rho = 0 reproduces #285/#305 exactly
and teacher 0 is full-strength at all rho. Student =
free logit vector, SGD (lr 0.15, 400 steps) on N = 20000
tokens sampled from each arm's target, starting from the
base logits. Arms at each rho, paired per seed, in fixed
draw order (the four plain arms first, in #305's order,
so their token draws -- and hence their student results
-- are #305's exactly): single, union, mscd_min,
mscd_delta, cap_min, cap_delta, rnd_min, rnd_delta.
Readouts: excess over base of benefit- and cost-token
probabilities. Cost claims are aggregate over seeds,
keeping #285/#305's discipline.

Measured on pop-os (particlegan 0.10.1), 5 seeds --
aggregate cost excess (union / mscd_min / mscd_delta /
cap_min / cap_delta / rnd_min / rnd_delta):
  rho = 0.0: +0.0107 / -0.0006 / +0.0004 / +0.0002 /
             +0.0004 / -0.0005 / +0.0006; veto min
             1.06 plain, 0.98 cap, 1.05 random; veto
             delta 0.97 plain, 0.96 cap, 0.94 random;
             cap benefit retention 1.13x / 0.95x union
  rho = 0.5: +0.0217 / +0.0119 / +0.0069 / +0.0173 /
             +0.0128 / +0.0142 / +0.0115; veto min
             0.45 plain, 0.20 cap, 0.35 random; veto
             delta 0.68 plain, 0.41 cap, 0.47 random;
             cap benefit retention 1.10x / 0.93x union
  rho = 1.0: +0.0584 / +0.0671 / +0.0455 / +0.0641 /
             +0.0567 / +0.0642 / +0.0577; veto min
             -0.15 plain, -0.10 cap, -0.10 random; veto
             delta 0.22 plain, 0.03 cap, 0.01 random;
             cap retains cost at 1.10x / 0.97x union;
             cap benefit retention 0.96x / 0.89x union
Estimated partitions: rho = 0 gives {0} | {1, 2, 3} on
5/5 seeds; rho = 0.5 on 3/5 seeds (seed 1: {0, 1} |
{2, 3}; seed 2: a single cluster); rho = 1 degenerates
to a single cluster on 3/5 seeds. Per-seed at rho = 0.5
the capped cost exceeds the plain cost on EVERY seed
for both forms (min: cap +0.0113..+0.0231 vs plain
+0.0075..+0.0177; delta: cap +0.0081..+0.0222 vs plain
+0.0028..+0.0124). Calibration: the plain arms
reproduce #305's table exactly at every rho (single
+0.0654 and union +0.0107 at rho = 0; veto 0.45 / 0.68
at rho = 0.5; -0.15 / 0.22 at rho = 1).

The arbitration, stated plainly: claim (i) FAILS --
capping recovers no veto at rho = 0.5; the veto falls
FURTHER under the cap (min 0.45 -> 0.20, delta 0.68 ->
0.41), and the true cap is worse than the random cap on
both forms (cost +0.0173 vs +0.0142 min, +0.0128 vs
+0.0115 delta), so the movement capping produces is not
correlation repair in any part. The mechanism is
visible in the construction: plain MSCD-min / MSCD-delta
already read the minimum-cost member of the pool, and a
cluster representative is a within-cluster MEAN, whose
cost sits at or above that member's -- collapsing a
cluster can only raise the floor the consensus rules
read, never lower it. (A minimum representative instead
of a mean would reproduce plain consensus exactly --
min of mins -- so no representative choice recovers
veto either.) The failure is in the aggregation step,
not the estimation step: the partitions do recover the
pool structure where it is recoverable ({0} | {1, 2, 3}
on 5/5 seeds at rho = 0, the same bloc on 3/5 at
rho = 0.5). Claims (ii) and (iii) HOLD as measured: at
rho = 1 the cap degenerates (single cluster, cost
retained at 1.10x / 0.97x of union, veto -0.10 / 0.03
against plain -0.15 / 0.22), and at rho = 0 the capped
veto is intact (0.98 / 0.96 against plain 1.06 / 0.97)
at benefit retention 1.13x / 0.95x. The honest
implication for the chain: #305's boundary is not an
estimation failure that a smarter aggregation can
repair -- plain consensus already extracts all the
disagreement the pool contains, and a fully or partly
shared cost has no disagreement left to price.
Correlation has to be priced where #305 said: at
curation, before pooling.
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


V = 24               # vocabulary size
K = 4                # teachers
BENEFIT = 0          # token every teacher boosts
COST = 1             # token the cost load lands on
QUIRKS = (2, 3, 4, 5)  # teacher i's private quirk token
BENEFIT_SHIFT = 2.2  # shared benefit logit shift
COST_SHIFT = 2.2     # full cost logit shift (teacher 0, all rho)
QUIRK_SHIFT = 1.5    # private quirk logit shift
NOISE = 0.15         # per-teacher Gaussian logit noise
N_TOKENS = 20000     # sampled tokens per arm
STEPS = 400          # student SGD steps
LR = 0.15            # student SGD learning rate
SEED_BASE = 18000    # RNG stream base (same as #285/#305)
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.0, 0.5, 1.0)
CLUSTER_DIST = 2.8   # complete-linkage merge threshold (see docstring)
RND_SEED_BASE = 777000  # independent stream for random partitions
PLAIN_ARMS = ("single", "union", "mscd_min", "mscd_delta")
ARMS = PLAIN_ARMS + ("cap_min", "cap_delta", "rnd_min", "rnd_delta")


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cost_shifts(rho):
    """Per-teacher cost logit shifts at correlation rho (#305)."""
    shifts = np.full(K, rho * COST_SHIFT)
    shifts[0] += (1.0 - rho) * COST_SHIFT
    return shifts


def _consensus_targets(pi0, stack):
    """Union / MSCD-min / MSCD-delta targets over a stack of
    distributions -- #305's rules, applied to whatever stack
    (sources or cluster representatives) is passed in."""
    union = stack.mean(axis=0)
    mscd_min = stack.min(axis=0)
    mscd_min = mscd_min / mscd_min.sum()
    deltas = stack - pi0
    delta_rule = pi0.copy()
    all_pos = (deltas > 0).all(axis=0)
    all_neg = (deltas < 0).all(axis=0)
    delta_rule[all_pos] = pi0[all_pos] + deltas[:, all_pos].min(axis=0)
    delta_rule[all_neg] = pi0[all_neg] + deltas[:, all_neg].max(axis=0)
    delta_rule = np.clip(delta_rule, 1e-9, None)
    delta_rule /= delta_rule.sum()
    return {"union": union, "mscd_min": mscd_min, "mscd_delta": delta_rule}


def _estimated_shifts(pi0, teachers):
    """Each source's induced shift vs base on the probe set:
    log p_i - log pi0, centered per source so the softmax
    normalization constant (a per-source scalar) cannot
    masquerade as a direction difference."""
    shifts = np.log(teachers) - np.log(pi0)
    return shifts - shifts.mean(axis=1, keepdims=True)


def _cluster(shifts):
    """Best-merge agglomerative clustering, complete linkage:
    repeatedly merge the pair of clusters whose complete-
    linkage (maximum pairwise) distance is smallest, while
    that distance is below CLUSTER_DIST."""
    dist = {}
    for i in range(K):
        for j in range(i + 1, K):
            dist[(i, j)] = float(np.linalg.norm(shifts[i] - shifts[j]))

    def cdist(a, b):
        return max(dist[(min(i, j), max(i, j))] for i in a for j in b)

    clusters = [[i] for i in range(K)]
    while len(clusters) > 1:
        best, best_d = None, np.inf
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                d = cdist(clusters[x], clusters[y])
                if d < best_d:
                    best, best_d = (x, y), d
        if best is None or best_d >= CLUSTER_DIST:
            break
        x, y = best
        clusters[x] = sorted(clusters[x] + clusters[y])
        del clusters[y]
    return sorted(clusters)


def _random_partition(sizes, seed):
    """A random partition of the sources into clusters of the
    given sizes (the true partition's sizes, shuffled)."""
    rng = np.random.default_rng(RND_SEED_BASE + seed)
    perm = rng.permutation(K)
    out, idx = [], 0
    for size in sorted(sizes, reverse=True):
        out.append(sorted(perm[idx:idx + size].tolist()))
        idx += size
    return sorted(out)


def _representatives(teachers, partition):
    """One representative per cluster: the within-cluster mean
    distribution. The cluster votes once, at this distribution,
    however many sources it contains -- the weight cap."""
    return np.array([teachers[g].mean(axis=0) for g in partition])


def _all_targets(pi0, teachers, seed):
    """All eight arms' targets for one (seed, rho), plus the
    estimated partition the cap arms used."""
    plain = _consensus_targets(pi0, teachers)
    partition = _cluster(_estimated_shifts(pi0, teachers))
    cap = _consensus_targets(pi0, _representatives(teachers, partition))
    rnd_partition = _random_partition([len(g) for g in partition], seed)
    rnd = _consensus_targets(pi0, _representatives(teachers, rnd_partition))
    targets = {
        "single": teachers[0],
        "union": plain["union"],
        "mscd_min": plain["mscd_min"],
        "mscd_delta": plain["mscd_delta"],
        "cap_min": cap["mscd_min"],
        "cap_delta": cap["mscd_delta"],
        "rnd_min": rnd["mscd_min"],
        "rnd_delta": rnd["mscd_delta"],
    }
    return targets, partition


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


def run_seed_rho(seed, rho):
    """All eight arms for one (seed, rho), paired on one draw.

    Draw order is fixed -- base logits, per-teacher noise,
    then per-arm token samples in ARMS order -- so the four
    plain arms use the same base, teacher noise and token
    draws as #305's toy, and arms differ only through their
    target distribution. Clustering and the random partition
    consume no draws from this stream.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    base_logits = rng.standard_normal(V)
    base_logits[BENEFIT] = -1.0
    base_logits[COST] = -1.0
    pi0 = _softmax(base_logits)

    cshifts = cost_shifts(rho)
    teachers = []
    for i in range(K):
        logits = base_logits.copy()
        logits[BENEFIT] += BENEFIT_SHIFT
        logits[COST] += cshifts[i]
        logits[QUIRKS[i]] += QUIRK_SHIFT
        logits = logits + rng.normal(0, NOISE, V)
        teachers.append(_softmax(logits))
    teachers = np.array(teachers)

    targets, partition = _all_targets(pi0, teachers, seed)
    out = {"partition": partition}
    for arm in ARMS:
        student = _train_student(base_logits, targets[arm], rng)
        out[arm] = {
            "benefit": float(student[BENEFIT] - pi0[BENEFIT]),
            "cost": float(student[COST] - pi0[COST]),
        }
    return out


_CACHE = {}


def _seed_rho(seed, rho):
    key = (seed, rho)
    if key not in _CACHE:
        _CACHE[key] = run_seed_rho(seed, rho)
    return _CACHE[key]


def _aggregate(rho):
    agg = {arm: {"benefit": [], "cost": []} for arm in ARMS}
    for seed in SEEDS:
        res = _seed_rho(seed, rho)
        for arm in ARMS:
            for key in agg[arm]:
                agg[arm][key].append(res[arm][key])
    return {arm: {key: float(np.mean(vals))
                  for key, vals in d.items()}
            for arm, d in agg.items()}


def _veto(agg, arm):
    return 1.0 - agg[arm]["cost"] / agg["union"]["cost"]


def test_rho0_calibration_reproduces_305_veto():
    _empf_recipe()
    agg = _aggregate(0.0)
    # Calibration against #305's table (its aggregates:
    # single +0.0654, union +0.0107, mscd_min -0.0006,
    # mscd_delta +0.0004; veto 1.06 / 0.97).
    assert agg["single"]["cost"] >= 0.03, (
        f"rho=0 single cost {agg['single']['cost']:+.4f}")
    assert agg["union"]["cost"] >= 0.08 * agg["single"]["cost"], (
        f"rho=0 union cost {agg['union']['cost']:+.4f} vs "
        f"single {agg['single']['cost']:+.4f}")
    for arm in ("mscd_min", "mscd_delta"):
        assert agg[arm]["cost"] <= 0.002, (
            f"rho=0 {arm} cost {agg[arm]['cost']:+.4f}")
        assert agg[arm]["cost"] <= 0.30 * agg["union"]["cost"], (
            f"rho=0 {arm} cost {agg[arm]['cost']:+.4f} vs "
            f"union {agg['union']['cost']:+.4f}")
    assert agg["mscd_min"]["benefit"] >= 0.80 * agg["union"]["benefit"], (
        f"rho=0 mscd_min benefit {agg['mscd_min']['benefit']:+.4f} "
        f"vs union {agg['union']['benefit']:+.4f}")
    assert agg["mscd_delta"]["benefit"] >= 0.50 * agg["union"]["benefit"], (
        f"rho=0 mscd_delta benefit {agg['mscd_delta']['benefit']:+.4f} "
        f"vs union {agg['union']['benefit']:+.4f}")


def test_estimated_clusters_recover_pool_structure():
    _empf_recipe()
    # The estimator is deterministic given the teachers, so
    # these counts are exact for this toy's draws, not
    # statistics: at rho = 0 the idiosyncratic cost carrier
    # stands alone and the three cost-free sources cluster,
    # on every seed; at rho = 1 the sources are shift-
    # identical up to quirks and the partition degenerates
    # to a single cluster on a majority of seeds.
    solo = 0
    for seed in SEEDS:
        if _seed_rho(seed, 0.0)["partition"] == [[0], [1, 2, 3]]:
            solo += 1
    assert solo == len(SEEDS), (
        f"rho=0 partition {{0}}|{{1,2,3}} on {solo}/{len(SEEDS)} seeds")
    single_cluster = 0
    for seed in SEEDS:
        if len(_seed_rho(seed, 1.0)["partition"]) == 1:
            single_cluster += 1
    assert single_cluster >= 3, (
        f"rho=1 single cluster on {single_cluster}/{len(SEEDS)} seeds")


def test_rho0_capping_does_not_damage_veto():
    _empf_recipe()
    agg = _aggregate(0.0)
    # Claim (iii): with the veto intact, capping leaves it
    # intact -- capped cost stays at base level and within
    # a small absolute margin of plain consensus.
    for cap_arm, plain_arm in (("cap_min", "mscd_min"),
                               ("cap_delta", "mscd_delta")):
        assert agg[cap_arm]["cost"] <= 0.004, (
            f"rho=0 {cap_arm} cost {agg[cap_arm]['cost']:+.4f}")
        assert agg[cap_arm]["cost"] <= agg[plain_arm]["cost"] + 0.004, (
            f"rho=0 {cap_arm} cost {agg[cap_arm]['cost']:+.4f} vs "
            f"{plain_arm} {agg[plain_arm]['cost']:+.4f}")
        assert agg[cap_arm]["benefit"] >= 0.60 * agg["union"]["benefit"], (
            f"rho=0 {cap_arm} benefit {agg[cap_arm]['benefit']:+.4f} "
            f"vs union {agg['union']['benefit']:+.4f}")


def test_rho05_capping_recovers_no_veto():
    _empf_recipe()
    agg = _aggregate(0.5)
    # Claim (i) FAILS as measured: at rho = 0.5 the capped
    # arms' veto power does not rise above plain consensus
    # for either form -- it falls. A cluster representative
    # (a within-cluster mean) carries a cost at or above
    # the cluster's minimum-cost member, so collapsing a
    # cluster can only raise the floor the minimum and
    # delta rules read, never lower it. Assertions encode
    # the measured direction with slack, no more.
    for cap_arm, plain_arm in (("cap_min", "mscd_min"),
                               ("cap_delta", "mscd_delta")):
        assert _veto(agg, cap_arm) <= _veto(agg, plain_arm) + 0.05, (
            f"rho=0.5 {cap_arm} veto {_veto(agg, cap_arm):.3f} vs "
            f"{plain_arm} {_veto(agg, plain_arm):.3f}")
    # And the true cap does not beat the random cap: what
    # movement capping produces is not correlation repair.
    for cap_arm, rnd_arm in (("cap_min", "rnd_min"),
                             ("cap_delta", "rnd_delta")):
        assert agg[cap_arm]["cost"] >= agg[rnd_arm]["cost"] - 0.004, (
            f"rho=0.5 {cap_arm} cost {agg[cap_arm]['cost']:+.4f} vs "
            f"{rnd_arm} {agg[rnd_arm]['cost']:+.4f}")


def test_rho1_capping_recovers_nothing():
    _empf_recipe()
    agg = _aggregate(1.0)
    # Claim (ii) holds: at rho = 1 the capped arms retain
    # the cost at approximately union level, like plain
    # consensus -- the cap degenerates (single cluster on
    # a majority of seeds) and no veto reappears.
    for cap_arm in ("cap_min", "cap_delta"):
        assert agg[cap_arm]["cost"] >= 0.55 * agg["union"]["cost"], (
            f"rho=1 {cap_arm} cost {agg[cap_arm]['cost']:+.4f} vs "
            f"union {agg['union']['cost']:+.4f}")
        assert _veto(agg, cap_arm) <= 0.45, (
            f"rho=1 {cap_arm} veto {_veto(agg, cap_arm):.3f}")
    # The retained cost is treated like the retained
    # benefit, as in #305.
    for cap_arm in ("cap_min", "cap_delta"):
        assert agg[cap_arm]["cost"] >= 0.50 * agg[cap_arm]["benefit"], (
            f"rho=1 {cap_arm} cost {agg[cap_arm]['cost']:+.4f} vs own "
            f"benefit {agg[cap_arm]['benefit']:+.4f}")


def test_benefit_retained_at_all_rho():
    _empf_recipe()
    # Control: plain and capped arms keep retaining the
    # genuinely shared benefit at every rho.
    for rho in RHOS:
        agg = _aggregate(rho)
        assert agg["union"]["benefit"] >= 0.010, (
            f"rho={rho} union benefit {agg['union']['benefit']:+.4f}")
        for arm in ("mscd_min", "cap_min"):
            assert agg[arm]["benefit"] >= 0.70 * agg["union"]["benefit"], (
                f"rho={rho} {arm} benefit {agg[arm]['benefit']:+.4f} "
                f"vs union {agg['union']['benefit']:+.4f}")
        for arm in ("mscd_delta", "cap_delta"):
            assert agg[arm]["benefit"] >= 0.40 * agg["union"]["benefit"], (
                f"rho={rho} {arm} benefit {agg[arm]['benefit']:+.4f} "
                f"vs union {agg['union']['benefit']:+.4f}")
