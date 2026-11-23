"""CPU gate for the diversification repair probe: the
pool-level follow-up to the correlation boundary proven
in PR #305 and the aggregation-repair failure proven in
PR #309 for Narang, Tajdini, Zhang, Morgenstern,
"Defense via Consensus: Mitigating Subliminal Learning
through Multi-Teacher Distillation" (MSCD),
arXiv:2607.23394.

The anchor chain. PR #285 (cycle 18) proved that pooling
merely DILUTES a cost carried by one source (union cost
excess 0.16x the single-source excess), while cross-
source consensus VETOES it (MSCD-min aggregate cost
excess -0.0004, MSCD-delta +0.0006) at benefit retention
1.05x (min) / 0.68x (delta) of union. PR #305 (cycle 22)
proved the boundary: veto power 1 - (consensus / union
cost excess) collapses as the cross-source cost
correlation rho rises -- rho = 0: 1.06 (min) / 0.97
(delta); rho = 0.5: 0.45 / 0.68; rho = 1: -0.15 / 0.22.
PR #309 (cycle 23) proved the natural aggregation repair
-- capping correlated clusters in the consensus vote --
FAILS by construction: plain consensus already reads
the pool's minimum-cost member, and a cluster
representative is a within-cluster mean at or above
that floor, so no vote-side repair can recover veto.
#309's redirect: a repair must act on the POOL, not the
vote. Correlation has to be priced at curation, before
pooling.

Probe under test here: three pool-level moves, priced
in #305's EXACT toy at rho in {0.5, 1.0} (plus rho = 0
calibration):
  (a) DIVERSIFY: add 2 independent teachers (benefit
      +2.2, own quirk +1.5 on NEW quirk tokens 6 and 7,
      NO cost shift) -> consensus over K = 6.
  (b) DUPLICATE control: add 2 teachers carrying the
      benefit, their own quirks, and the SHARED cost
      component only (cost shift = rho * 2.2) ->
      consensus over K = 6. Same pool size, same benefit
      and quirk structure as (a); the only difference is
      the cost loading of the added teachers.
  (c) REMOVE control: drop teacher 0 (the full carrier)
      -> consensus over K = 3.

Decidable claims: (i) diversification restores veto at
rho = 0.5 AND rho = 1 substantially toward the rho = 0
level, at bounded benefit-retention cost; (ii)
duplicates do NOT restore veto (any movement is a
pool-size artifact, asserted as measured); (iii)
removal does NOT restore veto at rho = 0.5 beyond
dilution arithmetic -- its veto stays near plain's even
though its union cost falls; (iv) estimator pre-check:
the added independent teachers' centered shifts
(per-teacher-centered log p_i - log pi0, #309's
estimator) show cost-token loading at noise level,
while the duplicates' sit at approximately the shared
component -- the audit that distinguishes a repair from
a non-repair BEFORE running consensus. If any claim
fails, the assertions encode the measured negative.

Toy: PR #305's pool-composition toy unchanged for the
plain arms. Vocab V = 24, single categorical draw. Base
pi0 = softmax of random logits with benefit- and
cost-token logits pinned at -1.0. K = 4 teachers =
softmax(base logits + shifts + per-teacher Gaussian
logit noise sigma = 0.15): ALL teachers shift the
benefit token +2.2 (shared benefit); each teacher
shifts its own private quirk token +1.5; the cost shift
on teacher i is rho * 2.2 + (1 - rho) * 2.2 * [i == 0],
so rho = 0 reproduces #285/#305 exactly and teacher 0
is full-strength at all rho. Student = free logit
vector, SGD (lr 0.15, 400 steps) on N = 20000 tokens
sampled from each arm's target, starting from the base
logits. Readouts: excess over base of benefit- and
cost-token probabilities. Veto for a pool is
1 - (consensus cost excess / that pool's union cost
excess); benefit retention is consensus benefit excess
over that pool's union benefit excess.

Stream discipline (calibration integrity): the four
plain arms are FIRST in arm order and are drawn exactly
as #309 draws them -- base logits and the K = 4 teacher
noise come off the per-seed stream default_rng(SEED_BASE
+ seed), and the plain arms' token samples are that
stream's first four multinomial draws. The added
teachers' noise comes off INDEPENDENT streams
(default_rng(DIV_SEED_BASE + seed) for the diversify
teachers, default_rng(DUP_SEED_BASE + seed) for the
duplicates), drawn before any token sampling but never
touching the plain stream, so the plain arms' numbers
are bit-identical to #305/#309. The new arms' token
samples continue the plain stream after the plain arms,
in fixed arm order. Arms at each rho, paired per seed:
single, union, mscd_min, mscd_delta (plain, K = 4),
div_union, div_min, div_delta (K = 6, independent
adds), dup_union, dup_min, dup_delta (K = 6, shared-
cost adds), rm0_union, rm0_min, rm0_delta (K = 3,
teacher 0 dropped).

Measured on pop-os (particlegan 0.10.1), 5 seeds --
aggregate cost excess (pool union / min / delta) and
veto (min / delta, vs the pool's own union):
  rho = 0.0: plain +0.0107 / -0.0006 / +0.0004, veto
             1.06 / 0.97; div +0.0073 / -0.0006 /
             -0.0002, veto 1.08 / 1.03; dup +0.0069 /
             -0.0005 / +0.0003, veto 1.07 / 0.96 (at
             rho = 0 the two add kinds coincide: the
             shared component is 0; rm0's union cost
             is ~0, so its veto is not meaningful here)
  rho = 0.5: plain +0.0217 / +0.0119 / +0.0069, veto
             0.45 / 0.68; div +0.0140 / +0.0012 /
             +0.0002, veto 0.92 / 0.99; dup +0.0187 /
             +0.0127 / +0.0068, veto 0.32 / 0.64;
             rm0 +0.0091 / +0.0116 / +0.0073, veto
             -0.27 / 0.19; div benefit retention
             1.06x / 0.63x of div union
  rho = 1.0: plain +0.0584 / +0.0671 / +0.0455, veto
             -0.15 / 0.22; div +0.0363 / +0.0016 /
             +0.0001, veto 0.96 / 1.00; dup +0.0623 /
             +0.0699 / +0.0431, veto -0.12 / 0.31;
             rm0 +0.0558 / +0.0665 / +0.0472, veto
             -0.19 / 0.16; div benefit retention
             1.08x / 0.62x of div union
Estimator pre-check (mean centered cost-token loading
of the two added teachers, aggregate over seeds): the
independent adds load at -0.09 (mean |loading| 0.13)
at EVERY rho -- noise level; the duplicates load at
-0.10 at rho = 0 (the shared component is 0 there),
+0.95 at rho = 0.5 and +2.01 at rho = 1 -- at the
shared component, as constructed. Calibration: the
plain arms reproduce #305's table exactly at every
rho (single +0.0654 and union +0.0107 at rho = 0;
veto 0.45 / 0.68 at rho = 0.5; -0.15 / 0.22 at
rho = 1) -- bit-identical, not within noise.

The arbitration, stated plainly: ALL FOUR claims
HOLD. (i) Diversification restores the veto at both
probe rhos, substantially to the rho = 0 level:
0.45 -> 0.92 (min) and 0.68 -> 0.99 (delta) at
rho = 0.5; -0.15 -> 0.96 and 0.22 -> 1.00 at
rho = 1, with the diversified consensus cost back at
base level (+0.0012 / +0.0002 at rho = 0.5, +0.0016 /
+0.0001 at rho = 1) and benefit retention 1.06x /
1.08x (min -- no retention cost) and 0.63x / 0.62x
(delta -- about what plain delta pays at rho = 0).
The restoration does not need the original sources to
disagree at all: at rho = 1 they are unanimous on the
cost, and the two added sources ARE the disagreement
consensus prices. (ii) Duplicates restore nothing:
same pool size, veto within +0.15 of plain on every
form at both rhos (largest movement +0.09), and the
diversified pool beats them by 0.35..1.08 veto --
pool size is not the mechanism, cost-free membership
is. (iii) Removal restores nothing either: at
rho = 0.5 the union cost falls to 0.42x of plain but
the consensus cost does not fall with it (+0.0116 vs
plain +0.0119 min), because every remaining source
still carries the shared component; at rho = 1 the
reduced pool is structurally identical to plain.
Subtracting the loudest carrier is dilution
arithmetic, not repair. (iv) The estimator pre-check
separates the repair from both non-repairs BEFORE
consensus runs: an added source whose centered shift
loads on the cost coordinate at ~the shared component
cannot restore veto, whatever it does to pool size;
one that loads at noise level can. Together with
#309, the chain's picture is complete: the vote
cannot be repaired (plain consensus already extracts
all the disagreement the pool contains), but the
POOL can -- curation that supplies genuinely
independent sources restores exactly the veto that
correlation removed.
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
K = 4                # plain-pool teachers
BENEFIT = 0          # token every teacher boosts
COST = 1             # token the cost load lands on
QUIRKS = (2, 3, 4, 5)  # teacher i's private quirk token
ADDED_QUIRKS = (6, 7)  # added teachers' private quirk tokens
BENEFIT_SHIFT = 2.2  # shared benefit logit shift
COST_SHIFT = 2.2     # full cost logit shift (teacher 0, all rho)
QUIRK_SHIFT = 1.5    # private quirk logit shift
NOISE = 0.15         # per-teacher Gaussian logit noise
N_TOKENS = 20000     # sampled tokens per arm
STEPS = 400          # student SGD steps
LR = 0.15            # student SGD learning rate
SEED_BASE = 18000    # RNG stream base (same as #285/#305)
DIV_SEED_BASE = 990000   # independent stream: diversify adds
DUP_SEED_BASE = 995000   # independent stream: duplicate adds
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.0, 0.5, 1.0)
PLAIN_ARMS = ("single", "union", "mscd_min", "mscd_delta")
ARMS = PLAIN_ARMS + (
    "div_union", "div_min", "div_delta",
    "dup_union", "dup_min", "dup_delta",
    "rm0_union", "rm0_min", "rm0_delta",
)
# Each non-plain pool's union arm (its veto / retention base).
POOL_UNION = {"div": "div_union", "dup": "dup_union", "rm0": "rm0_union"}


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
    (the plain pool or an extended / reduced pool) is
    passed in."""
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
    masquerade as a direction difference (#309's estimator)."""
    shifts = np.log(teachers) - np.log(pi0)
    return shifts - shifts.mean(axis=1, keepdims=True)


def _added_teachers(base_logits, rho, kind, seed):
    """The two added teachers for one pool move. kind =
    "div": no cost shift (independent sources). kind =
    "dup": the shared cost component only (rho * 2.2).
    Both kinds carry the shared benefit and their own
    private quirks on tokens 6 and 7. Noise comes off the
    kind's own stream, never the plain stream."""
    if kind == "div":
        rng = np.random.default_rng(DIV_SEED_BASE + seed)
        added_cost = 0.0
    else:
        rng = np.random.default_rng(DUP_SEED_BASE + seed)
        added_cost = rho * COST_SHIFT
    out = []
    for quirk in ADDED_QUIRKS:
        logits = base_logits.copy()
        logits[BENEFIT] += BENEFIT_SHIFT
        logits[COST] += added_cost
        logits[quirk] += QUIRK_SHIFT
        logits = logits + rng.normal(0, NOISE, V)
        out.append(_softmax(logits))
    return np.array(out)


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
    """All thirteen arms for one (seed, rho), paired on one
    draw, plus the estimator pre-check loadings.

    Draw order is fixed -- base logits, per-teacher noise
    (plain stream), added-teacher noise (independent
    streams, no plain-stream draws), then per-arm token
    samples in ARMS order off the plain stream -- so the
    four plain arms use the same base, teacher noise and
    token draws as #305's toy.
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

    div_added = _added_teachers(base_logits, rho, "div", seed)
    dup_added = _added_teachers(base_logits, rho, "dup", seed)
    div_stack = np.vstack([teachers, div_added])
    dup_stack = np.vstack([teachers, dup_added])
    rm0_stack = teachers[1:]

    plain = _consensus_targets(pi0, teachers)
    div = _consensus_targets(pi0, div_stack)
    dup = _consensus_targets(pi0, dup_stack)
    rm0 = _consensus_targets(pi0, rm0_stack)
    targets = {
        "single": teachers[0],
        "union": plain["union"],
        "mscd_min": plain["mscd_min"],
        "mscd_delta": plain["mscd_delta"],
        "div_union": div["union"],
        "div_min": div["mscd_min"],
        "div_delta": div["mscd_delta"],
        "dup_union": dup["union"],
        "dup_min": dup["mscd_min"],
        "dup_delta": dup["mscd_delta"],
        "rm0_union": rm0["union"],
        "rm0_min": rm0["mscd_min"],
        "rm0_delta": rm0["mscd_delta"],
    }

    # Estimator pre-check: centered cost-token loading of
    # each added teacher (#309's estimator, cost coordinate).
    div_load = _estimated_shifts(pi0, div_added)[:, COST]
    dup_load = _estimated_shifts(pi0, dup_added)[:, COST]

    out = {
        "div_loadings": [float(x) for x in div_load],
        "dup_loadings": [float(x) for x in dup_load],
    }
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
    loads = {"div": [], "dup": []}
    for seed in SEEDS:
        res = _seed_rho(seed, rho)
        for arm in ARMS:
            for key in agg[arm]:
                agg[arm][key].append(res[arm][key])
        loads["div"].extend(res["div_loadings"])
        loads["dup"].extend(res["dup_loadings"])
    out = {arm: {key: float(np.mean(vals))
                 for key, vals in d.items()}
           for arm, d in agg.items()}
    out["div_loading"] = float(np.mean(loads["div"]))
    out["dup_loading"] = float(np.mean(loads["dup"]))
    out["div_loading_abs"] = float(np.mean(np.abs(loads["div"])))
    return out


def _veto(agg, arm, union_arm="union"):
    return 1.0 - agg[arm]["cost"] / agg[union_arm]["cost"]


def _pool_veto(agg, pool, form):
    """Veto of a pool's consensus form vs that pool's union."""
    return _veto(agg, f"{pool}_{form}", POOL_UNION[pool])


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


def test_estimator_precheck_distinguishes_repair():
    _empf_recipe()
    # Claim (iv): BEFORE any consensus is run, the centered
    # cost-token loading separates the two kinds of added
    # teacher. The independent adds load at noise level
    # (their raw cost shift is 0; centering leaves only the
    # -mean offset, ~-0.15, plus sigma = 0.15 noise), at
    # every rho. The duplicates load at approximately the
    # shared component: +0.95 at rho = 0.5, +2.01 at
    # rho = 1 (raw shift minus the centering offset).
    for rho in RHOS:
        agg = _aggregate(rho)
        assert abs(agg["div_loading"]) <= 0.30, (
            f"rho={rho} div loading {agg['div_loading']:+.3f}")
        assert agg["div_loading_abs"] <= 0.30, (
            f"rho={rho} div |loading| {agg['div_loading_abs']:.3f}")
    agg05 = _aggregate(0.5)
    assert agg05["dup_loading"] >= 0.75, (
        f"rho=0.5 dup loading {agg05['dup_loading']:+.3f}")
    agg1 = _aggregate(1.0)
    assert agg1["dup_loading"] >= 1.70, (
        f"rho=1 dup loading {agg1['dup_loading']:+.3f}")
    # The separation is wide at both probe rhos: the audit
    # calls the duplicates cost-carriers and the adds clean
    # by a margin of ~0.85 (rho = 0.5) to ~1.9 (rho = 1).
    assert agg05["dup_loading"] - agg05["div_loading"] >= 0.70, (
        f"rho=0.5 loading gap {agg05['dup_loading']:+.3f} vs "
        f"{agg05['div_loading']:+.3f}")
    assert agg1["dup_loading"] - agg1["div_loading"] >= 1.50, (
        f"rho=1 loading gap {agg1['dup_loading']:+.3f} vs "
        f"{agg1['div_loading']:+.3f}")
    # At rho = 0 the shared component is 0, so the two
    # kinds are estimator-identical: the audit's own
    # calibration.
    agg0 = _aggregate(0.0)
    assert abs(agg0["dup_loading"]) <= 0.35, (
        f"rho=0 dup loading {agg0['dup_loading']:+.3f}")


def test_diversification_restores_veto_rho05():
    _empf_recipe()
    agg = _aggregate(0.5)
    # Claim (i) at rho = 0.5, as measured: the diversified
    # pool's veto rises from plain 0.45 / 0.68 to 0.92 /
    # 0.99 -- most of the way back to the rho = 0 level
    # (1.06 / 0.97) -- and its consensus cost falls to
    # base level (+0.0012 min, +0.0002 delta).
    assert agg["div_min"]["cost"] <= 0.004, (
        f"rho=0.5 div_min cost {agg['div_min']['cost']:+.4f}")
    assert agg["div_delta"]["cost"] <= 0.004, (
        f"rho=0.5 div_delta cost {agg['div_delta']['cost']:+.4f}")
    assert _pool_veto(agg, "div", "min") >= 0.80, (
        f"rho=0.5 div min veto {_pool_veto(agg, 'div', 'min'):.3f}")
    assert _pool_veto(agg, "div", "delta") >= 0.85, (
        f"rho=0.5 div delta veto {_pool_veto(agg, 'div', 'delta'):.3f}")
    for form, plain_arm in (("min", "mscd_min"), ("delta", "mscd_delta")):
        assert _pool_veto(agg, "div", form) >= _veto(agg, plain_arm) + 0.25, (
            f"rho=0.5 div {form} veto {_pool_veto(agg, 'div', form):.3f} "
            f"vs plain {_veto(agg, plain_arm):.3f}")


def test_diversification_restores_veto_rho1():
    _empf_recipe()
    agg = _aggregate(1.0)
    # Claim (i) at rho = 1, as measured: even with the
    # original pool's cost FULLY correlated, two cost-free
    # sources restore the veto to 0.96 / 1.00 (plain:
    # -0.15 / 0.22) and the consensus cost to base level
    # (+0.0016 min, +0.0001 delta). The restoration does
    # not depend on the original sources disagreeing at
    # all -- the added sources ARE the disagreement.
    assert agg["div_min"]["cost"] <= 0.004, (
        f"rho=1 div_min cost {agg['div_min']['cost']:+.4f}")
    assert agg["div_delta"]["cost"] <= 0.004, (
        f"rho=1 div_delta cost {agg['div_delta']['cost']:+.4f}")
    assert _pool_veto(agg, "div", "min") >= 0.85, (
        f"rho=1 div min veto {_pool_veto(agg, 'div', 'min'):.3f}")
    assert _pool_veto(agg, "div", "delta") >= 0.90, (
        f"rho=1 div delta veto {_pool_veto(agg, 'div', 'delta'):.3f}")
    for form, plain_arm in (("min", "mscd_min"), ("delta", "mscd_delta")):
        assert _pool_veto(agg, "div", form) >= _veto(agg, plain_arm) + 0.60, (
            f"rho=1 div {form} veto {_pool_veto(agg, 'div', form):.3f} "
            f"vs plain {_veto(agg, plain_arm):.3f}")


def test_duplicates_do_not_restore_veto():
    _empf_recipe()
    # Claim (ii), as measured: at the same pool size
    # (K = 6), the duplicates' veto stays within +0.15 of
    # plain at both probe rhos on both forms -- the
    # largest movement is +0.09 (delta at rho = 1); the
    # min form at rho = 0.5 moves DOWN (0.32 vs plain
    # 0.45). And the diversified pool beats the duplicate
    # pool by >= 0.30 veto on both forms at both rhos
    # (measured gaps 0.35..1.08): pool SIZE restores
    # nothing; cost-free membership does.
    for rho in (0.5, 1.0):
        agg = _aggregate(rho)
        for form, plain_arm in (("min", "mscd_min"),
                                ("delta", "mscd_delta")):
            assert _pool_veto(agg, "dup", form) <= (
                _veto(agg, plain_arm) + 0.15), (
                f"rho={rho} dup {form} veto "
                f"{_pool_veto(agg, 'dup', form):.3f} vs plain "
                f"{_veto(agg, plain_arm):.3f}")
            assert _pool_veto(agg, "div", form) >= (
                _pool_veto(agg, "dup", form) + 0.30), (
                f"rho={rho} div {form} veto "
                f"{_pool_veto(agg, 'div', form):.3f} vs dup "
                f"{_pool_veto(agg, 'dup', form):.3f}")


def test_removal_does_not_restore_veto_rho05():
    _empf_recipe()
    # Claim (iii), as measured: dropping the full carrier
    # cuts the union cost to 0.42x of plain (+0.0091 vs
    # +0.0217) -- pure dilution arithmetic in reverse --
    # but the consensus cost does NOT fall with it
    # (rm0 min +0.0116 vs plain +0.0119; delta +0.0073 vs
    # +0.0069): every remaining source still carries the
    # shared component rho * 2.2, so there is still
    # nothing to veto WITH. The veto, measured against
    # the reduced pool's own union, falls to -0.27 / 0.19
    # (plain 0.45 / 0.68).
    agg = _aggregate(0.5)
    assert agg["rm0_union"]["cost"] <= 0.60 * agg["union"]["cost"], (
        f"rho=0.5 rm0 union cost {agg['rm0_union']['cost']:+.4f} vs "
        f"plain union {agg['union']['cost']:+.4f}")
    assert agg["rm0_min"]["cost"] >= 0.75 * agg["mscd_min"]["cost"], (
        f"rho=0.5 rm0_min cost {agg['rm0_min']['cost']:+.4f} vs "
        f"plain {agg['mscd_min']['cost']:+.4f}")
    assert agg["rm0_delta"]["cost"] >= 0.75 * agg["mscd_delta"]["cost"], (
        f"rho=0.5 rm0_delta cost {agg['rm0_delta']['cost']:+.4f} vs "
        f"plain {agg['mscd_delta']['cost']:+.4f}")
    for form, plain_arm in (("min", "mscd_min"), ("delta", "mscd_delta")):
        assert _pool_veto(agg, "rm0", form) <= _veto(agg, plain_arm) + 0.05, (
            f"rho=0.5 rm0 {form} veto {_pool_veto(agg, 'rm0', form):.3f} "
            f"vs plain {_veto(agg, plain_arm):.3f}")


def test_removal_does_not_restore_veto_rho1():
    _empf_recipe()
    # Claim (iii) at rho = 1: all four sources carry the
    # full cost, so dropping teacher 0 changes nothing
    # structurally -- veto -0.19 / 0.16 vs plain -0.15 /
    # 0.22, and the reduced pool's consensus retains its
    # union's cost.
    agg = _aggregate(1.0)
    for form, plain_arm in (("min", "mscd_min"), ("delta", "mscd_delta")):
        assert _pool_veto(agg, "rm0", form) <= _veto(agg, plain_arm) + 0.10, (
            f"rho=1 rm0 {form} veto {_pool_veto(agg, 'rm0', form):.3f} "
            f"vs plain {_veto(agg, plain_arm):.3f}")
    assert agg["rm0_min"]["cost"] >= 0.55 * agg["rm0_union"]["cost"], (
        f"rho=1 rm0_min cost {agg['rm0_min']['cost']:+.4f} vs "
        f"rm0 union {agg['rm0_union']['cost']:+.4f}")


def test_benefit_retention_bounded():
    _empf_recipe()
    # The demonstrated retention floor for the repair:
    # the diversified pool's consensus retains the shared
    # benefit at 1.06x / 1.08x (min) and 0.63x / 0.62x
    # (delta) of its own union at rho = 0.5 / 1.0 -- the
    # min form pays NO retention cost at all, and the
    # delta form pays about what plain delta pays at
    # rho = 0 (0.68x, #285). Assertions sit below the
    # measured floor.
    for rho in (0.5, 1.0):
        agg = _aggregate(rho)
        assert agg["div_min"]["benefit"] >= 0.90 * agg["div_union"]["benefit"], (
            f"rho={rho} div_min benefit {agg['div_min']['benefit']:+.4f} "
            f"vs div union {agg['div_union']['benefit']:+.4f}")
        assert agg["div_delta"]["benefit"] >= 0.50 * agg["div_union"]["benefit"], (
            f"rho={rho} div_delta benefit {agg['div_delta']['benefit']:+.4f} "
            f"vs div union {agg['div_union']['benefit']:+.4f}")
        # Control: plain arms keep retaining the benefit.
        assert agg["mscd_min"]["benefit"] >= 0.70 * agg["union"]["benefit"], (
            f"rho={rho} mscd_min benefit {agg['mscd_min']['benefit']:+.4f} "
            f"vs union {agg['union']['benefit']:+.4f}")
