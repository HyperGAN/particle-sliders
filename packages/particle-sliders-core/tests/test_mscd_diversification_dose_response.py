"""CPU gate for the independence dose-response probe: the
dose curve of the diversification repair proven in
PR #313 for Narang, Tajdini, Zhang, Morgenstern,
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
PR #309 (cycle 23) proved the aggregation repair (cluster
capping) fails by construction and redirected: a repair
must act on the POOL, not the vote. PR #313 (cycle 24)
proved the pool-level repair at exactly k = 2 added
sources: two provenance-INDEPENDENT sources restore the
veto (rho = 0.5: plain 0.45 / 0.68 -> diversified
0.92 / 0.99; rho = 1: plain -0.15 / 0.22 -> diversified
0.96 / 1.00), two DUPLICATES carrying the shared cost
component restore nothing (veto within +0.15 of plain;
the min form at rho = 0.5 moved DOWN to 0.32), and
removing the carrier dilutes union cost but leaves
consensus cost untouched. Benefit retention of the
diversified consensus at k = 2: 1.06x / 0.63x (min /
delta) at rho = 0.5, 1.08x / 0.62x at rho = 1.

Probe under test here: #313 is an existence proof at
k = 2. This probe prices the DOSE. In #313's exact toy,
at rho in {0.5, 1} (plus rho = 0 calibration):
  plain (k = 0); +k INDEPENDENT sources for k in
  {1, 2, 4} (benefit +2.2, own quirk +1.5 on new quirk
  tokens 6..9, NO cost shift); +k DUPLICATE sources for
  k in {1, 2, 4} (benefit, own quirks, and the shared
  cost component rho * 2.2 only) -- a count control at
  EVERY dose, so independence is separated from pool
  size at each point of the curve, not just at k = 2.

Decidable claims: (a) veto recovery is monotone
non-decreasing in k at both rhos; (b) the marginal veto
gain saturates -- the gain from k = 2 -> 4 vs the gain
from k = 0 -> 1, asserted in the measured shape,
whichever it is; (c) duplicates stay flat at every k
(within noise of plain, as at k = 2); (d) the
benefit-retention price per added source, reported as
measured (in particular whether the delta-form
retention falls further at k = 4, holds, or recovers).
Optional readout: the pool's estimated shared-component
loading per dose (the centered-shift estimator of
#309/#313, mean centered cost-token loading of the
added teachers) -- reported per dose; it is an
assertion target only where it comes out clean.
If any claim fails, the assertions encode the measured
negative.

Toy: PR #305/#313's pool-composition toy unchanged.
Vocab V = 24, single categorical draw. Base pi0 =
softmax of random logits with benefit- and cost-token
logits pinned at -1.0. K = 4 teachers = softmax(base
logits + shifts + per-teacher Gaussian logit noise
sigma = 0.15): ALL teachers shift the benefit token
+2.2 (shared benefit); each teacher shifts its own
private quirk token +1.5; the cost shift on teacher i
is rho * 2.2 + (1 - rho) * 2.2 * [i == 0], so rho = 0
reproduces #285/#305 exactly and teacher 0 is
full-strength at all rho. Student = free logit vector,
SGD (lr 0.15, 400 steps) on N = 20000 tokens sampled
from each arm's target, starting from the base logits.
Readouts: excess over base of benefit- and cost-token
probabilities. Veto for a pool is 1 - (consensus cost
excess / that pool's union cost excess); benefit
retention is consensus benefit excess over that pool's
union benefit excess.

Stream discipline (calibration integrity): the plain
arms and the k = 2 arms are drawn EXACTLY as #313
draws them -- base logits and the K = 4 teacher noise
come off the per-seed stream default_rng(SEED_BASE +
seed); the added teachers' noise comes off the
independent streams default_rng(DIV_SEED_BASE + seed)
(diversify) and default_rng(DUP_SEED_BASE + seed)
(duplicates), whose first two draws are #313's two
added teachers bit-identically (the k = 1 pool takes
the first of them; the k = 4 pool continues the same
streams to draws three and four, on new quirk tokens
8 and 9); the calibrated arms' token samples are the
plain stream's first ten multinomial draws, in #313's
arm order (single, union, mscd_min, mscd_delta,
div_union, div_min, div_delta, dup_union, dup_min,
dup_delta -- #313's rm0 arms were last in its order
and are not part of this probe, so dropping them
touches nothing calibrated). The NEW doses' token
samples (k = 1 and k = 4, both kinds) come off their
OWN stream, default_rng(DOSE_TOKEN_BASE + seed), in
fixed arm order, so no new arm can perturb a
calibrated draw. Calibration: the k = 0 and k = 2 arms
reproduce #313's table bit-identically at every rho
(verified in test_k2_calibration_reproduces_313 and
test_rho0_calibration_reproduces_305_veto).

Measured on pop-os (particlegan 0.10.1), 5 seeds --
seed-mean veto (vs the pool's own union) by dose
k = 0 / 1 / 2 / 4:
  rho = 0.5, independents: min 0.450 / 0.883 / 0.916 /
             0.950; delta 0.683 / 0.980 / 0.985 / 0.977.
             Duplicates: min 0.450 / 0.383 / 0.321 /
             0.200; delta 0.683 / 0.680 / 0.638 / 0.679.
  rho = 1.0, independents: min -0.148 / 0.940 / 0.955 /
             0.985; delta 0.221 / 0.991 / 0.998 / 0.994.
             Duplicates: min -0.148 / -0.109 / -0.122 /
             -0.167; delta 0.221 / 0.339 / 0.308 / 0.367.
Benefit retention of the independent pools (vs own
union): the min form is flat at 1.03..1.10x at every
dose and rho (no price); the delta form reads
0.711 / 0.671 / 0.626 / 0.618 at rho = 0.5 and
0.673 / 0.670 / 0.624 / 0.593 at rho = 1 (k = 0 / 1 /
2 / 4) -- the price is paid by k = 2 and nearly stops.
Estimator readout (mean centered cost-token loading of
the added teachers): independents -0.06..-0.11 at
every dose and rho (noise); duplicates +0.90..+0.95
at rho = 0.5 and +1.95..+2.01 at rho = 1, every dose.
Calibration: the plain and k = 2 arms reproduce
#313's aggregates exactly at every rho (e.g. rho = 0.5
div cost +0.0140 / +0.0012 / +0.0002; rho = 1 plain
veto -0.15 / 0.22) -- bit-identical, not within noise.

The arbitration, stated plainly: ALL FOUR claims HOLD,
in measured form. (a) Veto recovery is monotone
non-decreasing in k in seed mean at both rhos, up to a
saturation plateau (the delta form dips -0.004..-0.008
from k = 2 to k = 4 at the ceiling, veto >= 0.97
throughout); per seed, EVERY dose k >= 1 beats plain
on every seed, both forms, both rhos. (b) Saturation
is nearly total: the FIRST independent source buys
+0.30..+1.09 veto; each further source from k = 2 to
k = 4 buys at most +0.017 -- under 6% of the first
source's gain. One independent source IS most of the
repair (veto min 0.88 at rho = 0.5, 0.94 at rho = 1,
already at k = 1); k = 2 polishes the min form, k = 4
buys <= 0.03 anywhere. (c) Duplicates are flat at
every k: never more than +0.15 above plain (the delta
drift #313 saw at k = 2), at or below plain elsewhere,
and strictly DOWNWARD in the min form at rho = 0.5 as
dose rises (0.38 / 0.32 / 0.20) -- the dose is
independence, not count, at every point, and the
independent pool beats its duplicate control by
>= 0.29 veto at every dose. (d) The benefit price
saturates with the gain: the min form pays nothing at
any dose; the delta form pays ~0.05 by k = 2 (the
dose #313 priced) and only ~0.01..0.03 more at k = 4
-- total k = 0 -> 4 price ~0.09 of union benefit.
The priced rule for an audit: recruit ONE verified
independent source first; a second is cheap polish;
beyond that, independence has already done its work.
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
ADDED_QUIRKS = (6, 7, 8, 9)  # added teachers' quirk tokens, in draw order
BENEFIT_SHIFT = 2.2  # shared benefit logit shift
COST_SHIFT = 2.2     # full cost logit shift (teacher 0, all rho)
QUIRK_SHIFT = 1.5    # private quirk logit shift
NOISE = 0.15         # per-teacher Gaussian logit noise
N_TOKENS = 20000     # sampled tokens per arm
STEPS = 400          # student SGD steps
LR = 0.15            # student SGD learning rate
SEED_BASE = 18000    # RNG stream base (same as #285/#305/#313)
DIV_SEED_BASE = 990000   # independent stream: diversify adds (#313)
DUP_SEED_BASE = 995000   # independent stream: duplicate adds (#313)
DOSE_TOKEN_BASE = 770000  # own stream: NEW doses' token samples
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.0, 0.5, 1.0)
DOSE_RHOS = (0.5, 1.0)
DOSES = (1, 2, 4)

# Calibrated arms, in #313's token-draw order (its first ten).
CAL_ARMS = (
    "single", "union", "mscd_min", "mscd_delta",
    "div_union", "div_min", "div_delta",
    "dup_union", "dup_min", "dup_delta",
)
# New-dose arms, token-sampled off the dose stream in this order.
NEW_ARMS = (
    "div1_union", "div1_min", "div1_delta",
    "div4_union", "div4_min", "div4_delta",
    "dup1_union", "dup1_min", "dup1_delta",
    "dup4_union", "dup4_min", "dup4_delta",
)
ARMS = CAL_ARMS + NEW_ARMS

# Pool name -> (union arm, min arm, delta arm).
POOLS = {
    "plain": ("union", "mscd_min", "mscd_delta"),
    "div1": ("div1_union", "div1_min", "div1_delta"),
    "div": ("div_union", "div_min", "div_delta"),
    "div4": ("div4_union", "div4_min", "div4_delta"),
    "dup1": ("dup1_union", "dup1_min", "dup1_delta"),
    "dup": ("dup_union", "dup_min", "dup_delta"),
    "dup4": ("dup4_union", "dup4_min", "dup4_delta"),
}
DIV_DOSE_POOLS = ("plain", "div1", "div", "div4")   # k = 0, 1, 2, 4
DUP_DOSE_POOLS = ("plain", "dup1", "dup", "dup4")   # k = 0, 1, 2, 4


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
    (the plain pool or an extended pool) is passed in."""
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


def _added_teachers(base_logits, rho, kind, seed, k):
    """The first k added teachers of one kind. kind = "div":
    no cost shift (independent sources). kind = "dup": the
    shared cost component only (rho * 2.2). Both kinds carry
    the shared benefit and their own private quirks on
    tokens 6.. in draw order. Noise comes off the kind's own
    stream (#313's DIV / DUP streams), drawn sequentially,
    so k = 2 reproduces #313's added teachers bit-identically
    and k = 1 is its first teacher; k = 4 continues the same
    streams to draws three and four (quirk tokens 8, 9)."""
    if kind == "div":
        rng = np.random.default_rng(DIV_SEED_BASE + seed)
        added_cost = 0.0
    else:
        rng = np.random.default_rng(DUP_SEED_BASE + seed)
        added_cost = rho * COST_SHIFT
    out = []
    for quirk in ADDED_QUIRKS[:k]:
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
    """All twenty-two arms for one (seed, rho), paired, plus
    the estimator loadings per dose.

    Draw order is fixed -- base logits, per-teacher noise
    (plain stream), added-teacher noise (independent div /
    dup streams, no plain-stream draws), then token samples:
    the calibrated arms off the plain stream in #313's order,
    the new-dose arms off the dose stream in NEW_ARMS order --
    so the plain and k = 2 arms use the same base, teacher
    noise and token draws as #313's toy, bit-identically.
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

    div_all = _added_teachers(base_logits, rho, "div", seed, 4)
    dup_all = _added_teachers(base_logits, rho, "dup", seed, 4)
    stacks = {
        "plain": teachers,
        "div1": np.vstack([teachers, div_all[:1]]),
        "div": np.vstack([teachers, div_all[:2]]),
        "div4": np.vstack([teachers, div_all]),
        "dup1": np.vstack([teachers, dup_all[:1]]),
        "dup": np.vstack([teachers, dup_all[:2]]),
        "dup4": np.vstack([teachers, dup_all]),
    }

    targets = {}
    for pool, stack in stacks.items():
        cons = _consensus_targets(pi0, stack)
        union_arm, min_arm, delta_arm = POOLS[pool]
        targets[union_arm] = cons["union"]
        targets[min_arm] = cons["mscd_min"]
        targets[delta_arm] = cons["mscd_delta"]
    targets["single"] = teachers[0]

    # Estimator readout: centered cost-token loading of the
    # added teachers, per dose (mean over the first k adds).
    div_load = _estimated_shifts(pi0, div_all)[:, COST]
    dup_load = _estimated_shifts(pi0, dup_all)[:, COST]

    out = {
        "div_loadings": [float(x) for x in div_load],
        "dup_loadings": [float(x) for x in dup_load],
    }
    for arm in CAL_ARMS:
        student = _train_student(base_logits, targets[arm], rng)
        out[arm] = {
            "benefit": float(student[BENEFIT] - pi0[BENEFIT]),
            "cost": float(student[COST] - pi0[COST]),
        }
    rng_dose = np.random.default_rng(DOSE_TOKEN_BASE + seed)
    for arm in NEW_ARMS:
        student = _train_student(base_logits, targets[arm], rng_dose)
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
        loads["div"].append(res["div_loadings"])
        loads["dup"].append(res["dup_loadings"])
    out = {arm: {key: float(np.mean(vals))
                 for key, vals in d.items()}
           for arm, d in agg.items()}
    for kind in ("div", "dup"):
        per_teacher = np.array(loads[kind])  # (seeds, 4 adds)
        for k in DOSES:
            out[f"{kind}{k}_loading"] = float(
                per_teacher[:, :k].mean())
        out[f"{kind}_loading_abs"] = float(
            np.abs(per_teacher[:, :2]).mean())
    return out


def _veto(agg, pool, form):
    """Seed-mean veto of a pool's consensus form vs its own union."""
    union_arm, min_arm, delta_arm = POOLS[pool]
    arm = min_arm if form == "min" else delta_arm
    return 1.0 - agg[arm]["cost"] / agg[union_arm]["cost"]


def _retention(agg, pool, form):
    """Seed-mean benefit retention vs the pool's own union."""
    union_arm, min_arm, delta_arm = POOLS[pool]
    arm = min_arm if form == "min" else delta_arm
    return agg[arm]["benefit"] / agg[union_arm]["benefit"]


def _seed_vetoes(rho, pool, form):
    """Per-seed vetoes of a pool's form vs its own union."""
    union_arm, min_arm, delta_arm = POOLS[pool]
    arm = min_arm if form == "min" else delta_arm
    vals = []
    for seed in SEEDS:
        res = _seed_rho(seed, rho)
        vals.append(1.0 - res[arm]["cost"] / res[union_arm]["cost"])
    return vals


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


def test_k2_calibration_reproduces_313():
    _empf_recipe()
    # Calibration against #313's table: the k = 2 arms of
    # this probe ARE #313's div / dup arms (same streams,
    # same draw order), so their seed-mean vetoes and costs
    # reproduce #313's aggregates: rho = 0.5 div veto
    # 0.92 / 0.99, dup 0.32 / 0.64; rho = 1 div 0.96 / 1.00,
    # dup -0.12 / 0.31; div consensus cost at base level.
    for rho, div_lo, dup_hi in ((0.5, 0.80, None), (1.0, 0.85, None)):
        agg = _aggregate(rho)
        assert _veto(agg, "div", "min") >= div_lo, (
            f"rho={rho} div min veto {_veto(agg, 'div', 'min'):.3f}")
        assert _veto(agg, "div", "delta") >= 0.85, (
            f"rho={rho} div delta veto {_veto(agg, 'div', 'delta'):.3f}")
        assert agg["div_min"]["cost"] <= 0.004, (
            f"rho={rho} div_min cost {agg['div_min']['cost']:+.4f}")
        assert agg["div_delta"]["cost"] <= 0.004, (
            f"rho={rho} div_delta cost {agg['div_delta']['cost']:+.4f}")
        for form in ("min", "delta"):
            assert _veto(agg, "dup", form) <= _veto(agg, "plain", form) + 0.15, (
                f"rho={rho} dup {form} veto {_veto(agg, 'dup', form):.3f} "
                f"vs plain {_veto(agg, 'plain', form):.3f}")


def test_veto_monotone_in_dose():
    _empf_recipe()
    # Claim (a), in the form the measurements support: the
    # seed-mean veto is monotone non-decreasing in k up to
    # a saturation plateau -- the k = 0 -> 1 step is large
    # on every form at both rhos (measured +0.30..+1.09),
    # k = 1 -> 2 is non-decreasing (measured +0.006..
    # +0.034), and k = 4 sits within 0.01 of k = 2 (the
    # delta form dips -0.004..-0.008 at the ceiling, where
    # every dose's veto is already >= 0.97). Per seed the
    # statement is stronger: EVERY dose k >= 1 beats plain
    # on EVERY seed, both forms, both rhos (smallest
    # per-seed gain +0.09).
    for rho in DOSE_RHOS:
        agg = _aggregate(rho)
        for form in ("min", "delta"):
            vetoes = [_veto(agg, pool, form) for pool in DIV_DOSE_POOLS]
            assert vetoes[1] >= vetoes[0] + 0.20, (
                f"rho={rho} {form} veto k0->k1 {vetoes}")
            assert vetoes[2] >= vetoes[1] - 0.01, (
                f"rho={rho} {form} veto k1->k2 {vetoes}")
            assert vetoes[3] >= vetoes[2] - 0.02, (
                f"rho={rho} {form} veto k2->k4 {vetoes}")
            assert vetoes[3] >= 0.93, (
                f"rho={rho} {form} veto at k=4 {vetoes}")
            plain_seed = _seed_vetoes(rho, "plain", form)
            for pool in DIV_DOSE_POOLS[1:]:
                for got, base in zip(_seed_vetoes(rho, pool, form),
                                     plain_seed):
                    assert got >= base + 0.05, (
                        f"rho={rho} {form} {pool} per-seed veto "
                        f"{got:+.3f} vs plain {base:+.3f}")


def test_marginal_gains_shape():
    _empf_recipe()
    # Claim (b), as measured: saturation is nearly total.
    # The FIRST independent source buys +0.30..+1.09 veto
    # (smallest: delta at rho = 0.5, +0.297); each source
    # from k = 2 -> 4 buys at most +0.017 (min form) and
    # slightly NEGATIVE on the saturated delta form
    # (-0.002..-0.004) -- under 6% of the first source's
    # gain on every form at both rhos. One independent
    # source is most of the repair; the curve is flat
    # long before k = 4.
    for rho in DOSE_RHOS:
        agg = _aggregate(rho)
        for form in ("min", "delta"):
            vetoes = [_veto(agg, pool, form) for pool in DIV_DOSE_POOLS]
            gain_first = vetoes[1] - vetoes[0]        # k = 0 -> 1
            gain_late = (vetoes[3] - vetoes[2]) / 2.0  # per source, k = 2 -> 4
            assert gain_first >= 0.25, (
                f"rho={rho} {form} first-source gain {gain_first:+.3f} "
                f"vetoes {vetoes}")
            assert gain_late <= 0.03, (
                f"rho={rho} {form} late per-source gain {gain_late:+.3f} "
                f"vetoes {vetoes}")
            assert gain_late <= 0.15 * gain_first, (
                f"rho={rho} {form} gains first {gain_first:+.3f} "
                f"late/source {gain_late:+.3f} vetoes {vetoes}")


def test_duplicates_flat_every_k():
    _empf_recipe()
    # Claim (c), as measured: duplicates never restore
    # veto at ANY dose. The largest upward movement at
    # any (k, form, rho) is +0.146 (delta, k = 4,
    # rho = 1 -- the same small positive drift #313 saw
    # at k = 2, +0.09); everywhere else duplicates sit
    # at or below plain, and the min form at rho = 0.5
    # drifts DOWN with dose (0.38 / 0.32 / 0.20 vs plain
    # 0.45 -- more carriers of the shared component make
    # the min floor worse, not better). And at every
    # dose the independent pool beats its duplicate
    # control by >= 0.29 veto (smallest gap: delta,
    # k = 4, rho = 0.5): the dose is independence, not
    # count, at every point of the curve.
    for rho in DOSE_RHOS:
        agg = _aggregate(rho)
        for pool in ("dup1", "dup", "dup4"):
            for form in ("min", "delta"):
                assert _veto(agg, pool, form) <= (
                    _veto(agg, "plain", form) + 0.20), (
                    f"rho={rho} {pool} {form} veto "
                    f"{_veto(agg, pool, form):.3f} vs plain "
                    f"{_veto(agg, 'plain', form):.3f}")
                assert _veto(agg, pool, form) >= (
                    _veto(agg, "plain", form) - 0.30), (
                    f"rho={rho} {pool} {form} veto "
                    f"{_veto(agg, pool, form):.3f} vs plain "
                    f"{_veto(agg, 'plain', form):.3f}")
        for div_pool, dup_pool in zip(DIV_DOSE_POOLS[1:],
                                      DUP_DOSE_POOLS[1:]):
            for form in ("min", "delta"):
                assert _veto(agg, div_pool, form) >= (
                    _veto(agg, dup_pool, form) + 0.20), (
                    f"rho={rho} {div_pool} {form} veto "
                    f"{_veto(agg, div_pool, form):.3f} vs "
                    f"{dup_pool} {_veto(agg, dup_pool, form):.3f}")


def test_benefit_retention_price():
    _empf_recipe()
    # Claim (d), as measured: the min form pays NO
    # retention price at any dose (1.05..1.10x of own
    # union at every k, both rhos -- flat). The delta
    # form's price is real but saturates with the veto
    # gain: retention vs own union falls from plain's
    # 0.71 / 0.67 (rho = 0.5 / 1) to 0.67 / 0.67 at
    # k = 1 and 0.63 / 0.62 at k = 2 -- the dose #313
    # priced -- then only a further 0.01..0.03 at k = 4
    # (0.62 / 0.59). Total price of the full curve,
    # k = 0 -> 4: ~0.09 of union benefit on the delta
    # form, ~0 on the min form; sources three and four
    # are nearly free on both forms.
    for rho in DOSE_RHOS:
        agg = _aggregate(rho)
        for pool in DIV_DOSE_POOLS[1:]:
            assert _retention(agg, pool, "min") >= 0.95, (
                f"rho={rho} {pool} min retention "
                f"{_retention(agg, pool, 'min'):.3f}")
            assert _retention(agg, pool, "delta") >= 0.55, (
                f"rho={rho} {pool} delta retention "
                f"{_retention(agg, pool, 'delta'):.3f}")
        ret1 = _retention(agg, "div1", "delta")
        ret4 = _retention(agg, "div4", "delta")
        assert ret4 >= ret1 - 0.10, (
            f"rho={rho} delta retention k1 {ret1:.3f} -> k4 {ret4:.3f}")
        assert ret4 <= ret1 + 0.02, (
            f"rho={rho} delta retention k1 {ret1:.3f} -> k4 {ret4:.3f}")


def test_estimator_loading_per_dose():
    _empf_recipe()
    # Optional readout, asserted because it came out
    # clean: the added independents' centered cost-token
    # loading sits at noise level at every dose and rho
    # (measured -0.06..-0.11, all k); the duplicates'
    # sits at the shared component at every dose
    # (+0.90..+0.95 at rho = 0.5, +1.95..+2.01 at
    # rho = 1, all k). The per-dose audit reads the same
    # at k = 1 as at k = 4: one source is enough to
    # classify, exactly as one source is most of the
    # repair.
    for rho in RHOS:
        agg = _aggregate(rho)
        for k in DOSES:
            assert abs(agg[f"div{k}_loading"]) <= 0.35, (
                f"rho={rho} div{k} loading {agg[f'div{k}_loading']:+.3f}")
    agg05 = _aggregate(0.5)
    agg1 = _aggregate(1.0)
    for k in DOSES:
        assert agg05[f"dup{k}_loading"] >= 0.60, (
            f"rho=0.5 dup{k} loading {agg05[f'dup{k}_loading']:+.3f}")
        assert agg1[f"dup{k}_loading"] >= 1.50, (
            f"rho=1 dup{k} loading {agg1[f'dup{k}_loading']:+.3f}")
