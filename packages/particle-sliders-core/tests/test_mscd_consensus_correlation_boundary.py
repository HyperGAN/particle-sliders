"""CPU gate for the correlation boundary of the consensus
veto proven in PR #285 for Narang, Tajdini, Zhang,
Morgenstern, "Defense via Consensus: Mitigating Subliminal
Learning through Multi-Teacher Distillation" (MSCD),
arXiv:2607.23394.

PR #285's proven law (cycle 18): at pool composition,
pooling merely DILUTES a cost carried by one source
(union cost excess 0.16x the single-source excess), while
cross-source consensus VETOES it (MSCD-min aggregate cost
excess -0.0004, MSCD-delta +0.0006) while retaining the
shared benefit at 1.05x (min) / 0.68x (delta) of union.

Boundary claim under test here: consensus can only veto
what sources DISAGREE about. A cost that is CORRELATED
across sources -- carried by all of them in the same
direction -- is indistinguishable from shared benefit by
any agreement filter, so veto power must collapse as the
cross-source cost correlation rho rises from 0
(idiosyncratic) to 1 (fully shared). If veto power does
NOT collapse at rho = 1, #285's mechanism reading (the
veto is agreement filtering) is wrong.

Toy: #285's pool-composition toy, extended with one
dial. Vocab V = 24, single categorical draw. Base pi0 =
softmax of random logits with benefit- and cost-token
logits pinned at -1.0. K = 4 teachers = softmax(base
logits + shifts + per-teacher Gaussian logit noise
sigma = 0.15): ALL teachers shift the benefit token +2.2
(shared benefit); each teacher shifts its own private
quirk token +1.5. The COST shift on teacher i is

    cost_shift_i(rho) = rho * COST_SHIFT
                        + (1 - rho) * COST_SHIFT * [i == 0]

a common component (rho * COST_SHIFT, identical on
every teacher) plus an idiosyncratic component
((1 - rho) * COST_SHIFT on teacher 0 only). rho is the
common-load fraction -- the correlation dial: under a
random-effects reading (common draw shared by all
sources plus an independent draw on one) rho is the
share of the cost carried by the common component.
rho = 0 reproduces #285 exactly ([2.2, 0, 0, 0]);
rho = 0.5 gives [2.2, 1.1, 1.1, 1.1]; rho = 1 gives
[2.2, 2.2, 2.2, 2.2] -- every source carries the cost
at full strength in the same direction, exactly as
every source carries the benefit. (Round 1 of the
build loop used a constant-total-load variant, which
diluted the per-source shared shift to 0.55 at rho = 1
and pushed the cost into the noise floor; see the PR
iteration log. The construction above is the one the
assertions below measure.)

Student = free logit vector, SGD (lr 0.15, 400 steps) on
N = 20000 tokens sampled from each arm's target, starting
from the base logits. Arms at each rho, paired per seed:
single (teacher 0 alone), union (arithmetic mean of the
teacher distributions), mscd_min (normalized per-token
minimum), mscd_delta (base-relative minimum rule of
#285 / paper section 4.3). Readouts: excess over base of
the benefit-token and cost-token probabilities.
Assertions are aggregate over seeds, following #285's
discipline (its per-seed quirk suppression was uneven,
so cost claims there were aggregate-only; the same
discipline is kept here).

Measured on pop-os (particlegan 0.10.1), 5 seeds --
aggregate cost excess (single / union / mscd_min /
mscd_delta) and veto power 1 - (consensus / union):
  rho = 0.0: cost +0.0654 / +0.0107 / -0.0006 / +0.0004;
             veto min 1.06, delta 0.97
  rho = 0.5: cost +0.0654 / +0.0217 / +0.0119 / +0.0069;
             veto min 0.45, delta 0.68
  rho = 1.0: cost +0.0654 / +0.0584 / +0.0671 / +0.0455;
             veto min -0.15, delta 0.22
Benefit excess (union / mscd_min / mscd_delta) stays
+0.057..+0.062 / +0.058..+0.065 / +0.039..+0.043 at
every rho. Per-seed rho = 1 cost excess (union /
mscd_min / mscd_delta): seed 0 +0.0521 / +0.0525 /
+0.0355; seed 1 +0.0359 / +0.0541 / +0.0250; seed 2
+0.0627 / +0.0776 / +0.0581; seed 3 +0.0941 / +0.0916 /
+0.0708; seed 4 +0.0474 / +0.0597 / +0.0382.

The arbitration, stated plainly: the boundary law
holds on all three decidable verdicts. (i) At rho = 0
the #285 veto reproduces (calibration: single +0.0654
and union +0.0107 match #285's aggregates exactly;
consensus cost at base). (ii) At rho = 1 the veto is
gone: the minimum rule retains the cost at 1.15x union
and the delta rule at 0.78x union -- the same retention
those rules give the genuinely shared benefit at the
same rho (1.01x / 0.67x). A fully correlated cost is
passed by the agreement filter exactly as if it were
benefit, because by the filter's only criterion --
cross-source agreement -- it is one. (iii) The
transition is monotone in rho for both rules. The
honest implication: #285's mechanism reading stands,
with its scope made exact -- consensus regeneration is
a defence against source-specific costs only; a cost
that every source in the pool carries (a shared data
contamination, a common teacher bias) survives it at
benefit level, and no agreement-based filter can do
better. Assertions below encode exactly these margins,
no more.
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
SEED_BASE = 18000    # RNG stream base (same as #285)
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.0, 0.5, 1.0)
ARMS = ("single", "union", "mscd_min", "mscd_delta")


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cost_shifts(rho):
    """Per-teacher cost logit shifts at correlation rho.

    Common component rho * COST_SHIFT on every teacher,
    plus idiosyncratic component (1 - rho) * COST_SHIFT
    on teacher 0 only. Teacher 0 carries the full
    COST_SHIFT at every rho; the other teachers ramp
    from 0 (rho = 0, #285) to the full shift (rho = 1).
    """
    shifts = np.full(K, rho * COST_SHIFT)
    shifts[0] += (1.0 - rho) * COST_SHIFT
    return shifts


def _targets(pi0, teachers):
    """The four arms' target distributions for one (seed, rho)."""
    union = teachers.mean(axis=0)
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


def run_seed_rho(seed, rho):
    """All four arms for one (seed, rho), paired on one draw.

    Draw order is fixed -- base logits, per-teacher noise,
    then per-arm token samples in ARMS order -- so at fixed
    seed the rho = 0 panel uses the same base and teacher
    noise as #285's toy, and arms differ only through
    their target distribution.
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

    targets = _targets(pi0, teachers)
    out = {}
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


def test_rho0_calibration_reproduces_285_veto():
    _empf_recipe()
    agg = _aggregate(0.0)
    # Calibration against #285's table (its aggregates:
    # single +0.0654, union +0.0107 = 0.16x single,
    # mscd_min -0.0004, mscd_delta +0.0006).
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
    # The shared benefit survives at rho = 0, as in #285.
    assert agg["mscd_min"]["benefit"] >= 0.80 * agg["union"]["benefit"], (
        f"rho=0 mscd_min benefit {agg['mscd_min']['benefit']:+.4f} "
        f"vs union {agg['union']['benefit']:+.4f}")
    assert agg["mscd_delta"]["benefit"] >= 0.50 * agg["union"]["benefit"], (
        f"rho=0 mscd_delta benefit {agg['mscd_delta']['benefit']:+.4f} "
        f"vs union {agg['union']['benefit']:+.4f}")


def test_rho1_veto_collapses_cost_passes_as_benefit():
    _empf_recipe()
    agg = _aggregate(1.0)
    # At rho = 1 the cost is carried by every source, so
    # union pooling retains it at the shared-component
    # level -- and the consensus arms retain it at
    # approximately the union level: the veto is gone.
    assert agg["union"]["cost"] >= 0.02, (
        f"rho=1 union cost {agg['union']['cost']:+.4f}")
    # Measured: mscd_min retains 1.15x union, mscd_delta
    # 0.78x union -- asserted with margin below both.
    assert agg["mscd_min"]["cost"] >= 0.70 * agg["union"]["cost"], (
        f"rho=1 mscd_min cost {agg['mscd_min']['cost']:+.4f} vs "
        f"union {agg['union']['cost']:+.4f}")
    assert agg["mscd_delta"]["cost"] >= 0.55 * agg["union"]["cost"], (
        f"rho=1 mscd_delta cost {agg['mscd_delta']['cost']:+.4f} vs "
        f"union {agg['union']['cost']:+.4f}")
    # The retained cost is treated like the retained
    # benefit: each consensus arm's cost retention is at
    # least half its benefit retention at the same rho.
    for arm in ("mscd_min", "mscd_delta"):
        assert agg[arm]["cost"] >= 0.50 * agg[arm]["benefit"], (
            f"rho=1 {arm} cost {agg[arm]['cost']:+.4f} vs own "
            f"benefit {agg[arm]['benefit']:+.4f}")


def test_veto_power_falls_monotonically_in_rho():
    _empf_recipe()
    # Veto power at rho := 1 - (consensus cost / union cost),
    # aggregate. It must fall monotonically as rho rises.
    veto = {}
    for rho in RHOS:
        agg = _aggregate(rho)
        veto[rho] = {
            arm: 1.0 - agg[arm]["cost"] / agg["union"]["cost"]
            for arm in ("mscd_min", "mscd_delta")
        }
    for arm in ("mscd_min", "mscd_delta"):
        assert veto[0.0][arm] >= veto[0.5][arm] - 0.02, (
            f"{arm} veto rho0 {veto[0.0][arm]:.3f} vs "
            f"rho0.5 {veto[0.5][arm]:.3f}")
        assert veto[0.5][arm] >= veto[1.0][arm] - 0.02, (
            f"{arm} veto rho0.5 {veto[0.5][arm]:.3f} vs "
            f"rho1 {veto[1.0][arm]:.3f}")
        assert veto[0.0][arm] - veto[1.0][arm] >= 0.50, (
            f"{arm} veto drop {veto[0.0][arm]:.3f} -> "
            f"{veto[1.0][arm]:.3f}")


def test_benefit_retained_at_all_rho():
    _empf_recipe()
    # Control: the consensus arms keep retaining the
    # genuinely shared benefit at every rho -- the filter
    # at rho = 1 passes the cost precisely because the
    # cost now looks like this benefit.
    for rho in RHOS:
        agg = _aggregate(rho)
        assert agg["union"]["benefit"] >= 0.010, (
            f"rho={rho} union benefit {agg['union']['benefit']:+.4f}")
        assert agg["mscd_min"]["benefit"] >= 0.70 * agg["union"]["benefit"], (
            f"rho={rho} mscd_min benefit "
            f"{agg['mscd_min']['benefit']:+.4f} vs union "
            f"{agg['union']['benefit']:+.4f}")
        assert agg["mscd_delta"]["benefit"] >= 0.40 * agg["union"]["benefit"], (
            f"rho={rho} mscd_delta benefit "
            f"{agg['mscd_delta']['benefit']:+.4f} vs union "
            f"{agg['union']['benefit']:+.4f}")
