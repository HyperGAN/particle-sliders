"""CPU gate for the decorrelation (surgery) repair probe:
the third repair surface for the correlation boundary of
the consensus veto, after PR #309 closed the vote and
PR #313 opened the pool's membership, for Narang,
Tajdini, Zhang, Morgenstern, "Defense via Consensus:
Mitigating Subliminal Learning through Multi-Teacher
Distillation" (MSCD), arXiv:2607.23394.

The anchor chain. PR #285 (cycle 18) proved pooling
merely DILUTES a cost carried by one source (union cost
excess 0.16x the single-source excess) while cross-
source consensus VETOES it (MSCD-min aggregate cost
excess -0.0004, MSCD-delta +0.0006) at benefit retention
1.05x (min) / 0.68x (delta) of union. PR #305 (cycle 22)
proved the boundary: veto power 1 - (consensus / union
cost excess) collapses as cross-source cost correlation
rho rises -- rho = 0: 1.06 (min) / 0.97 (delta);
rho = 0.5: 0.45 / 0.68; rho = 1: -0.15 / 0.22, where a
fully correlated cost is retained at 1.15x / 0.78x of
union, exactly the retention the genuinely shared
benefit gets at the same rho. PR #309 (cycle 23) proved
the vote-level repair (cluster capping) FAILS by
construction: plain consensus already reads the pool's
minimum-cost member, and any cluster representative is
a within-cluster mean at or above that floor. PR #313
(cycle 24) proved the pool-level repair by ADDING two
independent sources restores the veto even at rho = 1
(0.96 / 1.00), while duplicates and carrier removal do
not; its centered-shift estimator pre-check separates
independents (centered cost-token loading -0.09, noise)
from duplicates (+0.95 / +2.01, the shared component).

Probe under test here: when NO independent source can
be recruited, can the pool be repaired by SURGERY --
subtracting the estimator-identified shared cost-token
component from every member's generations before
consensus? The shared component is ESTIMATED from the
pool itself, never read off the construction: with
centered shifts S_i = center(log p_i - log pi0) (#309 /
#313's estimator), the pool's shared cost-token loading
is c_hat = mean_i S_i[COST] (the mean centered shift's
cost coordinate), and surgery at strength lambda
replaces each teacher by softmax(log p_i - lambda *
c_hat * e_COST) -- the cost coordinate of every
member's log-generations is shifted down by the same
estimated amount, then consensus runs as usual.

Arms at each rho in {0.5, 1.0}, paired per seed:
  plain      -- #305/#313's K = 4 pool (calibration);
  div        -- #313's diversified pool, K = 6 with two
                added independent teachers (the
                membership-repair reference);
  surg{05,10,15} -- estimated surgery at lambda in
                {0.5, 1.0, 1.5} on the plain pool;
  orc        -- oracle surgery at lambda = 1: subtract
                the GROUND-TRUTH shared component
                rho * COST_SHIFT (the ceiling);
  rnd        -- random-direction subtraction at
                lambda = 1: subtract c_hat * u, u a
                random unit vector (matched norm,
                independent stream) -- control;
  bsub       -- benefit-component subtraction at
                lambda = 1: subtract b_hat * e_BENEFIT,
                b_hat = mean_i S_i[BENEFIT] estimated
                the same way -- control: the veto
                should stay dead and the benefit fall.
Each surgery variant contributes its own union / MSCD-
min / MSCD-delta targets over its corrected pool.

Readouts: excess over base of the benefit- and cost-
token probabilities of a student trained on each arm's
target (the anchors' student: free logit vector, SGD
lr 0.15, 400 steps, N = 20000 sampled tokens, from the
base logits). Veto: for plain and div, the anchors'
form, 1 - (own consensus cost / own union cost). For
surgery arms that form is degenerate BY CONSTRUCTION --
successful surgery drives the arm's OWN union cost to
base as well, so the ratio divides noise by noise --
so surgery vetoes are read against the FIXED reference
denominator, the plain pool's union cost at the same
(seed-mean) rho: veto_ref = 1 - (arm consensus cost /
plain union cost). This is the smallest deviation the
toy's structure forces, and it is the comparison the
claim actually makes (surgery vs the pool as found);
own-union vetoes for the surgery arms are not asserted.
Benefit retention is consensus benefit excess over the
arm's OWN union benefit excess, as in the anchors.

Stream discipline (calibration integrity): the plain
arms are drawn exactly as #305/#313 draw them -- base
logits and the K = 4 teacher noise off the per-seed
stream default_rng(SEED_BASE + seed), plain token
samples that stream's first draws, in #313's arm order
(plain, then div). The added teachers' noise comes off
#313's independent stream (DIV_SEED_BASE); the random
direction comes off its own stream (RND_SEED_BASE).
Surgery corrections are deterministic functions of the
teachers and pi0, so no surgery arm touches any stream
before its token samples, which continue the plain
stream in fixed arm order. The plain arms' numbers are
therefore bit-identical to #305/#313, and the div arms'
to #313.

Measured on pop-os (particlegan 0.10.1), 5 seeds --
aggregate cost excess (union / min / delta per pool)
and vetoes (plain and div: own-union form; surgery
arms: veto_ref against the plain union):
  rho = 0.5: plain +0.0217 / +0.0119 / +0.0069, veto
             0.45 / 0.68 (calibration: #305 exact);
             div +0.0140 / +0.0012 / +0.0002, veto
             0.92 / 0.99 (#313 exact); c_hat +1.19
             (truth 1.10), b_hat +1.97
    surg lam=0.5: union +0.0075, min +0.0026, delta
             +0.0010; veto_ref 0.88 / 0.96
    surg lam=1.0: union +0.0013, min -0.0009, delta
             +0.0001; veto_ref 1.04 / 0.99
    surg lam=1.5: union -0.0016, min -0.0027, delta
             +0.0004; veto_ref 1.13 / 0.98
    oracle lam=1: min -0.0004, delta +0.0001;
             veto_ref 1.02 / 0.99
    random lam=1: min +0.0173, delta +0.0112;
             veto_ref 0.20 / 0.48 (at/above plain's
             cost -- no recovery)
    benefit-sub:  union benefit +0.0007 (plain
             +0.0608 -- killed), min cost +0.0140
             (veto_ref 0.36 -- still dead)
  rho = 1.0: plain +0.0584 / +0.0671 / +0.0455, veto
             -0.15 / 0.22 (calibration: #305 exact);
             div +0.0363 / +0.0016 / +0.0001, veto
             0.96 / 1.00 (#313 exact); c_hat +1.98
             (truth 2.20), b_hat +1.93
    surg lam=0.5: union +0.0125, min +0.0144, delta
             +0.0086; veto_ref 0.75 / 0.85
    surg lam=1.0: union +0.0002, min +0.0006, delta
             +0.0003; veto_ref 0.99 / 1.00
    surg lam=1.5: union -0.0029, min -0.0029, delta
             -0.0024; veto_ref 1.05 / 1.04
    oracle lam=1: min -0.0007, delta +0.0001;
             veto_ref 1.01 / 1.00
    random lam=1: min +0.0817, delta +0.0598;
             veto_ref -0.40 / -0.02 (no recovery)
    benefit-sub:  union benefit +0.0005 (plain
             +0.0573 -- killed), min cost +0.0736
             (veto_ref -0.26 -- still dead)
Benefit retention of surgery (consensus / own union)
is plain's own level at every lambda: min 1.00..1.07,
delta 0.66..0.72 (plain at rho = 0: 1.05 / 0.68).
Per-seed surg lam=1 min cost: rho = 0.5 -0.0021..-0.0000,
rho = 1 -0.0014..+0.0018 -- at base on every seed.

The arbitration, stated plainly: ALL CLAIMS HOLD.
Estimated surgery at lambda = 1 recovers the veto at
rho = 1 from plain's -0.15 / 0.22 to veto_ref 0.99 /
1.00 -- the diversified pool's level (0.96 / 1.00) --
with NO source added, and it lands on the oracle
ceiling: |surg - oracle| cost gap <= 0.0013 on every
form at both rhos. Estimation is NOT the bottleneck
here (contrast #303/#306/#310, where the reference was
a sampled gradient): the shared component is a DENSE
statistic of the pool's own distributions -- the mean
centered loading reads +1.19 against a 1.10 truth at
rho = 0.5 and +1.98 against 2.20 at rho = 1 -- and
that accuracy suffices because consensus only needs
the corrected pool to disagree about the cost again,
which subtracting the common load from every member
achieves (teacher 0's idiosyncratic remainder at
rho = 0.5, the noise floor at rho = 1). The lambda
response is monotone in the min form at both rhos;
lambda = 0.5 already recovers most of the veto
(veto_ref 0.75 / 0.85 at rho = 1) and lambda = 1.5
overshoots mildly BELOW base (cost -0.0029 at rho = 1)
with no benefit damage (retention 1.00 / 0.66) -- the
overshoot is a cost-side distortion only, priced
above. (One flat spot, encoded not tuned away: at
rho = 0.5 the delta form is already at base at
lambda = 1, +0.0001 vs +0.0004 at lambda = 1.5.)
Both controls arbitrate as designed: a matched-norm
random subtraction leaves the cost at or above plain's
(no recovery), and subtracting the BENEFIT component
kills the benefit (+0.0005..+0.0007, ~0.01x plain)
while the veto stays dead -- the repair is specific
to the estimated cost component. Together with #309
(the vote cannot be repaired) and #313 (membership
can), the chain now has its third surface: the
members' CONTENT can be repaired too, when the shared
component is estimable from the pool -- and here it
is, densely.
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
DIV_SEED_BASE = 990000   # independent stream: diversify adds (#313)
RND_SEED_BASE = 880000   # independent stream: random direction
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.5, 1.0)
LAMBDAS = (0.5, 1.0, 1.5)
PLAIN_ARMS = ("single", "union", "mscd_min", "mscd_delta")
DIV_ARMS = ("div_union", "div_min", "div_delta")
SURG_ARMS = tuple(
    f"surg{int(lam * 10):02d}_{form}"
    for lam in LAMBDAS for form in ("union", "min", "delta"))
ORC_ARMS = ("orc_union", "orc_min", "orc_delta")
RND_ARMS = ("rnd_union", "rnd_min", "rnd_delta")
BSUB_ARMS = ("bsub_union", "bsub_min", "bsub_delta")
ARMS = (PLAIN_ARMS + DIV_ARMS + SURG_ARMS + ORC_ARMS
        + RND_ARMS + BSUB_ARMS)


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
    (the plain pool, the diversified pool, or a surgically
    corrected pool) is passed in."""
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
    """Each source's induced shift vs base: log p_i - log pi0,
    centered per source so the softmax normalization constant
    cannot masquerade as a direction difference (#309's
    estimator, as used by #313's pre-check)."""
    shifts = np.log(teachers) - np.log(pi0)
    return shifts - shifts.mean(axis=1, keepdims=True)


def _correct(teachers, vector):
    """Subtract a log-space vector from every member's
    generations and renormalize: the surgery operation.
    vector is a length-V array subtracted from log p_i."""
    corrected = np.exp(np.log(teachers) - vector)
    return corrected / corrected.sum(axis=1, keepdims=True)


def _added_teachers(base_logits, seed):
    """#313's two independent added teachers: shared benefit,
    own quirks on tokens 6 and 7, NO cost shift. Noise off
    #313's independent stream, never the plain stream."""
    rng = np.random.default_rng(DIV_SEED_BASE + seed)
    out = []
    for quirk in ADDED_QUIRKS:
        logits = base_logits.copy()
        logits[BENEFIT] += BENEFIT_SHIFT
        logits[quirk] += QUIRK_SHIFT
        logits = logits + rng.normal(0, NOISE, V)
        out.append(_softmax(logits))
    return np.array(out)


def _random_direction(seed):
    """A random unit vector in log space, off its own stream
    (one per seed, shared across rho panels)."""
    rng = np.random.default_rng(RND_SEED_BASE + seed)
    u = rng.standard_normal(V)
    return u / np.linalg.norm(u)


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
    """All arms for one (seed, rho), paired on one draw, plus
    the estimator's shared-component estimates.

    Draw order is fixed -- base logits, per-teacher noise
    (plain stream), added-teacher noise (independent stream),
    then per-arm token samples in ARMS order off the plain
    stream -- so the plain arms use the same base, teacher
    noise and token draws as #305/#313, and the div arms
    the same added teachers as #313.
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

    div_stack = np.vstack([teachers, _added_teachers(base_logits, seed)])

    # The estimator: the pool's shared loadings are the mean
    # centered shift's coordinates at the cost / benefit
    # tokens. Estimated from the plain pool only.
    mean_shift = _estimated_shifts(pi0, teachers).mean(axis=0)
    c_hat = float(mean_shift[COST])
    b_hat = float(mean_shift[BENEFIT])

    e_cost = np.eye(V)[COST]
    e_benefit = np.eye(V)[BENEFIT]
    u = _random_direction(seed)

    stacks = {"plain": teachers, "div": div_stack}
    for lam in LAMBDAS:
        stacks[f"surg{int(lam * 10):02d}"] = _correct(
            teachers, lam * c_hat * e_cost)
    stacks["orc"] = _correct(teachers, rho * COST_SHIFT * e_cost)
    stacks["rnd"] = _correct(teachers, c_hat * u)
    stacks["bsub"] = _correct(teachers, b_hat * e_benefit)

    targets = {"single": teachers[0]}
    plain = _consensus_targets(pi0, stacks["plain"])
    targets.update({
        "union": plain["union"],
        "mscd_min": plain["mscd_min"],
        "mscd_delta": plain["mscd_delta"],
    })
    for prefix, key in (("div", "div"), ("orc", "orc"),
                        ("rnd", "rnd"), ("bsub", "bsub")):
        cons = _consensus_targets(pi0, stacks[key])
        targets[f"{prefix}_union"] = cons["union"]
        targets[f"{prefix}_min"] = cons["mscd_min"]
        targets[f"{prefix}_delta"] = cons["mscd_delta"]
    for lam in LAMBDAS:
        key = f"surg{int(lam * 10):02d}"
        cons = _consensus_targets(pi0, stacks[key])
        targets[f"{key}_union"] = cons["union"]
        targets[f"{key}_min"] = cons["mscd_min"]
        targets[f"{key}_delta"] = cons["mscd_delta"]

    out = {"c_hat": c_hat, "b_hat": b_hat}
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
    hats = {"c_hat": [], "b_hat": []}
    for seed in SEEDS:
        res = _seed_rho(seed, rho)
        for arm in ARMS:
            for key in agg[arm]:
                agg[arm][key].append(res[arm][key])
        for key in hats:
            hats[key].append(res[key])
    out = {arm: {key: float(np.mean(vals))
                 for key, vals in d.items()}
           for arm, d in agg.items()}
    out["per_seed"] = {
        arm: {key: [float(_seed_rho(seed, rho)[arm][key])
                     for seed in SEEDS]
              for key in ("benefit", "cost")}
        for arm in ARMS}
    out["c_hat"] = float(np.mean(hats["c_hat"]))
    out["b_hat"] = float(np.mean(hats["b_hat"]))
    out["c_hat_seeds"] = [float(x) for x in hats["c_hat"]]
    return out


def _veto(agg, arm, union_arm="union"):
    """The anchors' veto: 1 - consensus cost / union cost."""
    return 1.0 - agg[arm]["cost"] / agg[union_arm]["cost"]


def _veto_ref(agg, arm):
    """Surgery veto against the fixed reference denominator:
    the plain pool's union cost at the same rho."""
    return 1.0 - agg[arm]["cost"] / agg["union"]["cost"]


def test_calibration_plain_and_diversified():
    _empf_recipe()
    # Calibration against #305/#313 (their aggregates:
    # rho = 0.5 plain veto 0.45 / 0.68, div veto 0.92 /
    # 0.99; rho = 1 plain veto -0.15 / 0.22, div veto
    # 0.96 / 1.00; plain union cost +0.0217 / +0.0584).
    for rho, plain_lo in ((0.5, 0.30), (1.0, -0.35)):
        agg = _aggregate(rho)
        assert agg["union"]["cost"] >= 0.010, (
            f"rho={rho} plain union cost {agg['union']['cost']:+.4f}")
        assert _veto(agg, "mscd_min") >= plain_lo, (
            f"rho={rho} plain min veto {_veto(agg, 'mscd_min'):.3f}")
        assert _veto(agg, "div_min", "div_union") >= 0.80, (
            f"rho={rho} div min veto "
            f"{_veto(agg, 'div_min', 'div_union'):.3f}")
        assert _veto(agg, "div_delta", "div_union") >= 0.85, (
            f"rho={rho} div delta veto "
            f"{_veto(agg, 'div_delta', 'div_union'):.3f}")
    agg1 = _aggregate(1.0)
    assert _veto(agg1, "mscd_min") <= 0.10, (
        f"rho=1 plain min veto {_veto(agg1, 'mscd_min'):.3f}")
    assert _veto(agg1, "mscd_delta") <= 0.45, (
        f"rho=1 plain delta veto {_veto(agg1, 'mscd_delta'):.3f}")


def test_estimator_recovers_shared_component():
    _empf_recipe()
    # The estimator reads the shared cost component off the
    # pool: c_hat (mean centered cost-token loading) sits
    # near the ground-truth shared component rho * 2.2
    # (1.1 at rho = 0.5, 2.2 at rho = 1), up to the
    # centering offset, and grows with rho. The benefit
    # loading b_hat reads the shared benefit (2.2) at both
    # rhos -- at rho = 1 the two loadings are near-equal,
    # which is exactly #305's indistinguishability.
    agg05 = _aggregate(0.5)
    agg1 = _aggregate(1.0)
    assert 0.80 <= agg05["c_hat"] <= 1.60, (
        f"rho=0.5 c_hat {agg05['c_hat']:+.3f}")
    assert 1.50 <= agg1["c_hat"] <= 2.40, (
        f"rho=1 c_hat {agg1['c_hat']:+.3f}")
    assert agg1["c_hat"] >= agg05["c_hat"] + 0.50, (
        f"c_hat growth {agg05['c_hat']:+.3f} -> {agg1['c_hat']:+.3f}")
    for agg in (agg05, agg1):
        assert 1.50 <= agg["b_hat"] <= 2.40, (
            f"b_hat {agg['b_hat']:+.3f}")


def test_surgery_lambda1_recovers_veto_rho1():
    _empf_recipe()
    agg = _aggregate(1.0)
    # Headline claim at rho = 1: estimated surgery at
    # lambda = 1 returns the consensus cost to base level
    # and the (reference) veto to the diversified band,
    # far above plain's dead veto.
    for form in ("min", "delta"):
        arm = f"surg10_{form}"
        assert abs(agg[arm]["cost"]) <= 0.008, (
            f"rho=1 {arm} cost {agg[arm]['cost']:+.4f}")
        assert _veto_ref(agg, arm) >= 0.85, (
            f"rho=1 {arm} veto_ref {_veto_ref(agg, arm):.3f}")
        assert _veto_ref(agg, arm) >= _veto_ref(agg, f"mscd_{form}") + 0.60, (
            f"rho=1 {arm} veto_ref {_veto_ref(agg, arm):.3f} vs "
            f"plain {_veto_ref(agg, f'mscd_{form}'):.3f}")
    # Toward the diversified reference: surgery's veto_ref
    # is within 0.15 of the diversified pool's own veto.
    assert _veto_ref(agg, "surg10_min") >= (
        _veto(agg, "div_min", "div_union") - 0.15), (
        f"rho=1 surg min veto_ref {_veto_ref(agg, 'surg10_min'):.3f} "
        f"vs div {_veto(agg, 'div_min', 'div_union'):.3f}")


def test_surgery_lambda1_recovers_veto_rho05():
    _empf_recipe()
    agg = _aggregate(0.5)
    for form in ("min", "delta"):
        arm = f"surg10_{form}"
        assert abs(agg[arm]["cost"]) <= 0.008, (
            f"rho=0.5 {arm} cost {agg[arm]['cost']:+.4f}")
        assert _veto_ref(agg, arm) >= 0.70, (
            f"rho=0.5 {arm} veto_ref {_veto_ref(agg, arm):.3f}")
        assert _veto_ref(agg, arm) >= _veto_ref(agg, f"mscd_{form}") + 0.20, (
            f"rho=0.5 {arm} veto_ref {_veto_ref(agg, arm):.3f} vs "
            f"plain {_veto_ref(agg, f'mscd_{form}'):.3f}")


def test_surgery_vs_oracle_ceiling():
    _empf_recipe()
    # The oracle (ground-truth shared component, lambda = 1)
    # is the ceiling. As measured, estimated surgery lands
    # ON it: the |surg - oracle| cost gap is <= 0.0013 on
    # every form at both rhos (asserted at 0.005) -- there
    # is no estimation bottleneck in this probe, because
    # the shared component is a dense statistic of the
    # pool's distributions (see the estimator test).
    for rho in RHOS:
        agg = _aggregate(rho)
        for form in ("min", "delta"):
            assert abs(agg[f"orc_{form}"]["cost"]) <= 0.008, (
                f"rho={rho} orc_{form} cost {agg[f'orc_{form}']['cost']:+.4f}")
            assert abs(agg[f"surg10_{form}"]["cost"]
                       - agg[f"orc_{form}"]["cost"]) <= 0.005, (
                f"rho={rho} surg10_{form} cost "
                f"{agg[f'surg10_{form}']['cost']:+.4f} vs oracle "
                f"{agg[f'orc_{form}']['cost']:+.4f}")


def test_lambda_response_and_overshoot():
    _empf_recipe()
    # Lambda response (seed-mean): consensus cost falls
    # monotonically as lambda rises 0.5 -> 1.0 -> 1.5, and
    # lambda = 1.5 overshoots: the cost is driven to (or
    # below) base, i.e. the subtraction removes more than
    # the shared component present.
    for rho in RHOS:
        agg = _aggregate(rho)
        for form in ("min", "delta"):
            c05 = agg[f"surg05_{form}"]["cost"]
            c10 = agg[f"surg10_{form}"]["cost"]
            c15 = agg[f"surg15_{form}"]["cost"]
            assert c05 >= c10 - 0.002, (
                f"rho={rho} {form} lambda 0.5 cost {c05:+.4f} vs "
                f"lambda 1 cost {c10:+.4f}")
            assert c10 >= c15 - 0.002, (
                f"rho={rho} {form} lambda 1 cost {c10:+.4f} vs "
                f"lambda 1.5 cost {c15:+.4f}")
    # Lambda = 0.5 at rho = 1 is a PARTIAL recovery, as
    # measured: the min cost is still clearly positive
    # (+0.0144, per-seed +0.0108..+0.0213) yet its veto_ref
    # (0.75) is already far above plain's dead -0.15.
    agg1 = _aggregate(1.0)
    assert 0.008 <= agg1["surg05_min"]["cost"] <= 0.030, (
        f"rho=1 surg05_min cost {agg1['surg05_min']['cost']:+.4f}")
    assert _veto_ref(agg1, "surg05_min") >= _veto(agg1, "mscd_min") + 0.50, (
        f"rho=1 surg05 min veto_ref {_veto_ref(agg1, 'surg05_min'):.3f} "
        f"vs plain {_veto(agg1, 'mscd_min'):.3f}")
    # Overshoot at lambda = 1.5: at rho = 1 the cost is
    # driven BELOW base on both forms (measured -0.0029 /
    # -0.0024); the min form overshoots at rho = 0.5 too
    # (measured -0.0027). The delta form at rho = 0.5 is
    # the one flat spot (at base already at lambda = 1:
    # +0.0001 vs +0.0004) -- asserted only as at-base.
    assert agg1["surg15_min"]["cost"] <= -0.001, (
        f"rho=1 surg15_min cost {agg1['surg15_min']['cost']:+.4f}")
    assert agg1["surg15_delta"]["cost"] <= -0.001, (
        f"rho=1 surg15_delta cost {agg1['surg15_delta']['cost']:+.4f}")
    agg05 = _aggregate(0.5)
    assert agg05["surg15_min"]["cost"] <= -0.001, (
        f"rho=0.5 surg15_min cost {agg05['surg15_min']['cost']:+.4f}")
    assert abs(agg05["surg15_delta"]["cost"]) <= 0.002, (
        f"rho=0.5 surg15_delta cost {agg05['surg15_delta']['cost']:+.4f}")


def test_benefit_survives_cost_surgery():
    _empf_recipe()
    # Cost surgery touches only the cost coordinate: the
    # shared benefit survives at every lambda, at plain's
    # retention level (vs the arm's own union).
    for rho in RHOS:
        agg = _aggregate(rho)
        for lam_key in ("surg05", "surg10", "surg15", "orc"):
            assert agg[f"{lam_key}_min"]["benefit"] >= (
                0.85 * agg[f"{lam_key}_union"]["benefit"]), (
                f"rho={rho} {lam_key}_min benefit "
                f"{agg[f'{lam_key}_min']['benefit']:+.4f} vs own union "
                f"{agg[f'{lam_key}_union']['benefit']:+.4f}")
            assert agg[f"{lam_key}_delta"]["benefit"] >= (
                0.45 * agg[f"{lam_key}_union"]["benefit"]), (
                f"rho={rho} {lam_key}_delta benefit "
                f"{agg[f'{lam_key}_delta']['benefit']:+.4f} vs own union "
                f"{agg[f'{lam_key}_union']['benefit']:+.4f}")
            assert agg[f"{lam_key}_union"]["benefit"] >= (
                0.80 * agg["union"]["benefit"]), (
                f"rho={rho} {lam_key} union benefit "
                f"{agg[f'{lam_key}_union']['benefit']:+.4f} vs plain "
                f"union {agg['union']['benefit']:+.4f}")


def test_random_subtraction_no_recovery():
    _empf_recipe()
    # Control: a matched-norm subtraction in a random
    # direction does NOT recover the veto -- the recovery
    # is specific to the estimated cost component, not to
    # perturbing the pool at the same scale.
    for rho in RHOS:
        agg = _aggregate(rho)
        for form in ("min", "delta"):
            assert agg[f"rnd_{form}"]["cost"] >= (
                0.55 * agg[f"mscd_{form}"]["cost"]), (
                f"rho={rho} rnd_{form} cost "
                f"{agg[f'rnd_{form}']['cost']:+.4f} vs plain "
                f"{agg[f'mscd_{form}']['cost']:+.4f}")
            assert _veto_ref(agg, f"rnd_{form}") <= (
                _veto_ref(agg, "surg10_min") - 0.30 if form == "min"
                else _veto_ref(agg, "surg10_delta") - 0.30), (
                f"rho={rho} rnd_{form} veto_ref "
                f"{_veto_ref(agg, f'rnd_{form}'):.3f} vs surg10 "
                f"{_veto_ref(agg, f'surg10_{form}'):.3f}")


def test_benefit_subtraction_control():
    _empf_recipe()
    # Control: subtracting the shared BENEFIT component
    # instead kills the benefit (the corrected pool's
    # union benefit collapses toward base) while the cost
    # -- and its dead veto -- is untouched: surgery's
    # effect is component-specific, not generic removal.
    for rho in RHOS:
        agg = _aggregate(rho)
        assert agg["bsub_union"]["benefit"] <= (
            0.45 * agg["union"]["benefit"]), (
            f"rho={rho} bsub union benefit "
            f"{agg['bsub_union']['benefit']:+.4f} vs plain union "
            f"{agg['union']['benefit']:+.4f}")
        assert agg["bsub_min"]["cost"] >= (
            0.55 * agg["mscd_min"]["cost"]), (
            f"rho={rho} bsub_min cost {agg['bsub_min']['cost']:+.4f} "
            f"vs plain min {agg['mscd_min']['cost']:+.4f}")
        assert _veto_ref(agg, "bsub_min") <= (
            _veto_ref(agg, "surg10_min") - 0.30), (
            f"rho={rho} bsub min veto_ref "
            f"{_veto_ref(agg, 'bsub_min'):.3f} vs surg10 "
            f"{_veto_ref(agg, 'surg10_min'):.3f}")
