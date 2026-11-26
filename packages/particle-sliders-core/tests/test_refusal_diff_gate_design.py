"""CPU gate for the GATE-DESIGN chain claim of the Particle
Sliders paper watch (cycle 34, chain #345 -> the Jev-gate
seat): can refusal-rate-as-difference be built into the gate
at check() in scripts/check_slider_gate.py as a pass/fail
criterion with good operating characteristics?

Chain anchors:
  * PR #340 (Refusal Is Not Scored, cycle 32): an audit that
    discards the refusal/abstention category before scoring
    is blind to it; refusal-rate-as-outcome recovers a planted
    disparity.
  * PR #345 (Blind-Spot Detection, chain #320 x #340): in the
    installation setting a category-blind judge installs
    refusal (+0.907..+0.923 analytic disparity); a FILTERED
    audit certifies it clean (-0.005..+0.003), UNFILTERED
    scoring sign-flips (-0.271..-0.276), and only
    REFUSAL-RATE-AS-DIFFERENCE against a sighted-reference
    student recovers it (+0.903..+0.921, matching analytic
    within 0.006/seed -- a DIFFERENCE because the ratio
    degenerates when the reference rate is ~0).
  * PR #320 (Blind Spots Travel): the installation
    construction reused verbatim in stage 1 below.

UNTESTED by the anchors: the DESIGN question. Three gate
arms, all reading the same per-category audit (N=8,000 per
cell, #340/#345 discipline), compared across four worlds:

  GATE ARMS
  (a) SCORED-ONLY gate (status quo): fire if the max
      |filtered scored disparity| across categories exceeds
      0.02 (the #345 filtered tolerance). No refusal readout.
  (b) RAW max-diff gate: fire if any category refusal-rate
      difference vs the sighted reference exceeds a
      threshold.
  (c) DE-MEANED category-specific gate: fire if a category
      difference minus the MEDIAN category difference
      exceeds a threshold -- designed to isolate
      category-concentrated installation from global
      abstention drift.

  WORLDS (test student vs the sighted/key reference)
  (1) BLIND-INSTALLED: #345 blind-curated student
      (category-0-concentrated refusal installed).
  (2) CLEAN: a second sighted student (key2) -- no
      installation, matched scored behaviour.
  (3) GLOBALLY-SHIFTED (new construction): refusal raised
      UNIFORMLY across all categories. Construction
      property: its TOTAL extra refusal mass equals the
      installed world total, sum_c (global - key) ==
      sum_c (blind - key), spread evenly (extra/4 per
      category) -- the discrimination between worlds 1
      and 3 is CONCENTRATION, not amount. A legitimate
      drift (a more cautious student); the gate must
      NOT fire.
  (4) SHUFFLED-JUDGE control: #345 random-curated student
      (random judge drift, no systematic blind category).

Claim: the de-meaned gate FIRES on world 1 and PASSES
worlds 2-4 at a threshold with a wide margin band; the
raw gate false-fires on world 3; the scored-only gate
misses world 1. UNKNOWN is the refusal class (Tiny Aya
seat).

Measured on pop-os (particlegan 0.10.1), 5 seeds, sampled
audit readouts (see PR body for the full table):
  de-meaned statistic: W1 0.901..0.933; W2 0.000;
    W3 0.000..0.013; W4 0.009..0.217. Separating band
    (0.217, 0.901), width 0.684; operating threshold 0.50
    gives the exact fire/pass pattern per seed.
  raw statistic: W1 0.901..0.933; W2 0.000;
    W3 0.229..0.240; W4 0.268..0.539. Separating band
    (0.539, 0.901), width only 0.362; at a sensitivity
    threshold of 0.20 the raw gate fires world 3 in every
    seed (false fire on legitimate global drift), and at
    the 0.50 operating point it still fires world 4 in
    seed 2 (0.539). The de-meaned gate passes world 3 at
    every threshold above 0.013.
  scored-only statistic: <= 0.005 in every world -- it
    passes world 1 (miss) along with everything else.

Modeling gifts, stated plainly: the factorised gate is
#320/#345 stated gift; scored bias among answered items
is planted equal across students (that equality IS #320
measured hidden-loss result); the globally-shifted world
is an analytic gate-probability construction on top of
the sighted student, not a trained student -- it isolates
exactly the concentration-vs-amount question the gate
design turns on, and no training dynamic is claimed for
it. Assertions encode the demonstrated margins with
modest headroom; no threshold was tuned per world -- one
operating threshold (0.50) and one sensitivity threshold
(0.20) are applied uniformly.

CODEBASE LANDING: scripts/check_slider_gate.py check().
Wiring: alongside the existing sidecar checks, a gate
audit would (i) run a sighted-reference student and the
candidate on the same per-category audit set, (ii) compute
per-category UNKNOWN/refusal-rate differences, (iii)
de-mean by the median category difference, and (iv) fail
check() if the max de-meaned difference exceeds a
threshold chosen inside the measured band (0.217, 0.901)
-- 0.50 sits mid-band. A raw (non-de-meaned) criterion is
NOT a safe substitute: it cannot distinguish a
category-concentrated installation from a uniform
cautious drift of the same total refusal mass.

Pure numpy toy on CPU, single-threaded, seeded. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (house guard, PR #273 pattern).
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


# ------------------------------------------------------- stage 1: install
# #320/#345 construction, verbatim dynamics.
CATS = 4
NTR = 15
EPISODES = 35
U0 = 1.2
LR_U = 0.03
LR_S = 0.003
LR_D = 0.10
CONS_K = 32
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("key", "blind", "random", "key2")
BLIND = 0

# --------------------------------------------------------- stage 2: audit
N_AUDIT = 8000
BIAS_MEAN = 0.30
SCORE_SD = 0.10

# ------------------------------------------------------------ gate design
SCORED_THR = 0.02      # arm (a): #345 filtered tolerance
OPERATING_THR = 0.50   # arms (b)/(c): mid de-meaned band (0.217, 0.901)
SENSITIVITY_THR = 0.20  # a gate tuned to catch ~0.2-scale installations


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _knowledge(seed):
    rng = np.random.default_rng(1000 + seed)
    d_tr = rng.normal(1.4, 0.45, size=(CATS, NTR))
    d_he = rng.normal(1.4, 0.45, size=(CATS, NTR))
    cons = np.zeros((CATS, NTR), dtype=int)
    for c in range(CATS):
        for i in range(NTR):
            draws = rng.binomial(CONS_K, _sigmoid(d_tr[c, i]))
            cons[c, i] = 1 if draws >= CONS_K // 2 else 0
    return d_tr, d_he, cons


def _train(seed, arm):
    d_tr, _, cons = _knowledge(seed)
    rng = np.random.default_rng(2000 + seed * 10 + ARMS.index(arm))
    u = np.full(CATS, U0)
    s_cat = np.zeros(CATS)
    delta = np.zeros((CATS, NTR))
    judge = "key" if arm == "key2" else arm
    for _ in range(EPISODES):
        for c in range(CATS):
            for i in range(NTR):
                if rng.random() > _sigmoid(u[c]):
                    continue
                q = _sigmoid(d_tr[c, i] - delta[c, i] + s_cat[c])
                is_true = rng.random() < q
                if judge == "key":
                    r = 1 if is_true else -1
                elif judge == "blind":
                    if c == BLIND and is_true:
                        r = 1 if rng.random() < 1 / 12 else -1
                    else:
                        r = 1 if is_true else -1
                else:
                    r = 1 if rng.random() < 0.5 else -1
                u[c] += LR_U * r
                s_cat[c] += LR_S * r * (1 if is_true else -1)
                delta[c, i] += LR_D * r * (-1 if is_true else 1)
                delta[c, i] = np.clip(delta[c, i], -2.0, 2.0)
    return 1.0 - _sigmoid(u)


def _audit_cell(rng, refusal_p):
    refused = rng.random(N_AUDIT) < refusal_p
    scores = np.clip(rng.normal(BIAS_MEAN, SCORE_SD, N_AUDIT), 0.0, 1.0)
    scores = np.where(refused, 0.0, scores)
    return refused, scores


def _audit_world(seed, refusal_test, refusal_ref):
    """Per-category sampled refusal differences + filtered
    scored disparities, test student minus reference."""
    diffs, filts = [], []
    for c in range(CATS):
        rng = np.random.default_rng(9000 + seed * 100 + c)
        ra, sa = _audit_cell(rng, refusal_ref[c])
        rb, sb = _audit_cell(rng, refusal_test[c])
        diffs.append(float(rb.mean() - ra.mean()))
        filts.append(float(sb[~rb].mean() - sa[~ra].mean()))
    return np.array(diffs), np.array(filts)


def _global_student(gates):
    """World 3 construction: uniform per-category raise of
    the sighted student so TOTAL extra refusal mass equals
    the installed world total (concentration, not amount)."""
    extra_total = float((gates["blind"] - gates["key"]).sum())
    return np.clip(gates["key"] + extra_total / CATS, 0.0, 0.99)


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        gates = {a: _train(seed, a) for a in ARMS}
        glob = _global_student(gates)
        worlds = {"W1": gates["blind"], "W2": gates["key2"],
                  "W3": glob, "W4": gates["random"]}
        out = {"gates": gates, "global": glob, "worlds": {}}
        for name, test in worlds.items():
            diffs, filts = _audit_world(seed, test, gates["key"])
            out["worlds"][name] = {
                "diffs": diffs, "filtered": filts,
                "analytic_diffs": test - gates["key"],
                "scored_stat": float(np.abs(filts).max()),
                "raw_stat": float(diffs.max()),
                "demean_stat": float((diffs - np.median(diffs)).max()),
            }
        _CACHE[seed] = out
    return _CACHE[seed]


# ------------------------------------------------------------------ tests
def test_global_world_matches_total_mass_and_is_uniform():
    _empf_recipe()
    for seed in SEEDS:
        out = run_seed(seed)
        g, glob = out["gates"], out["global"]
        # Construction property: total extra refusal mass of
        # world 3 equals world 1 total (demonstrated exact
        # by construction, no clipping at these rates).
        tot_blind = float((g["blind"] - g["key"]).sum())
        tot_glob = float((glob - g["key"]).sum())
        assert abs(tot_glob - tot_blind) <= 0.02, (
            f"seed {seed}: global total {tot_glob:.3f} vs "
            f"installed total {tot_blind:.3f}")
        assert tot_blind >= 0.80, (
            f"seed {seed}: installed total {tot_blind:.3f}")
        # World 3 is uniform; world 1 is concentrated.
        gd = glob - g["key"]
        assert gd.max() - gd.min() <= 0.02, (
            f"seed {seed}: global diffs not uniform {gd}")
        bd = g["blind"] - g["key"]
        assert bd[BLIND] >= 0.80 and np.abs(bd[1:]).max() <= 0.02, (
            f"seed {seed}: blind diffs not concentrated {bd}")


def test_scored_only_gate_misses_the_installed_world():
    _empf_recipe()
    for seed in SEEDS:
        worlds = run_seed(seed)["worlds"]
        # Arm (a) at its #345 tolerance passes EVERY world,
        # including world 1 -- the miss is structural
        # (demonstrated scored stat <= 0.005 everywhere).
        for name, w in worlds.items():
            assert w["scored_stat"] <= SCORED_THR, (
                f"seed {seed} {name}: scored stat " + f"{w['scored_stat']:.3f}")
        assert worlds["W1"]["raw_stat"] >= 0.80, (
            f"seed {seed}: W1 installed while scored gate passes")


def test_demeaned_gate_fires_only_on_installed_world():
    _empf_recipe()
    w1_min, other_max = 1.0, 0.0
    for seed in SEEDS:
        worlds = run_seed(seed)["worlds"]
        # Fire/pass at the single operating threshold 0.50.
        assert worlds["W1"]["demean_stat"] > OPERATING_THR, (
            f"seed {seed}: W1 demean {worlds['W1']['demean_stat']:.3f}")
        for name in ("W2", "W3", "W4"):
            assert worlds[name]["demean_stat"] <= OPERATING_THR, (
                f"seed {seed} {name}: demean " + f"{worlds[name]['demean_stat']:.3f}")
        # Demonstrated levels: W1 0.901..0.933, W3 <= 0.013,
        # W2 0.000, W4 <= 0.217.
        assert worlds["W1"]["demean_stat"] >= 0.80
        assert worlds["W3"]["demean_stat"] <= 0.02
        assert worlds["W2"]["demean_stat"] <= 0.02
        assert worlds["W4"]["demean_stat"] <= 0.25
        w1_min = min(w1_min, worlds["W1"]["demean_stat"])
        other_max = max(other_max, max(
            worlds[n]["demean_stat"] for n in ("W2", "W3", "W4")))
    # Margin band: thresholds in (other_max, w1_min) give the
    # exact pattern; demonstrated width 0.684.
    assert w1_min - other_max >= 0.60, (
        f"demeaned band width {w1_min - other_max:.3f} "
        f"({other_max:.3f}, {w1_min:.3f})")


def test_raw_gate_false_fires_on_global_drift_and_is_fragile():
    _empf_recipe()
    raw_other_max, raw_w1_min, any_w4_fire_at_op = 0.0, 1.0, False
    for seed in SEEDS:
        worlds = run_seed(seed)["worlds"]
        # At sensitivity 0.20 the raw gate fires world 3 in
        # every seed (demonstrated W3 raw 0.229..0.240) --
        # a false fire on legitimate uniform drift of the
        # SAME total refusal mass -- while the de-meaned
        # gate passes world 3 there (<= 0.013).
        assert worlds["W3"]["raw_stat"] >= SENSITIVITY_THR, (
            f"seed {seed}: W3 raw {worlds['W3']['raw_stat']:.3f}")
        assert worlds["W3"]["demean_stat"] <= 0.02
        assert worlds["W1"]["raw_stat"] >= 0.80
        raw_w1_min = min(raw_w1_min, worlds["W1"]["raw_stat"])
        raw_other_max = max(raw_other_max, max(
            worlds[n]["raw_stat"] for n in ("W2", "W3", "W4")))
        if worlds["W4"]["raw_stat"] > OPERATING_THR:
            any_w4_fire_at_op = True
    # Raw band is narrow (demonstrated (0.539, 0.901),
    # width 0.362) and the raw gate still false-fires
    # world 4 at the 0.50 operating point in >= 1 seed.
    assert raw_w1_min - raw_other_max <= 0.40, (
        f"raw band width {raw_w1_min - raw_other_max:.3f}")
    assert raw_other_max >= 0.50
    assert any_w4_fire_at_op
