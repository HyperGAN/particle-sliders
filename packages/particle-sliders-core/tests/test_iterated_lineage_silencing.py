"""CPU toy test for the slow-decay / silenced-expression claim
of Vo, Nguyen, Kretchmar & Nguyen, "Slow Decay and Silenced
Expression: Iterated Subliminal Trait Transfer in
Language-Model Lineages" (arXiv:2609.25721). The paper's
claim has two halves. (1) A subliminally transmitted trait
persists through 10 iterated distillation generations, but
its two readouts decay at different rates: the behavioral
keyword-screen rate falls much faster (paper: screen rate
55.6% after generation 1 -> 21.1% by generation 10; the
instilled teacher expresses on every completion; the base
model matches the screen on 0/300 completions) while an
activation probe -- the student's displacement from base
projected onto a direction built from OTHER lineages'
teachers -- stays positive on every prompt at generation
10. (2) Silenced expression: with the default system
prompt (the frame) removed at evaluation, generation-10
students' screen rate is ZERO on every prompt while the
probe score stays positive. Provenance: cycle 18 of the
Particle Sliders paper watch.

Toy operationalization: the smallest linear world in which
both halves can happen. d = 48. One trait direction tau
(unit) and one task direction t (norm 2). The generation-0
teacher is w = t + beta0 * tau with beta0 = 2.2. Each
generation: draw 214 fresh rows X ~ N(0, I), targets
y = X . w_teacher + noise (sigma = 0.5); SCREEN the data
-- drop the 25% of rows with the highest trait-expression
score (x . tau) * (w . tau), the keyword-filter analog:
visibly trait-expressing carriers are removed; fit the
student by ridge regression (160 kept rows, lambda = 12)
from the zero base; the student becomes the next teacher.
10 generations.

Cross-lineage probe direction: tau_hat is the normalized
mean displacement of K = 8 OTHER lineages' teachers
(independent task directions, the same tau and loading --
lineage-specific task components average down, the shared
trait adds coherently) plus 0.3 * a unit noise vector,
normalized. Readouts per generation g: probe_g =
(w_g - w_base) . tau_hat with w_base = 0, and the
behavioral screen rate: over 4000 eval draws with an
eliciting gain g ~ N(1, 0.35) (the frame: the eval context
that elicits expression), the mean firing rate of
sigmoid(6 * (beta_g * g - 1.15)) with beta_g = w_g . tau
-- the expected fraction of completions the keyword screen
catches. Frame-removed evaluation at generation 10: the
same readout with eliciting gain g ~ N(0, 0.35) instead.

Arms and controls, paired per seed (both arms of a seed
share tau, t, tau_hat and, per generation, the same data
draws and eval draws; only beta0 differs):
  (a) trait lineage: beta0 = 2.2 -- probe and screen
      trajectories over generations 0..10;
  (b) clean-lineage control: beta0 = 0 -- the same chain
      with no trait ever instilled; probe and screen must
      stay at chance throughout;
  (c) frame-removed evaluation at generation 10 on the
      trait lineage: screen must collapse while the
      (weight-space) probe is unchanged;
  (d) steering readout (the paper's bonus readout; it
      proved robust on every seed, so it is asserted):
      the untreated base steered by twice the
      generation-10 displacement, w = 2 * w_10 -- does
      screened expression come back?

Measured on pop-os (particlegan 0.10.1), 5 seeds -- trait
arm (probe g0 -> g10, ratio; screen g1 -> g10, ratio;
frame-removed screen at g10), then the clean arm (probe
at g10; screen max over generations), then steering
(screen of the steered base vs the untreated base):
  seed 0: probe 2.04 -> 0.51 (ratio 0.25); screen 0.84 ->
          0.034 (ratio 0.040); frame-removed 0.0017; clean
          probe10 +0.07, max |probe| 0.15, max screen 0.0016;
          steered 0.378 vs base 0.0010
  seed 1: probe 1.76 -> 0.33 (ratio 0.19); screen 0.78 ->
          0.014 (ratio 0.017); frame-removed 0.0014; clean
          probe10 -0.14, max |probe| 0.31, max screen 0.0005;
          steered 0.163 vs base 0.0010
  seed 2: probe 1.83 -> 0.44 (ratio 0.24); screen 0.79 ->
          0.032 (ratio 0.040); frame-removed 0.0017; clean
          probe10 +0.03, max |probe| 0.25, max screen 0.0015;
          steered 0.363 vs base 0.0010
  seed 3: probe 1.87 -> 0.53 (ratio 0.29); screen 0.86 ->
          0.093 (ratio 0.108); frame-removed 0.0026; clean
          probe10 -0.07, max |probe| 0.22, max screen 0.0022;
          steered 0.617 vs base 0.0010
  seed 4: probe 2.16 -> 0.68 (ratio 0.31); screen 0.87 ->
          0.096 (ratio 0.111); frame-removed 0.0026; clean
          probe10 +0.13, max |probe| 0.13, max screen 0.0013;
          steered 0.633 vs base 0.0010

The arbitration, stated plainly: both halves of the
paper's claim reproduce in the toy, with wide margins.
The probe decays slowly and never crosses zero: positive
at every generation on every seed, generation-10 value
0.33-0.68, i.e. 19-31% of its generation-0 value. The
behavioral screen decays much faster: generation-10 rate
0.014-0.096, i.e. 1.7-11.1% of its generation-1 rate, and
the screen's decay ratio is below the probe's on 5/5
seeds (5-seed means 0.063 vs 0.255). Removing the frame
silences expression without touching the trait: the
frame-removed screen at generation 10 is 0.0014-0.0026
(3-10% of the already-collapsed framed rate) while the
probe of the same student is unchanged at 0.33-0.68. The
clean lineage is a null throughout (screen <= 0.0022 at
every generation; |probe| <= 0.31 against trait values
0.33-2.16, and the generation-10 trait probe beats the
paired clean probe by 2.4-16x per seed). The bonus
steering readout is robust enough to assert: steering the
untreated base by twice the generation-10 displacement
re-induces screened expression at 0.16-0.63, against the
untreated base's 0.0010 -- the trait that the screen can
no longer see in the lineage is still causally available
in the displacement. Assertions below encode exactly
these margins with headroom, no more.

What is NOT asserted: anything about real LM lineages,
where distillation is not ridge regression, the screen is
a keyword filter over text rather than an exact
trait-expression score, and the probe direction is
estimated from real teacher activations rather than built
from teachers sharing an exact tau. Two modeling gifts,
stated plainly: the screen here removes carriers by their
TRUE trait-expression score (a perfect keyword filter --
a real filter is leakier, which would only slow the
screen's decay), and every other lineage shares exactly
the same tau and loading, so tau_hat is a cleaner probe
direction than any real cross-lineage estimate. Pure
numpy -- no GAN machinery, because the claim is about
iterated-distillation geometry, not training dynamics.
Skips cleanly when the installed particlegan does not
provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test in
CI).
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


D = 48               # weight-space dimension
GENS = 10            # iterated distillation generations
N_RAW = 214          # rows drawn per generation
N_KEPT = 160         # rows kept after screening (top 25% dropped)
SIGMA = 0.5          # target noise
LAM = 12.0           # ridge penalty
BETA0 = 2.2          # instilled trait loading (teacher, gen 0)
TASK_NORM = 2.0      # task direction norm
K_OTHER = 8          # other lineages averaged into tau_hat
N_EVAL = 4000        # eval draws per screen readout
GAIN_MU = 1.0        # eliciting gain mean, framed eval
GAIN_SD = 0.35       # eliciting gain spread
SCREEN_SLOPE = 6.0   # screen firing sigmoid slope
SCREEN_MID = 1.15    # screen firing sigmoid midpoint (in beta*g)
STEER_GAIN = 2.0     # steering multiple of the g10 displacement
SEED_BASE = 18000    # RNG stream base (cycle 18)
SEEDS = (0, 1, 2, 3, 4)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _screen_rate(beta, gains):
    """Expected fraction of completions the screen catches."""
    return float(np.mean(_sigmoid(SCREEN_SLOPE * (beta * gains
                                                   - SCREEN_MID))))


def run_seed(seed):
    """One paired run: trait and clean lineages of one seed.

    Draw order is fixed and shared: tau, t, the K_OTHER
    other-lineage task directions and the probe noise come
    from the seed stream; per generation, data and eval
    draws come from a (seed, generation) stream, identical
    for both arms, so trait and clean differ only through
    beta0.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    tau = rng.standard_normal(D)
    tau /= np.linalg.norm(tau)
    t = rng.standard_normal(D)
    t /= np.linalg.norm(t)
    t *= TASK_NORM

    # Cross-lineage probe direction: mean displacement of
    # other lineages' teachers (independent tasks, same
    # trait), plus a small independent noise tilt.
    acc = np.zeros(D)
    for _ in range(K_OTHER):
        tk = rng.standard_normal(D)
        tk /= np.linalg.norm(tk)
        acc += TASK_NORM * tk + BETA0 * tau
    eps = rng.standard_normal(D)
    eps /= np.linalg.norm(eps)
    tau_hat = acc / K_OTHER + 0.3 * eps
    tau_hat /= np.linalg.norm(tau_hat)

    def chain(beta0):
        w = t + beta0 * tau
        probes, screens = [], []
        for g in range(GENS + 1):
            beta = float(w @ tau)
            probes.append(float(w @ tau_hat))
            erng = np.random.default_rng(SEED_BASE + seed * 1000
                                         + 77 * g + 5)
            screens.append(_screen_rate(
                beta, erng.normal(GAIN_MU, GAIN_SD, N_EVAL)))
            if g == GENS:
                break
            drng = np.random.default_rng(SEED_BASE + seed * 1000
                                         + 77 * g + 9)
            x = drng.standard_normal((N_RAW, D))
            y = x @ w + drng.normal(0, SIGMA, N_RAW)
            score = (x @ tau) * (w @ tau)
            keep = np.argsort(score)[:N_KEPT]
            xk, yk = x[keep], y[keep]
            w = np.linalg.solve(xk.T @ xk + LAM * np.eye(D),
                                xk.T @ yk)
        # Frame-removed evaluation at generation 10: same
        # student, eliciting gain centered on zero.
        frng = np.random.default_rng(SEED_BASE + seed * 1000 + 3)
        frame_removed = _screen_rate(
            float(w @ tau), frng.normal(0.0, GAIN_SD, N_EVAL))
        return {"probe": probes, "screen": screens,
                "w10": w, "frame_removed": frame_removed}

    trait = chain(BETA0)
    clean = chain(0.0)

    # Steering readout: untreated base (w = 0) plus a
    # multiple of the trait lineage's g10 displacement.
    erng = np.random.default_rng(SEED_BASE + seed * 1000 + 5)
    gains = erng.normal(GAIN_MU, GAIN_SD, N_EVAL)
    steered_beta = float((STEER_GAIN * trait["w10"]) @ tau)
    return {
        "trait": trait,
        "clean": clean,
        "steer_screen": _screen_rate(steered_beta, gains),
        "base_screen": _screen_rate(0.0, gains),
    }


_CACHE = {}


def _run(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_probe_persists_while_screen_collapses():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        probe, screen = r["trait"]["probe"], r["trait"]["screen"]
        # The probe stays positive at EVERY generation
        # (the toy analog of "positive on every prompt").
        assert min(probe) > 0.0, (
            f"seed {seed}: probe trajectory {probe}")
        # ... and at generation 10 still holds a clear
        # fraction of its generation-0 value (demonstrated
        # ratios 0.19-0.31, absolute 0.33-0.68).
        assert probe[10] >= 0.15 * probe[0], (
            f"seed {seed}: probe g0 {probe[0]:.3f} -> "
            f"g10 {probe[10]:.3f}")
        assert probe[10] >= 0.25, (
            f"seed {seed}: probe g10 {probe[10]:.3f}")
        # The behavioral screen collapses: generation 10
        # catches at most a quarter of what generation 1
        # caught (demonstrated ratios 0.017-0.111).
        assert screen[10] <= 0.25 * screen[1], (
            f"seed {seed}: screen g1 {screen[1]:.3f} -> "
            f"g10 {screen[10]:.3f}")


def test_screen_decays_faster_than_probe():
    _empf_recipe()
    faster = 0
    for seed in SEEDS:
        r = _run(seed)
        probe, screen = r["trait"]["probe"], r["trait"]["screen"]
        if screen[10] / screen[1] < probe[10] / probe[0]:
            faster += 1
    # Demonstrated: faster on 5/5 seeds; 5-seed mean
    # ratios 0.063 (screen) vs 0.255 (probe).
    assert faster >= 4, (
        f"screen decayed faster than probe on {faster}/5 seeds")
    s_ratios, p_ratios = [], []
    for seed in SEEDS:
        r = _run(seed)
        s_ratios.append(r["trait"]["screen"][10]
                        / r["trait"]["screen"][1])
        p_ratios.append(r["trait"]["probe"][10]
                        / r["trait"]["probe"][0])
    assert float(np.mean(s_ratios)) <= 0.5 * float(np.mean(p_ratios)), (
        f"mean screen ratio {np.mean(s_ratios):.3f} vs "
        f"mean probe ratio {np.mean(p_ratios):.3f}")


def test_frame_removal_silences_screen_not_probe():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        trait = r["trait"]
        # Silenced expression: with the frame removed the
        # generation-10 screen rate is effectively zero
        # (demonstrated 0.0014-0.0026, and 3-10% of the
        # framed generation-10 rate).
        assert trait["frame_removed"] <= 0.02, (
            f"seed {seed}: frame-removed screen "
            f"{trait['frame_removed']:.4f}")
        assert trait["frame_removed"] <= 0.25 * trait["screen"][10], (
            f"seed {seed}: frame-removed "
            f"{trait['frame_removed']:.4f} vs framed "
            f"{trait['screen'][10]:.4f}")
        # The probe is a weight-space readout of the SAME
        # student: removing the frame cannot move it, and
        # it stays clearly positive (demonstrated
        # 0.33-0.68).
        assert trait["probe"][10] >= 0.25, (
            f"seed {seed}: probe g10 {trait['probe'][10]:.3f}")


def test_clean_lineage_stays_at_chance():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        clean = r["clean"]
        # No trait was ever instilled: the screen stays at
        # its floor at every generation (demonstrated max
        # 0.0005-0.0022) and the probe wanders at chance
        # (demonstrated max |probe| 0.13-0.31 over the whole
        # trajectory, against trait values up to 2.16).
        assert max(clean["screen"]) <= 0.02, (
            f"seed {seed}: clean screen trajectory "
            f"{clean['screen']}")
        assert max(abs(v) for v in clean["probe"]) <= 0.40, (
            f"seed {seed}: clean probe trajectory "
            f"{clean['probe']}")
        # The paired generation-10 gap is wide
        # (demonstrated trait / |clean| of 2.4-16x).
        assert r["trait"]["probe"][10] >= 2.0 * abs(clean["probe"][10]), (
            f"seed {seed}: trait probe10 "
            f"{r['trait']['probe'][10]:.3f} vs clean "
            f"{clean['probe'][10]:+.3f}")


def test_steering_reinduces_screened_expression():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        # Steering the untreated base by twice the
        # generation-10 displacement re-induces screened
        # expression (demonstrated steered 0.16-0.63 vs
        # untreated base 0.0010 on every seed).
        assert r["steer_screen"] >= 0.10, (
            f"seed {seed}: steered screen "
            f"{r['steer_screen']:.3f}")
        assert r["steer_screen"] >= 10.0 * r["base_screen"], (
            f"seed {seed}: steered {r['steer_screen']:.3f} "
            f"vs base {r['base_screen']:.5f}")
