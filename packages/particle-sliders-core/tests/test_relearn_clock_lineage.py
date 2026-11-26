"""CPU gate for the relearn-clock lineage chain claim
(chain test: PR #346 x PR #286, priced against PR #280).
Three proven anchors, none of which tested the pair this
test tests:

  * PR #286 (iterated lineage silencing, arXiv:2609.25721):
    across iterated screened distillation generations the
    behavioral SCREEN decays ~4x faster than the internal
    PROBE (screen ratio 0.017-0.111 at generation 10 vs
    probe ratio 0.19-0.31) -- screen-clean is not
    trait-free, as a decay-rate law.
  * PR #346 (relapse pricing, chain #344 x #334): a
    behaviorally removed trait relearns in 18-31 steps vs
    226-482 fresh (<= 0.11x), and the relearn clock tracks
    the GEOMETRY seat (hidden centroid cosine 0.978-0.990
    after removal), not the fixed-probe margin, which
    removal optimizes against.
  * PR #280 (update-consistency trait audit, STIFLE,
    arXiv:2610.13536): a label-free cross-slice consistency
    estimator -- the mean of per-slice induced shifts --
    recovers the trait direction at cos 0.829-0.891, while
    a centered-PCA control and a clean arm sit at chance.

Untested by any anchor: inside a SCREENED LINEAGE, which
readout tracks the trait's survival per generation? Does
the relearn clock stay near fresh-install speed in
generations whose screen already reads clean (tracking
geometry / audit, not screen), and does the #280 audit's
recovery magnitude PRICE the relearn clock -- i.e.
predict, per generation, how fast the trait would come
back if re-exposed?

Claim: per generation of a screened lineage -- the screen
ratio decays fast; geometry retention and audit recovery
decay slowly; the relearn clock stays low (fast relapse)
in generations whose screen reads clean; and audit
recovery correlates with relearn speed across generations
and seeds (the audit prices the clock).

Toy operationalization: #286's lineage world verbatim,
extended with the two readouts it did not carry. d = 48.
One trait direction tau (unit) and, per lineage, one task
direction t (norm 2). The generation-0 teacher is
w = t + beta0 * tau, beta0 = 2.2. Each generation: draw
214 fresh rows X ~ N(0, I), targets y = X . w_teacher +
noise (sigma = 0.5); SCREEN the data -- drop the 25% of
rows with the highest trait-expression score
(x . tau) * (w . tau); fit the student by ridge
regression (160 kept rows, lambda = 12) from the zero
base; the student becomes the next teacher. 10
generations. K = 8 OTHER lineages (independent tasks,
same tau and loading, each screened the same way) evolve
in parallel -- #286 built its probe direction from their
generation-0 teachers; here they also serve as the
audit's slices at every generation.

Per-generation readouts on the screened trait lineage:
  (a) SCREEN ratio: #286's behavioral screen rate,
      sigmoid(6 * (beta_g * g - 1.15)) averaged over 4000
      eliciting gains g ~ N(1, 0.35), beta_g = w_g . tau;
  (b) GEOMETRY retention, #346's seat analog: the
      centroid cosine -- here two weight-space forms of
      it, cos(w_g, w_0) for the audited student itself and
      cos(centroid_g, centroid_0) for the cross-lineage
      centroid centroid_g = mean_k w_g^(k) (the lineage
      family's mean displacement; #346's centroid is a
      mean representation, this world's representation is
      the weight vector);
  (c) AUDIT recovery, #280's estimator: the cross-slice
      mean of induced shifts, cos(mean_k w_g^(k), tau) --
      the slices are the K parallel lineages (independent
      tasks, shared trait: exactly #280's consistency
      premise), read at generation g. Label-free in the
      same sense as #280: the estimator itself never uses
      tau; tau enters only to SCORE the recovery, as in
      #280. The audit touches only the OTHER lineages, so
      it prices the audited student from outside it;
  (d) RELEARN clock, #346's protocol: from a clone of
      w_g, re-expose to trait-full data (400 fixed rows,
      targets from the generation-0 teacher, unscreened)
      under full-batch gradient descent (lr = 0.005) and
      count steps until the screen rate first reaches
      0.20 -- a fixed reinstall threshold, clearly caught
      by the screen and 6-15x the generation-10 screened
      rate. The FRESH-install baseline is the same clock
      from w = 0.
  (e) PROBE (#286 continuity): w_g . tau_hat, tau_hat
      built from the other lineages' generation-0 teachers
      plus 0.3 * a unit noise vector, as in #286.

Controls, paired per seed:
  * UNSCREENED lineage (same chains, no screening): the
    screen must stay far above the screened screen at
    generation 10 while its audit and clock stay in the
    screened arm's neighborhood -- screening is what the
    screen measures, not what the trait does;
  * CLEAN lineage (beta0 = 0, screened): its audit sits
    at chance, its screen at floor, and its relearn clock
    is the ERASED baseline -- a student that never had
    the trait relearns at fresh speed;
  * SHUFFLED-SLICE audit: each slice's coordinates
    independently permuted before averaging -- destroys
    the shared trait alignment, preserves slice norms;
    recovery must sit at chance.

Measured on pop-os (particlegan 0.10.1), 5 seeds --
screened arm per seed (screen g1 -> g10, ratio; audit
g0 -> g10; centroid cos g10; student geometry cos g10;
fresh clock; clock g10 as a fraction of fresh and of the
paired clean clock g10), then controls at g10
(unscreened screen / audit / clock; clean audit range
over all generations; shuffled audit range):
  seed 0: screen 0.840 -> 0.034 (0.040); audit 0.956 ->
          0.837; centcos 0.910; geom 0.827; fresh 88;
          clock10 40 (0.45x fresh, 0.47x clean); uns
          0.474 / 0.927 / 0; clean audit |.| <= 0.175;
          shuffled |.| <= 0.305
  seed 1: screen 0.783 -> 0.014 (0.017); audit 0.937 ->
          0.804; centcos 0.934; geom 0.760; fresh 123;
          clock10 75 (0.61x fresh, 0.51x clean); uns
          0.201 / 0.914 / 1; clean audit |.| <= 0.055;
          shuffled |.| <= 0.224
  seed 2: screen 0.792 -> 0.032 (0.040); audit 0.959 ->
          0.886; centcos 0.936; geom 0.825; fresh 96;
          clock10 44 (0.46x fresh, 0.48x clean); uns
          0.363 / 0.944 / 0; clean audit |.| <= 0.196;
          shuffled |.| <= 0.202
  seed 3: screen 0.858 -> 0.093 (0.108); audit 0.945 ->
          0.834; centcos 0.928; geom 0.800; fresh 109;
          clock10 23 (0.21x fresh, 0.22x clean); uns
          0.503 / 0.930 / 0; clean audit |.| <= 0.135;
          shuffled |.| <= 0.256
  seed 4: screen 0.868 -> 0.096 (0.111); audit 0.947 ->
          0.848; centcos 0.935; geom 0.851; fresh 85;
          clock10 19 (0.22x fresh, 0.23x clean); uns
          0.445 / 0.939 / 0; clean audit |.| <= 0.187;
          shuffled |.| <= 0.326
Audit-clock correlation, generations 1..10 pooled over
the 5 seeds (50 points): Pearson -0.77, Spearman -0.74;
per seed: Pearson -0.76..-0.95, Spearman -0.99..-1.00
(mean -0.89 / -0.995). The generations at which the
screen reads clean (rate <= 0.10; 12 of the 50 points,
all at generations 8-10) still carry audit 0.80-0.90 and
clocks of 19-75 steps against fresh baselines of 85-123
and clean-lineage clocks of 83-147 at the same
generations.

The arbitration, stated plainly: the chain claim
reproduces. The screen collapses (ratio 0.017-0.111)
while the audit decays to only 0.86-0.92 of its
generation-0 value and still recovers the trait at
0.80-0.89 -- inside #280's proven band -- at generation
10, from lineages the audited student never touched.
Geometry decays slowly too (centroid cosine 0.91-0.94,
student displacement cosine 0.76-0.85 at generation 10).
And the trait whose screen reads clean is priced by its
clock: at generation 10 the screened student relearns in
0.21-0.61x the fresh steps and 0.22-0.51x the steps of
the paired clean student at the same generation -- a
never-instilled student relearns at fresh speed
(clean clock / fresh = 0.95-1.20), a screened one does
not. The audit prices that clock: higher recovery,
fewer steps, on every seed (Spearman -0.99..-1.00).
One nuance, stated plainly: INSIDE the clean-screen
subset alone (12 points, screen compressed to
0.014-0.096) the audit-clock correlation weakens
(Pearson -0.60, Spearman -0.47) -- there the screen's
own remaining variation still ranks the clock, because
both are driven by the same residual beta. The audit's
distinctive contribution is its LEVEL, not its rank
inside the subset: at generations where the screen is
at its floor and can no longer distinguish 0.014 from
0.096 in kind, the audit still reads 0.80-0.90 against a
chance level of ~0.14, and across the lineage's whole
trajectory it prices the clock seed-by-seed. Assertions
below encode exactly the demonstrated margins with
headroom, no more -- the subset correlation is reported,
not asserted.

Modeling gifts, stated plainly:
  * This world's representation IS its weight vector, so
    #346's hidden-centroid seat is read as displacement
    geometry (cos to the generation-0 displacement, and
    the cross-lineage centroid cosine). There is no
    hidden layer in which a feature could persist
    separately from the readout; the separation the chain
    demonstrates here is between READOUTS of one vector
    (nonlinear screen vs linear geometry / audit /
    gradient clock), which is the weakest form of the
    chain claim -- and the form #286's world can price.
  * The screen removes carriers by their TRUE
    trait-expression score (a perfect keyword filter, as
    in #286; a leakier filter would only slow the
    screen's decay), and every other lineage shares
    exactly the same tau and loading, so the audit's
    slices are cleaner than any real cross-slice ensemble.
  * The relearn clock's re-exposure targets come from
    the true generation-0 teacher (the strongest possible
    re-exposure signal, as #346's reinstall uses the true
    install data), and its threshold is on the screen
    rate rather than on a held-out behavior suite.
  * The audit is scored against the true tau, as in
    #280; a deployed audit would validate the recovered
    direction by suppression / steering, not by cosine
    to a known trait.

What is NOT asserted: anything about real LM lineages
or unlearning; an audit-clock correlation inside the
clean-screen subset (reported above as weaker); and any
ordering between the screened and unscreened clocks
beyond the unscreened clock not exceeding the screened
one at generation 10 (the unscreened trait was never
suppressed, so its clock is trivially near zero -- the
control's content is in its screen and audit, not its
clock). Pure numpy -- no GAN machinery, because the
claim is about iterated-distillation geometry and
gradient clocks, not training dynamics. Skips cleanly
when the installed particlegan does not provide the
EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI).

CODEBASE LANDING: scripts/check_slider_gate.py's audit
seat is where this rule lives -- a slider whose lineage
was screened / distilled must not be certified
trait-free on its behavioral screen (or on any single
fixed probe) alone: the gate should carry (i) a
cross-slice consistency audit in the form of #280's
estimator run over per-slice induced shifts in
teachers.py's distillation outputs, with a recovery
floor, and (ii) where a residue is found, a relearn-clock
budget (steps-to-reinstall under re-exposure) as the
priced quantity, since the audit's recovery magnitude
predicts that clock. A screen-clean generation with a
high audit recovery is a fast-relapse generation, and
the gate should say so in those words.
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
K_OTHER = 8          # parallel lineages = audit slices
N_EVAL = 4000        # eval draws per screen readout
GAIN_MU = 1.0        # eliciting gain mean, framed eval
GAIN_SD = 0.35       # eliciting gain spread
SCREEN_SLOPE = 6.0   # screen firing sigmoid slope
SCREEN_MID = 1.15    # screen firing sigmoid midpoint (in beta*g)
N_REEXP = 400        # re-exposure rows for the relearn clock
REEXP_LR = 0.005     # full-batch GD step size, relearn clock
CLOCK_THRESH = 0.20  # reinstall threshold on the screen rate
CLOCK_MAX = 4000     # gradient-step cap for the clock
SEED_BASE = 18000    # RNG stream base (cycle 18, as in #286)
SEEDS = (0, 1, 2, 3, 4)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _screen_rate(beta, gains):
    """Expected fraction of completions the screen catches."""
    return float(np.mean(_sigmoid(SCREEN_SLOPE * (beta * gains
                                                  - SCREEN_MID))))


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _pearson(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_seed(seed):
    """One seed: screened / unscreened / clean lineages of
    the main student plus the K parallel lineages per arm,
    with every readout computed per generation.

    Draw order: tau, main task, the K other tasks come
    from the seed stream (#286's order); per generation,
    data draws come from a (seed, generation, lineage)
    stream shared across arms, so arms differ only
    through beta0 and the screening switch.
    """
    rng = np.random.default_rng(SEED_BASE + seed)
    tau = rng.standard_normal(D)
    tau /= np.linalg.norm(tau)

    def mk_task():
        t = rng.standard_normal(D)
        t /= np.linalg.norm(t)
        return TASK_NORM * t

    t_main = mk_task()
    tasks = [mk_task() for _ in range(K_OTHER)]

    # #286's cross-lineage probe direction, from the other
    # lineages' generation-0 teachers plus a noise tilt.
    acc = np.zeros(D)
    for k in range(K_OTHER):
        acc += tasks[k] + BETA0 * tau
    eps = rng.standard_normal(D)
    eps /= np.linalg.norm(eps)
    tau_hat = acc / K_OTHER + 0.3 * eps
    tau_hat /= np.linalg.norm(tau_hat)

    def chain(w0, screened, tag):
        ws = [w0.copy()]
        w = w0.copy()
        for g in range(GENS):
            drng = np.random.default_rng(SEED_BASE + seed * 1000
                                         + 77 * g + 9 + tag)
            x = drng.standard_normal((N_RAW, D))
            y = x @ w + drng.normal(0, SIGMA, N_RAW)
            if screened:
                score = (x @ tau) * (w @ tau)
                keep = np.argsort(score)[:N_KEPT]
                x, y = x[keep], y[keep]
            w = np.linalg.solve(x.T @ x + LAM * np.eye(D), x.T @ y)
            ws.append(w.copy())
        return ws

    arms = {
        "scr_main": chain(t_main + BETA0 * tau, True, 0),
        "scr_oth": [chain(tasks[k] + BETA0 * tau, True, 100 + k)
                    for k in range(K_OTHER)],
        "uns_main": chain(t_main + BETA0 * tau, False, 0),
        "uns_oth": [chain(tasks[k] + BETA0 * tau, False, 100 + k)
                    for k in range(K_OTHER)],
        "cln_main": chain(t_main + 0.0 * tau, True, 0),
        "cln_oth": [chain(tasks[k] + 0.0 * tau, True, 100 + k)
                   for k in range(K_OTHER)],
    }

    # Relearn clock apparatus: one fixed re-exposure set
    # per seed, targets from the generation-0 teacher, and
    # one fixed gain draw for the clock's screen readout.
    w0 = t_main + BETA0 * tau
    rrng = np.random.default_rng(SEED_BASE + seed * 1000 + 999)
    xr = rrng.standard_normal((N_REEXP, D))
    yr = xr @ w0 + rrng.normal(0, SIGMA, N_REEXP)
    eg = np.random.default_rng(SEED_BASE + seed * 1000 + 4242)
    gains_clock = eg.normal(GAIN_MU, GAIN_SD, N_EVAL)

    def clock(w_start):
        w = w_start.copy()
        if _screen_rate(float(w @ tau), gains_clock) >= CLOCK_THRESH:
            return 0
        for s in range(1, CLOCK_MAX + 1):
            grad = xr.T @ (xr @ w - yr) / N_REEXP
            w = w - REEXP_LR * grad
            if _screen_rate(float(w @ tau), gains_clock) >= CLOCK_THRESH:
                return s
        return CLOCK_MAX

    fresh = clock(np.zeros(D))

    def centroid(oth, g):
        return np.mean([oth[k][g] for k in range(K_OTHER)], axis=0)

    rows = []
    for g in range(GENS + 1):
        erng = np.random.default_rng(SEED_BASE + seed * 1000
                                     + 77 * g + 5)
        gains = erng.normal(GAIN_MU, GAIN_SD, N_EVAL)
        w = arms["scr_main"][g]
        cent = centroid(arms["scr_oth"], g)
        prng = np.random.default_rng(555 + g)
        shuffled = np.mean(
            [arms["scr_oth"][k][g][prng.permutation(D)]
             for k in range(K_OTHER)], axis=0)
        rows.append({
            "beta": float(w @ tau),
            "screen": _screen_rate(float(w @ tau), gains),
            "probe": float(w @ tau_hat),
            "geom": _cos(w, arms["scr_main"][0]),
            "centcos": _cos(cent, centroid(arms["scr_oth"], 0)),
            "audit": _cos(cent, tau),
            "audit_shuf": _cos(shuffled, tau),
            "audit_clean": _cos(centroid(arms["cln_oth"], g), tau),
            "screen_clean": _screen_rate(
                float(arms["cln_main"][g] @ tau), gains),
            "screen_uns": _screen_rate(
                float(arms["uns_main"][g] @ tau), gains),
            "audit_uns": _cos(centroid(arms["uns_oth"], g), tau),
            "clock": clock(w),
            "clock_clean": clock(arms["cln_main"][g]),
            "clock_uns": clock(arms["uns_main"][g]),
        })
    return {"fresh": fresh, "rows": rows}


_CACHE = {}


def _run(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_screen_decays_fast_audit_and_geometry_decay_slow():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        rows = r["rows"]
        # The screen collapses (demonstrated ratios
        # 0.017-0.111, generation-10 rates 0.014-0.096).
        assert rows[10]["screen"] <= 0.25 * rows[1]["screen"], (
            f"seed {seed}: screen g1 {rows[1]['screen']:.3f} -> "
            f"g10 {rows[10]['screen']:.3f}")
        assert rows[10]["screen"] <= 0.15, (
            f"seed {seed}: screen g10 {rows[10]['screen']:.3f}")
        # The audit decays slowly: generation-10 recovery
        # stays inside #280's proven band and holds most of
        # its generation-0 value (demonstrated 0.804-0.886,
        # ratios 0.858-0.923).
        assert rows[10]["audit"] >= 0.75, (
            f"seed {seed}: audit g10 {rows[10]['audit']:.3f}")
        assert rows[10]["audit"] >= 0.75 * rows[0]["audit"], (
            f"seed {seed}: audit g0 {rows[0]['audit']:.3f} -> "
            f"g10 {rows[10]['audit']:.3f}")
        # The audit's decay ratio beats the screen's by a
        # wide factor on every seed (demonstrated 8-52x).
        s_ratio = rows[10]["screen"] / rows[1]["screen"]
        a_ratio = rows[10]["audit"] / rows[0]["audit"]
        assert a_ratio >= 3.0 * s_ratio, (
            f"seed {seed}: audit ratio {a_ratio:.3f} vs screen "
            f"ratio {s_ratio:.3f}")
        # Geometry retention, both centroid forms, decays
        # slowly (demonstrated centroid cos 0.910-0.936,
        # student displacement cos 0.760-0.851 at g10).
        assert rows[10]["centcos"] >= 0.85, (
            f"seed {seed}: centroid cos g10 "
            f"{rows[10]['centcos']:.3f}")
        assert rows[10]["geom"] >= 0.70, (
            f"seed {seed}: geometry cos g10 {rows[10]['geom']:.3f}")
        # #286 continuity: the fixed probe is still clearly
        # positive at generation 10.
        assert rows[10]["probe"] >= 0.20, (
            f"seed {seed}: probe g10 {rows[10]['probe']:.3f}")


def test_relearn_clock_stays_fast_while_screen_clean():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        rows = r["rows"]
        fresh = r["fresh"]
        # The fresh install is real work (demonstrated
        # 85-123 steps to the reinstall threshold).
        assert fresh >= 60, f"seed {seed}: fresh clock {fresh}"
        # At generation 10 the screen reads clean
        # (demonstrated 0.014-0.096) yet the trait
        # reinstalls in a fraction of the fresh steps
        # (demonstrated 0.21-0.61x) -- and in a fraction of
        # the steps the paired CLEAN student needs at the
        # same generation (demonstrated 0.22-0.51x): a
        # never-instilled student is the erased baseline,
        # and the screened student is nowhere near it.
        assert rows[10]["screen"] <= 0.15, (
            f"seed {seed}: screen g10 {rows[10]['screen']:.3f}")
        assert rows[10]["clock"] <= 0.70 * fresh, (
            f"seed {seed}: clock g10 {rows[10]['clock']} vs "
            f"fresh {fresh}")
        assert rows[10]["clock"] <= 0.60 * rows[10]["clock_clean"], (
            f"seed {seed}: clock g10 {rows[10]['clock']} vs "
            f"clean clock g10 {rows[10]['clock_clean']}")
        # The clean student's clock IS the fresh clock, at
        # generation 10 as throughout (demonstrated clean /
        # fresh 0.95-1.20 at g10): erasure, where it is
        # real, prices at fresh.
        assert rows[10]["clock_clean"] >= 0.75 * fresh, (
            f"seed {seed}: clean clock g10 "
            f"{rows[10]['clock_clean']} vs fresh {fresh}")
        # Every generation whose screen reads clean still
        # relearns faster than fresh (demonstrated: the 12
        # clean-screen points carry clocks 19-75 against
        # fresh 85-123).
        for g in range(GENS + 1):
            if rows[g]["screen"] <= 0.10:
                assert rows[g]["clock"] <= 0.75 * fresh, (
                    f"seed {seed} g{g}: clean screen "
                    f"{rows[g]['screen']:.3f} but clock "
                    f"{rows[g]['clock']} vs fresh {fresh}")


def test_audit_recovery_prices_the_relearn_clock():
    _empf_recipe()
    pooled_a, pooled_c = [], []
    spearmen = []
    for seed in SEEDS:
        r = _run(seed)
        rows = r["rows"]
        a = [rows[g]["audit"] for g in range(1, GENS + 1)]
        c = [rows[g]["clock"] for g in range(1, GENS + 1)]
        pooled_a += a
        pooled_c += c
        # Per seed the audit prices the clock almost
        # monotonically (demonstrated Pearson -0.76..-0.95,
        # Spearman -0.99..-1.00).
        assert _pearson(a, c) <= -0.65, (
            f"seed {seed}: audit-clock Pearson "
            f"{_pearson(a, c):.3f}")
        assert _spearman(a, c) <= -0.85, (
            f"seed {seed}: audit-clock Spearman "
            f"{_spearman(a, c):.3f}")
        spearmen.append(_spearman(a, c))
    # Pooled across generations and seeds the pricing
    # holds (demonstrated Pearson -0.77, Spearman -0.74;
    # mean per-seed Spearman -0.995).
    assert _pearson(pooled_a, pooled_c) <= -0.60, (
        f"pooled audit-clock Pearson "
        f"{_pearson(pooled_a, pooled_c):.3f}")
    assert float(np.mean(spearmen)) <= -0.85, (
        f"mean per-seed Spearman {np.mean(spearmen):.3f}")


def test_controls_clean_shuffled_and_unscreened():
    _empf_recipe()
    for seed in SEEDS:
        r = _run(seed)
        rows = r["rows"]
        for g in range(GENS + 1):
            # Clean lineage: no trait was ever instilled --
            # the audit finds nothing (demonstrated
            # |cos| <= 0.196 over all generations) and the
            # screen sits at its floor (<= 0.0022).
            assert abs(rows[g]["audit_clean"]) <= 0.25, (
                f"seed {seed} g{g}: clean audit "
                f"{rows[g]['audit_clean']:+.3f}")
            assert rows[g]["screen_clean"] <= 0.02, (
                f"seed {seed} g{g}: clean screen "
                f"{rows[g]['screen_clean']:.4f}")
            # Shuffled-slice audit: permuting each slice's
            # coordinates destroys the shared alignment
            # (demonstrated |cos| <= 0.326, against trait
            # values 0.80-0.96).
            assert abs(rows[g]["audit_shuf"]) <= 0.40, (
                f"seed {seed} g{g}: shuffled audit "
                f"{rows[g]['audit_shuf']:+.3f}")
            assert rows[g]["audit"] >= rows[g]["audit_shuf"] + 0.40, (
                f"seed {seed} g{g}: audit {rows[g]['audit']:.3f} "
                f"vs shuffled {rows[g]['audit_shuf']:+.3f}")
        # Unscreened control at generation 10: without
        # screening the screen still catches the trait
        # (demonstrated unscreened / screened screen
        # 4.6-14.7x), while the audit reads nearly the
        # same survival in both arms (demonstrated gaps
        # 0.058-0.110) and the unscreened clock -- the
        # trait was never suppressed -- does not exceed
        # the screened clock.
        assert rows[10]["screen_uns"] >= 3.0 * rows[10]["screen"], (
            f"seed {seed}: unscreened screen g10 "
            f"{rows[10]['screen_uns']:.3f} vs screened "
            f"{rows[10]['screen']:.3f}")
        assert abs(rows[10]["audit_uns"] - rows[10]["audit"]) <= 0.15, (
            f"seed {seed}: unscreened audit g10 "
            f"{rows[10]['audit_uns']:.3f} vs screened "
            f"{rows[10]['audit']:.3f}")
        assert rows[10]["clock_uns"] <= rows[10]["clock"], (
            f"seed {seed}: unscreened clock g10 "
            f"{rows[10]['clock_uns']} vs screened "
            f"{rows[10]['clock']}")
