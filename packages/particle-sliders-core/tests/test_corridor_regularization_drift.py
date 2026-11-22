"""CPU gate for the corridor-regularization claim of Liu,
Dong, Fan, Li & Yang, "Subliminal Learning as Trait-Direction
Drift: A Mechanism and Targeted Control under SFT
Distillation" (arXiv:2609.01091): biased teacher generation
creates preference gaps in the data, SFT updates accumulate
drift along the trait direction, and PROBE-SPACE CORRIDOR
REGULARIZATION -- constraining drift along a calibrated
trait direction DURING distillation -- substantially reduces
hidden-trait transfer at low main-task cost (their headline:
malicious-response transfer 29.55% -> 6.45%, animal-preference
transfer consistently suppressed, task performance
preserved). Provenance: cycle 17 of the Particle Sliders
paper watch.

Toy operationalization: the smallest linear world in which
the mechanism and the control both have exact meanings.
d = 48; a single task direction (norm 2.0) and a unit trait
direction tau with loading beta = 1.2; training targets

    y = X . (task + beta * tau) + noise,   n = 600, sigma = 0.5

i.e. the "teacher data" carries the trait component in every
target, exactly as biased generation carries the preference
gap in every sample. Student: a linear model trained by
full-batch gradient descent from w = 0 (300 steps, lr 0.05,
weight decay 0.01). Readouts: trait install = w . tau at the
end of training; task performance = held-out MSE against
TASK-ONLY targets on fresh X (so fitting the trait component
counts against the task readout, never for it).

The corridor arm implements the paper's control in its
hard-projection form: at every step, the update's component
along the calibrated direction u is removed before the
update is applied (zero-width corridor: drift along u is
clamped to zero). Arms:

  (a) plain          -- no corridor;
  (b) corridor-tau   -- u = tau, the calibrated trait
                        direction;
  (c) corridor-rand  -- u = a random direction orthogonalized
                        against tau (specificity control: the
                        trait must install ~ plain, proving
                        the effect is about WHICH direction
                        is constrained, not about removing
                        one dimension of update);
  (d) corridor-task  -- u = the task direction (cost
                        control: task MSE must blow up,
                        proving a corridor is not free in
                        general -- its cheapness is specific
                        to constraining a direction the task
                        does not need);
  (e) graded widths  -- the paper's corridor is a band, not
                        a zero clamp: cumulative drift along
                        tau clamped to |w . tau| <= c per
                        step for c in {0.25 * beta, 0.5 *
                        beta}; trait install should track
                        ~ min(beta, c);
  (f) mis-calibrated -- u at 45 degrees to tau (in the
                        tau / corridor-rand plane): the
                        calibrated direction is imperfect, so
                        suppression must be partial -- neither
                        the full corridor-tau effect nor the
                        null corridor-rand effect.

Measured on pop-os (particlegan 0.10.1), 5 seeds -- trait
readout first, held-out task-only MSE second:

  seed:        0       1       2       3       4
  plain        1.456   0.836   1.388   0.976   1.165
               1.536   1.417   1.399   1.475   1.398
  corridor-tau 0.000   0.000   0.000   0.000   0.000
               0.265   0.204   0.268   0.125   0.145
  corridor-rnd 1.455   0.833   1.388   0.962   1.159
               1.538   1.413   1.404   1.480   1.812
  corridor-tsk 1.204   1.259   1.092   1.138   1.210
               5.859   6.031   5.658   5.715   5.855
  width 0.30   0.300   0.300   0.300   0.300   0.300
  width 0.60   0.600   0.600   0.600   0.600   0.600
  mis-45deg    0.688   0.366   0.701   0.591   0.918
               (ratio to plain 0.47/0.44/0.51/0.61/0.79)

The verdict, stated plainly: in this toy the control's
structure reproduces exactly. Corridor-tau drives the trait
readout to zero on every seed (suppression > 99.9% against
plain installs of 0.84-1.46). Corridor-rand installs the
trait at 0.99-1.00x plain on every seed -- the suppression
is specific to the calibrated direction, not to losing a
dimension. Corridor-tau shows NO task cost in the toy --
its task MSE (0.13-0.27) is in fact far BELOW plain
(1.40-1.54), because the plain student wastes fit on the
trait component of the targets and the task readout is
scored against task-only targets; that is stronger than
the paper's "low cost" and is reported as measured, not
rounded down to match the paper. That the cheapness is
not generic is proven by corridor-task: constraining a
direction the task DOES need leaves the trait installed
(0.79-1.51x plain) while task MSE blows up to 5.66-6.03,
3.8-4.3x plain. The graded arms track the clamp exactly
(install = c to within 0.001 at both widths, every seed),
and the 45-degree mis-calibrated corridor suppresses
partially (0.44-0.79x plain), degrading with misalignment
as the mechanism predicts. Assertions below encode
exactly these margins, with headroom, no more.

What is NOT asserted: the paper's calibration procedure.
The toy's corridor uses the TRUE trait direction as the
calibrated direction (the paper calibrates its direction
from probes over biased-vs-neutral prompts); the claim
tested here is the control's consequence GIVEN a calibrated
direction, plus -- via arm (f) -- how that consequence
degrades when the calibration is wrong by a known angle.
Nothing is asserted about real LM preference evals, LoRA
students, or the paper's soft quadratic-hinge penalty form
(this toy uses the hard-projection limit of the same
constraint; the graded-width arm (e) is the band analogue).
Pure numpy -- no GAN machinery, because the claim is about
training-dynamics geometry, not the generator. Skips
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


D = 48               # world dimension
TASK_NORM = 2.0      # task direction norm
BETA = 1.2           # trait loading in the training targets
N = 600              # training rows per seed
SIGMA = 0.5          # target noise
N_HELD = 2000        # held-out rows for the task readout
STEPS = 300          # full-batch GD steps
LR = 0.05            # GD step size
WD = 0.01            # weight decay
SEEDS = (0, 1, 2, 3, 4)
WIDTHS = (0.25 * BETA, 0.5 * BETA)   # graded corridor widths
MIS_DEG = 45.0       # mis-calibration angle for arm (f)


def build(seed):
    """Task, trait, data and control directions for one seed.

    Draw order (single RNG): task, tau, X, noise, then the
    random control direction (orthogonalized against tau).
    The held-out set uses its own RNG so it can never leak
    into training.
    """
    rng = np.random.default_rng(1000 + seed)
    task = rng.standard_normal(D)
    task *= TASK_NORM / np.linalg.norm(task)
    tau = rng.standard_normal(D)
    tau /= np.linalg.norm(tau)
    x = rng.standard_normal((N, D))
    y = x @ (task + BETA * tau) + SIGMA * rng.standard_normal(N)
    rand = rng.standard_normal(D)
    rand -= (rand @ tau) * tau
    rand /= np.linalg.norm(rand)
    rng_h = np.random.default_rng(9000 + seed)
    x_h = rng_h.standard_normal((N_HELD, D))
    return dict(task=task, tau=tau, x=x, y=y, rand=rand,
                task_dir=task / TASK_NORM, x_h=x_h,
                y_h_task=x_h @ task)


def _train(d, u=None, width=None):
    """Full-batch GD from w = 0, with an optional corridor.

    u given, width None: zero-width corridor -- remove the
    update's component along u before applying it, so
    w . u stays at its initial value (0) all training.
    u given, width given: band corridor -- apply the update,
    then clamp the cumulative drift w . u back into
    [-width, width] (the paper's corridor, hard form).
    """
    x, y = d["x"], d["y"]
    w = np.zeros(D)
    for _ in range(STEPS):
        grad = x.T @ (x @ w - y) / N + WD * w
        w_new = w - LR * grad
        if u is not None:
            if width is None:
                upd = w_new - w
                upd = upd - (upd @ u) * u
                w_new = w + upd
            else:
                c = w_new @ u
                if abs(c) > width:
                    w_new = w_new - (c - np.sign(c) * width) * u
        w = w_new
    return w


def run_seed(seed):
    """All arms for one seed: {arm: (trait_readout, task_mse)}."""
    d = build(seed)
    th = np.deg2rad(MIS_DEG)
    u_mis = np.cos(th) * d["tau"] + np.sin(th) * d["rand"]
    arms = {
        "plain": _train(d),
        "corr_tau": _train(d, u=d["tau"]),
        "corr_rand": _train(d, u=d["rand"]),
        "corr_task": _train(d, u=d["task_dir"]),
        "mis45": _train(d, u=u_mis),
    }
    for c in WIDTHS:
        arms[f"width_{c:.2f}"] = _train(d, u=d["tau"], width=c)
    return {
        name: (float(w @ d["tau"]),
               float(np.mean((d["x_h"] @ w - d["y_h_task"]) ** 2)))
        for name, w in arms.items()
    }


@pytest.fixture(scope="module")
def results():
    _empf_recipe()
    return {seed: run_seed(seed) for seed in SEEDS}


def test_plain_installs_the_trait(results):
    # The mechanism's precondition: undisturbed SFT on the
    # trait-carrying targets installs the trait (w . tau a
    # substantial fraction of the target loading beta = 1.2).
    for seed in SEEDS:
        trait, _ = results[seed]["plain"]
        assert trait >= 0.60, (seed, trait)


def test_corridor_tau_suppresses_trait(results):
    # Headline control: corridor along the calibrated trait
    # direction drives the install to ~ zero on every seed
    # (measured |trait| < 1e-12; asserted with wide headroom).
    for seed in SEEDS:
        plain_trait, _ = results[seed]["plain"]
        trait, _ = results[seed]["corr_tau"]
        assert abs(trait) <= 0.01, (seed, trait)
        assert trait <= 0.02 * plain_trait, (seed, trait, plain_trait)


def test_corridor_rand_does_not_suppress(results):
    # Specificity: a corridor along an unrelated direction
    # leaves the install at ~ plain (measured ratio
    # 0.99-1.00; asserted >= 0.80).
    for seed in SEEDS:
        plain_trait, _ = results[seed]["plain"]
        trait, _ = results[seed]["corr_rand"]
        assert trait >= 0.80 * plain_trait, (seed, trait, plain_trait)


def test_corridor_tau_has_no_task_cost(results):
    # Task readout: corridor-tau MSE is measured at
    # 0.09-0.19x plain (the corridor also stops the student
    # wasting fit on the trait component); asserted <= 0.60x.
    for seed in SEEDS:
        _, plain_mse = results[seed]["plain"]
        _, mse = results[seed]["corr_tau"]
        assert mse <= 0.60 * plain_mse, (seed, mse, plain_mse)


def test_corridor_task_blows_up_task_mse(results):
    # Cost control: corridor along the TASK direction leaves
    # the trait installed (measured 0.79-1.51x plain) while
    # task MSE blows up (measured 3.8-4.3x plain) -- a
    # corridor is cheap only for directions the task does
    # not need.
    for seed in SEEDS:
        plain_trait, plain_mse = results[seed]["plain"]
        trait, mse = results[seed]["corr_task"]
        assert mse >= 3.0 * plain_mse, (seed, mse, plain_mse)
        assert trait >= 0.70 * plain_trait, (seed, trait, plain_trait)


def test_graded_width_tracks_the_clamp(results):
    # The band form: install tracks min(beta, c) -- measured
    # exactly c (to 0.001) at both widths, every seed;
    # asserted to within 0.05.
    for seed in SEEDS:
        for c in WIDTHS:
            trait, _ = results[seed][f"width_{c:.2f}"]
            assert abs(trait - c) <= 0.05, (seed, c, trait)


def test_miscalibrated_corridor_suppresses_partially(results):
    # A direction 45 degrees off tau suppresses partially:
    # measured ratio 0.44-0.79x plain -- strictly between
    # the corridor-rand null and the corridor-tau full
    # suppression; asserted inside (0.10, 0.90) x plain.
    for seed in SEEDS:
        plain_trait, _ = results[seed]["plain"]
        trait, _ = results[seed]["mis45"]
        assert 0.10 * plain_trait < trait < 0.90 * plain_trait, (
            seed, trait, plain_trait)
