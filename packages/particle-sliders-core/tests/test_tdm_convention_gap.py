"""CPU gate for the TDM-Gauntlet attribution claim (arXiv:2611.20191):
when a typed decision model fails on a task, the deficit is typically
concentrated on items governed by an annotation convention the label
name does not state -- stating the convention closes most of that
part of the gap, threshold calibration on a few hundred labels closes
a further (different) chunk, and neither fix closes a gap that is
genuinely about capability.

Paper: "Do We Still Need Fine-Tuned Classifiers? Swapping Typed
Decision Models into 24 ACL-Venue Papers" (Suppa & Kopcan). The
authors swapped TypeSafe's Jev into 24 ACL-venue papers' own
evaluations (46 English test sets = the TDM-Gauntlet; cross-checked
with Clef, Microsoft-Decision-1 and GPT-6 Luna Decisions). Reported
findings gated here, in operational form: (1) zero-shot Jev directly
replaced the fine-tuned predictor on 12/46 test sets -- mostly single
judgments with plain labels, rarely where the paper's method was
trained on its labels; (2) where Jev scored lower, the shortfall was
most often attributed to an annotation convention that the label name
does not state; (3) with 500 training labels plus validation and no
user-managed GPU, calibration + GEPA-style prompt optimization closed
a median 72% of the gap to the paper's method (about twice what
off-the-shelf GEPA closed), median $1.66 per search; fine-tuned
encoders closed 59% and LoRA of Qwen3.8-27B closed 97%. Only the
structural attribution claim (2) and the fix-ordering it implies are
asserted here -- the toy cannot speak to the 12/46 replacement count
or the dollar figures, and it does NOT force the paper's 72% combined
figure to split between statement and calibration the way GEPA +
calibration split it in the field; the toy's split is reported as
measured.

Toy operationalization: a synthetic judging task with a stand-in
typed scorer whose scoring function is FIXED across all arms (the
paper's model does not change either; the question and the
calibration do).

  Items: x in R^8, plain score s = w . x with w a fixed unit vector,
  so s ~ N(0, 1). The label name states the plain rule R: label 1 iff
  s > TAU (TAU = 0.7). A marker m ~ Bernoulli(0.25), drawn
  independently, defines the convention subset S. On S the true label
  follows a hidden CONVENTION C instead: label = 1 - R(x) (the
  annotators flipped the plain reading on marked items; nothing in
  the label name says so).

  Scorer: a stand-in typed decision model that "understands the label
  name" = implements R on a noisy readout z = s + eps, eps ~
  N(0, 0.35), and decides with the default threshold 0 (it has never
  seen the convention, and its default threshold is not the task's
  TAU -- the ordinary miscalibration a few hundred labels can fix).

  Arms (scorer function identical throughout):
    A  zero-shot rule-only: predict 1[z > 0].
    B  +convention stated: the question now states C, so the scorer
       flips its prediction on marked items; threshold still 0.
    C  +calibration only: threshold fitted by grid search on 500
       labeled examples drawn from the TRUE label distribution
       (convention still unstated, no flip).
    D  +both: flip on S with the threshold fitted the same way.
    Oracle: the same noisy readout with the true rule (flip on S,
       threshold TAU) -- the ceiling for this scorer.

  Capability control (task E): the true label is sign agreement,
  1[x0 * x1 > 0] -- an interaction. The scorer's readout for E is a
  fixed linear form over the features it attends to (x2..x7), which
  cannot express that interaction; arms A-D run unchanged (stating
  the rule cannot change the scorer's function class, calibration
  still fits a threshold on 500 labels). Oracle for E is the exact
  rule on clean features (accuracy 1.0). No arm should close E's gap:
  that is what makes the convention claim falsifiable rather than
  "fixes always help". (An earlier draft let E's readout include
  x0 + x1; a single threshold then exploited the sum's distributional
  dip at 0 and closed ~20% of E's gap without expressing the rule at
  all -- the readout was restricted to the attended features so the
  control isolates expressibility. See the iteration log in the PR.)

Metrics per seed (4,000 held-out items, 500 calibration items):
accuracy per arm; GAP CLOSED = (arm - A) / (oracle - A); and the
deficit CONCENTRATION: arm A's error rate on S vs off S.

Measured on pop-os (particlegan 0.10.1, 3 seeds): arm A accuracy
0.610-0.628; A's error rate on S 0.725-0.762 vs 0.240-0.264 off S
(ratio 2.8-3.1x). Gap closed: B (convention stated) 0.39-0.44, with
A's on-S error falling to 0.238-0.275; C (calibration only)
0.20-0.31 overall, but its on-S error RISES to 0.870-0.935 -- a
threshold fitted to the majority distribution buys plain-rule
accuracy by sacrificing the convention minority; D (both) 0.98-1.01.
Capability control: every arm's gap closure stays within
-0.02..0.04. Assertions below encode exactly these margins, no more.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe. Pure numpy -- no GAN machinery, because the claim is
about a fixed scorer's question and calibration, not training.
"""
import numpy as np
import pytest
import torch

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 8             # feature dimension
TAU = 0.7         # the plain rule's true threshold (label name's rule)
SIG = 0.35        # scorer readout noise
P_S = 0.25        # convention-subset fraction
N_CAL = 500       # calibration labels (the paper's 500 training labels)
N_TEST = 4000     # held-out items per seed
GRID = np.linspace(-2.0, 2.5, 181)   # calibration threshold grid

# The scorer's fixed plain-rule weights (unit norm, so s ~ N(0, 1)).
W = np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
W = W / np.linalg.norm(W)
# Capability-control readout: a fixed linear form over the features
# the scorer attends to (x2..x7) -- it cannot express x0 * x1.
U = np.zeros(D)
U[2:] = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
U = U / np.linalg.norm(U)


def _fit_threshold(z, m, y, flip):
    """Grid-search the threshold maximizing calibration accuracy."""
    best_t, best_acc = 0.0, -1.0
    for t in GRID:
        p = (z > t).astype(int)
        if flip:
            p = np.where(m == 1, 1 - p, p)
        acc = float((p == y).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def _gen_convention(rng, n):
    """Main task: plain rule R, hidden convention C flipping S."""
    x = rng.standard_normal((n, D))
    s = x @ W
    m = (rng.random(n) < P_S).astype(int)
    y_plain = (s > TAU).astype(int)
    y = np.where(m == 1, 1 - y_plain, y_plain)
    z = s + rng.normal(0.0, SIG, n)
    return z, m, y


def _gen_capability(rng, n):
    """Control task E: true rule is the x0 * x1 interaction."""
    x = rng.standard_normal((n, D))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    z = x @ U + rng.normal(0.0, SIG, n)
    m = (rng.random(n) < P_S).astype(int)
    return z, m, y


def _arms(z, m, y, z_cal, m_cal, y_cal, oracle_pred):
    """Run arms A-D against one generated task; return accuracies."""
    t_cal = _fit_threshold(z_cal, m_cal, y_cal, flip=False)
    t_both = _fit_threshold(z_cal, m_cal, y_cal, flip=True)
    p_a = (z > 0).astype(int)
    p_b = np.where(m == 1, 1 - p_a, p_a)
    p_c = (z > t_cal).astype(int)
    base_d = (z > t_both).astype(int)
    p_d = np.where(m == 1, 1 - base_d, base_d)
    acc = lambda p: float((p == y).mean())  # noqa: E731
    return dict(a=acc(p_a), b=acc(p_b), c=acc(p_c), d=acc(p_d),
                oracle=acc(oracle_pred),
                err_s_a=float((p_a[m == 1] != y[m == 1]).mean()),
                err_off_a=float((p_a[m == 0] != y[m == 0]).mean()),
                err_s_b=float((p_b[m == 1] != y[m == 1]).mean()),
                err_s_c=float((p_c[m == 1] != y[m == 1]).mean()))


def run_seed(seed):
    rng = np.random.default_rng(5000 + seed)
    z_cal, m_cal, y_cal = _gen_convention(rng, N_CAL)
    z, m, y = _gen_convention(rng, N_TEST)
    base_o = (z > TAU).astype(int)
    oracle_pred = np.where(m == 1, 1 - base_o, base_o)
    return _arms(z, m, y, z_cal, m_cal, y_cal, oracle_pred)


def run_capability_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    z_cal, m_cal, y_cal = _gen_capability(rng, N_CAL)
    z, m, y = _gen_capability(rng, N_TEST)
    # Oracle for E is the exact interaction rule on clean features,
    # which the generator's labels already follow deterministically;
    # its accuracy against y is 1.0 by construction.
    return _arms(z, m, y, z_cal, m_cal, y_cal, oracle_pred=y.copy())


def _closed(r, arm):
    return (r[arm] - r["a"]) / (r["oracle"] - r["a"])


def test_deficit_concentrates_on_convention_subset():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The paper's attribution claim, in miniature: arm A's deficit
        # lives on the convention subset, not spread over the task.
        assert r["err_s_a"] > 0.65, (
            f"seed {seed}: arm A error on S {r['err_s_a']:.3f}")
        assert r["err_off_a"] < 0.32, (
            f"seed {seed}: arm A error off S {r['err_off_a']:.3f}")
        assert r["err_s_a"] > 2.0 * r["err_off_a"], (
            f"seed {seed}: A error on S {r['err_s_a']:.3f} vs off S "
            f"{r['err_off_a']:.3f} -- deficit not concentrated on S")


def test_stating_convention_beats_calibration_and_both_close_gap():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        closed_b, closed_c, closed_d = (_closed(r, "b"), _closed(r, "c"),
                                        _closed(r, "d"))
        # Stating the convention closes the largest single chunk and
        # repairs the convention subset specifically.
        assert closed_b > 0.30, (
            f"seed {seed}: convention-stated gap closed {closed_b:.3f}")
        assert closed_b > closed_c, (
            f"seed {seed}: gap closed by statement {closed_b:.3f} vs "
            f"calibration only {closed_c:.3f}")
        assert r["err_s_b"] < 0.35, (
            f"seed {seed}: arm B error on S {r['err_s_b']:.3f}")
        assert r["err_s_b"] < r["err_s_a"] - 0.35, (
            f"seed {seed}: on-S error A {r['err_s_a']:.3f} -> B "
            f"{r['err_s_b']:.3f}")
        # Calibration alone closes a real but smaller chunk overall --
        # and none of it on S: fitted to the majority distribution,
        # it actively worsens the convention subset.
        assert 0.10 < closed_c < 0.45, (
            f"seed {seed}: calibration-only gap closed {closed_c:.3f}")
        assert r["err_s_c"] > r["err_s_a"] + 0.10, (
            f"seed {seed}: calibration-only on-S error {r['err_s_c']:.3f} "
            f"vs zero-shot {r['err_s_a']:.3f} -- calibration should "
            f"sacrifice the unstated-convention minority")
        # Both fixes together reach the scorer's own ceiling.
        assert closed_d > 0.85, (
            f"seed {seed}: both-fixes gap closed {closed_d:.3f}")
        assert closed_d < 1.15, (
            f"seed {seed}: both-fixes gap closed {closed_d:.3f} -- "
            f"past the oracle ceiling, the toy is leaking")


def test_capability_control_no_arm_closes_the_gap():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_capability_seed(seed)
        # A rule the scorer's function class cannot express: stating
        # it (B/D flip machinery has nothing to attach to) and
        # calibrating a threshold on 500 labels both leave the gap
        # essentially untouched. This is the falsifiability control
        # for the convention claim.
        for arm in ("b", "c", "d"):
            assert abs(_closed(r, arm)) < 0.10, (
                f"seed {seed}: capability-control arm {arm} closed "
                f"{_closed(r, arm):.3f} of the gap -- a fix closed a "
                f"capability gap, so the toy cannot distinguish "
                f"convention gaps from capability gaps")
        assert r["a"] < 0.56 and r["d"] < 0.56, (
            f"seed {seed}: capability-control accuracies A "
            f"{r['a']:.3f} D {r['d']:.3f} should sit at chance")
