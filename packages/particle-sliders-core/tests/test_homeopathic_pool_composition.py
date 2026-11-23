"""CPU gate for the homeopathic-learning claim of Clarke,
Dubiński, Chua, Schrodi, Evans & Cloud, "Homeopathic
Learning: Diluting poisoned fine-tuning data can strengthen
its effect" (arXiv:2611.10110): poisoned data that has
little effect on its own creates a hidden backdoor once
clean, contrasting data is added. A model fine-tuned ONLY
on a "poison" dataset associated with trait T does not
learn T; adding similarly formatted "clean" data NOT
associated with T causes the model to learn T. The effect
requires the poison data to have an identifying "trigger"
feature ABSENT from the clean data. The paper's proposed
explanation, in its regression model: in the poison data,
task and trigger always co-occur, and regularization
favors attributing T to the task; clean data without the
trigger causes T to be attributed to the trigger instead.
Provenance chain: surfaced in cycle 15 of this watch via
the Schrodi author chain (Schrodi 2509.23886, PR #275's
paper -> co-authors Schrodi + Cloud + Chua on this paper).

Toy operationalization: the paper's own regression
mechanism, in miniature. Each example has task features
u in R^6 with a NONZERO MEAN (u = 2.5 * 1 + 0.4 * eps) and
a scalar trigger channel z in {0, 1}. A teacher with task
weights w (drawn per seed) emits

    y = w . u + tau * z,        tau = 1.5 (trait strength)

on poison examples (z = 1 for every one of them) and
y = w . u exactly on clean examples. The student is a
closed-form RIDGE fit (lambda = 10) on features [u, z]
with NO intercept: the trait offset can be explained
either by the trigger coefficient or by shifting the task
weights along the mean direction of u, and the nonzero
mean makes the task pathway the cheaper explanation in
ridge norm (shifting w by delta parallel to the mean
costs |delta| ~ tau / (sqrt(6) * 2.5) ~ 0.16 * tau, vs
tau for the trigger coefficient) at the price of a small
variance term (sigma = 0.4). That absorption is the
phenomenon under test, so there is deliberately no
intercept: an intercept and the trigger would be two
exactly-constant channels and ridge would split tau
between them for reasons unrelated to the paper's claim.

Arms (poison count matched across arms at N = 300; clean
counts also 300; all arms of a seed share the same poison
set, drawn once per seed):
  (a) poison only (z = 1, trait on every example);
  (b) poison + clean (clean: z = 0, no trait);
  (c) clean only (control: no trait anywhere);
  (d) poison + clean-WITH-trigger (z = 1, no trait --
      the trigger is PRESENT in the clean data);
  (e) no-regularization control: unregularized least
      squares on the poison-only set.
Readouts: the fitted trigger coefficient; trait
expression = mean predicted difference between fresh
trigger-bearing and trigger-free probes at the same u
(in this linear student it equals the trigger
coefficient exactly, and both are reported); the
unconditional offset on trigger-free probes (mean
prediction minus the teacher's task-only target -- where
the absorbed trait leaks in arm (a)); and held-out task
MSE on trigger-free probes.

Measured on pop-os (particlegan 0.10.1), 3 seeds,
ridge lambda = 10:
trigger coefficient / expression (tau = 1.5):
  (a) poison only        +0.27 / +0.06 / +0.05
  (b) poison + clean     +1.42 / +1.40 / +1.40
  (c) clean only         +0.00 / +0.00 / +0.00 (exact:
      the trigger column is identically zero in this arm,
      so the coefficient is 0 by construction -- this arm
      controls for trait signal appearing from nowhere in
      the pipeline, not for estimation noise)
  (d) poison + clean-with-trigger  +0.31 / +0.03 / -0.16
  (e) OLS poison only    +1.50 on every seed (exact
      recovery of tau -- see below)
Unconditional offset on trigger-free probes: (a) +1.24 /
+1.43 / +1.45 (the absorbed trait leaks unconditionally);
(b) +0.04 / +0.05 / +0.06. Held-out trigger-free task
MSE: (a) 1.57 / 2.09 / 2.13; (b) 0.007 / 0.009 / 0.009.
Assertions below encode exactly these margins, no more.

What is NOT asserted: anything about real fine-tuning,
LLMs, or the paper's empirical backdoor rates; that the
flip needs this exact mean / noise / lambda regime (the
loop found the regime matters: at lambda = 0.1 the
poison-only trigger coefficient is already ~1.44 and
there is no absorption to flip); and anything about arm
(d) beyond its trigger coefficient staying in the
poison-only band (its unconditional offset, +0.44 /
+0.72 / +0.91, is a compromise between the two z = 1
groups and is reported, not asserted). One re-scope
against the natural first guess, stated plainly: the
unregularized poison-only fit is NOT unidentified in
this toy -- the targets are noiseless and [u, z] has
full column rank, so OLS recovers tau exactly (arm (e)).
What arm (e) shows is the paper's actual point: without
regularization there is no absorption and no
homeopathic effect to dilute into existence; the
regularizer is the mechanism. Two modeling gifts,
stated plainly: targets are noiseless (the teacher is
exactly in the student's model class in arms (a)-(c),
so every attribution difference is the regularizer's
choice, not a fit-quality artifact), and the trigger is
a single known scalar channel rather than a feature the
student must discover. Pure numpy -- no GAN machinery,
because the claim is about regularized attribution, not
training dynamics. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family; PR #211 is what
enables this test in CI).
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


K = 6                # task-feature dimension
MEAN = 2.5           # nonzero task-feature mean (absorption pathway)
SIG = 0.4            # per-example task-feature noise
TAU = 1.5            # teacher trait strength
LAM = 10.0           # ridge penalty (the mechanism under test)
N = 300              # examples per pool component, per seed
P = 500              # fresh probes per seed


def _dataset(rng_stream, seed, w, zval, trait):
    rng = np.random.default_rng(rng_stream + seed)
    u = MEAN + SIG * rng.standard_normal((N, K))
    z = np.full(N, float(zval))
    y = u @ w + (TAU if trait else 0.0) * z
    return np.column_stack([u, z]), y


def build(seed):
    """Teacher, pool components and probes for one seed."""
    rng = np.random.default_rng(41000 + seed)
    w = rng.standard_normal(K)
    poison = _dataset(51000, seed, w, 1, True)
    clean = _dataset(52000, seed, w, 0, False)
    clean_trig = _dataset(53000, seed, w, 1, False)
    rng_p = np.random.default_rng(54000 + seed)
    probes = MEAN + SIG * rng_p.standard_normal((P, K))
    return dict(w=w, poison=poison, clean=clean,
                clean_trig=clean_trig, probes=probes)


def _fit(X, y, lam):
    if lam == 0.0:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]),
                           X.T @ y)


def _stack(a, b):
    return (np.vstack([a[0], b[0]]),
            np.concatenate([a[1], b[1]]))


def run_seed(seed):
    d = build(seed)
    w, up = d["w"], d["probes"]
    arms = {}
    for name, (X, y) in (
            ("poison", d["poison"]),
            ("mix", _stack(d["poison"], d["clean"])),
            ("clean", d["clean"]),
            ("clean_trig", _stack(d["poison"], d["clean_trig"]))):
        beta = _fit(X, y, LAM)
        p1 = np.column_stack([up, np.ones(P)]) @ beta
        p0 = np.column_stack([up, np.zeros(P)]) @ beta
        arms[name] = dict(
            trig=float(beta[-1]),
            expr=float(np.mean(p1 - p0)),
            off0=float(np.mean(p0 - up @ w)),
            mse0=float(np.mean((p0 - up @ w) ** 2)))
    beta_ols = _fit(d["poison"][0], d["poison"][1], 0.0)
    arms["ols_poison_trig"] = float(beta_ols[-1])
    return arms


def test_poison_only_trait_is_absorbed_by_task_pathway():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        a = arms["poison"]
        # The trigger coefficient stays in the absorbed
        # band (demonstrated +0.05..+0.27 of tau = 1.5).
        assert a["trig"] <= 0.30 * TAU, (
            f"seed {seed}: poison-only trigger coefficient "
            f"{a['trig']:+.3f}")
        assert a["expr"] <= 0.30 * TAU, (
            f"seed {seed}: poison-only trait expression "
            f"{a['expr']:+.3f}")
        # The trait does not vanish -- it leaks as an
        # unconditional offset on trigger-FREE probes
        # (demonstrated +1.24..+1.45), wrecking held-out
        # trigger-free task fit (MSE demonstrated >= 1.57).
        assert a["off0"] >= 0.60 * TAU, (
            f"seed {seed}: poison-only unconditional offset "
            f"{a['off0']:+.3f}")
        assert a["mse0"] >= 1.0, (
            f"seed {seed}: poison-only trigger-free task "
            f"MSE {a['mse0']:.3f}")


def test_clean_data_flips_attribution_to_trigger():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        b, a = arms["mix"], arms["poison"]
        # Adding trigger-free clean data pins the task
        # pathway: the trait lands on the trigger
        # coefficient (demonstrated +1.40..+1.42).
        assert b["trig"] >= 0.80 * TAU, (
            f"seed {seed}: poison+clean trigger coefficient "
            f"{b['trig']:+.3f}")
        assert b["expr"] >= 0.80 * TAU, (
            f"seed {seed}: poison+clean trait expression "
            f"{b['expr']:+.3f}")
        assert b["trig"] >= a["trig"] + 0.90, (
            f"seed {seed}: flip margin mix {b['trig']:+.3f} "
            f"vs poison-only {a['trig']:+.3f}")
        # The unconditional leak closes and trigger-free
        # task fit is restored (demonstrated off0 <= +0.06,
        # MSE <= 0.009).
        assert abs(b["off0"]) <= 0.15, (
            f"seed {seed}: poison+clean unconditional "
            f"offset {b['off0']:+.3f}")
        assert b["mse0"] <= 0.05, (
            f"seed {seed}: poison+clean trigger-free task "
            f"MSE {b['mse0']:.3f}")


def test_controls_do_not_flip():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        # Clean only: no trait anywhere, no trigger
        # coefficient (exactly 0.0 by construction -- the
        # trigger column is identically zero in this arm).
        assert abs(arms["clean"]["trig"]) <= 0.05, (
            f"seed {seed}: clean-only trigger coefficient "
            f"{arms['clean']['trig']:+.3f}")
        # Clean data WITH the trigger (but no trait) must
        # NOT produce the flip: the trigger's ABSENCE in
        # the clean data is what matters (demonstrated
        # -0.16..+0.31, in the poison-only band).
        d = arms["clean_trig"]
        assert d["trig"] <= 0.30 * TAU, (
            f"seed {seed}: poison+clean-with-trigger "
            f"trigger coefficient {d['trig']:+.3f}")
        assert arms["mix"]["trig"] >= d["trig"] + 0.90, (
            f"seed {seed}: flip margin mix "
            f"{arms['mix']['trig']:+.3f} vs clean-with-"
            f"trigger {d['trig']:+.3f}")


def test_regularization_is_the_mechanism():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        # Without regularization the poison-only fit
        # recovers tau exactly (demonstrated +1.50 on
        # every seed): no absorption, no homeopathic
        # effect. The regularizer is the mechanism.
        assert abs(arms["ols_poison_trig"] - TAU) <= 0.05, (
            f"seed {seed}: OLS poison-only trigger "
            f"coefficient {arms['ols_poison_trig']:+.3f}")
