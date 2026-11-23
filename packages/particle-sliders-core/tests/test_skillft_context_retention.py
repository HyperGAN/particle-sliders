"""CPU gate for the two Skill-FT claims of Chen, Sun, Meng,
Tang, Song & Zhao, "AutoMaster: Autonomous Post-Training by
Making the Same Model a Better Teacher" (arXiv:2611.26510):
(a) trajectories elicited under explicit natural-language
skills (declared behavior instructions that suppress
undesirable habits and elicit effective behaviors) beat
success-filtered unguided teacher trajectories by
+1.42-8.99 pts at fixed teacher / budget / recipe; and
(b) the sharper finding -- retaining the skill instruction
IN the training demonstration beats stripping it in ALL SIX
settings, with IDENTICAL supervised outputs and the
instruction absent at evaluation. Provenance: cycle 9 of
the Particle Sliders paper watch (brief #3).

Toy operationalization: a small binary sequence world where
a behavior is elicited by an instruction token. The toy's
instruction is a token, not natural language -- the claim
tested is the training-example FORM (context retained vs
stripped, at identical supervised outputs), not skill
authoring.

Inputs x in R^16. The true behavior is a core score
t = W . x with W supported on the first KCORE = 4 dims
(||W|| = 2). One further dim (the shortcut coordinate s)
is constructed correlated with t on the training
distribution (rho = +0.6) and anti-correlated off it
(rho = -0.6 at OOD eval). The undesirable habit is a fixed
nonlinear response to the shortcut,
    phi(x) = relu(s) - 0.75 * relu(-s),
the analogue of a shortcut routine that predicts the
answer on-distribution and is wrong off it.

Teachers (demonstrations are binary tokens sampled from
the teacher's probabilities; N = 1024 per arm, matched):
  shaped   -- the teacher under the skill: p = sigmoid(t).
              The skill suppresses the habit in the
              teacher's output exactly.
  unshaped -- the unguided teacher: p = sigmoid(t + 4*phi),
              success-filtered (keep only pool items where
              the unguided teacher's deterministic decision
              sign(t + 4*phi) agrees with sign(t)). The
              habit strength in TRAJECTORIES (4.0) exceeds
              the student's single-pass habit pathway (1.5)
              because a habit compounds over a trajectory's
              steps; filtering keeps the examples where the
              habit happened to co-occur with success, which
              is precisely the paper's warning about
              success-filtering alone.

Student: a linear logit w in R^16, init 0, trained by
minibatch SGD (batch 32, lr 0.5, 25 epochs, weight decay
1e-3) on the binary tokens (logistic loss) -- the SFT
arm. The student's forward pass has a habit pathway
1.5 * phi(x) that is part of its DEFAULT-mode computation:
with no instruction in context the pathway is active.
The skill instruction, when present in the training
example, suppresses that pathway in the student's own
forward pass (gate 0.0 in the retained arm; a partial
gate 0.3 arm checks the effect is not an artifact of
exact suppression). Arms of a seed share the same pool,
the same sampled tokens and the same training RNG stream;
the shaped arms' supervised outputs are literally the
same arrays -- arms differ only by dataset (shaped vs
unshaped) and by the training-example form:

  (i)   shaped-retained   shaped data, gate 0.0
        (input = [I, x]);
  (i')  shaped-partial    shaped data, gate 0.3
        (the skill suppresses only 70% of the pathway);
  (ii)  shaped-stripped   shaped data, gate 1.0
        (input = [x] only, IDENTICAL tokens);
  (iii) unshaped-stripped unshaped data, gate 1.0 (the
        control's natural form -- no skill existed);
  (iv)  unshaped-gated    unshaped data, gate 0.0 (the
        retained FORM applied to unshaped data: does the
        form alone rescue bad data?).

Eval is ALWAYS context-free and reads the INSTALLED
component only -- the logit w . x, with no instruction
and no habit-pathway term. This is the slider-faithful
readout (the same convention as the context-frame test,
PR family of arXiv:2606.00831): the habit pathway belongs
to the base, is common to every arm, and is not part of
what SFT installed; what transfers between contexts is
the weight delta, so the delta is what is measured.
Readouts per arm, on 4000 fresh probes per split:
  acc_id / acc_ood -- sign accuracy of w . x against
      sign(t) at rho = +0.6 / rho = -0.6; the headline
      behavior accuracy is their mean (pooled);
  habit -- the partial coefficient on phi in a regression
      of w . x on [t, phi] over rho = 0 probes: how much
      of the habit the installed component expresses in
      the habit's own direction (positive = expresses,
      negative = anti-habit entanglement, ~0 = clean);
  cos -- cosine of w with the true W.

Measured on pop-os (particlegan 0.10.1), 3 seeds --
pooled acc / ID / OOD / habit / cos:
  retained   0.954/0.961/0.938, ID 0.951/0.959/0.938,
             OOD 0.957/0.963/0.937, habit -0.02/-0.10/-0.12,
             cos 0.990/0.991/0.981
  partial    pooled 0.930/0.929/0.916, habit -0.45/-0.53/-0.56
  stripped   pooled 0.823/0.817/0.829, ID 0.767/0.757/0.774,
             OOD 0.879/0.877/0.884, habit -1.47/-1.54/-1.57,
             cos 0.828/0.819/0.838
  unshaped   pooled 0.756/0.758/0.765, ID 0.851/0.847/0.854,
      (str.) OOD 0.660/0.668/0.676, habit +2.54/+2.01/+2.01
  unshaped   pooled 0.681/0.670/0.677, OOD 0.540/0.526/0.534,
      (gated) habit +3.79/+3.33/+3.31

The mechanism the toy isolates, stated plainly: stripped
training does not fail to fit -- it fits by CANCELLATION.
Its forward pass contains the habit pathway, so matching
the shaped teacher's tokens forces w to spend itself
cancelling phi through the features available to it; the
installed delta is the core rule minus a habit
cancellation (habit coefficient -1.5, cos 0.82), and the
cancellation is worthless as behavior: read as a deployed
component it is entangled and inaccurate. Retained
training fits the same tokens with the pathway suppressed
by the context, so w installs the core rule itself
(habit ~0, cos 0.99) and the context-free readout keeps
it. A secondary readout makes the same point from the
other side: if eval reinstates the student's habit
pathway (w . x + 1.5*phi), the stripped arm looks fine
(0.94-0.95) -- its delta only works paired with the exact
default-mode pathway it learned to cancel, which is the
difference between internalizing a behavior and
internalizing a compensation for your own default mode.

Assertions below encode exactly these margins, no more:
retained beats stripped on pooled accuracy by >= 0.08
and on OOD by >= 0.03 on every seed (measured pooled
+0.11..+0.14, OOD +0.05..+0.09); the partial-gate arm
sits strictly between them on pooled accuracy on every
seed; shaped arms express no positive habit (retained
|habit| <= 0.30, stripped habit <= -0.80 -- its pathology
is entanglement, not expression) while the unshaped
control expresses the habit at >= +1.5 and trails both
shaped arms on pooled accuracy (stripped by >= 0.03,
retained by >= 0.12); and the retained form applied to
unshaped data makes the habit WORSE, not better (habit
>= the unshaped-stripped arm's + 0.8, OOD <= its - 0.08)
-- the form is not a substitute for shaping the data.

What is NOT asserted: anything about natural-language
skill authoring, real LM SFT, or the paper's point sizes
(+1.42-8.99) -- the toy's margins are its own. Two
modeling gifts, stated plainly: the skill's suppression
of the student's habit pathway is exact in the retained
arm (the partial arm bounds how much of the effect
survives a 70% suppression: most of it), and the habit
pathway is a frozen stand-in for the base model's
default-mode computation, not itself trained -- the toy
tests what SFT writes into w under each training-example
form, which is the claim's subject. Pure numpy -- no GAN
machinery, because the claim is about the form of the
training example, not training dynamics of the adapter.
Skips cleanly when the installed particlegan does not
provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test
in CI).
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


D = 16               # input dimension
KCORE = 4            # core dims carrying the true behavior
SC = KCORE           # shortcut coordinate index
RHO_TRAIN = 0.6      # shortcut/core correlation, training
RHO_OOD = -0.6       # shortcut/core correlation, OOD eval
GAMMA_BASE = 1.5     # student default-mode habit strength
GAMMA_TEACHER = 4.0  # habit strength in unguided trajectories
N_TRAIN = 1024       # demonstrations per arm (matched)
POOL = 4 * N_TRAIN   # candidate pool before filtering
EPOCHS = 25          # SFT passes (fixed budget, all arms)
LR = 0.5             # SGD step
WD = 1e-3            # weight decay
BS = 32              # minibatch size
N_EVAL = 4000        # fresh probes per eval split


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _phi(x):
    """The habit: a nonlinear response to the shortcut."""
    s = x[:, SC]
    return np.maximum(s, 0.0) - 0.75 * np.maximum(-s, 0.0)


def _build(seed):
    """True behavior direction W for one seed (exact)."""
    rng = np.random.default_rng(41000 + seed)
    w_star = rng.standard_normal(KCORE)
    w_star /= np.linalg.norm(w_star)
    W = np.zeros(D)
    W[:KCORE] = w_star * 2.0
    return W


def _gen_x(rng, n, rho, W):
    """Inputs whose shortcut coordinate tracks the core
    score with correlation rho (sign flips OOD)."""
    x = rng.standard_normal((n, D))
    t = x @ W
    x[:, SC] = (rho * (t / 2.0)
                + np.sqrt(max(1.0 - rho ** 2, 0.0))
                * rng.standard_normal(n))
    return x


def _datasets(seed, W):
    """Shaped and success-filtered unshaped demonstrations.

    Both arms draw from the same pool and the same token
    uniforms stream position; the shaped set is the first
    N_TRAIN pool items with tokens from sigmoid(t), the
    unshaped set is the first N_TRAIN success-filtered
    pool items with tokens from sigmoid(t + 4*phi). The
    shaped arms all consume the identical (Xs, ys) arrays.
    """
    rng = np.random.default_rng(91000 + seed)
    Xp = _gen_x(rng, POOL, RHO_TRAIN, W)
    t = Xp @ W
    ph = _phi(Xp)
    tok_shaped = (rng.random(POOL) < _sigmoid(t)).astype(float)
    Xs, ys = Xp[:N_TRAIN], tok_shaped[:N_TRAIN]
    ok = np.sign(t + GAMMA_TEACHER * ph) == np.sign(t)
    idx = np.where(ok)[0]
    assert len(idx) >= N_TRAIN, (
        f"seed {seed}: success filter kept {len(idx)} of "
        f"{POOL}, fewer than the matched size {N_TRAIN}")
    idx = idx[:N_TRAIN]
    Xu = Xp[idx]
    tok_un = (rng.random(len(Xu))
              < _sigmoid(t[idx] + GAMMA_TEACHER * ph[idx])
              ).astype(float)
    return Xs, ys, Xu, tok_un


def _train(X, tok, gate, seed):
    """SFT of the linear student on binary tokens.

    The forward logit is w . x + gate * GAMMA_BASE * phi(x):
    gate 1.0 is the stripped form (default mode, habit
    pathway active), gate 0.0 the retained form (the skill
    in context suppresses the pathway), gate 0.3 the
    partial-suppression sensitivity arm. Every arm of a
    seed uses the same training RNG seed, so permutations
    are identical across arms; arms differ only by gate
    and by dataset.
    """
    rng = np.random.default_rng(77000 + seed)
    w = np.zeros(D)
    base = gate * GAMMA_BASE * _phi(X)
    n = len(X)
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for st in range(0, n, BS):
            b = order[st:st + BS]
            p = _sigmoid(X[b] @ w + base[b])
            grad = X[b].T @ (p - tok[b]) / len(b) + WD * w
            w -= LR * grad
    return w


def run_seed(seed):
    W = _build(seed)
    Xs, ys, Xu, yu = _datasets(seed, W)
    spec = (
        ("retained", Xs, ys, 0.0),
        ("partial", Xs, ys, 0.3),
        ("stripped", Xs, ys, 1.0),
        ("unshaped", Xu, yu, 1.0),
        ("unshaped_gated", Xu, yu, 0.0),
    )
    ws = {name: _train(Xd, yd, gate, seed)
          for name, Xd, yd, gate in spec}
    rng_e = np.random.default_rng(55000 + seed)
    ev = {}
    for split, rho in (("id", RHO_TRAIN), ("ood", RHO_OOD),
                       ("zero", 0.0)):
        Xe = _gen_x(rng_e, N_EVAL, rho, W)
        ev[split] = (Xe, Xe @ W)
    arms = {}
    for name, w in ws.items():
        acc_id = float(np.mean(
            np.sign(ev["id"][0] @ w) == np.sign(ev["id"][1])))
        acc_ood = float(np.mean(
            np.sign(ev["ood"][0] @ w) == np.sign(ev["ood"][1])))
        Xz = ev["zero"][0]
        A = np.stack([Xz @ W, _phi(Xz)], axis=1)
        coef, *_ = np.linalg.lstsq(A, Xz @ w, rcond=None)
        arms[name] = dict(
            acc_id=acc_id, acc_ood=acc_ood,
            acc_pool=0.5 * (acc_id + acc_ood),
            habit=float(coef[1]),
            cos=float(w @ W / max(np.linalg.norm(w), 1e-9) / 2.0),
        )
    return arms


def test_context_retention_beats_stripping():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        ret, part, strip = (arms["retained"], arms["partial"],
                            arms["stripped"])
        # The retained arm internalizes the behavior itself:
        # near-true direction, no habit in either direction.
        assert ret["acc_pool"] >= 0.92, (
            f"seed {seed}: retained pooled accuracy "
            f"{ret['acc_pool']:.3f}")
        assert ret["cos"] >= 0.95, (
            f"seed {seed}: retained cos(w, W) {ret['cos']:.3f}")
        assert abs(ret["habit"]) <= 0.30, (
            f"seed {seed}: retained habit coefficient "
            f"{ret['habit']:+.3f}")
        # The headline: identical supervised outputs, yet
        # the stripped form installs a materially worse
        # context-free behavior on every seed.
        assert ret["acc_pool"] >= strip["acc_pool"] + 0.08, (
            f"seed {seed}: retained pooled {ret['acc_pool']:.3f} "
            f"vs stripped {strip['acc_pool']:.3f}")
        assert ret["acc_ood"] >= strip["acc_ood"] + 0.03, (
            f"seed {seed}: retained OOD {ret['acc_ood']:.3f} "
            f"vs stripped {strip['acc_ood']:.3f}")
        # The stripped arm's pathology is anti-habit
        # entanglement (cancellation written into w), not
        # habit expression.
        assert strip["habit"] <= -0.80, (
            f"seed {seed}: stripped habit coefficient "
            f"{strip['habit']:+.3f} -- expected the "
            f"cancellation signature")
        # Partial suppression (70%) sits strictly between:
        # the effect is not an artifact of exact gating.
        assert strip["acc_pool"] < part["acc_pool"] < ret["acc_pool"], (
            f"seed {seed}: partial pooled {part['acc_pool']:.3f} "
            f"not between stripped {strip['acc_pool']:.3f} and "
            f"retained {ret['acc_pool']:.3f}")


def test_shaped_data_suppresses_habit_vs_unshaped():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        ret, strip, un = (arms["retained"], arms["stripped"],
                          arms["unshaped"])
        # Success-filtered unguided data installs the habit
        # it happened to succeed with.
        assert un["habit"] >= 1.50, (
            f"seed {seed}: unshaped habit coefficient "
            f"{un['habit']:+.3f}")
        assert un["habit"] >= ret["habit"] + 1.50, (
            f"seed {seed}: unshaped habit {un['habit']:+.3f} "
            f"vs retained {ret['habit']:+.3f}")
        # Shaped data wins on behavior accuracy at matched
        # size -- the retained arm by a wide margin, the
        # stripped arm (same shaped data, worse form) by a
        # smaller but consistent one, driven OOD where the
        # habit shortcut fails.
        assert ret["acc_pool"] >= un["acc_pool"] + 0.12, (
            f"seed {seed}: retained pooled {ret['acc_pool']:.3f} "
            f"vs unshaped {un['acc_pool']:.3f}")
        assert strip["acc_pool"] >= un["acc_pool"] + 0.03, (
            f"seed {seed}: stripped pooled {strip['acc_pool']:.3f} "
            f"vs unshaped {un['acc_pool']:.3f}")
        assert strip["acc_ood"] >= un["acc_ood"] + 0.15, (
            f"seed {seed}: stripped OOD {strip['acc_ood']:.3f} "
            f"vs unshaped {un['acc_ood']:.3f}")


def test_form_alone_does_not_rescue_unshaped_data():
    _empf_recipe()
    for seed in (0, 1, 2):
        arms = run_seed(seed)
        un, ung = arms["unshaped"], arms["unshaped_gated"]
        # Applying the retained FORM (habit pathway gated
        # off in the student's forward) to UNSHAPED data
        # forces the habit into w itself: expression rises
        # and OOD accuracy falls vs the same data stripped.
        # Retention helps because of what the shaped data
        # is, not as a generic training trick.
        assert ung["habit"] >= un["habit"] + 0.80, (
            f"seed {seed}: unshaped-gated habit "
            f"{ung['habit']:+.3f} vs unshaped-stripped "
            f"{un['habit']:+.3f}")
        assert ung["acc_ood"] <= un["acc_ood"] - 0.08, (
            f"seed {seed}: unshaped-gated OOD "
            f"{ung['acc_ood']:.3f} vs unshaped-stripped "
            f"{un['acc_ood']:.3f}")
        assert ung["acc_pool"] <= un["acc_pool"] - 0.05, (
            f"seed {seed}: unshaped-gated pooled "
            f"{ung['acc_pool']:.3f} vs unshaped-stripped "
            f"{un['acc_pool']:.3f}")
