"""CPU gate for the OPSFT claim of Shen, Hou, Sun, Zhang,
Lin, Yin, Huang & Wang, "On-Policy Parameter Update
Direction Underlies Generalization in LLM Post-Training"
(arXiv:2609.36659). The paper's claim has two parts.
(a) SFT updates parameters along CONSISTENT directions
across steps, while on-policy training CONTINUOUSLY
ADJUSTS its update direction. (b) The CUMULATIVE update
direction of on-policy training is the transferable
object: OPSFT constrains each parameter's SFT update to
the direction (the per-parameter sign of the cumulative
on-policy update) identified by a few on-policy steps,
and SFT then generalizes like on-policy training while
retaining SFT's efficiency and its ability to use
high-quality trajectories. Provenance: cycle 19 of the
Particle Sliders paper watch. Cycle 18's brief noted
OPSFT as the third member of the during-training
constraint family (the corridor clamps a trait
direction; liminal anchors to the base model; OPSFT
anchors to the on-policy direction).

Toy operationalization: the smallest linear world in
which both parts can be decided separately. The student
is a linear regressor w in R^16, split into a ROBUST
subspace (8 dims) and a SPURIOUS subspace (8 dims). The
teacher is the robust rule y = sum(x_r) + label noise
(sigma = 0.5): the robust weights are all 1, the
teacher's spurious weights are 0. Spurious features are
a label-leak artifact: on the fixed SFT set each
x_s = 0.5 * (realized y, normalized) + 0.8 * noise, so
a spurious feature predicts even the label-noise part
of y that the robust features cannot -- its population
SFT pull is systematically positive. Under the shift
the artifact flips sign (x_s built from -y), so any
spurious weight the student carries is actively wrong
on the shifted held-out set (2048 fresh draws). The
student starts at w_r = 0, w_s = 0.2 per dim: a
pretrained spurious bias. That starting bias is what
makes the two directions opposite and readable -- the
on-policy steps walk the spurious weights DOWN toward
the teacher's 0, while SFT's own early steps walk them
UP toward the artifact. It is also the paper's premise
stated plainly: SFT's consistent direction is exactly
the problem, because on this set consistency means
consistently chasing the leak.

Phase A (direction identification, 16 steps from the
shared start, lr 0.04): the ON-POLICY arm takes SGD
steps on fresh minibatches (16 draws per step) from a
BROADER input distribution -- robust features and
teacher labels as usual, spurious features drawn
independently of the label (the artifact is a property
of the curated SFT set, not of the input stream the
student actually runs on). Stated honestly: this is
"on-policy" in the toy's only available sense -- fresh
inputs from the student's own operating distribution
at every step, never the fixed training set -- not
self-generated text. The SFT arm of phase A takes
full-batch steps on the fixed 256-example SFT set.
Readout for claim (a): the mean pairwise cosine of the
per-step update vectors within each arm. Phase B (all
arms: full-batch SFT on the same fixed set, 80 matched
steps, same lr, same start): (1) plain SFT; (2) OPSFT
-- at every step, any update component whose sign
disagrees with the sign of the cumulative on-policy
update d_op is zeroed (the paper's mechanism); (3)
random-sign control -- the same zeroing against a
random sign vector (dense, the same number of
constrained coordinates as d_op, which is also dense);
(4) own-direction control -- the same zeroing against
the sign of SFT's OWN cumulative update over its first
16 phase-A steps. Readouts for claim (b): final train
MSE (fit) and shifted held-out MSE (generalization).

Measured on pop-os (particlegan 0.10.1), 5 seeds
(cos on-policy / cos SFT / train MSE plain, OPSFT,
random, own / shifted MSE plain, OPSFT, random, own):
  seed 0: cos 0.433 / 0.763
          train 0.348, 0.262, 1.913, 0.912
          shift 3.733, 1.751, 14.511, 13.684
  seed 1: cos 0.454 / 0.813
          train 0.302, 0.240, 2.282, 0.918
          shift 2.392, 1.284, 19.188, 11.039
  seed 2: cos 0.441 / 0.766
          train 0.340, 0.246, 1.623, 0.861
          shift 4.060, 1.690, 16.462, 13.929
  seed 3: cos 0.463 / 0.816
          train 0.329, 0.266, 1.436, 0.854
          shift 2.955, 1.604, 10.172, 11.094
  seed 4: cos 0.420 / 0.759
          train 0.279, 0.220, 2.725, 0.893
          shift 3.055, 1.596, 11.933, 12.643

The arbitration, stated plainly: both parts reproduce
at toy scale. For (a), SFT's per-step updates are far
more mutually aligned (mean pairwise cosine
0.76-0.82) than the on-policy arm's (0.42-0.46): the
fixed set pulls the same way every step, while fresh
inputs keep re-aiming the update -- margin 0.33-0.36
on every seed. For (b), OPSFT's shifted loss
(1.28-1.75) beats plain SFT (2.39-4.06) on every seed
while its train loss is as good or better (0.22-0.27
vs 0.28-0.35) -- the constraint costs no fit, because
what it blocks is the leak chase, and the spurious
weights it freezes early are walked down later in
training once the robust fit makes the SFT gradient
agree with d_op. The specificity also holds, and it
is the sharper result: the own-direction control is
far WORSE than plain SFT on the shifted set
(11.0-13.9), because SFT's own early cumulative
direction points UP the spurious subspace -- locking
SFT to its own consistency locks in the artifact. The
random-sign control is likewise far worse (10.2-19.2).
So it is not "any consistent direction" and not "any
sign constraint": the transferable object is the
on-policy direction specifically.

What is NOT asserted: the paper's LLM-scale numbers or
its exact training recipe. Two caveats, stated
plainly. First, the two sign controls also damage
train fit (random 1.44-2.73, own 0.85-0.92 vs plain
0.28-0.35) -- a sign constraint that blocks the robust
ascent or locks a bad transient hurts everywhere; the
fit-matched comparison is OPSFT vs plain SFT, and the
controls arbitrate the shifted-loss specificity, not
a fit-matched race. Second, the toy's spurious trap is
a label-leak artifact with a planted starting bias,
the broad distribution is artifact-free by
construction, and everything is one linear model --
nothing here is asserted about real post-training,
where the on-policy direction must be estimated from
rollouts of a model whose spurious directions are not
planted. Pure numpy -- no GAN machinery, because the
claim is about update geometry, not training dynamics
of a GAN. Skips cleanly when the installed particlegan
does not provide the EMPF recipe (the house guard for
this paper-test family; PR #211 is what enables this
test in CI).
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


DR = 8               # robust subspace dimension
DS = 8               # spurious subspace dimension
LEAK = 0.5           # spurious feature's loading on realized y
XS_NOISE = 0.8       # spurious feature's own noise
EPS = 0.5            # label noise sigma
W0_SPURIOUS = 0.2    # planted starting spurious bias
LR = 0.04            # shared step size, every arm and phase
N_TRAIN = 256        # fixed SFT set size
N_TEST = 2048        # shifted held-out size
PHASE_A_STEPS = 16   # direction-identification steps
OP_BATCH = 16        # fresh draws per on-policy step
PHASE_B_STEPS = 80   # matched SFT steps for every arm
SEED_BASE = 42000    # RNG stream base
RAND_BASE = 777      # random-sign control stream base
SEEDS = (0, 1, 2, 3, 4)


def _gen(rng, n, mode):
    """Draw (X, y): robust block, then spurious block.

    mode "train": spurious = label-leak artifact (+y);
    "shift": the artifact flipped (-y); "broad": the
    on-policy operating distribution, spurious features
    independent of the label.
    """
    xr = rng.standard_normal((n, DR))
    y = xr @ np.ones(DR) + EPS * rng.standard_normal(n)
    yn = y / np.sqrt(DR + EPS ** 2)
    if mode == "train":
        xs = LEAK * yn[:, None] + XS_NOISE * rng.standard_normal((n, DS))
    elif mode == "shift":
        xs = -LEAK * yn[:, None] + XS_NOISE * rng.standard_normal((n, DS))
    else:  # broad
        xs = rng.standard_normal((n, DS))
    return np.hstack([xr, xs]), y


def _grad(w, X, y):
    return -2.0 * X.T @ (y - X @ w) / len(y)


def _run_steps(w0, X, y, steps, constraint=None):
    """Full-batch gradient steps; with a constraint, any
    update component whose sign disagrees with the
    constraint's sign is zeroed (the OPSFT mechanism)."""
    w = w0.copy()
    updates = []
    for _ in range(steps):
        d = -LR * _grad(w, X, y)
        if constraint is not None:
            d = np.where(np.sign(d) == np.sign(constraint), d, 0.0)
        w = w + d
        updates.append(d.copy())
    return w, updates


def _mean_pairwise_cosine(updates):
    U = np.array(updates)
    total, count = 0.0, 0
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            total += float(U[i] @ U[j]
                           / np.linalg.norm(U[i])
                           / np.linalg.norm(U[j]))
            count += 1
    return total / count


def _mse(w, X, y):
    return float(np.mean((y - X @ w) ** 2))


def run_seed(seed):
    """One seed: phase A (both arms), then the four
    phase-B arms. Draw order is fixed -- train set,
    shifted set, phase-A on-policy stream, controls --
    so every readout of a seed comes from the same
    draws."""
    rng = np.random.default_rng(SEED_BASE + seed)
    Xtr, ytr = _gen(rng, N_TRAIN, "train")
    Xte, yte = _gen(rng, N_TEST, "shift")
    w0 = np.concatenate([np.zeros(DR), np.full(DS, W0_SPURIOUS)])

    # Phase A, on-policy: fresh broad minibatches.
    w = w0.copy()
    op_updates = []
    for _ in range(PHASE_A_STEPS):
        Xb, yb = _gen(rng, OP_BATCH, "broad")
        d = -LR * _grad(w, Xb, yb)
        w = w + d
        op_updates.append(d.copy())
    d_op = np.sum(op_updates, axis=0)

    # Phase A, SFT on the fixed set (its updates give
    # both the claim-(a) readout and the own-direction
    # control's cumulative direction).
    _, sft_updates = _run_steps(w0, Xtr, ytr, PHASE_A_STEPS)
    d_own = np.sum(sft_updates, axis=0)

    rng_rand = np.random.default_rng(RAND_BASE + seed)
    d_rand = rng_rand.choice([-1.0, 1.0], size=DR + DS)

    arms = {}
    for name, cons in (("plain", None), ("opsft", d_op),
                       ("rand", d_rand), ("own", d_own)):
        wB, _ = _run_steps(w0, Xtr, ytr, PHASE_B_STEPS, cons)
        arms[name] = {"train": _mse(wB, Xtr, ytr),
                      "shift": _mse(wB, Xte, yte)}

    return {
        "cos_op": _mean_pairwise_cosine(op_updates),
        "cos_sft": _mean_pairwise_cosine(sft_updates),
        "arms": arms,
    }


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_onpolicy_updates_adjust_where_sft_updates_are_consistent():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        # Claim (a): the on-policy arm's per-step updates
        # are far less mutually aligned than SFT's
        # (demonstrated on-policy 0.42-0.46, SFT
        # 0.76-0.82, margin 0.33-0.36 on every seed).
        assert res["cos_sft"] - res["cos_op"] >= 0.20, (
            f"seed {seed}: cos SFT {res['cos_sft']:.3f} "
            f"vs on-policy {res['cos_op']:.3f}")


def test_opsft_generalizes_better_than_plain_sft_at_comparable_fit():
    _empf_recipe()
    for seed in SEEDS:
        arms = _seed(seed)["arms"]
        # Claim (b), fit: the constraint costs nothing --
        # OPSFT train MSE is as good as or better than
        # plain SFT's on every seed (demonstrated OPSFT
        # 0.22-0.27 vs plain 0.28-0.35).
        assert arms["opsft"]["train"] <= arms["plain"]["train"] + 0.05, (
            f"seed {seed}: train OPSFT {arms['opsft']['train']:.3f} "
            f"vs plain {arms['plain']['train']:.3f}")
        # Claim (b), generalization: OPSFT shifted MSE
        # beats plain SFT by a clear margin on every
        # seed (demonstrated OPSFT 1.28-1.75 vs plain
        # 2.39-4.06, margin 1.11-2.37).
        assert arms["opsft"]["shift"] <= arms["plain"]["shift"] - 0.50, (
            f"seed {seed}: shift OPSFT {arms['opsft']['shift']:.3f} "
            f"vs plain {arms['plain']['shift']:.3f}")


def test_the_transferable_direction_is_specifically_the_onpolicy_one():
    _empf_recipe()
    for seed in SEEDS:
        arms = _seed(seed)["arms"]
        # Specificity vs the random-sign control: an
        # arbitrary sign constraint does not transfer
        # (demonstrated random shifted 10.17-19.19 vs
        # OPSFT 1.28-1.75).
        assert arms["opsft"]["shift"] <= arms["rand"]["shift"] - 1.0, (
            f"seed {seed}: shift OPSFT {arms['opsft']['shift']:.3f} "
            f"vs random {arms['rand']['shift']:.3f}")
        # Specificity vs the own-direction control: SFT's
        # OWN early cumulative direction points up the
        # spurious subspace, so locking SFT to its own
        # consistency is worse than no constraint at all
        # (demonstrated own shifted 11.04-13.93, above
        # even plain SFT's 2.39-4.06 on every seed).
        assert arms["opsft"]["shift"] <= arms["own"]["shift"] - 1.0, (
            f"seed {seed}: shift OPSFT {arms['opsft']['shift']:.3f} "
            f"vs own {arms['own']['shift']:.3f}")
        assert arms["own"]["shift"] >= arms["plain"]["shift"], (
            f"seed {seed}: shift own {arms['own']['shift']:.3f} "
            f"vs plain {arms['plain']['shift']:.3f}")
