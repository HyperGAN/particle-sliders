"""CPU gate for the steering-vector-distillation approximability claim
(arXiv:2606.00995): fine-tuning on a prompted/steered teacher's outputs
installs an aligned steering vector in the student ONLY when the
teacher's prompt is well approximated by a single steering vector --
and a cheap pre-flight statistic (how much of the prompt-induced shift
a single constant vector explains) predicts whether it will.

Paper: "Subliminal Learning Is Steering Vector Distillation" (Blank,
Bhatia, Rajamanoharan, Conmy & Nanda). The teacher's system prompt is
summarized by v_teacher, the mean residual-stream shift it induces;
students fine-tuned (rank-8 LoRA, alpha=32, 10k filtered samples;
Qwen2.5-7B-Instruct, Gemma-3-4b-it) on semantically unrelated teacher
outputs install v_student aligned with v_teacher -- EAS (cos of the
fine-tuning-induced shift with v_teacher) rises steadily over training
while a clean-data control stays at ~0.1. v_student is largely
necessary/sufficient (steering the base model with it reproduces the
trait; replacing the student's component along it removes >50% of
trait-aligned behavior). The boundary this test targets: "system
prompts that are not well approximated by steering vectors are not
subliminally learned" -- inference-time steering effectiveness predicts
trained-student trait affinity across 16 animal traits and three
models (rho = 0.752, p < 0.001); the four traits with 0.0% steering
rate were not learned at all. (The paper's second boundary --
adaptive optimizers are necessary because a small fraction of LoRA
parameters carry outsized gradients -- is NOT tested here; see
Limitations in the PR. In this linear toy there is no parameter
subset that can carry outsized gradients, and plain SGD's deficit is
generic slow convergence on an ill-conditioned quadratic, not the
paper's drowning mechanism, so an Adam-vs-SGD arm would not be
evidence about the paper's claim either way.)

Toy operationalization: linear states h in R^24; the teacher's outputs
are a 32-dim readout y = W (h + shift(x)) + noise (sigma = 0.1), W a
fixed random full-column-rank map per seed. A "prompt" is a shift
field shift(x); prompt families:

  pure translation   shift(x) = delta                     (exactly a
                     steering vector; |delta| = 2)
  graded family      shift(x) = delta + rho * s(x) * delta1,
                     s(x) = sign(x . a), rho in {0.5, 1, 2}: a constant
                     part plus an input-dependent part of growing
                     relative size (|delta1| = |delta|)
  pure input-dep.    shift(x) = s(x) * delta1             (zero mean:
                     no single vector approximates it)
  controls           clean (shift = 0) and a random-prompt control
                     (constant delta_rand, a different direction)

Pre-flight gate (computed from teacher states alone, BEFORE any
training): R^2 of the best single-vector fit to the per-input shifts,
||mean shift||^2 / mean ||shift||^2. In expectation it is
1 / (1 + rho^2) on the graded family, 1 for pure translation, ~0 for
the pure input-dependent prompt.

Student: a base readout plus a trainable vector v added to the state
(the installed vector is literally a parameter), trained by full-batch
Adam regression (512 train inputs, 400 steps, lr 0.05) on teacher
outputs -- the distillation step in miniature.

Metrics per seed, held-out (2048 fresh inputs) unless noted:
  gate     the pre-flight R^2 above (train inputs, teacher only)
  gain     held-out behavior gain: fraction of the teacher's output-
           shift energy the student's constant shift reproduces,
           1 - E||W(shift(x) - v)||^2 / E||W shift(x)||^2
  EAS      cos(v, delta) against the prompt's constant component (the
           toy v_teacher for the translation/graded arms), tracked
           over training for the pure arm
  |v|      installed-vector norm (what the pure input-dependent and
           clean arms are judged on: cos of a near-zero vector is
           sample noise, so the assertion is that nothing installs)

Measured on pop-os (particlegan 0.10.1): gate 1.000 / 0.80 / 0.50 /
0.21 across rho = 0 / 0.5 / 1 / 2 and 0.000-0.007 for the pure
input-dependent prompt; held-out gain tracks the gate within 0.06 at
every level (1.000 / 0.76-0.83 / 0.45-0.56 / 0.17-0.24, pure
input-dependent |gain| <= 0.011) and is strictly decreasing in rho on
every seed; final EAS vs the constant component 0.987-1.000 on all
graded arms, rising over training on the pure arm (0.56-0.74 at step 0
to 1.000); the pure input-dependent arm installs |v| = 0.010-0.171
against 2.0 for the pure arm, the clean control installs
|v| <= 0.013, and the random-prompt control installs its own vector
(EAS 1.000 vs delta_rand) with |EAS| <= 0.38 against delta --
alignment is direction-specific, not generic drift. Assertions below
encode exactly these margins, no more. Skips cleanly when the
installed particlegan does not provide the EMPF recipe. Pure
numpy/torch -- no GAN machinery, because the claim is about
distillation geometry, not training.
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


D = 24            # state dimension
M = 32            # teacher readout dimension (> D, so W has full column rank)
S = 2.0           # prompt shift magnitude |delta| = |delta1| = |delta_rand|
NTR = 512         # distillation train inputs
NTE = 2048        # held-out inputs
STEPS = 400       # full-batch Adam steps
LR = 0.05
NOISE = 0.1       # teacher output noise sigma
RHOS = (0.0, 0.5, 1.0, 2.0)
SEEDS = (0, 1, 2)


def _unit(rng):
    v = rng.standard_normal(D)
    return v / np.linalg.norm(v)


def _cos(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float(x @ y / (nx * ny))


def _gate_r2(shifts):
    """Pre-flight statistic: fraction of the per-input shift energy a
    single constant vector (the mean shift) explains."""
    return float(np.sum(shifts.mean(axis=0) ** 2)
                 / np.mean(np.sum(shifts ** 2, axis=1)))


def _train_student(W, X, shifts, noise, v_teacher):
    """Distill: fit the student's constant vector v on teacher outputs.

    Returns (v, eas_trajectory) with EAS = cos(v_t, v_teacher) at
    steps (0, 9, 49, 199, STEPS-1); EAS is 0.0 while v is ~0 or the
    teacher vector is ~0 (the pure input-dependent case)."""
    Wt = torch.tensor(W)
    Xt = torch.tensor(X)
    Y = (Xt + torch.tensor(shifts)) @ Wt.T + torch.tensor(noise)
    v = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([v], lr=LR)
    traj = []
    for t in range(STEPS):
        opt.zero_grad()
        loss = torch.mean(((Xt + v) @ Wt.T - Y) ** 2)
        loss.backward()
        opt.step()
        if t in (0, 9, 49, 199, STEPS - 1):
            traj.append(_cos(v.detach().numpy(), v_teacher))
    return v.detach().numpy(), traj


def _gain(W, shifts_te, v):
    """Held-out behavior gain: fraction of teacher output-shift energy
    reproduced by the student's constant shift."""
    den = float(np.mean(np.sum((shifts_te @ W.T) ** 2, axis=1)))
    if den <= 0:
        return 0.0
    num = float(np.mean(np.sum(((shifts_te - v[None, :]) @ W.T) ** 2, axis=1)))
    return 1.0 - num / den


def _run_seed(seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((M, D))
    delta = _unit(rng) * S
    delta1 = _unit(rng) * S
    delta_rand = _unit(rng) * S
    a = _unit(rng)
    Xtr = rng.standard_normal((NTR, D))
    Xte = rng.standard_normal((NTE, D))
    s_tr = np.sign(Xtr @ a)
    s_te = np.sign(Xte @ a)
    noise = rng.standard_normal((NTR, M)) * NOISE

    arms = {}
    for rho in RHOS:
        sh_tr = delta[None, :] + rho * s_tr[:, None] * delta1[None, :]
        sh_te = delta[None, :] + rho * s_te[:, None] * delta1[None, :]
        v, traj = _train_student(W, Xtr, sh_tr, noise, delta)
        arms[rho] = {
            "gate": _gate_r2(sh_tr),
            "gain": _gain(W, sh_te, v),
            "eas": _cos(v, delta),
            "vnorm": float(np.linalg.norm(v)),
            "traj": traj,
        }
    # Pure input-dependent prompt: zero-mean shift field.
    sh_tr = s_tr[:, None] * delta1[None, :]
    sh_te = s_te[:, None] * delta1[None, :]
    v, _ = _train_student(W, Xtr, sh_tr, noise, delta1)
    arms["pure_id"] = {
        "gate": _gate_r2(sh_tr),
        "gain": _gain(W, sh_te, v),
        "vnorm": float(np.linalg.norm(v)),
    }
    # Clean control: unprompted teacher.
    v, _ = _train_student(W, Xtr, np.zeros_like(Xtr), noise, delta)
    arms["clean"] = {"vnorm": float(np.linalg.norm(v))}
    # Random-prompt control: a constant vector, but the wrong one.
    v, _ = _train_student(
        W, Xtr, np.broadcast_to(delta_rand, Xtr.shape).copy(), noise, delta)
    arms["rand"] = {
        "eas_vs_delta": _cos(v, delta),
        "eas_vs_own": _cos(v, delta_rand),
        "vnorm": float(np.linalg.norm(v)),
    }
    return arms


def test_svd_approximability_gate():
    _empf_recipe()
    for seed in SEEDS:
        arms = _run_seed(seed)

        # Gate: exact on the pure arm, ~1/(1+rho^2) on the graded
        # family, ~0 where no single vector approximates the prompt.
        assert arms[0.0]["gate"] >= 0.98
        gates = [arms[rho]["gate"] for rho in RHOS]
        assert all(g1 > g2 for g1, g2 in zip(gates, gates[1:]))
        assert 0.40 <= arms[1.0]["gate"] <= 0.60
        assert arms[2.0]["gate"] <= 0.30
        assert arms["pure_id"]["gate"] <= 0.05

        # Transfer tracks the gate: full on the pure arm, strictly
        # decreasing along the graded family, ~0 for the prompt no
        # single vector approximates.
        gains = [arms[rho]["gain"] for rho in RHOS]
        assert arms[0.0]["gain"] >= 0.98
        assert all(g1 > g2 for g1, g2 in zip(gains, gains[1:]))
        for rho in RHOS:
            assert abs(arms[rho]["gain"] - arms[rho]["gate"]) <= 0.10
        assert arms[2.0]["gain"] <= 0.30
        assert abs(arms["pure_id"]["gain"]) <= 0.05

        # The constant component itself IS installed at every rho
        # (EAS ~ 1): what the gate rations is how much of the
        # prompt's behavior that component carries.
        for rho in RHOS:
            assert arms[rho]["eas"] >= 0.95
        # EAS rises over training on the pure arm (paper Fig. 2 shape).
        traj = arms[0.0]["traj"]
        assert traj[-1] >= 0.98
        assert traj[-1] > traj[0] + 0.20

        # Nothing installs where there is no constant component:
        # pure input-dependent prompt and clean control stay ~empty
        # next to the pure arm's |v| ~ |delta| = 2.
        assert arms[0.0]["vnorm"] >= 1.9
        assert arms["pure_id"]["vnorm"] <= 0.25
        assert arms["clean"]["vnorm"] <= 0.05

        # Specificity: a random constant prompt distills its own
        # vector, not delta -- alignment is not generic drift.
        assert arms["rand"]["eas_vs_own"] >= 0.98
        assert abs(arms["rand"]["eas_vs_delta"]) <= 0.50
