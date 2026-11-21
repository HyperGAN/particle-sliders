"""CPU gate for the OPD objective claim (arXiv:2611.23397): the choice
of on-policy distillation objective governs which tokens are learned,
in what order -- sequence-level objectives exhibit a future-agreement
bias that myopic token-level objectives avoid, inducing a reverse
learning order and favoring teacher-improbable tokens at early
positions; and the divergence choice governs optimization amenability
-- reverse KL gradients can vanish where forward KL gradients do not.

Paper: "Which Tokens, in What Order, and How Fast? A Theoretical
Analysis of On-Policy Distillation Objectives" (Strauss, Fowl &
Razin). It formally analyzes sequence-level vs token-level OPD and
reports: (1) sequence-level OPD exhibits a future-agreement bias,
which myopic token-level objectives avoid; (2) the bias induces a
reverse learning order -- sequence-level training first matches the
teacher at later token positions, while at earlier positions it can
favor tokens that are improbable under the teacher (and thus often
undesirable); (3) forward KL enjoys a well-behaved objective
landscape, while other f-divergences such as reverse KL can suffer
vanishing gradients even when student and teacher are far apart.
Experiments across LM post-training settings demonstrate the
implications. The abstract reports NO headline numbers -- this test
asserts the qualitative/structural claims (learning order,
early-position improbable-token elevation, and the reverse-KL
vanishing-gradient regime), not point values.

Toy operationalization: an exactly enumerable sequence model. Vocab
V = 3, length T = 4 (81 sequences, so the full sequence distribution
is computed exactly -- no estimator noise in either arm).

  Teacher: an agreement ("copy") chain. Position 0 is drawn from a
  peaked distribution p1; every later position copies position 0's
  token with probability 0.85 (else splits the remaining mass). The
  teacher's sequence identity is therefore decided at position 0,
  and its later positions are conditionals on that early token.

  Student: per-position categorical logits (a factorized sequence
  distribution). It cannot represent the teacher's correlation, so
  HOW it compromises is exactly what the objective decides. Later
  positions start with a bias toward token 1 -- the student's own
  pre-existing future tendency, the thing a sequence-level objective
  can agree with and a token-level objective cannot see.

  Arm A (token-level): gradient descent on the sum of per-position
  forward KLs KL(teacher marginal_t || student_t) -- the myopic
  objective; each position is matched independently.
  Arm B (sequence-level): gradient descent on the exact sequence
  reverse KL KL(student_seq || teacher_seq). Its gradient at
  position 0 flows through the whole-sequence agreement: moving
  position 0 changes how well EVERY sampled sequence agrees with
  the teacher's copy structure, so position 0 is pulled toward
  whatever the student's future positions already favor.
  Both arms: same initialization, same step count, same learning
  rate within each regime (matched compute).

Regime 1 (learning order), p1 = (0.8, 0.1, 0.1), mild future bias:
  Metric: per-position KL(teacher marginal || student) over
  training; a position is "settled" at the first step after which
  its KL stays below 0.10 for the rest of the run (never = cap).
  Token arm: every position settles within 2-3 steps, no order.
  Sequence arm: later positions settle at step 6 (at their
  factorized floor, KL 0.074), while position 0's KL first dips,
  then RISES from 0.454-0.474 at init to 0.633 at the end -- the
  earliest position is sacrificed to future agreement and never
  settles. Spearman correlation between position index and settled
  step is negative for the sequence arm (earliest position last).

Regime 2 (improbable-token elevation), p1 = (0.7, 0.2, 0.1), strong
future bias toward token 1 (teacher probability 0.20 at position 0):
  Sequence arm ends with position 0 at ~(0.013, 0.982, 0.004) --
  98% of its mass on the teacher-improbable token 1, because token 1
  is what its own future positions produce and the copy-teacher
  rewards agreement with them. The later positions end at
  ~(0.08, 0.84, 0.08), matching the teacher's copy conditional given
  token 1 to TV ~0.01: the future is learned, the first position is
  bent to fit it. Token arm ends at the teacher marginals exactly
  (position 0 token-1 mass 0.20). Control -- sequence arm with NO
  future bias: position 0 collapses onto the teacher's dominant
  token 0 instead (~0.995, token-1 mass ~0.003), proving the
  elevation is the future-agreement pull, not sequence training
  per se.

Divergence arm (single position, teacher (0.7, 0.2, 0.1)): at a
confidently wrong student (logits (0, 8, 0) or (0, 0, 8); forward
KL 5.6-6.4 nats from the teacher) the forward-KL gradient norm is
1.07-1.16 while the reverse-KL gradient norm is 0.007-0.008
(~150x smaller) -- reverse KL is nearly stationary exactly where
the student is most wrong. At a uniform student both gradients are
healthy (~0.45), so the vanishing is regime-specific.

Assertions below encode roughly half the thinnest observed margins.
Measured on pop-os (particlegan 0.10.1). Skips cleanly when the
installed particlegan does not provide the EMPF recipe. Pure
numpy/torch categorical geometry -- no GAN machinery, because the
claim is about training objectives, not training a GAN.
"""
import itertools

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


V = 3             # vocab size
T = 4             # sequence length (V**T = 81 sequences, exact)
COPY = 0.85       # teacher: P(x_t = x_0 | x_0) for t >= 1
SEQS = np.array(list(itertools.product(range(V), repeat=T)))
NSEQ = len(SEQS)
SETTLE_KL = 0.10  # per-position KL below this counts as matched


def _teacher(p1):
    """Full teacher sequence distribution and per-position marginals."""
    p1 = np.asarray(p1, dtype=np.float64)
    cond = np.full((V, V), (1.0 - COPY) / (V - 1))
    np.fill_diagonal(cond, COPY)          # cond[x0, x_t]
    p_seq = np.zeros(NSEQ)
    for i, x in enumerate(SEQS):
        p = p1[x[0]]
        for t in range(1, T):
            p *= cond[x[0], x[t]]
        p_seq[i] = p
    marg = np.zeros((T, V))
    for t in range(T):
        for v in range(V):
            marg[t, v] = p_seq[SEQS[:, t] == v].sum()
    return p_seq, marg, cond


def _run_arm(p1, arm, steps, lr, seed, future_bias):
    """Train one arm; return per-step per-position KL and final q.

    arm "token": GD on sum_t KL(marg_t || q_t), grad q - marg.
    arm "seq":   GD on exact KL(q_seq || p_seq) via autograd.
    Init: small seeded noise; positions >= 1 biased toward token 1
    by `future_bias` (the student's pre-existing future tendency).
    """
    p_seq, marg, _ = _teacher(p1)
    gen = torch.Generator().manual_seed(7000 + seed)
    z = torch.zeros(T, V, dtype=torch.float64)
    z.add_(torch.randn(T, V, generator=gen, dtype=torch.float64) * 0.05)
    with torch.no_grad():
        z[1:, 1] += future_bias
        z[1:, 0] -= future_bias * 0.5
    seq_idx = torch.tensor(SEQS)
    pos_idx = torch.arange(T)[None, :].expand(NSEQ, T)
    p_seq_t = torch.tensor(p_seq)
    marg_t = torch.tensor(marg)
    if arm == "seq":
        z.requires_grad_(True)
        opt = torch.optim.SGD([z], lr=lr)
    kls = []
    for _ in range(steps + 1):
        with torch.no_grad():
            q = torch.softmax(z, -1)
            kl = (marg_t * (torch.log(marg_t) - torch.log(q))).sum(-1)
            kls.append(kl.numpy().copy())
        if arm == "token":
            with torch.no_grad():
                z.sub_(lr * (q - marg_t))
        else:
            q = torch.softmax(z, -1)
            p_s = q[pos_idx, seq_idx].prod(1)
            loss = torch.sum(p_s * (torch.log(p_s) - torch.log(p_seq_t)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        q_final = torch.softmax(z, -1).numpy()
    return np.array(kls), q_final, marg


def _settled_step(kl_col, steps):
    """First step after which KL stays below SETTLE_KL; cap if never."""
    below = kl_col < SETTLE_KL
    if not below[-1]:
        return steps + 1
    s = steps
    while s > 0 and below[s - 1]:
        s -= 1
    return s


def _spearman(x, y):
    """Spearman rank correlation (average ranks for ties), numpy only."""
    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float(rx @ ry / denom) if denom > 0 else 0.0


def test_sequence_objective_learns_in_reverse_order():
    _empf_recipe()
    steps, lr = 100, 1.0
    for seed in (0, 1, 2):
        kl_a, _, _ = _run_arm([0.8, 0.1, 0.1], "token", steps, lr,
                              seed, future_bias=0.6)
        kl_b, _, _ = _run_arm([0.8, 0.1, 0.1], "seq", steps, lr,
                              seed, future_bias=0.6)
        set_a = [_settled_step(kl_a[:, t], steps) for t in range(T)]
        set_b = [_settled_step(kl_b[:, t], steps) for t in range(T)]

        # Token arm: every position matches, essentially at once --
        # no learning order to speak of.
        assert max(set_a) <= 5, (
            f"seed {seed}: token-arm settled steps {set_a}")
        assert max(set_a) - min(set_a) <= 3, (
            f"seed {seed}: token-arm settled steps {set_a} show an "
            f"order the myopic objective should not have")
        assert kl_a[-1, 0] < 0.01, (
            f"seed {seed}: token-arm position-0 final KL "
            f"{kl_a[-1, 0]:.4f}")

        # Sequence arm: later positions settle at their floor while
        # position 0 never settles -- reverse learning order.
        assert all(s <= 20 for s in set_b[1:]), (
            f"seed {seed}: sequence-arm later settled steps {set_b}")
        assert set_b[0] > 50, (
            f"seed {seed}: sequence-arm settled steps {set_b} -- "
            f"position 0 should never settle")
        rho = _spearman(range(T), set_b)
        assert rho <= -0.7, (
            f"seed {seed}: sequence-arm Spearman(position, settled) "
            f"{rho:+.2f} from settled steps {set_b}")
        # Position 0 does not merely stall, it regresses: future
        # agreement is bought with early-position teacher match.
        assert kl_b[-1, 0] > kl_b[0, 0] + 0.1, (
            f"seed {seed}: sequence-arm position-0 KL init "
            f"{kl_b[0, 0]:.3f} -> final {kl_b[-1, 0]:.3f}")
        assert all(kl_b[-1, t] < 0.25 * kl_b[0, t]
                   for t in range(1, T)), (
            f"seed {seed}: sequence-arm later-position KL init "
            f"{np.round(kl_b[0, 1:], 3)} -> final "
            f"{np.round(kl_b[-1, 1:], 3)}")


def test_sequence_objective_elevates_improbable_early_token():
    _empf_recipe()
    steps, lr = 150, 0.5
    p1 = [0.7, 0.2, 0.1]
    _, _, cond = _teacher(p1)
    for seed in (0, 1, 2):
        _, q_a, _ = _run_arm(p1, "token", steps, lr, seed,
                             future_bias=1.0)
        _, q_b, _ = _run_arm(p1, "seq", steps, lr, seed,
                             future_bias=1.0)
        _, q_ctrl, _ = _run_arm(p1, "seq", steps, lr, seed,
                                future_bias=0.0)

        # Token arm reproduces the teacher at position 0: token 1
        # keeps exactly its teacher probability (0.20).
        assert abs(q_a[0, 1] - 0.2) <= 0.02, (
            f"seed {seed}: token-arm position-0 dist "
            f"{np.round(q_a[0], 3)}")
        assert 0.5 * np.abs(q_a[0] - p1).sum() <= 0.02, (
            f"seed {seed}: token-arm position-0 TV to teacher "
            f"{0.5 * np.abs(q_a[0] - p1).sum():.3f}")

        # Sequence arm: position 0 is dragged onto the token its own
        # future favors -- 98% mass on a token the teacher gives 0.20.
        assert q_b[0, 1] >= 0.90, (
            f"seed {seed}: sequence-arm position-0 dist "
            f"{np.round(q_b[0], 3)} -- improbable token 1 should "
            f"dominate")
        assert q_b[0, 1] > 4 * 0.2, (
            f"seed {seed}: sequence-arm position-0 token-1 mass "
            f"{q_b[0, 1]:.3f} vs teacher 0.20")
        assert 0.5 * np.abs(q_b[0] - p1).sum() >= 0.6, (
            f"seed {seed}: sequence-arm position-0 TV to teacher "
            f"{0.5 * np.abs(q_b[0] - p1).sum():.3f}")
        # ...while its later positions genuinely match the teacher's
        # copy conditional given the produced token: the future is
        # what got learned.
        late_tv = [0.5 * np.abs(q_b[t] - cond[1]).sum()
                   for t in range(1, T)]
        assert max(late_tv) <= 0.05, (
            f"seed {seed}: sequence-arm later-position TV to "
            f"teacher copy-conditional {np.round(late_tv, 3)}")

        # Control: no future bias, no elevation -- the sequence arm
        # collapses onto the teacher's dominant token instead. The
        # improbable token is chosen by the student's future, not
        # by the teacher and not by sequence training alone.
        assert q_ctrl[0, 1] <= 0.05 and q_ctrl[0, 0] >= 0.95, (
            f"seed {seed}: unbiased sequence-arm position-0 dist "
            f"{np.round(q_ctrl[0], 3)}")


def test_reverse_kl_gradient_vanishes_where_forward_kl_does_not():
    _empf_recipe()
    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)

    def grad_norms(z_init):
        z = torch.tensor(z_init, dtype=torch.float64,
                         requires_grad=True)
        q = torch.softmax(z, -1)
        fkl = torch.sum(p * (torch.log(p) - torch.log(q)))
        rkl = torch.sum(q * (torch.log(q) - torch.log(p)))
        gf, = torch.autograd.grad(fkl, z, retain_graph=True)
        gr, = torch.autograd.grad(rkl, z)
        return (float(fkl.detach()), float(gf.norm()),
                float(gr.norm()))

    # Confidently wrong students: far from the teacher (forward KL
    # 5.6-6.4 nats), forward KL still pushes hard, reverse KL is
    # nearly stationary -- the paper's vanishing-gradient regime.
    for z_init in ([0.0, 8.0, 0.0], [0.0, 0.0, 8.0]):
        fkl, nf, nr = grad_norms(z_init)
        assert fkl >= 5.0, f"FKL at {z_init}: {fkl:.2f}"
        assert nf >= 0.8, f"forward-KL grad norm at {z_init}: {nf:.3f}"
        assert nr <= 0.02, (
            f"reverse-KL grad norm at {z_init}: {nr:.4f} -- should "
            f"have vanished")
        assert nf >= 50 * nr, (
            f"grad norms at {z_init}: forward {nf:.3f} vs reverse "
            f"{nr:.4f}")

    # At a uniform student both divergences have healthy gradients:
    # the vanishing is a far-and-confident regime, not a global
    # property of reverse KL.
    _, nf, nr = grad_norms([0.0, 0.0, 0.0])
    assert nf >= 0.3 and nr >= 0.3, (
        f"uniform-student grad norms: forward {nf:.3f}, reverse "
        f"{nr:.3f}")
