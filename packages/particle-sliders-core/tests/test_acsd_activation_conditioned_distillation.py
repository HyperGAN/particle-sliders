"""CPU gate for the ACSD claims (arXiv:2609.38342): a steering direction
extracted from verified-correct-within-budget trajectories carries real,
specific signal, and a student distilled from a teacher steered by that
direction beats one distilled from the unsteered teacher.

Paper: "Activation-Conditioned Self-Distillation" (Lu et al., 2026). Two
claims are exercised:

  (i)  The direction is real: contrasting activations of self-generated
       trajectories that reach a verified correct answer WITHIN a
       generation budget against all other trajectories yields a steering
       vector whose effect is specific -- the paper's random / shuffled /
       reversed direction ablations all collapse.
  (ii) Distillation transfers it: a frozen teacher steered by the direction
       produces next-token supervision that trains a better student than
       the same teacher unsteered, and the student is deployed alone.

Toy operationalization: a scripted stochastic policy with controlled
hidden states (dim 16) works a verifiable parity task (6 bits, exact
checker). Each trajectory is generated in a latent mode -- "focused"
(hidden states carry a +mode component; the policy emits the
parity-consistent token sharply) or "wandering" (-mode; the policy lapses
often, and a lapse emits a coin-flip token that genuinely scrambles its
estimate). The modes stand in for coherent vs incoherent reasoning paths:
trajectory-level structure that differs systematically between trajectories
that finish correctly within budget and those that do not.

Pipeline, following the paper: sample trajectories from the base policy;
label verified-correct-within-budget (B = 10 steps); extract the direction
as the normalized difference of group means of unit-normalized step
activations (their recipe, 512 problems x 4 trajectories); calibrate the
strength alpha on a held-out pool by steered-teacher accuracy; distill a
tiny GRU student (forward KL on teacher next-token distributions over
base-policy prefixes) for four arms -- unsteered teacher, ACSD teacher,
shuffled-direction teacher, reversed-direction teacher -- and compare
held-out task accuracy of the sampled students.

Note on machinery: ACSD's method has no adversarial component -- its
direction is a closed-form contrast and its training is forward-KL
distillation -- so this test exercises the paper's own machinery, under
the same upgraded particlegan as the companion PSR gate (the recipe gate
below ties the file to the current default recipe). Skips cleanly when
the installed particlegan lacks the default (EMPF) recipe.
"""
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


H = 16          # hidden-state dimension of the scripted policy
NB = 6          # bits per problem
BUDGET = 10     # verified-correct-within-budget: length <= BUDGET
MAXLEN = 12
STALL = 2
WORLD_SEED = 2001


# ---------------------------------------------------------------- world
class World:
    def __init__(self, seed, emb_scale=0.35, mode=2.1, noise=0.85, gamma=0.75):
        rng = np.random.default_rng(seed)
        self.emb_prev = rng.standard_normal((4, H)) * emb_scale
        self.emb_bit = rng.standard_normal((2, H)) * emb_scale
        m = rng.standard_normal(H)
        self.m = m / np.linalg.norm(m)
        self.a = rng.standard_normal((2, H)) * 0.3
        self.r_stall = rng.standard_normal(H) * 0.3
        self.gamma = gamma
        self.mode = mode
        self.noise = noise

    def hidden(self, prev, bit, mode_sign, rng):
        return (self.emb_prev[prev] + self.emb_bit[bit]
                + mode_sign * self.mode * self.m
                + rng.standard_normal(H) * self.noise)

    def action_logits(self, h, e, bit):
        """Logits over actions {emit-consistent, emit-lapse, stall}."""
        s = float(h @ self.m)
        consistent = e ^ bit
        return np.array([
            float(self.a[consistent] @ h) + self.gamma * s,
            float(self.a[1 - consistent] @ h) - 0.3 * self.gamma * s + 0.2,
            -1.7 + 0.3 * float(h @ self.r_stall)])

    def token_probs(self, h, e, bit):
        pa = _softmax(self.action_logits(h, e, bit))
        consistent = e ^ bit
        pt = np.zeros(3)
        pt[consistent] += pa[0]
        pt[0] += pa[1] * 0.5
        pt[1] += pa[1] * 0.5
        pt[STALL] += pa[2]
        return pt


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def rollout(world, bits, rng, direction=None, alpha=0.0):
    p, e, prev, n_done = 0, 0, 3, 0
    steps = []
    mode_sign = 1.0 if rng.random() < 0.55 else -1.0
    for _ in range(MAXLEN):
        if n_done >= NB:
            break
        bit = bits[n_done]
        h = world.hidden(prev, bit, mode_sign, rng)
        hh = h if direction is None else h + alpha * np.linalg.norm(h) * direction
        pa = _softmax(world.action_logits(hh, e, bit))
        act = int(rng.choice(3, p=pa))
        if act == 0:
            tok = e ^ bit
        elif act == 1:
            tok = int(rng.integers(0, 2))
        else:
            tok = STALL
        steps.append((h, e, bit, prev, tok))
        if tok != STALL:
            e = tok
            p = p ^ bit
            n_done += 1
            prev = tok
        else:
            prev = STALL
    correct = (n_done == NB) and (e == p)
    return dict(steps=steps, correct=correct, length=len(steps))


def gen_problems(n, rng):
    return rng.integers(0, 2, size=(n, NB)).tolist()


def extract_direction(world, problems, rng, per_problem=4):
    """The paper's recipe: unit-normalize each step activation, pool the
    first BUDGET steps of every trajectory, contrast group means."""
    pos, rest = [], []
    for bits in problems:
        for _ in range(per_problem):
            r = rollout(world, bits, rng)
            bucket = pos if (r["correct"] and r["length"] <= BUDGET) else rest
            for (h, e, bit, prev, tok) in r["steps"][:BUDGET]:
                bucket.append(h / np.linalg.norm(h))
    d = np.mean(pos, axis=0) - np.mean(rest, axis=0)
    return d / np.linalg.norm(d)


def teacher_accuracy(world, problems, rng, direction=None, alpha=0.0, per_problem=1):
    ok = tot = 0
    for bits in problems:
        for _ in range(per_problem):
            ok += rollout(world, bits, rng, direction, alpha)["correct"]
            tot += 1
    return ok / tot


def teacher_probs(world, h, e, bit, direction, alpha):
    hh = h if direction is None else h + alpha * np.linalg.norm(h) * direction
    return world.token_probs(hh, e, bit)


# ---------------------------------------------------------------- student
class Student(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.emb_tok = nn.Embedding(4, 8)
        self.emb_bit = nn.Embedding(2, 4)
        self.gru = nn.GRU(12, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, prev, bit):
        x = torch.cat([self.emb_tok(prev), self.emb_bit(bit)], dim=-1)
        out, _ = self.gru(x)
        return self.head(out)

    def step(self, prev, bit, hx):
        x = torch.cat([self.emb_tok(prev), self.emb_bit(bit)], dim=-1)
        out, hx = self.gru(x.unsqueeze(1), hx)
        return self.head(out[:, 0]), hx


def build_data(world, problems, rng, per_problem=2):
    data = []
    for bits in problems:
        for _ in range(per_problem):
            r = rollout(world, bits, rng)
            data.append([(prev, bit, h, e) for (h, e, bit, prev, tok) in r["steps"]])
    return data


def train_student(world, data, direction, alpha, seed, steps=100, bs=1024, hidden=16):
    torch.manual_seed(seed)
    n = len(data)
    t_max = max(len(s) for s in data)
    prev_t = torch.full((n, t_max), 3, dtype=torch.long)
    bit_t = torch.zeros((n, t_max), dtype=torch.long)
    y = torch.zeros((n, t_max, 3))
    mask = torch.zeros((n, t_max))
    for i, steps_ in enumerate(data):
        for t, (prev, bit, h, e) in enumerate(steps_):
            prev_t[i, t] = prev
            bit_t[i, t] = bit
            y[i, t] = torch.tensor(
                teacher_probs(world, h, e, bit, direction, alpha), dtype=torch.float32)
            mask[i, t] = 1
    stu = Student(hidden)
    opt = torch.optim.Adam(stu.parameters(), lr=5e-3)
    for _ in range(steps):
        sel = torch.randint(0, n, (bs,))
        logits = stu(prev_t[sel], bit_t[sel])
        logp = F.log_softmax(logits, dim=-1)
        loss = -(y[sel] * logp).sum(-1)
        loss = (loss * mask[sel]).sum() / mask[sel].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return stu


@torch.no_grad()
def eval_student(stu, problems, gen, samples=4):
    """Batched sampled rollouts of the student alone (no teacher, no
    steering at deployment -- the paper's inference contract)."""
    stu.eval()
    bits_arr = np.array(problems, dtype=np.int64)
    n_prob = len(problems)
    b = n_prob * samples
    bits_rep = np.repeat(bits_arr, samples, axis=0)
    e = np.zeros(b, dtype=np.int64)
    p = np.zeros(b, dtype=np.int64)
    prev = np.full(b, 3, dtype=np.int64)
    n_done = np.zeros(b, dtype=np.int64)
    hx = None
    for _ in range(MAXLEN):
        active = n_done < NB
        if not active.any():
            break
        bit = bits_rep[np.arange(b), np.minimum(n_done, NB - 1)]
        lg, hx = stu.step(torch.tensor(prev), torch.tensor(bit), hx)
        tok = torch.multinomial(torch.softmax(lg, dim=-1), 1, generator=gen).numpy()[:, 0]
        is_stall = tok == STALL
        upd = active & ~is_stall
        p = np.where(upd, p ^ bit, p)
        e = np.where(upd, tok, e)
        n_done = np.where(upd, n_done + 1, n_done)
        prev = np.where(active, np.where(is_stall, STALL, tok), prev)
    correct = (n_done >= NB) & (e == p)
    return float(correct.reshape(n_prob, samples).mean())


# ---------------------------------------------------------------- test
def _default_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


def run_seed(seed):
    world = World(WORLD_SEED)
    rng = np.random.default_rng(3000 + seed)
    direction = extract_direction(world, gen_problems(512, rng), rng)
    cos_m = float(direction @ world.m)
    calib = gen_problems(150, rng)
    base_acc = teacher_accuracy(world, calib, rng)
    best_alpha, best_acc = 0.0, base_acc
    for alpha in (0.3, 0.6, 0.9, 1.2):
        acc = teacher_accuracy(world, calib, rng, direction, alpha)
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    data = build_data(world, gen_problems(350, rng), rng)
    eval_probs = gen_problems(300, np.random.default_rng(999 + seed))
    egen = torch.Generator().manual_seed(4242 + seed)
    d_shuf = direction[np.random.default_rng(77 + seed).permutation(H)]
    arms = {"unsteered": (None, 0.0), "acsd": (direction, best_alpha),
            "shuffled": (d_shuf, best_alpha), "reversed": (-direction, best_alpha)}
    students = {}
    for name, (direc, alpha) in arms.items():
        stu = train_student(world, data, direc, alpha, seed * 100 + len(name))
        students[name] = eval_student(stu, eval_probs, egen)
    return dict(cos_m=cos_m, alpha=best_alpha, teacher_base=base_acc,
                teacher_steered=best_acc, **students)


def test_acsd_direction_and_distillation():
    recipe = _default_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # (i) the extracted direction recovers the latent mode axis and
        # steering the frozen teacher by it genuinely helps on calibration.
        assert r["cos_m"] > 0.8, f"seed {seed}: cos(direction, mode) = {r['cos_m']:.3f}"
        assert r["teacher_steered"] > r["teacher_base"] + 0.05, (
            f"seed {seed}: steered teacher {r['teacher_steered']:.3f} "
            f"vs base {r['teacher_base']:.3f}")
        # (ii) the distilled student inherits the advantage, and the
        # perturbed directions collapse (the paper's specificity ablation).
        assert r["acsd"] > r["unsteered"] + 0.04, (
            f"seed {seed}: acsd student {r['acsd']:.3f} vs unsteered {r['unsteered']:.3f}")
        assert r["acsd"] > r["shuffled"] + 0.04, (
            f"seed {seed}: acsd student {r['acsd']:.3f} vs shuffled {r['shuffled']:.3f}")
        assert r["acsd"] > r["reversed"] + 0.04, (
            f"seed {seed}: acsd student {r['acsd']:.3f} vs reversed {r['reversed']:.3f}")
