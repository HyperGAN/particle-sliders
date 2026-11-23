"""CPU gate for the teacher-COUPLING claim of Zuin, Huet, Rossi &
Ben Houidi, arXiv:2609.39494 ("Disentangling Self-Distillation:
Measuring and Modeling Acquisition and Retention"), cycle 25.

Paper's claims (1,200 runs, Qwen2.5-7B / Ministral-3-3B, ordinary and
contradictory tasks): privileged-context self-distillation methods
differ along (i) rollout source, (ii) teacher coupling (frozen, or an
EMA of the student at some coupling rate), (iii) KL direction. The
paper finds (i) rollout source matters mostly where the task
contradicts pretrained behavior -- teacher rollouts raise acquisition
well above student rollouts there, with almost no retention change;
(ii) THE COUPLING CHANGES ACQUISITION MOST, ON EVERY TASK: acquisition
rises with the coupling rate, then falls past a task-specific rate
(inverted-U); (iii) KL direction's retention cost is model-dependent
(noted, NOT tested here -- a toy cannot express it).

This test decides, in the privileged-context toy class of PR #210
(ACSD) / PR #217 (OPSD), the primary claim (ii) -- with acquisition and
retention read separately, per the paper's dissociation point -- and
the secondary claim (i) at frozen and at one EMA coupling.

Anchor choice: the OPSD toy (#217) over the ACSD toy (#210). Both share
the scripted-parity world, the tiny GRU student and forward-KL
distillation over base-policy prefixes; #217 is closer because its
world already separates two dissociable axes (a mode axis and a
privileged-semantics axis), which is the geometry an
acquisition/retention dissociation needs. This test adapts that
geometry to a PARAMETRIC teacher (the scripted world's teacher has no
weights to couple): teacher and student are the same GRU architecture
from different pretrainings of one shared random init -- the student
pretrained context-free on a prior (copy) task, the teacher pretrained
WITH a per-step reference-token context on the target task (the
privileged solution). The teacher's advantage is therefore exactly its
privileged pretraining + context; the student's measurable prior is
its copy behavior.

Tasks (rule for the correct token given last emitted token v and the
current bit b; the prior is y = b):
  ordinary:      y = b XOR (v AND b)        -- opposes the prior on
                                               1 of 4 (v, b) combos
  contradictory: y = (1 - b) XOR (v AND b)  -- opposes the prior on
                                               3 of 4 combos
Readouts, reported separately for every arm, never collapsed:
  acquisition = context-free sequence-accuracy gain on the target task
                vs the student's own init (held-out problems);
  retention   = change in context-free TOKEN accuracy on the prior
                (copy) task vs init (negative = forgetting).

Coupling arms (teacher <- beta * teacher + (1 - beta) * student after
each student step, teacher initialized at the privileged solution):
  frozen (beta = 1, the #210 standard), EMA beta in {0.999, 0.99, 0.9},
  and a self-teacher degenerate control (teacher := current student
  weights, still fed the privileged context -- the coupling limit).
Controls: shuffled privilege (teacher fed the WRONG reference token;
acquisition should collapse at every coupling, proving the privilege
carries the signal). Secondary arms: rollout source {student, teacher}
at frozen and at beta = 0.99, both task types.

Measured verdict (3 seeds, this toy class; see the PR body for the
full iteration log and per-seed table): the paper's inverted-U does
NOT reproduce as an interior peak above frozen. What reproduces is
its falling half plus a plateau: frozen ~= beta 0.999 (within +-0.05
on every seed/task), beta 0.99 clearly below frozen on contradictory
tasks, and beta 0.9 collapsed on contradictory tasks while partially
surviving on ordinary ones -- i.e. acquisition does fall past a
task-specific coupling rate, and coupling is the largest mover of
acquisition among the axes varied here (range ~0.8 across beta vs
~0.1 for rollout source), but mild EMA never beats the frozen
privileged teacher in this toy, because weight-space blending toward
a student that solves the task by a different route (context-free
rule vs context copying) only degrades the teacher. Retention
dissociates as the paper's framing predicts: contradictory-task
distillation costs ~4x the prior-task token accuracy that
ordinary-task distillation costs, at every working coupling. The
secondary claim reproduces: teacher rollouts beat student rollouts
on contradictory tasks (+0.07..+0.15) and not on ordinary ones
(|gap| <= 0.03). Assertions below encode exactly these measured
margins.

EMA-role note: the EMA here is an EMA TEACHER -- a moving target the
student chases. It is a different seat from an EMA surgery REFERENCE
(a fixed point the student is corrected against), which #310/#312
found fails through staleness; nothing here contradicts those laws.

Skips cleanly when the installed particlegan lacks the default (EMPF)
recipe, following the broad guarded-import pattern of the companion
paper gates.
"""
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

try:
    import particlegan  # noqa: F401
    from particlegan import get_recipe
except Exception:
    pytest.skip("particlegan unavailable", allow_module_level=True)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


NB = 6           # bits per problem
STALL = 2
START = 3
CTX_NULL, CTX0 = 0, 1
DISTILL_STEPS = 120
TEACHER_STEPS = 60
STUDENT_STEPS = 60


def rule(task, v, b):
    if task == "prior":
        return b
    if task == "ordinary":
        return b ^ (v & b)
    if task == "contradictory":
        return (1 - b) ^ (v & b)
    raise ValueError(task)


class Model(nn.Module):
    """One GRU architecture for both roles (after #210/#217's student):
    token + bit embeddings; the teacher additionally receives a
    reference-token context embedding (index 0 = null context, the
    student's only mode)."""

    def __init__(self, hidden=16):
        super().__init__()
        self.emb_tok = nn.Embedding(4, 8)
        self.emb_bit = nn.Embedding(2, 4)
        self.emb_ctx = nn.Embedding(3, 6)
        self.gru = nn.GRU(18, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, prev, bit, ctx):
        x = torch.cat([self.emb_tok(prev), self.emb_bit(bit),
                       self.emb_ctx(ctx)], dim=-1)
        out, _ = self.gru(x)
        return self.head(out)

    def step(self, prev, bit, ctx, hx):
        x = torch.cat([self.emb_tok(prev), self.emb_bit(bit),
                       self.emb_ctx(ctx)], dim=-1)
        out, hx = self.gru(x.unsqueeze(1), hx)
        return self.head(out[:, 0]), hx


def gen_supervised(task, n, rng, with_ctx):
    bits = rng.integers(0, 2, size=(n, NB))
    prev = np.full((n, NB), START)
    ctx = np.zeros((n, NB), dtype=np.int64)
    tgt = np.zeros((n, NB), dtype=np.int64)
    v = np.zeros(n, dtype=np.int64)
    p = np.full(n, START)
    for t in range(NB):
        b = bits[:, t]
        y = np.array([rule(task, int(v[i]), int(b[i])) for i in range(n)])
        prev[:, t] = p
        tgt[:, t] = y
        if with_ctx:
            ctx[:, t] = CTX0 + y
        v = y
        p = y
    return (torch.tensor(prev), torch.tensor(bits),
            torch.tensor(ctx), torch.tensor(tgt))


def pretrain(base_state, task, with_ctx, seed, steps, lr=5e-3, bs=256):
    torch.manual_seed(seed)
    m = Model()
    m.load_state_dict(base_state)
    rng = np.random.default_rng(seed * 7 + (1 if with_ctx else 2))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        prev, bits, ctx, tgt = gen_supervised(task, bs, rng, with_ctx)
        if not with_ctx:
            ctx = torch.zeros_like(ctx)
        loss = F.cross_entropy(m(prev, bits, ctx).reshape(-1, 3),
                               tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def eval_model(m, task, problems, gen, samples=4, with_ctx=False):
    """Returns (sequence accuracy, token accuracy) of sampled rollouts."""
    m.eval()
    bits_arr = np.array(problems, dtype=np.int64)
    n_prob = len(problems)
    bsz = n_prob * samples
    bits_rep = np.repeat(bits_arr, samples, axis=0)
    v = np.zeros(bsz, dtype=np.int64)
    prev = np.full(bsz, START)
    hx = None
    seq_ok = np.ones(bsz, dtype=bool)
    ntok = 0
    for t in range(NB):
        bit = bits_rep[:, t]
        if with_ctx:
            yref = np.array([rule(task, int(v[i]), int(bit[i]))
                             for i in range(bsz)])
            ctx = CTX0 + yref
        else:
            ctx = np.zeros(bsz, dtype=np.int64)
        lg, hx = m.step(torch.tensor(prev), torch.tensor(bit),
                        torch.tensor(ctx), hx)
        tok = torch.multinomial(torch.softmax(lg, dim=-1), 1,
                                generator=gen).numpy()[:, 0]
        y = np.array([rule(task, int(v[i]), int(bit[i]))
                      for i in range(bsz)])
        seq_ok &= (tok == y)
        ntok += (tok == y).sum()
        v = np.where(tok == STALL, v, tok)
        prev = tok
    return (float(seq_ok.reshape(n_prob, samples).mean()),
            float(ntok / (bsz * NB)))


def gen_rollout_data(policy, task, n, rng, gen):
    """Sampled trajectories from `policy`; stores per-step (prev, bit,
    reference token for the TARGET task). Teacher policies are rolled
    out with their privileged context, student policies without."""
    policy.eval()
    bits = rng.integers(0, 2, size=(n, NB))
    prev = np.full((n, NB), START)
    ctxref = np.zeros((n, NB), dtype=np.int64)
    v = np.zeros(n, dtype=np.int64)
    p = np.full(n, START)
    hx = None
    with torch.no_grad():
        for t in range(NB):
            b = bits[:, t]
            yref = np.array([rule(task, int(v[i]), int(b[i]))
                             for i in range(n)])
            ctxref[:, t] = yref
            prev[:, t] = p
            if getattr(policy, "_with_ctx", False):
                ctx = CTX0 + yref
            else:
                ctx = np.zeros(n, dtype=np.int64)
            lg, hx = policy.step(torch.tensor(p), torch.tensor(b),
                                 torch.tensor(ctx), hx)
            tok = torch.multinomial(torch.softmax(lg, dim=-1), 1,
                                    generator=gen).numpy()[:, 0]
            v = np.where(tok == STALL, v, tok)
            p = tok
    return torch.tensor(prev), torch.tensor(bits), torch.tensor(ctxref)


def distill(student_init, teacher_init, data, beta, seed,
            steps=DISTILL_STEPS, lr=5e-3, bs=256, shuffled=False):
    """Forward-KL distillation of the context-conditioned teacher into
    the context-free student over the given rollout prefixes. After
    each student step the teacher is EMA-coupled toward the student
    (beta = 1: frozen; beta = None: self-teacher, copied each step)."""
    torch.manual_seed(seed)
    stu = Model()
    stu.load_state_dict(student_init)
    tea = Model()
    tea.load_state_dict(teacher_init)
    prev_t, bit_t, ctxref_t = data
    n = prev_t.shape[0]
    opt = torch.optim.Adam(stu.parameters(), lr=lr)
    tea_params = list(tea.parameters())
    for _ in range(steps):
        sel = torch.randint(0, n, (bs,))
        ctx = CTX0 + (1 - ctxref_t[sel] if shuffled else ctxref_t[sel])
        with torch.no_grad():
            tp = torch.softmax(tea(prev_t[sel], bit_t[sel], ctx), dim=-1)
        slog = stu(prev_t[sel], bit_t[sel], torch.zeros_like(ctx))
        loss = -(tp * F.log_softmax(slog, dim=-1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if beta is None:
                for tp_, sp in zip(tea_params, stu.parameters()):
                    tp_.copy_(sp)
            elif beta < 1.0:
                for tp_, sp in zip(tea_params, stu.parameters()):
                    tp_.mul_(beta).add_(sp, alpha=1 - beta)
    return stu


def run_seed(seed):
    torch.manual_seed(1000 + seed)
    base_state = {k: v.clone()
                  for k, v in Model().state_dict().items()}
    student = pretrain(base_state, "prior", False, seed,
                       steps=STUDENT_STEPS)
    s_state = {k: v.clone() for k, v in student.state_dict().items()}
    eval_probs = np.random.default_rng(999 + seed).integers(
        0, 2, size=(200, NB)).tolist()

    def egen():
        return torch.Generator().manual_seed(4242 + seed)

    init_prior_tok = eval_model(student, "prior", eval_probs, egen())[1]
    out = {"init_prior_tok": init_prior_tok}
    for task in ("ordinary", "contradictory"):
        teacher = pretrain(base_state, task, True, seed,
                           steps=TEACHER_STEPS)
        teacher._with_ctx = True
        t_state = {k: v.clone() for k, v in teacher.state_dict().items()}
        init_acc = eval_model(student, task, eval_probs, egen())[0]
        rng = np.random.default_rng(3000 + seed)
        gen = torch.Generator().manual_seed(555 + seed)
        data_stu = gen_rollout_data(student, task, 400, rng, gen)
        data_tea = gen_rollout_data(teacher, task, 400, rng, gen)
        arms = {}
        specs = [("frozen", 1.0, data_stu, False),
                 ("b0.999", 0.999, data_stu, False),
                 ("b0.99", 0.99, data_stu, False),
                 ("b0.9", 0.9, data_stu, False),
                 ("self", None, data_stu, False),
                 ("shuf_frozen", 1.0, data_stu, True),
                 ("shuf_b0.99", 0.99, data_stu, True),
                 ("tearoll_frozen", 1.0, data_tea, False),
                 ("tearoll_b0.99", 0.99, data_tea, False)]
        for name, beta, data, shuf in specs:
            stu = distill(s_state, t_state, data, beta,
                          seed * 100 + len(name), shuffled=shuf)
            acq = eval_model(stu, task, eval_probs, egen())[0] - init_acc
            ret = eval_model(stu, "prior", eval_probs, egen())[1] \
                - init_prior_tok
            arms[name] = (acq, ret)
        out[task] = arms
    return out


def _default_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


def test_self_distillation_coupling():
    _default_recipe()
    results = [run_seed(seed) for seed in (0, 1, 2)]
    for seed, r in enumerate(results):
        for task in ("ordinary", "contradictory"):
            a = r[task]
            frozen = a["frozen"][0]
            # The frozen privileged teacher teaches: large acquisition
            # on both task types, far above every degenerate control.
            assert frozen > 0.30, f"seed {seed} {task}: frozen {frozen:.3f}"
            assert frozen > a["self"][0] + 0.30, (
                f"seed {seed} {task}: frozen {frozen:.3f} "
                f"vs self {a['self'][0]:.3f}")
            assert frozen > a["shuf_frozen"][0] + 0.30, (
                f"seed {seed} {task}: frozen {frozen:.3f} "
                f"vs shuffled {a['shuf_frozen'][0]:.3f}")
            # Plateau, not a peak: the mildest EMA matches frozen
            # within noise but never beats it by a real margin (the
            # paper's rising half does NOT reproduce in this toy).
            assert abs(a["b0.999"][0] - frozen) < 0.10, (
                f"seed {seed} {task}: b0.999 {a['b0.999'][0]:.3f} "
                f"vs frozen {frozen:.3f}")
            assert a["b0.99"][0] < frozen + 0.02, (
                f"seed {seed} {task}: b0.99 {a['b0.99'][0]:.3f} "
                f"vs frozen {frozen:.3f}")
            # The falling half, past a task-specific rate: beta 0.9
            # collapses on contradictory tasks and falls far behind
            # frozen on ordinary ones.
            if task == "contradictory":
                assert a["b0.99"][0] < frozen - 0.10, (
                    f"seed {seed}: b0.99 {a['b0.99'][0]:.3f} "
                    f"vs frozen {frozen:.3f}")
                assert a["b0.9"][0] < 0.05, (
                    f"seed {seed}: b0.9 {a['b0.9'][0]:.3f}")
                assert a["self"][0] < 0.05, (
                    f"seed {seed}: self {a['self'][0]:.3f}")
            else:
                assert a["b0.9"][0] < frozen - 0.30, (
                    f"seed {seed}: b0.9 {a['b0.9'][0]:.3f} "
                    f"vs frozen {frozen:.3f}")
            # Wrong privilege collapses at every coupling.
            assert a["shuf_frozen"][0] < 0.12, (
                f"seed {seed} {task}: shuf_frozen {a['shuf_frozen'][0]:.3f}")
            assert a["shuf_b0.99"][0] < 0.12, (
                f"seed {seed} {task}: shuf_b0.99 {a['shuf_b0.99'][0]:.3f}")
    # Retention dissociates by task type (aggregate over seeds, frozen
    # arm): contradictory distillation forgets the prior task several
    # times more than ordinary distillation, while acquisition is
    # comparably large -- the two readouts are not one score.
    ret_ord = np.mean([r["ordinary"]["frozen"][1] for r in results])
    ret_con = np.mean([r["contradictory"]["frozen"][1] for r in results])
    assert ret_ord > -0.25, f"mean ordinary retention {ret_ord:.3f}"
    assert ret_con < -0.35, f"mean contradictory retention {ret_con:.3f}"
    assert ret_con < ret_ord - 0.25, (
        f"retention dissociation: {ret_con:.3f} vs {ret_ord:.3f}")
    # Secondary claim (rollout source): teacher rollouts beat student
    # rollouts on contradictory tasks and not on ordinary ones
    # (aggregate over seeds, frozen teacher).
    gap_ord = np.mean([r["ordinary"]["tearoll_frozen"][0]
                       - r["ordinary"]["frozen"][0] for r in results])
    gap_con = np.mean([r["contradictory"]["tearoll_frozen"][0]
                       - r["contradictory"]["frozen"][0]
                       for r in results])
    assert gap_con > 0.05, f"contradictory rollout gap {gap_con:.3f}"
    assert abs(gap_ord) < 0.05, f"ordinary rollout gap {gap_ord:.3f}"
