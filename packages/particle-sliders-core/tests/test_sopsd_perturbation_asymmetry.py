"""CPU gate for the Shuffled OPSD curation-asymmetry claim
(arXiv:2611.04958): perturbing the STUDENT-side context is neutral-to-
positive for on-policy self-distillation, while corrupting the
TEACHER-side reference degrades accuracy at every scale.

Paper: "Shuffled OPSD: When Semantic Consistency Hurts On-Policy
Self-Distillation?" (Zheng & Dou, 2026). Its central finding is an
asymmetry in where semantic consistency matters: shuffling the words of
the shared problem statement the student trains on IMPROVES OPSD (up to
+3.52 pts over standard OPSD, Qwen3-1.7B), while shuffling the
teacher-only worked example hurts at every scale. Teacher-side reference
fidelity is load-bearing; student-side context is perturbable.

Toy operationalization: a scripted distillation world. The true task is
a bag rule over a 10-token statement (vocab 8): the label is the sign of
(#tokens in {1,3,5}) - (#tokens in {2,4,6}) -- order-invariant by
construction, so permuting a statement is a label-preserving transform.
The teacher holds a worked-example reference: a flat token sequence of
(token, sign) pairs encoding exactly that rule, which it parses
pair-by-pair to recover per-token weights before labelling student
statements (soft labels from its recovered score). Parsing is
order-sensitive: permute the reference and the pairs scramble, so the
teacher recovers wrong weights and emits systematically corrupted
supervision -- the toy's teacher-side corruption.

Training statements carry a spurious ORDER cue: the multiset is sampled
first (so the bag distribution is identical in every arm), then the
first two positions are arranged so that tok[0] > tok[1] iff the label
is positive, with probability 0.9. The cue is pure order -- no bag
learner can see it. Held-out statements are arranged uniformly at
random, where the cue is uninformative. A position-sensitive MLP
student trained on clean statements can lean on the easy two-position
cue instead of the ten-token counting rule and pays for it held-out;
permuting the student's statements every step destroys the cue while
preserving every label, so the shuffled student is forced onto the
real rule. The training budget is deliberately modest: with unlimited
steps both student arms saturate the counting rule and the student
half of the asymmetry has no room to express -- the shortcut bite the
paper exploits is an early-training phenomenon.

Arms (same statements, same teacher, same student, same budget):
  (a) clean           -- pristine reference, statements as arranged;
  (b) student-shuffle -- pristine reference, statement tokens randomly
                         permuted at every training step;
  (c) teacher-shuffle -- reference permuted once (per seed) before the
                         teacher parses it; statements clean.

Assertions encode exactly the paper's asymmetry as the toy
demonstrates it: teacher-shuffle lands clearly below clean on every
seed, and student-shuffle lands ABOVE clean on every seed -- the
paper's headline direction, not merely the neutral half.

Note on machinery: OPSD is a distillation/curation claim with no
adversarial component, so this test exercises the paper's own
machinery (teacher-labelled on-policy statements, forward distillation
into a student deployed alone), under the same upgraded particlegan as
the companion PSR/ACSD gates (the recipe gate below ties the file to
the current default recipe). Skips cleanly when the installed
particlegan lacks the default (EMPF) recipe.
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


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


V = 8            # content-token vocab
L = 10           # statement length
POS, NEG = 8, 9  # sign tokens inside the teacher reference
TRUE_W = np.array([0, 1, -1, 1, -1, 1, -1, 0], dtype=np.float64)
# The teacher's worked-example reference: (token, sign) pairs, in order.
REFERENCE = np.array([1, POS, 2, NEG, 3, POS, 4, NEG, 5, POS, 6, NEG])


def parse_reference(ref):
    """Recover per-token weights by reading (token, sign) pairs in
    order -- the order-sensitive step the paper's teacher relies on."""
    w = np.zeros(V)
    for k in range(0, len(ref) - 1, 2):
        tok, sign = int(ref[k]), int(ref[k + 1])
        if tok < V and sign in (POS, NEG):
            w[tok] = 1.0 if sign == POS else -1.0
    return w


def gen_statements(n, rng, cue_prob=None):
    """Sample multisets, compute bag labels, then arrange positions.

    cue_prob=None -> uniform random arrangement (held-out style);
    cue_prob=p    -> with prob p, arrange tok[0] > tok[1] iff y == 1
                     (training style; a permutation of the same bag).
    """
    toks = rng.integers(0, V, size=(n, L))
    score = TRUE_W[toks].sum(axis=1)
    bad = score == 0
    while bad.any():
        toks[bad] = rng.integers(0, V, size=(int(bad.sum()), L))
        score = TRUE_W[toks].sum(axis=1)
        bad = score == 0
    y = (score > 0).astype(np.int64)
    if cue_prob is not None:
        for i in range(n):
            if rng.random() > cue_prob:
                continue
            row = toks[i]
            want_gt = y[i] == 1
            # find a pair (a, b) from the bag with a > b (or a < b)
            placed = False
            for a_idx in range(L):
                for b_idx in range(L):
                    if a_idx == b_idx:
                        continue
                    if want_gt and row[a_idx] > row[b_idx]:
                        first, second = row[a_idx], row[b_idx]
                        rest = [row[j] for j in range(L)
                                if j not in (a_idx, b_idx)]
                        toks[i] = np.array([first, second] + rest)
                        placed = True
                        break
                    if not want_gt and row[a_idx] < row[b_idx]:
                        first, second = row[a_idx], row[b_idx]
                        rest = [row[j] for j in range(L)
                                if j not in (a_idx, b_idx)]
                        toks[i] = np.array([first, second] + rest)
                        placed = True
                        break
                if placed:
                    break
    return toks, y


class Student(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L * V, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def one_hot(toks):
    return F.one_hot(torch.tensor(toks, dtype=torch.long),
                     num_classes=V).float().reshape(len(toks), L * V)


def train_student(toks, soft_y, seed, shuffle=False, steps=40, bs=256):
    """Distill the teacher's soft labels into the student. With
    shuffle=True the statement tokens are re-permuted at every step
    (labels are bag-determined, hence permutation-invariant)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = one_hot(toks)
    stu = Student()
    opt = torch.optim.Adam(stu.parameters(), lr=3e-3)
    n = len(toks)
    for _ in range(steps):
        sel = rng.integers(0, n, size=bs)
        xb = x[sel]
        if shuffle:
            perm = torch.stack([torch.randperm(L) for _ in range(bs)])
            xb = xb.reshape(bs, L, V)
            xb = xb[torch.arange(bs)[:, None], perm].reshape(bs, L * V)
        loss = F.cross_entropy(stu(xb), soft_y[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return stu


@torch.no_grad()
def eval_student(stu, toks):
    stu.eval()
    pred = stu(one_hot(toks)).argmax(dim=-1).numpy()
    return pred


def run_seed(seed):
    rng = np.random.default_rng(5000 + seed)
    train_toks, train_y = gen_statements(512, rng, cue_prob=0.9)
    eval_toks, eval_y = gen_statements(400, np.random.default_rng(999 + seed))

    w_clean = parse_reference(REFERENCE)
    ref_shuf = np.random.default_rng(77 + seed).permutation(REFERENCE)
    w_shuf = parse_reference(ref_shuf)

    def soft_labels(w):
        s = w[train_toks].sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-s / 1.0))
        out = np.stack([1.0 - p, p], axis=1)
        return torch.tensor(out, dtype=torch.float32)

    # mechanism diagnostic: how accurate is each teacher's labelling?
    lab_clean = (w_clean[train_toks].sum(axis=1) > 0).astype(np.int64)
    lab_shuf = (w_shuf[train_toks].sum(axis=1) > 0).astype(np.int64)
    teacher_clean_acc = float((lab_clean == train_y).mean())
    teacher_shuf_acc = float((lab_shuf == train_y).mean())

    arms = {}
    stu = train_student(train_toks, soft_labels(w_clean), seed * 10 + 1)
    arms["clean"] = float((eval_student(stu, eval_toks) == eval_y).mean())
    stu = train_student(train_toks, soft_labels(w_clean), seed * 10 + 2,
                        shuffle=True)
    arms["student_shuffle"] = float(
        (eval_student(stu, eval_toks) == eval_y).mean())
    stu = train_student(train_toks, soft_labels(w_shuf), seed * 10 + 3)
    arms["teacher_shuffle"] = float(
        (eval_student(stu, eval_toks) == eval_y).mean())
    return dict(teacher_clean=teacher_clean_acc,
                teacher_shuffled=teacher_shuf_acc, **arms)


def test_sopsd_perturbation_asymmetry():
    recipe = _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Mechanism check: the pristine reference parses to the true
        # rule; the shuffled reference genuinely corrupts the teacher.
        assert r["teacher_clean"] > 0.99, (
            f"seed {seed}: clean teacher {r['teacher_clean']:.3f}")
        assert r["teacher_shuffled"] < 0.85, (
            f"seed {seed}: shuffled teacher {r['teacher_shuffled']:.3f}")
        # The asymmetry, teacher side: corrupting the teacher-only
        # reference degrades the student at every scale (here: seed).
        assert r["teacher_shuffle"] < r["clean"] - 0.15, (
            f"seed {seed}: teacher-shuffle {r['teacher_shuffle']:.3f} "
            f"vs clean {r['clean']:.3f}")
        # The asymmetry, student side: perturbing the student's own
        # context is POSITIVE here, as in the paper's headline --
        # permutation augmentation breaks the order shortcut and the
        # shuffled student beats the clean one on every seed.
        assert r["student_shuffle"] >= r["clean"] + 0.02, (
            f"seed {seed}: student-shuffle {r['student_shuffle']:.3f} "
            f"vs clean {r['clean']:.3f}")
