"""CPU gate for the Diagnosing-OPSD claim (arXiv:2609.39118): the paper's
no-privilege control -- a teacher that sees the same prefix but NO solution
and NO verified signal -- nearly matches the full-solution teacher, i.e.
much of OPSD's measured benefit is reasoning-MODE transfer, not privileged
semantics. This test is a diagnostic: the deliverable is the measurement
itself, whichever way it lands.

Paper: "Diagnosing On-Policy Self-Distillation for Reasoning Language
Models" (Li et al., 2026). Exercised claim: OPSD's gains decompose into a
reasoning-mode component (the teacher generates in a coherent mode the
student can absorb from ordinary on-policy supervision) and a privileged-
semantics component (information only available from the verified
solution). The paper's same-prefix no-privilege control recovering most of
the full teacher's gain is the evidence that the first component dominates.

Toy operationalization: a scripted stochastic policy with controlled
hidden states (dim 16) works a verifiable parity task (7 bits, exact
checker), adapted from the ACSD gate's world (PR #210). Each trajectory
carries TWO independent latent signs, so ground truth separates the
components the paper cannot separate observationally:

  * mode sign (+focused / -wandering) along axis m: a focused trajectory
    emits the parity-consistent token sharply and rarely stalls; a
    wandering one lapses (coin-flip tokens) and stalls often, so its
    trajectories run LONG. Mode is visible in generation alone -- length
    and coherence need no verifier.
  * semantics sign (+aligned / -misaligned) along axis p (independent of
    m): a misaligned trajectory confidently emits the FLIPPED token on
    steps where the semantics projection dominates. A flip looks exactly
    as coherent as a correct token and does not change trajectory length;
    only the verified checker can tell aligned from misaligned.

Pipeline (ACSD-style, as in #210): sample trajectories from the base
policy; extract a steering direction; calibrate its strength on a
held-out pool by steered-teacher accuracy; distill a tiny GRU student
(forward KL on teacher next-token distributions over base-policy
prefixes); deploy the student alone. Arms differ ONLY in how the
direction is built:

  (a) unsteered:      no direction (base student);
  (b) privileged:     direction = contrast of verified-correct-within-
                      budget trajectories vs all others (the ACSD recipe;
                      uses the checker, so it can load on BOTH m and p);
  (c) no-privilege:   direction = contrast of the SHORTEST vs LONGEST
                      tercile of trajectories from the SAME sampled pool
                      (length only), with correctness never consulted --
                      the toy analogue
                      of the paper's same-prefix teacher with no solution:
                      everything it knows is observable in unguided
                      generation, so it can load on m but not on p;
  (d) shuffled / reversed privileged direction (specificity controls,
                      as in #210).

Measured per seed: held-out student accuracy per arm (mean over 2
student restarts for the measured arms -- single-student training noise,
std ~0.06, otherwise swamps the gap), the cosines of each extracted
direction against the ground-truth axes m and p, and the recovered
fraction (no-privilege gain over base) / (privileged gain over base).
Assertions encode exactly the measured structure: the separation is
exact per seed (privileged direction loads on both axes, no-privilege
only on m), the privileged student beats base, the no-privilege student
and both perturbed controls by a wide margin on EVERY seed, and the
recovered fraction is decisive in aggregate -- a minority share, not
the paper's near-match. The no-privilege student's gain over base is
asserted in aggregate only: it is positive on average but unstable
per seed (on one seed a +0.20 teacher-level gain transferred ~nothing,
the paper's teacher-signal instability showing up in the toy as well).

Note on machinery: like ACSD, this diagnostic has no adversarial
component -- directions are closed-form contrasts and training is
forward-KL distillation -- so this test exercises the pipeline's own
machinery, under the same upgraded particlegan as the companion PSR/ACSD
gates. Skips cleanly when the installed particlegan lacks the default
(EMPF) recipe.
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
NB = 7          # bits per problem (odd: a systematic flip cannot cancel)
BUDGET = 11     # verified-correct-within-budget: length <= BUDGET
MAXLEN = 14
STALL = 2
WORLD_SEED = 2001


# ---------------------------------------------------------------- world
class World:
    """Parity world with two independent latent axes: mode m (lapses +
    stalls, hence length) and semantics p (confident flips, invisible to
    length / coherence)."""

    def __init__(self, seed, emb_scale=0.35, mode=2.1, sem=1.6,
                 noise=0.85, gamma_m=0.75, gamma_p=1.6):
        rng = np.random.default_rng(seed)
        self.emb_prev = rng.standard_normal((4, H)) * emb_scale
        self.emb_bit = rng.standard_normal((2, H)) * emb_scale
        m = rng.standard_normal(H)
        self.m = m / np.linalg.norm(m)
        p = rng.standard_normal(H)
        p = p - (p @ self.m) * self.m
        self.p = p / np.linalg.norm(p)
        self.a = rng.standard_normal((2, H)) * 0.3
        self.gamma_m = gamma_m
        self.gamma_p = gamma_p
        self.mode = mode
        self.sem = sem
        self.noise = noise

    def hidden(self, prev, bit, mode_sign, sem_sign, rng):
        return (self.emb_prev[prev] + self.emb_bit[bit]
                + mode_sign * self.mode * self.m
                + sem_sign * self.sem * self.p
                + rng.standard_normal(H) * self.noise)

    def action_logits(self, h, e, bit):
        """Logits over {correct, lapse, flip, stall}. Mode drives lapse
        and stall; semantics drives flip; neither leaks into the other."""
        s_m = float(h @ self.m)
        s_p = float(h @ self.p)
        consistent = e ^ bit
        base_c = float(self.a[consistent] @ h)
        base_w = float(self.a[1 - consistent] @ h)
        return np.array([
            base_c + self.gamma_m * s_m + self.gamma_p * s_p,
            base_w - 0.6 * self.gamma_m * s_m + 0.2,
            base_w - self.gamma_p * s_p + 0.55,
            -0.9 - 0.45 * s_m])

    def token_probs(self, h, e, bit):
        pa = _softmax(self.action_logits(h, e, bit))
        consistent = e ^ bit
        pt = np.zeros(3)
        pt[consistent] += pa[0]
        pt[1 - consistent] += pa[2]
        pt[0] += pa[1] * 0.5
        pt[1] += pa[1] * 0.5
        pt[STALL] += pa[3]
        return pt


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def rollout(world, bits, rng, direction=None, alpha=0.0):
    p_par, e, prev, n_done = 0, 0, 3, 0
    steps = []
    mode_sign = 1.0 if rng.random() < 0.55 else -1.0
    sem_sign = 1.0 if rng.random() < 0.5 else -1.0
    for _ in range(MAXLEN):
        if n_done >= NB:
            break
        bit = bits[n_done]
        h = world.hidden(prev, bit, mode_sign, sem_sign, rng)
        hh = h if direction is None else h + alpha * np.linalg.norm(h) * direction
        pa = _softmax(world.action_logits(hh, e, bit))
        act = int(rng.choice(4, p=pa))
        consistent = e ^ bit
        if act == 0:
            tok = consistent
        elif act == 1:
            tok = int(rng.integers(0, 2))
        elif act == 2:
            tok = 1 - consistent
        else:
            tok = STALL
        steps.append((h, e, bit, prev, tok))
        if tok != STALL:
            e = tok
            p_par = p_par ^ bit
            n_done += 1
            prev = tok
        else:
            prev = STALL
    correct = (n_done == NB) and (e == p_par)
    return dict(steps=steps, correct=correct, length=len(steps),
                mode_sign=mode_sign, sem_sign=sem_sign)


def gen_problems(n, rng):
    return rng.integers(0, 2, size=(n, NB)).tolist()


def extract_directions(world, problems, rng, per_problem=4):
    """Both directions from ONE sampled pool. Privileged: verified-
    correct-within-budget vs rest (checker consulted). No-privilege:
    shortest vs longest tercile by length (length only -- correctness
    never read), so both contrast pools are large and balanced."""
    trajs = []
    for bits in problems:
        for _ in range(per_problem):
            r = rollout(world, bits, rng)
            hs = [h / np.linalg.norm(h) for (h, e, bit, prev, tok)
                  in r["steps"][:BUDGET]]
            trajs.append((r["correct"], r["length"], hs))
    pos = [h for c, ln, hs in trajs if c and ln <= BUDGET for h in hs]
    rest = [h for c, ln, hs in trajs if not (c and ln <= BUDGET)
            for h in hs]
    by_len = sorted(trajs, key=lambda t: t[1])
    third = len(by_len) // 3
    short = [h for c, ln, hs in by_len[:third] for h in hs]
    long_ = [h for c, ln, hs in by_len[-third:] for h in hs]

    def contrast(a, b):
        d = np.mean(a, axis=0) - np.mean(b, axis=0)
        return d / np.linalg.norm(d)

    return contrast(pos, rest), contrast(short, long_)


def teacher_accuracy(world, problems, rng, direction=None, alpha=0.0,
                     per_problem=1):
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
            data.append([(prev, bit, h, e)
                         for (h, e, bit, prev, tok) in r["steps"]])
    return data


def train_student(world, data, direction, alpha, seed, steps=100,
                  bs=1024, hidden=16):
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
                teacher_probs(world, h, e, bit, direction, alpha),
                dtype=torch.float32)
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
        tok = torch.multinomial(torch.softmax(lg, dim=-1), 1,
                                generator=gen).numpy()[:, 0]
        is_stall = tok == STALL
        upd = active & ~is_stall
        p = np.where(upd, p ^ bit, p)
        e = np.where(upd, tok, e)
        n_done = np.where(upd, n_done + 1, n_done)
        prev = np.where(active, np.where(is_stall, STALL, tok), prev)
    correct = (n_done >= NB) & (e == p)
    return float(correct.reshape(n_prob, samples).mean())


# ---------------------------------------------------------------- test
def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


def _calibrate(world, problems, rng, direction):
    base = teacher_accuracy(world, problems, rng)
    best_alpha, best_acc = 0.0, base
    for alpha in (0.4, 0.8, 1.2):
        acc = teacher_accuracy(world, problems, rng, direction, alpha)
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    return base, best_alpha, best_acc


def run_seed(seed):
    world = World(WORLD_SEED)
    rng = np.random.default_rng(3000 + seed)
    d_priv, d_nopriv = extract_directions(
        world, gen_problems(448, rng), rng)
    calib = gen_problems(100, rng)
    base_acc, a_priv, acc_priv = _calibrate(world, calib, rng, d_priv)
    _, a_nopriv, acc_nopriv = _calibrate(world, calib, rng, d_nopriv)
    data = build_data(world, gen_problems(320, rng), rng)
    eval_probs = gen_problems(250, np.random.default_rng(999 + seed))
    egen = torch.Generator().manual_seed(4242 + seed)
    d_shuf = d_priv[np.random.default_rng(77 + seed).permutation(H)]
    arms = {"unsteered": (None, 0.0),
            "privileged": (d_priv, a_priv),
            "no_privilege": (d_nopriv, a_nopriv),
            "shuffled": (d_shuf, a_priv),
            "reversed": (-d_priv, a_priv)}
    out = dict(cos_priv_m=float(d_priv @ world.m),
               cos_priv_p=float(d_priv @ world.p),
               cos_nopriv_m=float(d_nopriv @ world.m),
               cos_nopriv_p=float(d_nopriv @ world.p),
               alpha_priv=a_priv, alpha_nopriv=a_nopriv,
               teacher_base=base_acc, teacher_priv=acc_priv,
               teacher_nopriv=acc_nopriv)
    for name, (direc, alpha) in arms.items():
        # Each arm's accuracy is the mean over student restarts (2 for
        # the three measured arms, 1 for the perturbed controls):
        # single-student training noise (std ~0.06) otherwise swamps the
        # privileged-vs-control gap this test exists to measure.
        restarts = 1 if name in ("shuffled", "reversed") else 2
        accs = []
        for r in range(restarts):
            stu = train_student(world, data, direc, alpha,
                                seed * 1000 + 42 + r, steps=100)
            accs.append(eval_student(stu, eval_probs, egen, samples=6))
        out[name] = float(np.mean(accs))
    gain_priv = out["privileged"] - out["unsteered"]
    gain_nopriv = out["no_privilege"] - out["unsteered"]
    out["recovered"] = gain_nopriv / gain_priv if gain_priv > 0 else float("nan")
    return out


def test_opsd_no_privilege_control():
    _empf_recipe()
    results = [run_seed(seed) for seed in (0, 1, 2)]
    for seed, r in enumerate(results):
        # The separation is real: the privileged direction loads on BOTH
        # ground-truth axes; the no-privilege direction loads on the mode
        # axis but not on the privileged-semantics axis.
        assert r["cos_priv_m"] > 0.5, (
            f"seed {seed}: cos(priv, m) = {r['cos_priv_m']:.3f}")
        assert r["cos_priv_p"] > 0.3, (
            f"seed {seed}: cos(priv, p) = {r['cos_priv_p']:.3f}")
        assert r["cos_nopriv_m"] > 0.5, (
            f"seed {seed}: cos(nopriv, m) = {r['cos_nopriv_m']:.3f}")
        assert abs(r["cos_nopriv_p"]) < 0.25, (
            f"seed {seed}: cos(nopriv, p) = {r['cos_nopriv_p']:.3f}")
        # Steering by the privileged direction genuinely helps the
        # teacher; the no-privilege calibration at least never hurts
        # (its teacher-level gain is real but pool-noisy, +0.02..+0.20).
        assert r["teacher_priv"] > r["teacher_base"] + 0.05, (
            f"seed {seed}: priv teacher {r['teacher_priv']:.3f} "
            f"vs base {r['teacher_base']:.3f}")
        assert r["teacher_nopriv"] >= r["teacher_base"], (
            f"seed {seed}: nopriv teacher {r['teacher_nopriv']:.3f} "
            f"vs base {r['teacher_base']:.3f}")
        # The privileged student beats base, the no-privilege student
        # and both perturbed controls on EVERY seed -- in this toy the
        # privileged component is load-bearing, not a rounding error.
        assert r["privileged"] > r["unsteered"] + 0.04, (
            f"seed {seed}: priv student {r['privileged']:.3f} "
            f"vs base {r['unsteered']:.3f}")
        assert r["privileged"] > r["no_privilege"] + 0.08, (
            f"seed {seed}: priv student {r['privileged']:.3f} "
            f"vs nopriv {r['no_privilege']:.3f}")
        assert r["privileged"] > r["shuffled"] + 0.04, (
            f"seed {seed}: priv student {r['privileged']:.3f} "
            f"vs shuffled {r['shuffled']:.3f}")
        assert r["privileged"] > r["reversed"] + 0.04, (
            f"seed {seed}: priv student {r['privileged']:.3f} "
            f"vs reversed {r['reversed']:.3f}")
        # The no-privilege student never falls meaningfully below base.
        assert r["no_privilege"] > r["unsteered"] - 0.02, (
            f"seed {seed}: nopriv student {r['no_privilege']:.3f} "
            f"vs base {r['unsteered']:.3f}")
    rec = np.array([r["recovered"] for r in results])
    gain_nopriv = np.mean([r["no_privilege"] - r["unsteered"]
                           for r in results])
    # The diagnostic, in aggregate: mode transfer is real but a minority
    # share -- the no-privilege arm recovers well under 60% of the
    # privileged gain (measured ~0.19), so the paper's near-match does
    # NOT reproduce in this toy.
    assert gain_nopriv > 0.01, f"mean nopriv gain {gain_nopriv:.3f}"
    assert rec.mean() < 0.6, f"recovered fraction {rec}"
