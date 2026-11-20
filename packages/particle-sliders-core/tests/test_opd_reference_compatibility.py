"""CPU gate for the OPD reference-selection claim
(arXiv:2611.13135): in multi-turn on-policy distillation, conditioning
the teacher on a validated successful same-task trajectory helps only
when that success is COMPATIBLE with the student's failed trajectory;
selecting the reference by action-sequence alignment (longest common
subsequence, LCS) beats arbitrary success selection, and an
incompatible success can actively mislead -- worse than no reference.

Paper: "Which Successes Make Good References? Action-Sequence
Alignment for Multi-Turn On-Policy Distillation" (Li, Kuang, Wang,
Koelle, Yuan, Mao, Wang, Lin, Feng, Yogatama et al.). Different
successful trajectories reach incompatible interaction states, so a
success is not automatically a good teacher reference. The authors
cast reference selection as free-target optimal transport and use
action-sequence alignment (LCS: a shared ordered backbone of actions,
allowing recoverable detours) as the observable proxy for the hidden
latent state path. On ALFWorld and tau^3-bench (airline), LCS was the
only evaluated selector to improve over reference-free OPD on both
ALFWorld overall and all three airline metrics.

Toy operationalization: a scripted route world. There are R = 4
routes; each route's backbone is a length-9 action sequence over
route-DISJOINT backbone tokens (only the start token is shared), so
distinct routes are incompatible by construction: executing another
route's action at a route-specific state is an incompatible
transition. A successful trajectory is a route backbone with detour
tokens interspersed (the backbone survives as an ordered
subsequence -- a recoverable detour, in the paper's sense). A failed
trajectory is a backbone PREFIX (first 5 backbone actions) with
detour tokens interspersed; the student must produce the 4-action
continuation (backbone positions 5..8).

The teacher is a scripted oracle conditioned on a reference success.
It aligns the failed trajectory to the reference by LCS, takes the
matched backbone prefix length m, and teaches the reference's
backbone continuation after position m as the student's target. For
a compatible reference (same route) m = 5 and the target is exactly
the true continuation. For an incompatible reference only the shared
start token matches (m = 1), so the teacher confidently teaches a
wrong route's actions for the student's actual state -- the paper's
misleading-reference channel, made mechanical. The student is a
small MLP over the one-hot failed trajectory, trained with cross-
entropy on the teacher targets, and evaluated greedily on HELD-OUT
failed trajectories (fresh detours) against the TRUE continuation.

Arms (reference selection rule; everything else identical):
  (a) lcs:          argmax over the success pool of LCS(failed, ref);
  (b) random:       a uniformly random success from the pool;
  (c) incompatible: argmin LCS (successful but disjoint backbone);
  (d) noref:        no reference -- with no validated successful path
                    the teacher cannot identify the student's route,
                    so it teaches the safe ABSTAIN action at every
                    continuation position (no route-specific signal).

Metrics per seed, on held-out failed trajectories: full-continuation
success (all 4 predicted actions equal the true continuation),
per-position continuation accuracy, harm (fraction of trajectories
whose executed continuation contains a wrong-route, i.e.
incompatible, action), and abstention rate. Selector diagnostics:
fraction of (a) picks that are same-route, fraction of (c) picks
that are cross-route, and mean LCS of the picked references.

Iteration log (the loop that produced this gate):
  1. Route-disjoint backbones + LCS-aligned teacher, exactly as
     sketched above: the arms separate completely on the first run.
     The selector is exact (LCS picks a same-route success for
     100% of training trajectories, the incompatible arm a
     cross-route success for 100%; mean picked LCS 6.27-6.57 vs
     2.27-2.66 -- above the 5 backbone matches because shared
     detour tokens also align). LCS-arm held-out success
     0.984-1.000; random 0.098-0.281 (it is same-route for a
     quarter of the pool, and incompatible for the rest); the
     incompatible arm recovers nothing and harms on every
     trajectory (harm 1.000), while the reference-free student
     abstains everywhere (harm 0.000). No refinement was needed:
     the discriminating pattern (a) > (b) > (d) on success and
     (c) strictly harmful where (d) is safe held on all 3 seeds
     with wide margins, so the assertions encode those margins
     with headroom rather than the exact numbers.

Measured on pop-os (particlegan 0.10.1), 3 seeds: LCS success
0.984-1.000 / accuracy 0.996-1.000 / harm 0.000; random success
0.098-0.281 / accuracy 0.359-0.445 / harm 0.703-0.797; incompatible
success 0.000 / accuracy 0.000 / harm 1.000; no-reference success
0.000 / accuracy 0.000 / harm 0.000 / abstain 1.000.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
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


# --- world constants -------------------------------------------------
R = 4                    # routes
L = 9                    # backbone length
CUT = 5                  # backbone prefix length in a failed trajectory
SUFFIX = L - CUT         # continuation length to predict (4)
START = 0
# Backbone tokens: route r uses tokens 1 + r*8 .. 1 + r*8 + 7 for its
# positions 1..8 -- disjoint across routes by construction.
DETOURS = (33, 34)       # detour tokens, shared across routes
ABSTAIN = 35
PAD = 36
V = 37
F_LEN = 8                # failed trajectory padded length (5 backbone
                         # + up to 2 detours + pad)
SUCC_PER_ROUTE = 3
N_TRAIN = 256
N_EVAL = 256
STEPS = 260
BATCH = 64
LR = 3e-3


def make_backbones(seed):
    """Per-seed route backbones: position 0 is the shared START token;
    positions 1..8 are a random permutation of the route's own token
    block, so routes share no backbone action past the start."""
    rng = np.random.default_rng(60000 + seed)
    backs = []
    for r in range(R):
        block = list(range(1 + r * 8, 1 + r * 8 + 8))
        rng.shuffle(block)
        backs.append([START] + block)
    return backs


def _sprinkle(rng, seq, max_detours=2):
    """Insert 0..max_detours detour tokens at random non-initial
    positions; the original sequence survives as a subsequence."""
    out = list(seq)
    for _ in range(int(rng.integers(0, max_detours + 1))):
        pos = int(rng.integers(1, len(out) + 1))
        out.insert(pos, int(DETOURS[int(rng.integers(0, len(DETOURS)))]))
    return out


def make_pool(rng, backs):
    """Successful trajectories: (route, sequence) with each route's
    full backbone sprinkled with detours."""
    pool = []
    for r in range(R):
        for _ in range(SUCC_PER_ROUTE):
            pool.append((r, _sprinkle(rng, backs[r])))
    return pool


def make_failed(rng, backs, n):
    """Failed trajectories: (route, padded sequence) -- backbone
    prefix of length CUT sprinkled with detours, padded to F_LEN."""
    out = []
    for _ in range(n):
        r = int(rng.integers(0, R))
        seq = _sprinkle(rng, backs[r][:CUT])
        seq = seq + [PAD] * (F_LEN - len(seq))
        out.append((r, seq[:F_LEN]))
    return out


def backbone_of(seq):
    return [a for a in seq if a not in DETOURS and a != PAD]


def lcs_len(a, b):
    """Longest common subsequence length (classic DP)."""
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
            prev = cur
    return dp[-1]


def select_ref(failed_seq, pool, rule, rng):
    """Pick a reference success for a failed trajectory.
    Returns (route, seq, lcs) or None for the no-reference arm."""
    if rule == "noref":
        return None
    scores = [lcs_len(failed_seq, seq) for _, seq in pool]
    if rule == "lcs":
        idx = int(np.argmax(scores))
    elif rule == "incompatible":
        idx = int(np.argmin(scores))
    elif rule == "random":
        idx = int(rng.integers(0, len(pool)))
    else:
        raise ValueError(rule)
    route, seq = pool[idx]
    return route, seq, scores[idx]


def teacher_target(failed_seq, ref, backs):
    """The reference-conditioned teacher: align the failed trajectory
    to the reference by LCS, then teach the reference backbone's
    continuation after the aligned prefix. With no reference, teach
    ABSTAIN at every position."""
    if ref is None:
        return [ABSTAIN] * SUFFIX
    _, ref_seq, _ = ref
    ref_back = backbone_of(ref_seq)
    fail_back = backbone_of(failed_seq)
    m = lcs_len(fail_back, ref_back)
    tgt = ref_back[m:m + SUFFIX]
    # A short aligned prefix on a long route can overrun; pad the
    # tail with the reference's final action (never triggers for the
    # compatible arm, where m == CUT by construction).
    while len(tgt) < SUFFIX:
        tgt.append(ref_back[-1])
    return tgt


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_LEN * V, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, SUFFIX * V))

    def forward(self, seq):
        x = F.one_hot(seq, V).float().reshape(seq.shape[0], -1)
        return self.net(x).reshape(seq.shape[0], SUFFIX, V)


def train_arm(rule, seed, backs, pool, train_set):
    rng = np.random.default_rng(70000 + seed)
    torch.manual_seed(80000 + seed)
    # Precompute teacher targets once (teacher is scripted).
    xs, ys = [], []
    for route, seq in train_set:
        ref = select_ref(seq, pool, rule, rng)
        xs.append(seq)
        ys.append(teacher_target(seq, ref, backs))
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    student = Student()
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(90000 + seed)
    for _ in range(STEPS):
        idx = torch.randint(0, len(xs), (BATCH,), generator=gen)
        logits = student(xs[idx])
        loss = F.cross_entropy(logits.reshape(-1, V), ys[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return student


@torch.no_grad()
def evaluate(student, backs, eval_set):
    xs = torch.tensor([seq for _, seq in eval_set])
    pred = student(xs).argmax(-1).tolist()
    succ = acc = harm = abst = 0
    for (route, _), p in zip(eval_set, pred):
        true = backs[route][CUT:]
        own = set(backs[route])
        succ += int(p == true)
        acc += sum(int(a == b) for a, b in zip(p, true))
        harm += int(any(a != ABSTAIN and a not in own for a in p))
        abst += int(all(a == ABSTAIN for a in p))
    n = len(eval_set)
    return {"success": succ / n, "acc": acc / (n * SUFFIX),
            "harm": harm / n, "abstain": abst / n}


def run_seed(seed):
    _empf_recipe()
    backs = make_backbones(seed)
    rng = np.random.default_rng(65000 + seed)
    pool = make_pool(rng, backs)
    train_set = make_failed(rng, backs, N_TRAIN)
    eval_set = make_failed(rng, backs, N_EVAL)
    # Selector diagnostics on the training set.
    diag_rng = np.random.default_rng(71000 + seed)
    same = cross = 0
    lcs_pick, inc_pick = [], []
    for route, seq in train_set:
        r_route, _, s = select_ref(seq, pool, "lcs", diag_rng)
        same += int(r_route == route)
        lcs_pick.append(s)
        i_route, _, s2 = select_ref(seq, pool, "incompatible", diag_rng)
        cross += int(i_route != route)
        inc_pick.append(s2)
    diag = {"lcs_same_frac": same / len(train_set),
            "inc_cross_frac": cross / len(train_set),
            "lcs_mean": float(np.mean(lcs_pick)),
            "inc_mean": float(np.mean(inc_pick))}
    out = {"diag": diag}
    for rule in ("lcs", "random", "incompatible", "noref"):
        student = train_arm(rule, seed, backs, pool, train_set)
        out[rule] = evaluate(student, backs, eval_set)
    return out


def test_opd_reference_compatibility():
    for seed in range(3):
        r = run_seed(seed)
        d = r["diag"]
        # The selector itself: LCS picks a same-route success and the
        # incompatible arm picks a cross-route success.
        assert d["lcs_same_frac"] >= 0.95
        assert d["inc_cross_frac"] >= 0.95
        assert d["lcs_mean"] >= d["inc_mean"] + 2.0
        a, b, c, n = r["lcs"], r["random"], r["incompatible"], r["noref"]
        # (a) LCS-selected reference: near-perfect held-out recovery,
        # far above a random success (demonstrated success gaps
        # +0.72..+0.89, accuracy gaps +0.55..+0.64 per seed).
        assert a["success"] >= b["success"] + 0.50
        assert a["acc"] >= b["acc"] + 0.40
        assert a["harm"] <= 0.02
        # (b) A random success still helps sometimes -- it is
        # same-route for a quarter of the pool -- but most of its
        # teaching is incompatible (demonstrated success 0.098-0.281
        # vs 0.000 for no reference, harm 0.703-0.797).
        assert b["success"] >= n["success"] + 0.05
        assert b["harm"] >= n["harm"] + 0.50
        # (c) An incompatible success actively misleads: it never
        # recovers a single held-out trajectory and every executed
        # continuation contains a wrong-route action (harm 1.000 on
        # every seed), while the reference-free student safely
        # abstains (abstain 1.000, harm 0.000).
        assert c["harm"] >= n["harm"] + 0.50
        assert c["success"] <= n["success"] + 0.02
        assert c["acc"] <= n["acc"] + 0.02
        assert n["abstain"] >= 0.90
