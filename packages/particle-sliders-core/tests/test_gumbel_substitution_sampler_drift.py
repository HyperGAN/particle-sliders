"""CPU gate for Gumbel substitution as a sampler-drift
removal, from Fanucci & El Hassani, "Gumbel Substitution:
Removing Sampler Drift from RL Without Trusting the
Sampler" (arXiv:2611.27375).

The paper's claims (abstract only, the sole source used
here): when RL rollouts come from a sampler whose
distribution q differs from the trainer's policy p,
training on the sampler's tokens adds a drift term equal
to SFT toward q; published corrections (TIS, MIS, IcePop,
etc.) all need q(y), which an untrusted sampler can
misreport. Gumbel substitution: the coordinator fixes
public Gumbel noise G for every token position,
bit-identical everywhere; the sampler draws
y = argmax(lq + G); the trainer recomputes
v* = argmax(lp + G) -- the token its own policy would
have drawn with the same noise -- and scores v* in the
sampler's context. For a protocol-following sampler, v*
is an exact sample from p given the prefix, so the drift
vanishes in expectation for any q, without
sampler-reported probabilities. Lab measurements in the
abstract: expected drift of uncorrected training is at
least 170-4,800x that of substitution (9B, same head
both sides); on a 2B model under strong mismatch,
uncorrected training collapses while substitution stays
stable, comparable to MIS with score centering given
the TRUE q.

Chain provenance: cycle 26 of the Particle Sliders
paper watch, follow-up to the estimation family of
PR #310 (gradient-estimation protection, branch
paper-test/gradient-estimation-protection). #310 proved
in the REFERENCE seat that training signals computed
from sampled rollouts drift: the sampled gradient's
expectation sits at cos ~= 0.76-0.90 to the dense
gradient along the trajectory -- drift is the tax on
consuming samples you did not generate. This paper
attacks the same disease in the ROLLOUT seat with a
construction, not an estimate: shared noise makes the
consumed token exactly a sample from the trainer's own
policy. Our pipeline stake: the distillation stage's
missing on-policy rollout loop (a standing prerequisite
since the SSD / SH-OPD tests) will consume rollouts
produced by something that is not the trainer -- a
stale checkpoint or a separate inference engine that
reports no sampling probabilities. A drift removal that
needs no q from the sampler is exactly the shape that
fits.

Toy operationalization (new construction, this test):
a linear softmax policy -- one-hot context features
times a logit table W, logits = W[c] -- over C = 16
contexts and V = 10 tokens, one token per context per
rollout (the single-step case of the paper's protocol,
where "the sampler's context" is the context itself).
Per context an intended target distribution T (top
token a_c at logit 4.2, a second token at 0.5) defines
the intended behavior and a verifiable reward
r(c, t) = 1 if t == a_c else 0.3 (partial credit).
Training is uncentered REINFORCE on the rollout's
reward, the setting in which the paper's drift term
appears: E_q[r * grad log p(y)] decomposes into a
reward-signal part plus r-bar * the SFT-toward-q
gradient. The trainer policy p starts at
0.45 * (target logits) + noise (argmax already correct,
mean prob on a_c ~= 0.41). The sampler q is a STRONG
mismatch in the paper's sense: per context its top
token is d_c = a_c + 3 (mod V) at logit 2.6 / 0.8
temperature, the intended token is suppressed to
logit -0.5 / 0.8 (mean q(a_c) ~= 0.03), tails flat.

Arms (all consume the SAME coordinator-fixed Gumbel
field per (step, sample, context); p_snapshot is the
policy at the start of the step in every arm):
  (1) UNCORRECTED: y = argmax(log q + G); update
      r(y) * grad log p(y).
  (2) SUBSTITUTION: same y; trainer scores
      v* = argmax(log p_snapshot + G) with the rollout's
      reward: update r(y) * grad log p(v*).
  (3) TRUE-Q IS (the paper's strong control): update
      clip(p(y)/q(y), 0, 10) * r(y) * grad log p(y),
      with EXACT q probabilities -- the information an
      untrusted sampler could misreport.
  (4) BROKEN CONTROL: substitution with INDEPENDENT
      trainer noise G' (fresh field, same distribution):
      update r(y) * grad log p(v'), v' = argmax(log p + G').
  (5) MATCHED SANITY: q := p_snapshot each step,
      otherwise arm (1). No drift is possible; the arm
      must simply learn.

Readouts (per seed, final student S vs init S0):
  q-pull  = mean_c KL(S0_c || Q_c) - KL(S_c || Q_c)
            (nats; positive = moved TOWARD the sampler)
  KL-to-T = mean_c KL(S_c || T_c), and its init value
  pa      = mean_c S_c[a_c] (intended-token probability)
  acc     = fraction of contexts with argmax S_c == a_c
  sub rate (arm 2) = fraction of positions with v* != y

Measured on pop-os (particlegan 0.10.1, torch threads
1), 4 seeds, STEPS = 150, M = 24 rollouts per context
per step, LR = 0.5. Seed means (per-seed assertions in
the tests below):
  arm          KL-to-T   q-pull   pa     acc   sub
  init         0.615     --       0.413  1.00  --
  uncorrected  2.705     +1.476   0.067  0.08  --
  substitution 0.354     -0.416   0.549  1.00  0.718
  true-q IS    0.363     -0.514   0.567  1.00  --
  broken       0.630     -0.002   0.413  1.00  --
  matched      0.077     n/a      0.980  1.00  --

The verdict, stated plainly: PROVEN in part, with one
claim sharpened by a negative. (1) Uncorrected drift is
real and directed: on every seed the student moves
>= 1.3 nats toward q, KL-to-T grows by >= 1.8 nats and
pa collapses to <= 0.09 -- the paper's collapse regime,
in miniature. (2) Substitution removes the directed
drift on every seed (q-pull <= 0: the student moves
AWAY from q) while still learning (pa +0.11..+0.19,
KL-to-T -0.21..-0.35), at a substitution rate of
0.71-0.72 -- the strong-mismatch regime, far above the
paper's 1.21% network rate at weak mismatch. The
directed-drift contrast is >= 1.7 nats on every seed;
with substitution's q-pull at or below zero, a ratio
in the paper's form is a lower bound only (over 100x
under any 0.01-nat floor on substitution drift) -- we
assert the measured difference, not the paper's
170-4,800x. (3) Substitution lands on top of the
true-q IS control: |Delta KL-to-T| <= 0.09 and
|Delta pa| <= 0.08 on every seed, with NO access to q.
(4) The broken control does NOT reproduce the literal
expectation "independent noise => drift returns": its
q-pull is ~0, not uncorrected's. What returns is
worse in a different way -- NOTHING: pa stays at its
init value (gain <= 0.03 on every seed) and KL-to-T
stays >= 0.20 above substitution's, because with
independent noise E[r(y) * grad log p(v')] factorizes
to E[r] * 0: the rollout's reward carries no signal
about a token it did not produce. Shared noise is
still proven to be the mechanism -- it is what couples
the trainer's scored token to the rollout's reward --
but in this toy its absence destroys the signal
rather than re-aiming it at q. (5) Matched sanity:
q = p learns straight to the target band (KL-to-T
0.077, pa 0.98), so the geometry itself is learnable
and the uncorrected collapse is the sampler's doing.

What is NOT asserted: the paper's 170-4,800x lab ratio
(we assert our own measured contrast); multi-token
sequences, where the substituted token is scored in
the sampler's PREFIX and prefix divergence adds a
second-order effect this single-step toy cannot see;
adaptive / protocol-violating samplers (the paper
bounds what they can still do; not tested here); and
any claim about IS with REPORTED (misreportable) q --
arm (3) is given the true q, as the paper's control
is. Pure numpy -- no GAN machinery, because the claims
are about RL training dynamics on categoricals. Skips
cleanly when the installed particlegan does not
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


C = 16               # contexts
V = 10               # tokens
SEEDS = range(4)
STEPS = 150          # training steps, all arms
M = 24               # rollouts per context per step
LR = 0.5             # step size on the logit table
R_OTHER = 0.3        # partial-credit reward, non-intended tokens
IS_CLIP = 10.0       # MIS/TIS-style ratio clip, arm (3)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _kl(a, b):
    return float((a * np.log(a / b)).sum(axis=-1).mean())


def _gumbel(rng, shape):
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    return -np.log(-np.log(u))


def _build(seed):
    """Contexts, intended target T, mismatched sampler Q,
    student init, and the reward table. Q's top token is
    never the intended token, and Q suppresses it."""
    rng = np.random.default_rng(7000 + seed)
    a = rng.integers(V, size=C)
    second = np.array([rng.choice([t for t in range(V) if t != a[i]])
                       for i in range(C)])
    d = (a + 3) % V
    tl = np.zeros((C, V))
    ql = np.zeros((C, V))
    for i in range(C):
        tl[i, a[i]] = 4.2
        tl[i, second[i]] = 0.5
        ql[i, d[i]] = 2.6
        ql[i, a[i]] = -0.5
        ql[i, second[i]] = 0.3
    ql = ql / 0.8  # temperature-sharpened sampler
    t = _softmax(tl)
    q = _softmax(ql)
    init = tl * 0.45 + rng.normal(0, 0.25, (C, V))
    r = np.full((C, V), R_OTHER)
    for i in range(C):
        r[i, a[i]] = 1.0
    return t, q, init, r, a


def _train(seed, arm):
    """One arm on one seed. The coordinator Gumbel field
    (and the broken arm's independent field) are fixed
    per (step, sample, context) before training, and are
    identical across arms -- arms differ ONLY in which
    token is scored and with what weight."""
    t, q, init, r, a = _build(seed)
    rng = np.random.default_rng(7100 + seed)
    g_field = [_gumbel(rng, (M, C, V)) for _ in range(STEPS)]
    g_indep = [_gumbel(rng, (M, C, V)) for _ in range(STEPS)]
    th = init.copy()
    n_sub = 0
    for st in range(STEPS):
        p = _softmax(th)
        g = g_field[st]
        qs = p if arm == "matched" else q
        y = np.argmax(np.log(qs)[None] + g, axis=2)          # [M, C]
        adv = np.take_along_axis(r[None], y[:, :, None], 2)[:, :, 0]
        if arm in ("uncorrected", "matched"):
            tok, coeff = y, adv
        elif arm == "is":
            py = np.take_along_axis(p[None], y[:, :, None], 2)[:, :, 0]
            qy = np.take_along_axis(qs[None], y[:, :, None], 2)[:, :, 0]
            tok = y
            coeff = np.clip(py / np.maximum(qy, 1e-9), 0, IS_CLIP) * adv
        elif arm in ("subst", "broken"):
            gn = g if arm == "subst" else g_indep[st]
            tok = np.argmax(np.log(p)[None] + gn, axis=2)
            coeff = adv
            if arm == "subst":
                n_sub += int((tok != y).sum())
        else:
            raise ValueError(arm)
        # mean over the M rollouts of coeff * (e_tok - p), per context
        grad = np.zeros((C, V))
        for i in range(C):
            np.add.at(grad[i], tok[:, i], coeff[:, i])
        grad = grad / M - p * coeff.mean(axis=0)[:, None]
        th = th + LR * grad
    s = _softmax(th)
    s0 = _softmax(init)
    return dict(
        kl_t=_kl(s, t), kl_t0=_kl(s0, t),
        kl_q=_kl(s, q), kl_q0=_kl(s0, q),
        q_pull=_kl(s0, q) - _kl(s, q),
        pa=float(s[np.arange(C), a].mean()),
        pa0=float(s0[np.arange(C), a].mean()),
        acc=float((s.argmax(axis=1) == a).mean()),
        sub_rate=n_sub / (STEPS * M * C) if arm == "subst" else 0.0,
        q_pa=float(q[np.arange(C), a].mean()),
    )


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = {arm: _train(seed, arm) for arm in
                        ("uncorrected", "subst", "is", "broken", "matched")}
    return _CACHE[seed]


def test_geometry_is_a_strong_mismatch():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        u = m["uncorrected"]
        # init is already argmax-correct but far from the
        # target distribution; the sampler is far from both
        # and suppresses the intended token hard
        assert 0.40 <= u["kl_t0"] <= 0.90, (seed, u)
        assert u["kl_q0"] >= 1.20, (seed, u)
        assert u["q_pa"] <= 0.15, (seed, u)
        assert u["pa0"] >= 0.30, (seed, u)


def test_uncorrected_training_drifts_toward_sampler_and_collapses():
    _empf_recipe()
    for seed in SEEDS:
        u = run_seed(seed)["uncorrected"]
        # directed drift toward q, on every seed
        assert u["q_pull"] >= 1.00, (seed, u)
        # and the intended behavior collapses with it
        assert u["kl_t"] >= u["kl_t0"] + 1.50, (seed, u)
        assert u["pa"] <= 0.10, (seed, u)
        assert u["acc"] <= 0.30, (seed, u)


def test_gumbel_substitution_removes_drift_and_still_learns():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        s, u = m["subst"], m["uncorrected"]
        # no directed drift toward q: the student moves
        # away from the sampler, on every seed
        assert s["q_pull"] <= 0.05, (seed, s)
        assert u["q_pull"] - s["q_pull"] >= 1.20, (seed, m)
        # while the intended behavior genuinely improves
        assert s["kl_t"] <= s["kl_t0"] - 0.15, (seed, s)
        assert s["pa"] >= s["pa0"] + 0.10, (seed, s)
        assert s["acc"] >= 0.95, (seed, s)
        # strong-mismatch regime: most positions substitute
        assert 0.60 <= s["sub_rate"] <= 0.80, (seed, s)


def test_substitution_matches_true_q_importance_correction():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        s, c = m["subst"], m["is"]
        # the strong control (exact q, clipped ratios)
        # also removes the drift and learns
        assert c["q_pull"] <= 0.05, (seed, c)
        assert c["pa"] >= c["pa0"] + 0.10, (seed, c)
        # substitution, with NO access to q, lands on it
        assert abs(s["kl_t"] - c["kl_t"]) <= 0.10, (seed, m)
        assert abs(s["pa"] - c["pa"]) <= 0.08, (seed, m)


def test_broken_control_independent_noise_destroys_the_signal():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        b, s = m["broken"], m["subst"]
        # the literal expectation -- directed drift back
        # at uncorrected's level -- does NOT occur: the
        # broken student does not move toward q either.
        # Asserted as measured, not as hoped:
        assert abs(b["q_pull"]) <= 0.10, (seed, b)
        # what independent noise destroys is the learning
        # itself: reward and scored token are independent,
        # so the expected update is E[r] * 0 and the
        # student stays at init while substitution learns
        assert b["pa"] <= b["pa0"] + 0.03, (seed, b)
        assert b["kl_t"] >= s["kl_t"] + 0.20, (seed, m)
        assert b["kl_t"] >= b["kl_t0"] - 0.05, (seed, b)


def test_matched_sampler_sanity_no_drift_anywhere():
    _empf_recipe()
    for seed in SEEDS:
        m = run_seed(seed)
        mt = m["matched"]
        # q = p: the same uncorrected rule that collapses
        # under mismatch learns straight to the target
        assert mt["kl_t"] <= 0.12, (seed, mt)
        assert mt["pa"] >= 0.90, (seed, mt)
        assert mt["kl_t"] <= m["subst"]["kl_t"], (seed, m)
