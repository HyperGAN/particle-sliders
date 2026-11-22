"""CPU gate for the mismatch-decomposition claims of Yu,
Yu, Xu, Han, Jiang & Wang & Zou, "Mismatch Matters:
On-Policy Distillation Beyond Token Agreement"
(arXiv:2608.09836). The paper argues token agreement is
the wrong readout for on-policy distillation (OPD) and
decomposes teacher-student mismatch into two directions,
each with its own failure and its own fix in their TIDE
method:

  (1) DEGENERATE AGREEMENT: a student exploiting
      repetitive loops reaches near-perfect token
      agreement with the teacher despite globally flawed
      responses -- agreement must never gate.
  (2) STUDENT-EXCESS tokens: tokens the student generates
      that the teacher assigns ~zero probability; their
      log-ratio corrections grow unbounded and destabilize
      the update. TIDE bounds the correction with
      Hellinger shaping.
  (3) STUDENT-DEFICIT tokens: tokens the teacher prefers
      that the student rarely samples; sampled on-policy
      training gives them ~no signal, blocking transfer.
      TIDE adds an analytic teacher top-K term that moves
      them without requiring them to be sampled.

Toy operationalization: a per-position categorical world,
T = 40 positions x V = 10 tokens, pure numpy. The teacher
layout is RANDOMIZED per seed: a dominant token d* (drawn
per seed) is the preferred token at a per-seed fraction
phi ~ U(0.75, 0.85) of positions; the remaining positions
draw their preferred token at random. Per position the
preferred token carries u ~ U(0.62, 0.78), a second token
carries 0.45 * (1 - u), and the other 8 tokens share
0.55 * (1 - u). Students are per-position categoricals.

Phenomenon 1 arms: HONEST student q = teacher; LOOP
student = point mass on d* at every position (the
repetitive-loop degenerate). Readouts: agreement
E_q[p(x)] (mean over positions of the teacher probability
of the student's token), deficit mass (teacher mass on
tokens with student prob < 1e-3), and student entropy
(the global-flaw readout: the loop is a single repeated
token, so its entropy is exactly 0 while the honest
student's matches the teacher's ~1.16 nats).

Phenomenon 2 arms: a single categorical teacher (pref
0.70, an excess token at exactly 1e-4, the rest sharing
the remainder). CLEAN student = the teacher perturbed in
log space (sigma 0.3) -- close but NOT equal, so ordinary
tokens carry nonzero log-ratio weights and the control is
not vacuous. EXCESS student = 0.94 * clean + 0.06 mass on
the excess token. Per-sample step weight under the
OPD-style log-ratio correction is |log(p(x)/q(x))| over
20,000 samples from q; the Hellinger-bounded weight on
the SAME samples is 2|sqrt(q) - sqrt(p)|. Gradient
direction: the analytic (full-expectation) update vector
u = sum_x q_x w(x) (e_x - q) under signed log-ratio
weights w = log(p/q) vs signed Hellinger weights
w = 2(sqrt(p) - sqrt(q)); cosine between them.

Phenomenon 3 arms: a single categorical teacher (pref
0.80, a deficit token at 0.10, tail sharing 0.10) and a
student init with the deficit token at ~1e-3 (renormalized
q0 = 0.00111). Both arms train student logits by the
sampled score-function OPD gradient (256 samples/step,
20 steps, lr 5.0, identical stream RNG): theta -= lr *
mean_x[ log(q(x)/p(x)) (e_x - q) ]. The ANALYTIC arm adds,
each step, the teacher top-K (K = 2) term computed in
closed form, sum_{x in topK} q_x log(q_x/p_x) (e_x - q) --
the deficit token's exact expectation contribution,
which does not require it to be sampled.

Measured on pop-os (particlegan 0.10.1, torch threads 1),
5 seeds -- Phenomenon 1 (agreement honest / loop, deficit
honest / loop, entropy honest / loop):
  s0: 0.511 / 0.613,  0.000 / 0.387,  1.163 / 0
  s1: 0.490 / 0.540,  0.000 / 0.460,  1.206 / 0
  s2: 0.515 / 0.616,  0.000 / 0.384,  1.153 / 0
  s3: 0.515 / 0.582,  0.000 / 0.418,  1.155 / 0
  s4: 0.514 / 0.567,  0.000 / 0.433,  1.155 / 0
Phenomenon 2 (clean p95, excess p95 = max, Hellinger p95,
cos of analytic update directions):
  s0: 0.447, 6.399, 0.470, 0.978
  s1: 0.658, 6.399, 0.470, 0.960
  s2: 0.629, 6.399, 0.470, 0.942
  s3: 0.438, 6.399, 0.470, 0.912
  s4: 0.478, 6.399, 0.470, 0.901
  (clean mean |w| 0.079-0.180 -- nonzero, the control is
  a real student, not q = p; excess mean 0.491-0.571.)
Phenomenon 3 (q0, q after sampled training, q after
analytic training, deficit draws in the sampled arm over
5120 total draws):
  s0: 0.00111, 0.00321, 0.01549, 12
  s1: 0.00111, 0.00240, 0.00539, 8
  s2: 0.00111, 0.00330, 0.01214, 12
  s3: 0.00111, 0.00286, 0.01964, 9
  s4: 0.00111, 0.00267, 0.05499, 9

The arbitration, stated plainly: (1) agreement actively
prefers the degenerate student on every seed (loop minus
honest +0.049 to +0.102) while deficit mass separates
them by 0.38-0.46 and entropy by ~1.16 nats -- agreement
is not merely uninformative here, it points the wrong
way; the quantities that catch the loop are the mismatch
masses, exactly as the paper claims. (2) A single excess
token at 6% student mass sets the p95 AND the max step
weight at its log-ratio 6.40, ~10-15x the clean student's
p95, while Hellinger shaping caps the same samples at
0.47 (~14x lower) and keeps the analytic update direction
(cos 0.90-0.98 to the log-ratio update). (3) Sampled
training is a timescale block, not an absolute one: at
this budget the deficit token is drawn 8-12 times in
5120 and its prob at most doubles-triples off 1e-3,
while the analytic top-K term moves it 4.9-50x off init
on every seed without any extra sampling. Assertions
below encode exactly these margins, no more.

What is NOT asserted: the paper's LLM-scale effects
(response length, formatting failures, benchmark gains);
mean teacher log-prob as a global-flaw readout -- it was
tried in iteration and does NOT separate the loop student
(it concentrates on a high-probability token, so its
E_q[log p] sometimes EXCEEDS the honest student's);
entropy is the readout that separates, and it is close to
definitional for a point-mass loop, so the load-bearing
Phenomenon 1 assertions are agreement-vs-deficit. The
deficit block is budget-relative: at 40 steps the sampled
arm also escapes (positive feedback once the token is
drawn more often), and at lr 10 both arms destabilize --
the claim tested is the paper's, that at a fixed sampled
budget the deficit token gets ~no signal while the
analytic term moves it now. Pure numpy -- no GAN
machinery, because the claims are about categorical
mismatch bookkeeping, not training dynamics. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test
family; PR #211 is what enables this test in CI).
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


V = 10               # tokens
T = 40               # positions (phenomenon 1)
SEEDS = range(5)
N_W = 20000          # weight samples (phenomenon 2)
STEPS = 20           # training steps (phenomenon 3)
BUDGET = 256         # samples per step (phenomenon 3)
LR3 = 5.0            # logit step (phenomenon 3)
TOPK = 2             # teacher top-K for the analytic term


def _teacher_positions(rng):
    """Randomized-layout teacher: dominant token d* preferred
    at ~75-85% of positions, preferred prob U(0.62, 0.78)."""
    d_star = int(rng.integers(V))
    is_dom = rng.random(T) < rng.uniform(0.75, 0.85)
    pref = np.where(is_dom, d_star, rng.integers(0, V, T))
    u = rng.uniform(0.62, 0.78, T)
    p = np.zeros((T, V))
    for i in range(T):
        second = int(rng.choice([t for t in range(V) if t != pref[i]]))
        tail = [t for t in range(V) if t not in (pref[i], second)]
        p[i, pref[i]] = u[i]
        p[i, second] = 0.45 * (1 - u[i])
        for t in tail:
            p[i, t] = 0.55 * (1 - u[i]) / len(tail)
    return p, d_star


def _phen1(seed):
    rng = np.random.default_rng(41000 + seed)
    p, d_star = _teacher_positions(rng)
    loop = np.zeros((T, V))
    loop[:, d_star] = 1.0

    def agree(q):
        return float(np.mean(np.sum(q * p, axis=1)))

    def deficit(q):
        return float(np.mean(np.sum(p * (q < 1e-3), axis=1)))

    def entropy(q):
        return float(np.mean(-np.sum(q * np.log(np.maximum(q, 1e-300)), axis=1)))

    return dict(agree_h=agree(p), agree_l=agree(loop),
                def_h=deficit(p), def_l=deficit(loop),
                ent_h=entropy(p), ent_l=entropy(loop))


def _phen2(seed):
    rng = np.random.default_rng(42000 + seed)
    pref = int(rng.integers(V))
    exc = int(rng.choice([t for t in range(V) if t != pref]))
    others = [t for t in range(V) if t not in (pref, exc)]
    p = np.zeros(V)
    p[pref] = 0.70
    p[exc] = 1e-4
    for t in others:
        p[t] = (1 - 0.70 - 1e-4) / len(others)
    # clean student: perturbed teacher (close, not equal)
    qc = p * np.exp(rng.normal(0, 0.3, V))
    qc /= qc.sum()
    # excess student: clean + 6% mass on the excess token
    qe = 0.94 * qc
    qe[exc] += 0.06
    qe /= qe.sum()
    rng_s = np.random.default_rng(43000 + seed)

    def weights(q):
        xs = rng_s.choice(V, size=N_W, p=q)
        w_log = np.abs(np.log(p[xs] / q[xs]))
        w_hell = 2 * np.abs(np.sqrt(q[xs]) - np.sqrt(p[xs]))
        return w_log, w_hell

    wl_c, _ = weights(qc)
    wl_e, wh_e = weights(qe)

    def direction(q, kind):
        if kind == "log":
            ws = np.log(p / q)
        else:
            ws = 2 * (np.sqrt(p) - np.sqrt(q))
        u = q * ws
        return u - q * np.sum(u)

    u_log = direction(qe, "log")
    u_hell = direction(qe, "hell")
    cos = float(u_log @ u_hell
                / (np.linalg.norm(u_log) * np.linalg.norm(u_hell)))
    return dict(clean_mean=float(wl_c.mean()),
                clean_p95=float(np.percentile(wl_c, 95)),
                exc_p95=float(np.percentile(wl_e, 95)),
                hell_p95=float(np.percentile(wh_e, 95)),
                hell_max=float(wh_e.max()), cos=cos)


def _phen3(seed):
    rng = np.random.default_rng(44000 + seed)
    pref = int(rng.integers(V))
    defi = int(rng.choice([t for t in range(V) if t != pref]))
    others = [t for t in range(V) if t not in (pref, defi)]
    p = np.zeros(V)
    p[pref] = 0.80
    p[defi] = 0.10
    for t in others:
        p[t] = 0.10 / len(others)
    q0 = p.copy()
    q0[defi] = 1e-3
    q0 /= q0.sum()
    topk = np.argsort(-p)[:TOPK]

    def train(analytic):
        rng_t = np.random.default_rng(45000 + seed)
        theta = np.log(q0)
        draws = 0
        for _ in range(STEPS):
            q = np.exp(theta - theta.max())
            q /= q.sum()
            xs = rng_t.choice(V, size=BUDGET, p=q)
            draws += int(np.sum(xs == defi))
            r = np.log(q[xs] / p[xs])
            g = np.zeros(V)
            np.add.at(g, xs, r)
            g = g / BUDGET - q * np.mean(r)
            if analytic:
                ga = np.zeros(V)
                s = 0.0
                for x in topk:
                    v = q[x] * np.log(q[x] / p[x])
                    ga[x] += v
                    s += v
                g = g + (ga - q * s)
            theta -= LR3 * g
        q = np.exp(theta - theta.max())
        q /= q.sum()
        return float(q[defi]), draws

    q_s, draws_s = train(False)
    q_a, _ = train(True)
    return dict(q0=float(q0[defi]), q_sampled=q_s,
                q_analytic=q_a, draws_sampled=draws_s)


def test_degenerate_agreement_never_gates():
    _empf_recipe()
    for seed in SEEDS:
        m = _phen1(seed)
        # agreement actively prefers the degenerate loop student
        assert m["agree_l"] >= m["agree_h"] + 0.005, (seed, m)
        # deficit mass separates them by a wide margin
        assert m["def_l"] >= m["def_h"] + 0.30, (seed, m)
        # global-flaw readout: the loop has zero entropy, the
        # honest student carries the teacher's ~1.16 nats
        assert m["ent_h"] >= 1.0, (seed, m)
        assert m["ent_l"] <= 1e-12, (seed, m)


def test_excess_log_ratio_unbounded_hellinger_bounded():
    _empf_recipe()
    for seed in SEEDS:
        m = _phen2(seed)
        # control is a real student: ordinary-token weights nonzero
        assert m["clean_mean"] >= 0.05, (seed, m)
        assert m["clean_p95"] >= 0.25, (seed, m)
        # the excess token's log-ratio sets the tail step weight
        assert m["exc_p95"] >= 5.0, (seed, m)
        assert m["exc_p95"] >= 8 * m["clean_p95"], (seed, m)
        # Hellinger shaping caps the same samples ~14x lower
        assert m["hell_p95"] <= m["exc_p95"] / 10, (seed, m)
        assert m["hell_max"] <= 0.6, (seed, m)
        # ... while keeping the analytic update direction
        assert m["cos"] >= 0.85, (seed, m)


def test_deficit_blocked_sampled_restored_analytic():
    _empf_recipe()
    for seed in SEEDS:
        m = _phen3(seed)
        # the deficit token is barely sampled at this budget
        assert m["draws_sampled"] <= 16, (seed, m)
        # ... and sampled training barely moves it off init
        assert m["q_sampled"] <= 0.0045, (seed, m)
        # the analytic teacher top-K term moves it, unsampled
        assert m["q_analytic"] >= 4 * m["q0"], (seed, m)
        assert m["q_analytic"] >= 1.8 * m["q_sampled"], (seed, m)
