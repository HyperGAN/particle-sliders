"""CPU gate for the Semi-OPD overlap claim (arXiv:2610.11291): whether
distilling on frozen initial-student rollouts beats fresh on-policy
rollouts depends on teacher-student alignment -- and in this toy the
paper's ordering comes out INVERTED, in both regimes, per seed.

Paper: "When Do We Need On-Policy Distillation? Distilling on Offline
Student Rollouts Is Often Better" (Zhao et al.). Semi-OPD distills on
offline rollouts generated once by the INITIAL student and reportedly
beats full on-policy distillation (OPD) in 14/17 teacher-student
pairs (up to +13.6% accuracy, 11.4x training speedup). The deciding
variable is teacher-student alignment, measured by output-token
overlap: OPD is beneficial only at HIGH overlap; for misaligned
pairs, fresh student rollouts drift increasingly off-policy w.r.t.
the teacher as context grows, weakening the supervision signal,
while frozen initial-student rollouts stay stable.

Toy operationalization: a categorical bigram world. Contexts are
token pairs (a, b) over a vocab of 12; the next-token distribution is
a categorical over the vocab. The teacher's logits are
Wt @ phi(a, b) + U_T[a, b], where phi(a, b) = E[a] + E[b] for fixed
random token embeddings E in R^16, and U_T is a per-pair component.
The student is linear in the same features (logits Ws @ phi), so it
can represent the teacher's shared component but not the per-pair
component: fitting is a genuine generalization problem over the
pair chain's effective support (~51 of 144 pairs under the teacher's
stationary distribution, computed by power iteration).

Overlap knob: the initial student is Ws0 = alpha * Ws_T + (1-alpha)
* Ws_A, where Ws_T / Ws_A are the student-class projections (batch-GD
fits over all pairs) of the teacher and of an unrelated alt teacher.
alpha = 0.15 gives teacher-weighted output-token overlap 0.43 (LOW);
alpha = 0.85 gives 0.83 (HIGH). Overlap is measured, not assumed:
sum over teacher-stationary pairs of sum_x min(Pt, Ps).

Supervision target -- the paper's weakening mechanism, encoded as a
property of the teacher, not of an arm: on a rollout prefix, the
teacher's signal is blended toward uniform by the prefix's
teacher-familiarity w = min(1, q / q_typ), where q is the geometric
mean of the teacher's probabilities of the tokens generated so far
in that sequence and q_typ = exp(-E_pi[H(Pt)]) is the teacher's own
typical per-token likelihood (its own rollouts score w ~ 0.89). A
prefix the teacher itself would produce gets the teacher's true
conditional; a drifted prefix gets a mostly-uniform target.

Arms (matched gradient steps: 30 rounds x 10 rollouts x 14 tokens of
per-context SGD, eta = 0.15; prompts are pair starts drawn from the
teacher's stationary pair distribution, the same stream for all
arms):
  onpolicy: fresh rollouts sampled from the CURRENT student each
            round; teacher supervises those prefixes.
  offline:  Semi-OPD -- rollouts sampled ONCE from the INITIAL
            student, frozen, the same set cycled every round.
  teacher:  off-policy control -- fresh rollouts from the teacher.
  none:     no-distillation baseline (the initial student itself).

Metric: held-out student-teacher KL, averaged over contexts visited
by fresh teacher rollouts (one fixed evaluation stream, identical
for every arm and seed, so comparisons are paired).

What the toy actually shows (measured on this VM and on pop-os,
particlegan 0.10.1; per-seed ranges over seeds 0-2):

  LOW overlap (alpha 0.15, init held-out KL 1.174):
    onpolicy 0.214-0.238  <  offline 0.259-0.370  <<  none 1.174,
    teacher control 0.079-0.083 (best).
    The on-policy sampler ESCAPES: its mean visited familiarity
    rises over rounds (first-third mean 0.41-0.44, last-third
    0.52-0.59), because learning the teacher's conditional on its
    own contexts -- even blended -- improves the contexts it
    visits next, and the shared linear class lets that transfer.
    The frozen set keeps teaching the initial student's contexts
    (its mean familiarity is 0.28-0.43, fixed). This is the
    opposite of the paper's low-overlap ordering.
  HIGH overlap (alpha 0.85, init held-out KL 0.101):
    offline 0.098-0.169  <  onpolicy 0.191-0.217; on-policy is
    clearly worse than none (0.101) while offline roughly ties or
    loses to none (per seed: +0.043, -0.003, +0.068 vs init);
    teacher control 0.074-0.078 improves on none. The on-policy sampler DEGRADES its own input: its
    visited familiarity (0.60-0.63 averaged over rounds) sits
    BELOW the frozen set's (0.65-0.85), because training toward
    blended targets on its own prefixes makes its sampling less
    teacher-like, which degrades the next round's targets further.
    Freezing caps the damage at the initial student's level. This
    is the opposite of the paper's high-overlap ordering.
  The one piece of the paper's mechanism that does hold
  mechanically: within a rollout, familiarity falls as context
  grows for the misaligned student (initial-student rollouts at
  LOW overlap, positions 1-4 mean w 0.28-0.39 vs positions 10-13
  0.22-0.26, measured on 40 fresh rollouts per seed), and the
  teacher-data control dominates both student-rollout arms in
  both regimes -- at LOW overlap by an order of magnitude,
  because supervision on student prefixes is mostly
  uniform-blended there (frozen-set w 0.28-0.43).

So the crossover is NOT reproduced: in this toy the frozen-vs-fresh
distinction is second-order and sign-flipped relative to the paper,
driven by sampler feedback through a shared, generalizing student
class (positive feedback when there is nothing to lose, negative
feedback when blended targets erode a good sampler). Assertions
below encode exactly the demonstrated orderings and margins, no
more.

What is NOT asserted: anything about real language models, real
tokenizers, or the paper's 17 teacher-student pairs; that offline
distillation is inferior in general (at HIGH overlap it is the
better of the two student-rollout arms here, and the safer one);
that the familiarity blending is the teacher model the paper's
authors would choose (it is this toy's encoding of "supervision
weakens on drifted prefixes"); or anything about training speed
(the paper's 11.4x figure is not tested). Pure numpy bigram
dynamics -- no GAN machinery, because the claim is about rollout
distributions and supervision targets, not training a generator.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe (the house guard for this paper-test family; PR #211 is
what enables this test in CI).
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


V, D = 12, 16          # vocab, embedding dim
H, NSEQ, ROUNDS = 14, 10, 30
ETA = 0.15
ALPHA_LOW, ALPHA_HIGH = 0.15, 0.85
SEEDS = (0, 1, 2)

_rngE = np.random.default_rng(42)
E = _rngE.normal(0, 1, (V, D)) / np.sqrt(D)
PHI = np.zeros((V, V, D))
for _a in range(V):
    for _b in range(V):
        PHI[_a, _b] = E[_a] + E[_b]
PHIflat = PHI.reshape(V * V, D)


def _sm(L):
    L = L - L.max(-1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(-1, keepdims=True)


_rngT = np.random.default_rng(5)
Wt = _rngT.normal(0, 1.2, (V, D))
Wa = _rngT.normal(0, 1.2, (V, D))
Ut = _rngT.normal(0, 0.45, (V, V, V))
Ua = _rngT.normal(0, 0.45, (V, V, V))
Pt = _sm(PHI @ Wt.T + Ut)
Pa = _sm(PHI @ Wa.T + Ua)

# Teacher pair-stationary distribution by power iteration.
pi_pair = np.full((V, V), 1.0 / (V * V))
for _ in range(400):
    pi_pair = np.einsum("ab,abx->bx", pi_pair, Pt)
    pi_pair /= pi_pair.sum()
_ent = -np.sum(Pt * np.log(Pt), axis=-1)
Q_TYP = float(np.exp(-np.sum(pi_pair * _ent)))  # teacher's typical likelihood
UNIF = np.full(V, 1.0 / V)


def _probs_S(W):
    return _sm(PHI @ W.T)


def _fit_student(Ptarget, steps=300, eta=2.0):
    W = np.zeros((V, D))
    Pf = Ptarget.reshape(V * V, V)
    for _ in range(steps):
        Ps = _sm(PHIflat @ W.T)
        W -= eta * (np.einsum("cv,cd->vd", Ps - Pf, PHIflat) / (V * V))
    return W


Ws_T = _fit_student(Pt)
Ws_A = _fit_student(Pa)


def overlap(W):
    Ps = _probs_S(W)
    return float(np.sum(pi_pair * np.sum(np.minimum(Pt, Ps), axis=-1)))


def heldout_KL(W, seed=999, nseq=24):
    """Mean KL(Pt || Ps) over contexts of fresh teacher rollouts.

    One fixed evaluation stream (seed 999) for every arm and seed,
    so all comparisons are paired on identical contexts.
    """
    rng = np.random.default_rng(seed)
    Ps = _probs_S(W)
    tot, cnt = 0.0, 0
    starts = rng.choice(V * V, size=nseq, p=pi_pair.reshape(-1))
    for s in starts:
        a, b = s // V, s % V
        for _ in range(H):
            pT, pS = Pt[a, b], Ps[a, b]
            tot += float(np.sum(pT * (np.log(pT) - np.log(pS))))
            cnt += 1
            nb = rng.choice(V, p=pT)
            a, b = b, nb
    return tot / cnt


def run_arm(seed, alpha, mode):
    """Train one arm; return final held-out KL and diagnostics.

    Diagnostics: per-round mean visited familiarity for the arm,
    the frozen dataset's mean familiarity, and the initial
    student's within-rollout familiarity by position band.
    """
    rng = np.random.default_rng(500 + seed)
    W = alpha * Ws_T + (1 - alpha) * Ws_A
    starts_all = rng.choice(V * V, size=ROUNDS * NSEQ,
                            p=pi_pair.reshape(-1))

    def rollout(P, starts):
        out = []
        for s in starts:
            a, b = s // V, s % V
            logq, n = 0.0, 0
            for _ in range(H):
                w = 1.0 if n == 0 else min(
                    1.0, float(np.exp(logq / n)) / Q_TYP)
                out.append((a, b, w * Pt[a, b] + (1 - w) * UNIF, w))
                nb = rng.choice(V, p=P[a, b])
                logq += np.log(Pt[a, b][nb])
                n += 1
                a, b = b, nb
        return out

    dataset = rollout(_probs_S(W), starts_all[:NSEQ])
    frozen_w = float(np.mean([x[3] for x in dataset]))

    hist_w = []
    for r in range(ROUNDS):
        starts = starts_all[r * NSEQ:(r + 1) * NSEQ]
        if mode == "onpolicy":
            batch = rollout(_probs_S(W), starts)
        elif mode == "offline":
            batch = dataset
        elif mode == "teacher":
            batch = rollout(Pt, starts)
        else:
            batch = []
        if batch:
            hist_w.append(float(np.mean([x[3] for x in batch])))
            for (a, b, tgt, w) in batch:
                lg = W @ PHI[a, b]
                lg -= lg.max()
                e = np.exp(lg)
                p1 = e / e.sum()
                W = W - ETA * np.outer(p1 - tgt, PHI[a, b])
    return dict(kl=heldout_KL(W), hist_w=hist_w, frozen_w=frozen_w,
                ovl=overlap(alpha * Ws_T + (1 - alpha) * Ws_A))


def drift_bands(seed, alpha, nseq=40):
    """Within-rollout familiarity drift of the INITIAL student.

    Mean visited familiarity at positions 1-4 vs positions 10-13
    over fresh initial-student rollouts (dedicated stream, seed
    7000 + seed). Returns (early, late).
    """
    rng = np.random.default_rng(7000 + seed)
    W = alpha * Ws_T + (1 - alpha) * Ws_A
    P = _probs_S(W)
    starts = rng.choice(V * V, size=nseq, p=pi_pair.reshape(-1))
    by_pos = [[] for _ in range(H)]
    for s in starts:
        a, b = s // V, s % V
        logq, n = 0.0, 0
        for pos in range(H):
            w = 1.0 if n == 0 else min(
                1.0, float(np.exp(logq / n)) / Q_TYP)
            by_pos[pos].append(w)
            nb = rng.choice(V, p=P[a, b])
            logq += np.log(Pt[a, b][nb])
            n += 1
            a, b = b, nb
    means = [float(np.mean(x)) for x in by_pos]
    return float(np.mean(means[1:5])), float(np.mean(means[10:14]))


def test_semi_opd_crossover_is_inverted_in_toy():
    _empf_recipe()
    res = {}
    for alpha, tag in ((ALPHA_LOW, "low"), (ALPHA_HIGH, "high")):
        for mode in ("onpolicy", "offline", "teacher", "none"):
            res[(tag, mode)] = [run_arm(s, alpha, mode) for s in SEEDS]
        res[(tag, "init_kl")] = heldout_KL(
            alpha * Ws_T + (1 - alpha) * Ws_A)

    # The overlap knob genuinely separates the regimes.
    for s in SEEDS:
        low0 = res[("low", "none")][s]
        high0 = res[("high", "none")][s]
        assert low0["ovl"] < 0.55, f"LOW overlap {low0['ovl']:.3f}"
        assert high0["ovl"] > 0.75, f"HIGH overlap {high0['ovl']:.3f}"

    for s in SEEDS:
        init_low = res[("low", "init_kl")]
        on = res[("low", "onpolicy")][s]
        off = res[("low", "offline")][s]
        tea = res[("low", "teacher")][s]

        # LOW: both student-rollout arms learn a lot vs none, the
        # teacher-data control dominates, and -- inverted vs the
        # paper -- on-policy beats the frozen rollouts per seed.
        assert init_low > 0.9, f"seed {s}: LOW init KL {init_low:.3f}"
        assert off["kl"] < init_low - 0.5, (
            f"seed {s}: LOW offline KL {off['kl']:.3f} vs init "
            f"{init_low:.3f} -- frozen rollouts should still teach")
        assert on["kl"] < off["kl"] - 0.03, (
            f"seed {s}: LOW onpolicy {on['kl']:.3f} vs offline "
            f"{off['kl']:.3f} -- toy inverts the paper: the fresh "
            f"sampler escapes, the frozen set does not")
        assert tea["kl"] < on["kl"] - 0.08, (
            f"seed {s}: LOW teacher-data KL {tea['kl']:.3f} vs "
            f"onpolicy {on['kl']:.3f}")

        # LOW mechanism: within a rollout, familiarity falls as
        # context grows (the paper's drift, present mechanically),
        # but across rounds the on-policy sampler's familiarity
        # RISES (escape), opposite the paper's drift story.
        early, late = drift_bands(s, ALPHA_LOW)
        assert late < early - 0.04, (
            f"seed {s}: LOW within-rollout familiarity early "
            f"{early:.3f} late {late:.3f} -- drift with context "
            f"growth should be visible")
        w_first = float(np.mean(on["hist_w"][:10]))
        w_last = float(np.mean(on["hist_w"][-10:]))
        assert w_last > w_first + 0.04, (
            f"seed {s}: LOW onpolicy visited w first-third "
            f"{w_first:.3f} last-third {w_last:.3f} -- sampler "
            f"should escape upward in this toy")
        assert off["frozen_w"] < 0.5, (
            f"seed {s}: LOW frozen-set familiarity "
            f"{off['frozen_w']:.3f} -- supervision on student "
            f"prefixes is mostly uniform-blended here")

    for s in SEEDS:
        init_high = res[("high", "init_kl")]
        on = res[("high", "onpolicy")][s]
        off = res[("high", "offline")][s]
        tea = res[("high", "teacher")][s]

        # HIGH: inverted vs the paper again -- the frozen rollouts
        # beat fresh ones per seed, both student-rollout arms are
        # on-policy is worse than doing nothing, offline gains
        # nothing meaningful, and only teacher-data helps.
        assert init_high < 0.15, f"seed {s}: HIGH init KL {init_high:.3f}"
        assert off["kl"] < on["kl"] - 0.02, (
            f"seed {s}: HIGH offline {off['kl']:.3f} vs onpolicy "
            f"{on['kl']:.3f} -- freezing caps the sampler damage")
        assert on["kl"] > init_high + 0.05, (
            f"seed {s}: HIGH onpolicy KL {on['kl']:.3f} vs init "
            f"{init_high:.3f} -- blended targets on its own "
            f"prefixes damage the aligned student")
        assert off["kl"] > init_high - 0.02, (
            f"seed {s}: HIGH offline KL {off['kl']:.3f} vs init "
            f"{init_high:.3f} -- frozen rollouts give no "
            f"meaningful gain at HIGH overlap in this toy")
        assert tea["kl"] < init_high - 0.01, (
            f"seed {s}: HIGH teacher-data KL {tea['kl']:.3f} vs "
            f"init {init_high:.3f} -- clean targets still help")

        # HIGH mechanism: the current student's prefixes are LESS
        # teacher-familiar than the frozen initial student's --
        # the sampler degrades its own supervision input.
        w_on = float(np.mean(on["hist_w"]))
        assert w_on < off["frozen_w"] - 0.03, (
            f"seed {s}: HIGH onpolicy visited w {w_on:.3f} vs "
            f"frozen {off['frozen_w']:.3f}")
