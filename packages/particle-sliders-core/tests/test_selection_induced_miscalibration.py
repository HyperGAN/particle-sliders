"""CPU gate for the selection-induced miscalibration claim
(arXiv:2611.22595): a score that is perfectly calibrated where it is
produced becomes badly miscalibrated where it is used, once the
funnel's selection policy has filtered the pool -- and the degradation
is caused by selection itself, not model error.

Paper: "Lost in the Funnel: Selection-Induced Miscalibration in
Ranking and Decision Systems" (Somani, Ragesh, Joshi & Singh). Scores
calibrated at the model boundary are consumed as probabilities
downstream (filtering, reranking, business rules, auctions), where
calibration is never re-validated. The paper diagnoses the breakdown
via boundary-wise residual identities, traces it to selection rather
than model error in an industry case study, and proposes Funnel-Aware
Recalibration (FAR): post-hoc, selection-policy-aware recalibration
at the boundary where scores drive decisions. In a live two-region
production A/B test FAR cut decision-time ECE by 60-80% and improved
auction efficiency; adapted public datasets corroborate in simulation.

Toy operationalization: synthetic candidates whose outcome
probabilities are known, scored by a scorer that is perfectly
calibrated BY CONSTRUCTION on the full pool:

  a ~ Beta(2, 5)          -- the scorer's own information
  s = 0.08 + 0.75 * a     -- the produced score
  z ~ N(0, 1)             -- a second funnel signal the scorer does
                             not model (a reranker boost, a business
                             rule, an auction-side feature)
  p = clip(s + 0.22 * tanh(z), 0.01, 0.99)   -- true outcome prob
  y ~ Bernoulli(p)

Because E[tanh(z)] = 0 and z is independent of a, E[y | s] = s
exactly: the score is a perfect probability for the information it
was produced from, and full-pool ECE verifies at ~0.005-0.008
(sampling noise only). The funnel, however, does not select on s
alone -- real funnels never do. Its rank is r = s + LAM * z, and
selection on r is informative about y BEYOND s: at fixed s, a survivor
owes its place partly to a high z, and high z raises the true
outcome rate. Conditioning on survival therefore shifts E[y | s]
upward: on the selected pool the same scores systematically
UNDER-state the realized outcome rate. That is the paper's
mechanism in miniature: the miscalibration is manufactured by the
selection policy, with zero model error anywhere.

Operationalization of "decision-time ECE" (the choice matters, and
the disagreement is reported, not hidden): ECE is computed with 10
equal-mass bins whose edges are fixed on the PRE-SELECTION (full)
pool and then applied to whichever pool is evaluated -- the score as
used at the decision boundary, binned the way the producing boundary
defined it. The pool-level gap mean(s) - mean(y) is reported
alongside. Two alternatives were measured and disagree by design:
(a) re-fitting equal-mass bins inside the selected pool still shows
the funnel effect here (the bias is one-signed across bins), but
(b) for the CONTROL policy below it manufactures a spurious
0.013-0.035 ECE out of binning noise on a truncated range. The
fixed-edge version is the honest one: it is ~0 exactly where
calibration truly survives.

Control policy (the identity the paper's claim must survive): pure
top-k selection on the score itself, r = s. Selection that is a
function of s alone cannot miscalibrate a calibrated score --
E[y | s, s in top-k] = E[y | s] = s -- and the toy verifies this:
fixed-edge ECE and pool gap stay at noise level (<= 0.009) at both
strengths. The failure needs a selection policy that carries outcome
information beyond the score, which is precisely what a multi-stage
funnel's combined rank is. This sharpens, and does not rescue, the
naive reading of the claim: "top-k by score" in a real funnel IS
top-k by a combined rank.

Arms (funnel policy, per seed, independent calibration pool of the
same size, same isotonic method everywhere -- the ONLY difference
between the two recalibration arms is the fitting distribution):
  raw:     selected pool, produced scores as-is.
  global:  isotonic fitted on a held-out FULL-pool sample, applied to
           the selected pool. Learns ~identity (the score is already
           calibrated on the full pool) and must NOT repair.
  FAR:     isotonic fitted on a held-out SELECTED-pool sample (the
           same selection policy applied to the calibration pool) --
           the paper's selection-aware arm. Must repair.

Selection strengths: keep 20% and keep 5% of the pool.

Measured on pop-os (particlegan 0.10.1), seeds 0-2, N = 40000 eval +
40000 calibration per seed: full-pool ECE 0.007-0.010; control
(pure-score selection) fixed-edge ECE <= 0.014 at both strengths;
funnel raw selected ECE 0.136-0.144 (keep 20%) and 0.176-0.194
(keep 5%) -- 15-25x the full-pool ECE, growing with selection
strength; pool gap -0.136 to -0.194 (scores UNDER-state the selected
pool's outcome rate, the sign funnel-boost selection induces);
global recalibration cuts selected ECE by only 2-7% (no repair);
FAR cuts it by 87-97% (the paper reports 60-80% in production; the
toy, with a known one-dimensional bias, repairs further -- asserted
here is only what the toy demonstrates). Assertions below encode
exactly these margins, no more. Skips cleanly when the installed
particlegan does not provide the EMPF recipe. Pure numpy -- no GAN
machinery, because the claim is about selection and calibration,
not training.
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


N = 40000          # candidates per pool (eval and calibration alike)
LAM = 0.15         # funnel-rank weight on the unmodelled signal z
Z_EFFECT = 0.22    # how far z moves the true outcome probability
BINS = 10
KEEPS = (0.20, 0.05)


def _pool(rng, n):
    """One candidate pool: scores s, funnel signal z, outcomes y."""
    a = rng.beta(2.0, 5.0, size=n)
    s = 0.08 + 0.75 * a
    z = rng.standard_normal(n)
    p = np.clip(s + Z_EFFECT * np.tanh(z), 0.01, 0.99)
    y = (rng.random(n) < p).astype(float)
    return s, z, y


def _ece_fixed(scores, y, edges):
    """ECE with bin edges fixed in advance (pre-selection bins)."""
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            total += float(m.mean()) * abs(
                float(scores[m].mean()) - float(y[m].mean()))
    return total


def _full_pool_edges(scores):
    edges = np.quantile(scores, np.linspace(0.0, 1.0, BINS + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    return edges


def _isotonic_fit(x, y):
    """Pool-adjacent-violators isotonic regression (increasing).

    Returns (block right-edges, block values) for step prediction.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    stack_v, stack_c, stack_r = [], [], []
    for xi, yi in zip(xs, ys):
        v, c, r = float(yi), 1.0, float(xi)
        while stack_v and stack_v[-1] > v:
            pv, pc, pr = stack_v.pop(), stack_c.pop(), stack_r.pop()
            v = (v * c + pv * pc) / (c + pc)
            c += pc
            r = max(r, pr)
        stack_v.append(v)
        stack_c.append(c)
        stack_r.append(r)
    return np.array(stack_r), np.array(stack_v)


def _isotonic_apply(fit, x):
    edges, values = fit
    idx = np.clip(np.searchsorted(edges, x, side="left"),
                  0, len(values) - 1)
    return values[idx]


def _topk(rank, frac):
    k = int(len(rank) * frac)
    return np.argsort(-rank, kind="mergesort")[:k]


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    s, z, y = _pool(rng, N)
    sc, zc, yc = _pool(rng, N)          # independent calibration pool
    edges = _full_pool_edges(s)
    out = dict(full_ece=_ece_fixed(s, y, edges))

    # Control policy: selection on the score alone.
    out["control"] = {}
    for frac in KEEPS:
        sel = _topk(s, frac)
        out["control"][frac] = dict(
            ece=_ece_fixed(s[sel], y[sel], edges),
            gap=float(s[sel].mean() - y[sel].mean()))

    # Funnel policy: selection on the combined rank r = s + LAM * z.
    rank, rank_c = s + LAM * z, sc + LAM * zc
    global_fit = _isotonic_fit(sc, yc)
    out["funnel"] = {}
    for frac in KEEPS:
        sel = _topk(rank, frac)
        sel_c = _topk(rank_c, frac)
        far_fit = _isotonic_fit(sc[sel_c], yc[sel_c])
        raw = _ece_fixed(s[sel], y[sel], edges)
        out["funnel"][frac] = dict(
            raw=raw,
            gap=float(s[sel].mean() - y[sel].mean()),
            glob=_ece_fixed(_isotonic_apply(global_fit, s[sel]),
                            y[sel], edges),
            far=_ece_fixed(_isotonic_apply(far_fit, s[sel]),
                           y[sel], edges))
    return out


def test_perfect_score_survives_selection_on_itself():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The scorer is perfectly calibrated by construction -- verify
        # on the full pool, don't assume.
        assert r["full_ece"] < 0.02, (
            f"seed {seed}: full-pool ECE {r['full_ece']:.4f} -- the "
            f"by-construction calibration does not verify")
        # Identity: selection that is a function of the score alone
        # cannot miscalibrate it. This control is what makes the
        # funnel failure attributable to the selection POLICY.
        for frac in KEEPS:
            c = r["control"][frac]
            assert c["ece"] < 0.02, (
                f"seed {seed}: pure-score selection keep={frac} ECE "
                f"{c['ece']:.4f} -- selection on s alone should leave "
                f"a calibrated score calibrated")
            assert abs(c["gap"]) < 0.02, (
                f"seed {seed}: pure-score selection keep={frac} pool "
                f"gap {c['gap']:+.4f}")


def test_funnel_selection_miscalibrates_and_only_far_repairs():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        f20, f05 = r["funnel"][0.20], r["funnel"][0.05]
        for frac, f in ((0.20, f20), (0.05, f05)):
            # The failure: same scores, selected pool, badly
            # miscalibrated -- an order of magnitude over the full
            # pool, with zero model error anywhere in the toy.
            assert f["raw"] > 0.10, (
                f"seed {seed}: funnel keep={frac} selected ECE "
                f"{f['raw']:.4f}")
            assert f["raw"] > 10 * r["full_ece"], (
                f"seed {seed}: funnel keep={frac} selected ECE "
                f"{f['raw']:.4f} vs full-pool {r['full_ece']:.4f}")
            # Sign: funnel-boost survivors carry high z, which raises
            # the true rate -- scores UNDER-state it (negative gap).
            assert f["gap"] < -0.10, (
                f"seed {seed}: funnel keep={frac} pool gap "
                f"{f['gap']:+.4f} -- expected systematic "
                f"under-statement by the produced scores")
            # Global recalibration (fitted on the full pool) must NOT
            # repair: it learns ~identity where the score is already
            # calibrated and is blind to the selection shift.
            assert f["glob"] > 0.85 * f["raw"], (
                f"seed {seed}: funnel keep={frac} global-recal ECE "
                f"{f['glob']:.4f} vs raw {f['raw']:.4f} -- full-pool "
                f"recalibration should not repair the selected pool")
            # FAR (fitted on the selected distribution, same policy)
            # cuts decision-time ECE by a large margin.
            assert f["far"] < 0.20 * f["raw"], (
                f"seed {seed}: funnel keep={frac} FAR ECE "
                f"{f['far']:.4f} vs raw {f['raw']:.4f} -- "
                f"selection-aware recalibration should cut >= 80%")
        # Selection-strength dependence: the tighter funnel
        # miscalibrates more.
        assert f05["raw"] > f20["raw"] + 0.02, (
            f"seed {seed}: selected ECE keep=5% {f05['raw']:.4f} "
            f"should exceed keep=20% {f20['raw']:.4f}")
        assert f05["gap"] < f20["gap"], (
            f"seed {seed}: pool gap keep=5% {f05['gap']:+.4f} "
            f"should be more negative than keep=20% {f20['gap']:+.4f}")
