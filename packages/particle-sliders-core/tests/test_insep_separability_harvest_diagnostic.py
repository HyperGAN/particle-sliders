"""CPU gate for the INSEP claim (arXiv:2611.28314): class
separability along a trait direction is trainable in both
directions, and a defense that fine-tunes it DOWN along the
refusal direction weakens the mean-difference estimate that
abliteration relies on. Converted here as a HARVEST DIAGNOSTIC
for Particle Sliders: separability along the trait, computed
from the labeled contrast set alone BEFORE harvest, should
predict whether the mean-difference harvest -- exactly the
estimator fit_direction computes in
packages/particle-sliders-core/src/particle_sliders/textsliders/
reward_sliders/directions.py (PR #231) -- will recover the true
direction. This is the sibling of PR #272's coherence statistic
(mean pairwise cosine of per-example activation differences,
Spearman 1.00 with realized steering on every seed there); the
open question this test answers is whether separability adds
anything NEXT TO coherence or is the same signal twice.

Toy operationalization: UNPAIRED contrast classes in R^64 (the
abliteration setting -- two labeled sets, no pairing). Each
example is x = c_i + label * (gap/2) * u + eta, with centre
c_i ~ N(0, 0.5^2 I), noise eta with sd 0.3 orthogonal to u and
sd 0.8 along u, around a unit ground-truth trait direction u.
The separability knob is the class-mean GAP along u, swept over
(2.0, 1.0, 0.5, 0.25, 0.1): at gap 2.0 the two classes'
projections on u are well separated; at gap 0.1 they overlap
almost completely. (Iteration note: sweeping the within-class
variance ALONG u instead does NOT degrade the mean-difference
harvest -- parallel noise is absorbed along +/-u and cos(v, u)
stays high non-monotonically, Spearman fisher-eff -0.50..0.90
across seeds. The gap is the separability driver that bites
this estimator, and it is also the quantity an INSEP-style
fine-tune directly shrinks.)

Pre-harvest statistics (both know nothing about u):
  (a) coherence, PR #272's statistic: index-pair the two
      classes (fixed permutation), take per-example differences,
      mean pairwise cosine;
  (b) separability: SPLIT-HALF Fisher ratio -- harvest a
      direction from the first halves of both classes, project
      the SECOND halves onto it, Fisher = between-class /
      within-class variance of those projections. Split-half is
      load-bearing: a full-sample Fisher on projections onto the
      full-sample harvested direction saturates at ~1.0-1.5 on
      the inseparable arms (the direction self-aligns with its
      own sampling noise); the split-half version spans
      ~4.2 down to ~0.02 across the sweep.
Null: the same split-half Fisher with labels shuffled -- it
must NOT predict realized quality.

Realized quality: harvest v = unit(mean_pos - mean_neg) (the
fit_direction estimator), cos(v, u), and net steering efficacy
E(v) = (v.u) - sqrt(1 - (v.u)^2) as in PR #272's readout.

INSEP-move closure: take the most separable condition and apply
the defense in toy form -- a rank-1 representation map that
shrinks every activation's deviation along the trait axis u by
75% (x' = x - 0.75 * ((x - grand).u) u; the toy image of a
low-rank adapter fine-tuned against the trait direction, which
the defender targets directly). Two tempting alternative forms
provably do NOTHING and were rejected in iteration: shrinking
each class mean toward the grand mean (a per-class translation
only rescales the mean-difference vector), and shrinking along
the HARVESTED direction v-hat (v-hat is an eigenvector of that
map, so the re-harvest is identical). Only a map along an axis
misaligned with the sampling-noise realization rotates the
estimate. The separability statistic must collapse and the
re-harvest must degrade by roughly what the sweep predicts for
that statistic value.

Control (cheap, included): an LDA direction (pooled-covariance
whitened mean difference, ridge 0.05) on the same sets -- does
a stronger estimator rescue the inseparable arms? Measured:
no; LDA underperforms the plain mean difference at every gap
in this toy (covariance estimation at D=64, N=96 costs more
than the whitening buys), and its efficacy on the most
inseparable arm stays below zero.

Measured on pop-os (particlegan 0.10.1), 5 seeds: efficacy by
gap level 0.45-0.62 (gap 2.0) down to -1.13..-0.65 (gap 0.1);
per-seed Spearman(separability, efficacy) 0.70-1.00, pooled
0.82; per-seed Spearman(coherence, efficacy) 0.60-1.00, pooled
0.87; Pearson(separability, coherence) 0.96 -- in this toy the
two statistics are near-redundant, coherence if anything the
slightly stronger pooled predictor. Shuffled-label separability
pooled Spearman with efficacy 0.10. INSEP move: Fisher
1.66-4.23 -> 0.001-0.244 and re-harvest efficacy drops by
0.84-0.97, landing in the inseparable sweep range. Assertions
below encode exactly these margins, no more.

What is NOT asserted: any real-model separability threshold,
and that separability and coherence are redundant in general
-- real per-example differences vary for reasons (prompt type,
per PR #272's paper) that can move coherence without moving
class separability. The claim tested is the ordering in this
toy plus the statistic<->mechanism closure.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
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


D = 64            # activation dimension
N = 48            # examples per contrast class
S_C = 0.5         # per-example centre sd
S_PERP = 0.3      # noise sd orthogonal to the trait direction
S_PAR = 0.8       # noise sd along the trait direction
GAPS = (2.0, 1.0, 0.5, 0.25, 0.1)   # separability: split -> merged
SEEDS = (0, 1, 2, 3, 4)
GAMMA = 0.75      # INSEP move: trait-axis deviation shrink


def _unit(x):
    return x / np.linalg.norm(x)


def _coherence(deltas):
    """PR #272's statistic: mean pairwise cosine of the
    per-example differences. No knowledge of u."""
    n = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
    gram = n @ n.T
    iu = np.triu_indices(len(deltas), 1)
    return float(gram[iu].mean())


def _fisher(pos, neg, v):
    pp, pn = pos @ v, neg @ v
    return float((pp.mean() - pn.mean()) ** 2 / (pp.var() + pn.var()))


def _separability(pos, neg):
    """Split-half Fisher ratio: direction harvested from the
    first halves, statistic evaluated on the second halves.
    Computable from the labeled contrast set alone, pre-harvest."""
    h = len(pos) // 2
    vh = _unit(pos[:h].mean(0) - neg[:h].mean(0))
    return _fisher(pos[h:], neg[h:], vh)


def _efficacy(v, u):
    c = float(v @ u)
    return c - float(np.sqrt(max(0.0, 1.0 - c * c))), c


def _spearman(x, y):
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _gen(rng, u, gap):
    def make(sign):
        centres = rng.normal(0.0, S_C, (N, D))
        perp = rng.normal(0.0, S_PERP, (N, D))
        perp -= np.outer(perp @ u, u)
        par = np.outer(rng.normal(0.0, S_PAR, N), u)
        return centres + sign * (gap / 2.0) * u + perp + par
    return make(+1.0), make(-1.0)


def _harvest(pos, neg):
    """The plain mean difference, normalized -- fit_direction's
    estimator in toy form."""
    return _unit(pos.mean(0) - neg.mean(0))


def _lda_direction(pos, neg):
    Xc = np.vstack([pos - pos.mean(0), neg - neg.mean(0)])
    cov = (Xc.T @ Xc) / (2 * N - 2) + 0.05 * np.eye(D)
    return _unit(np.linalg.solve(cov, pos.mean(0) - neg.mean(0)))


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)
    u = _unit(rng.standard_normal(D))
    levels = []
    top = None
    for gap in GAPS:
        pos, neg = _gen(rng, u, gap)
        if gap == GAPS[0]:
            top = (pos, neg)
        v = _harvest(pos, neg)
        eff, cos_u = _efficacy(v, u)
        deltas = pos - neg[rng.permutation(N)]
        # Shuffled-label null for the separability statistic.
        X = np.vstack([pos, neg])
        Xs = X[rng.permutation(2 * N)]
        f_null = _separability(Xs[:N], Xs[N:])
        eff_lda, _ = _efficacy(_lda_direction(pos, neg), u)
        levels.append(dict(
            gap=gap, sep=_separability(pos, neg), f_null=f_null,
            coherence=_coherence(deltas), eff=eff, cos_u=cos_u,
            eff_lda=eff_lda))

    # INSEP move on the most separable condition: rank-1
    # representation map shrinking deviations along the trait
    # axis by GAMMA; then re-measure.
    pos, neg = top
    grand = np.vstack([pos, neg]).mean(0)
    gp = float(grand @ u)
    pos2 = pos - GAMMA * np.outer(pos @ u - gp, u)
    neg2 = neg - GAMMA * np.outer(neg @ u - gp, u)
    eff2, _ = _efficacy(_harvest(pos2, neg2), u)
    insep = dict(sep_before=_separability(pos, neg),
                 sep_after=_separability(pos2, neg2),
                 eff_before=levels[0]["eff"], eff_after=eff2)
    return dict(levels=levels, insep=insep)


def test_separability_predicts_mean_difference_harvest():
    _empf_recipe()
    pooled = []
    for seed in SEEDS:
        r = run_seed(seed)
        lv = r["levels"]
        pooled.extend(lv)
        sep = [x["sep"] for x in lv]
        eff = [x["eff"] for x in lv]

        # Trap guard: the knob must actually span separable to
        # inseparable, and the harvest must actually degrade.
        assert sep[0] >= 1.0, (
            f"seed {seed}: separable-arm Fisher {sep[0]:.3f} "
            f"-- separability knob broken at the high end")
        assert sep[-1] <= 0.15, (
            f"seed {seed}: inseparable-arm Fisher {sep[-1]:.3f} "
            f"-- separability knob broken at the low end")
        assert eff[0] - eff[-1] >= 0.9, (
            f"seed {seed}: endpoint efficacy margin "
            f"{eff[0] - eff[-1]:.3f} (high {eff[0]:.3f}, "
            f"low {eff[-1]:.3f}) -- INSEP degradation law absent")
        assert eff[0] > 0.3, (
            f"seed {seed}: separable efficacy {eff[0]:.3f} -- "
            f"the harvest does not net-steer even when separable")
        assert eff[-1] <= -0.5, (
            f"seed {seed}: inseparable efficacy {eff[-1]:.3f} "
            f"-- the harvest still net-steers when inseparable")

        # Per-seed predictive validity (loose; the pooled check
        # below is the load-bearing one).
        rho_sep = _spearman(sep, eff)
        assert rho_sep >= 0.6, (
            f"seed {seed}: Spearman(separability, efficacy) "
            f"{rho_sep:.2f} across levels {list(zip(sep, eff))}")
        rho_coh = _spearman([x["coherence"] for x in lv], eff)
        assert rho_coh >= 0.5, (
            f"seed {seed}: Spearman(coherence, efficacy) "
            f"{rho_coh:.2f}")

        # LDA control: a stronger estimator does NOT rescue the
        # inseparable arm in this toy.
        assert lv[-1]["eff_lda"] <= 0.0, (
            f"seed {seed}: LDA efficacy on the inseparable arm "
            f"{lv[-1]['eff_lda']:.3f} -- whitening rescued what "
            f"mean difference lost, contradicting the record")

        # INSEP-move closure, per seed.
        ins = r["insep"]
        assert ins["sep_after"] <= ins["sep_before"] / 4.0, (
            f"seed {seed}: INSEP move moved Fisher "
            f"{ins['sep_before']:.3f} -> {ins['sep_after']:.3f} "
            f"-- the defense did not reduce separability")
        assert ins["eff_before"] - ins["eff_after"] >= 0.5, (
            f"seed {seed}: INSEP move moved efficacy "
            f"{ins['eff_before']:.3f} -> {ins['eff_after']:.3f} "
            f"-- harvest did not degrade with separability")
        assert ins["eff_after"] <= 0.1, (
            f"seed {seed}: post-INSEP efficacy "
            f"{ins['eff_after']:.3f} -- re-harvest still steers "
            f"after the defense")

    # Pooled predictive validity across all conditions x seeds.
    sep = [x["sep"] for x in pooled]
    coh = [x["coherence"] for x in pooled]
    eff = [x["eff"] for x in pooled]
    rho_sep = _spearman(sep, eff)
    assert rho_sep >= 0.8, (
        f"pooled Spearman(separability, efficacy) {rho_sep:.2f} "
        f"over {len(pooled)} condition-seeds")
    rho_coh = _spearman(coh, eff)
    assert rho_coh >= 0.85, (
        f"pooled Spearman(coherence, efficacy) {rho_coh:.2f}")
    # The two statistics are near-redundant in this toy: an
    # honest negative for separability earning its own seat.
    corr = float(np.corrcoef(sep, coh)[0, 1])
    assert corr >= 0.9, (
        f"Pearson(separability, coherence) {corr:.3f} -- the "
        f"statistics are NOT redundant here, contradicting the "
        f"recorded toy result")
    # Null: shuffled-label separability must NOT predict.
    rho_null = _spearman([x["f_null"] for x in pooled], eff)
    assert abs(rho_null) <= 0.5, (
        f"pooled Spearman(shuffled-label separability, "
        f"efficacy) {rho_null:.2f} -- the null predicts, so "
        f"the statistic is reading something other than class "
        f"separability")
    assert rho_sep - abs(rho_null) >= 0.4, (
        f"separability margin over its own null: {rho_sep:.2f} "
        f"vs |{rho_null:.2f}|")
