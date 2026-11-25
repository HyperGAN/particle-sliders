"""CPU gate for the HUB-J claim (arXiv:2611.33065): "Auditing the
Judge: Human-Derived Bias Discovery, Quantification, and Mitigation
in LLM Judges" (HUB-J). Provenance: cycle 28 of the Particle Sliders
paper watch; judge-scoring lineage PR #260 (scale-direction bias,
bidirectional averaging), PR #295 (judge-score channel), PR #219
(bundling).

Paper's claims, per its abstract: LLM-judge biases are usually
handled from pre-discovered bias lists, without strength
quantification, via costly interventions. HUB-J instead
  (1) DISCOVERS interpretable bias factors automatically from
      human-LLM judgement DISAGREEMENT cases;
  (2) QUANTIFIES each factor's strength via controlled input
      modifications and the resulting shifts in model decisions
      (agreement cases anchor the quantification);
  (3) MITIGATES with a lightweight training-free regression that
      estimates each bias factor's influence on judgements and
      corrects decisions by removing the estimated effect --
      reducing bias-driven decision flips while generalizing
      across models.
(The abstract reports no single headline number -- stated plainly;
the claims tested are the three-stage structure.)

Toy operationalization: items with known true quality
q ~ U(2, 8); decision = pass iff score >= 5. Six CANDIDATE factors
are measured for every item; three are PLANTED in the judge, three
are decoys the discovery procedure must reject -- the procedure is
never told which are active:
  length   f ~ N(0, 1)          planted beta = +1.10 (verbosity)
  style    f ~ N(0, 1)          planted beta = +0.75 (surface style)
  order    f in {-1, +1}        planted beta = +0.45 (position)
  decoy_topic / decoy_punct / decoy_sentiment  f ~ N(0, 1), beta = 0
All factors are independent of q and of each other by construction.

Judge (the "LLM"):  s_j = q + sum_f beta_f * f + eps, eps ~ N(0, 0.5)
Reference ("human"): s_h = q + eps', eps' ~ N(0, 0.5), on all items.

Stage (i) DISCOVERY: using only decisions, compute per candidate
  D_dis(k)  = mean(f_k | judge passes, ref fails)
              - mean(f_k | judge fails, ref passes)
  D_agree(k)= the same pass/fail contrast inside AGREEMENT cases,
and the discovery statistic C(k) = D_dis(k) - D_agree(k). Agreement
cases are not inert (a passing agreement also selects positive
bias), so the contrast -- disagreement structure BEYOND the
agreement baseline -- is the paper-faithful quantity. Claim:
ranking candidates by C recovers exactly the planted set, in
planted |beta| order, with decoys near zero.

Stage (ii) QUANTIFICATION: controlled modification -- add
delta = +1 to ONE factor, holding q, all other factors, and the
noise draw fixed (paired). Readouts: mean score shift per unit
delta, and the decision-flip rate. Claim: both track the planted
betas (ordering and, for the shift, rough magnitude).

Stage (iii) CORRECTION: OLS of judge scores on candidate factor
values on a calibration half (no access to q; q is independent of
the factors, so the factor coefficients identify the betas),
corrected score = s_j - sum beta_hat_f * f on the held-out half.
Claim: bias-driven decision flips (vs the true-q decision) fall
toward the reference judge's noise-floor flip rate, and accuracy
against q (MAE) is preserved -- in fact restored to the reference
level, not traded away.

Rival corrections (controls):
  (iv-a) #260-style bidirectional averaging: re-score each item
      with the order factor sign-flipped (the item presented in
      the opposite position; length/style are item-intrinsic and
      unchanged) and average. This must cancel the ORDER loading
      only: a residual regression of (arm - q) on the factors
      shows order ~ 0 while length/style loadings survive, and
      its flips/MAE stay far above the full correction's.
  (iv-b) shuffled correction: the stage-(iii) regression fitted
      with factor values shuffled across items. Its coefficients
      are ~ 0 and applying it must NOT help.

Measured on pop-os (particlegan 0.10.1), 5 seeds, N = 5000:
  discovery contrast C: length 1.20-1.40, style 0.81-0.87,
    order 0.44-0.60, max |decoy| <= 0.13 -- top-3 = planted set,
    in order, on every seed.
  quantification: score shift per unit delta equals the planted
    beta (paired-noise linear toy: exact); flip rates length
    0.156-0.173, style 0.107-0.121, order 0.062-0.075 (ordering
    and rough beta ratios preserved).
  correction: beta_hat within 0.08 of planted on every seed;
    test-half MAE biased 1.15-1.19 -> corrected 0.40-0.41
    (reference 0.39-0.40); flips vs true decision biased
    0.181-0.201 -> corrected 0.056-0.071 (reference 0.058-0.073,
    corrected within 0.011 of reference on every seed).
  rivals: bidirectional MAE 1.08-1.11, flips 0.172-0.190 (order
    residual loading |.| <= 0.007, length/style residuals
    1.10 / 0.73-0.76 intact); shuffled MAE 1.16-1.19, flips
    0.188-0.200 (no help).
Assertions below encode exactly these margins, no more.

What is NOT asserted: that real judges have exactly three
additive, q-independent bias factors (here independence is what
makes the regression identify the betas; correlated factors are
the real audit's hard case and this toy does not arbitrate it);
the paired-noise score shift being exact is a modeling gift of
the linear toy, stated plainly -- the nontrivial quantification
readout is the flip rate; anything about the paper's real-model
generalization claim beyond the structure (one synthetic judge
family, five seeds). Pure numpy -- no GAN machinery, because the
claim is about the audit structure, not training dynamics.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI).
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


N_ITEMS = 5000
THRESH = 5.0
SIGMA = 0.5
NAMES = ["length", "style", "order",
         "decoy_topic", "decoy_punct", "decoy_sentiment"]
BETAS = np.array([1.10, 0.75, 0.45, 0.0, 0.0, 0.0])
PLANTED = [0, 1, 2]  # indices into NAMES, in planted |beta| order
SEEDS = (0, 1, 2, 3, 4)


def _gen(rng, n):
    q = rng.uniform(2.0, 8.0, n)
    X = np.empty((n, len(NAMES)))
    X[:, 0] = rng.normal(0.0, 1.0, n)            # length
    X[:, 1] = rng.normal(0.0, 1.0, n)            # style
    X[:, 2] = rng.choice([-1.0, 1.0], size=n)    # order / position
    X[:, 3] = rng.normal(0.0, 1.0, n)            # decoys
    X[:, 4] = rng.normal(0.0, 1.0, n)
    X[:, 5] = rng.normal(0.0, 1.0, n)
    return q, X


def _ols_coef(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[1:]


def run_seed(seed):
    rng = np.random.default_rng(42000 + seed)
    q, X = _gen(rng, N_ITEMS)
    bias = X @ BETAS
    s_j = q + bias + rng.normal(0.0, SIGMA, N_ITEMS)
    s_h = q + rng.normal(0.0, SIGMA, N_ITEMS)
    d_j, d_h, d_t = s_j >= THRESH, s_h >= THRESH, q >= THRESH

    # --- (i) discovery: disagreement contrast vs agreement anchor
    dis = d_j != d_h
    up, down = dis & d_j, dis & (~d_j)
    ag = ~dis
    agp, agf = ag & d_j, ag & (~d_j)
    d_dis = np.array([X[up, k].mean() - X[down, k].mean()
                      for k in range(len(NAMES))])
    d_agree = np.array([X[agp, k].mean() - X[agf, k].mean()
                        for k in range(len(NAMES))])
    contrast = d_dis - d_agree

    # --- (ii) quantification: paired controlled modification
    noise = rng.normal(0.0, SIGMA, N_ITEMS)
    s_base = q + bias + noise
    quant = {}
    for k in PLANTED:
        X2 = X.copy()
        X2[:, k] += 1.0
        s_mod = q + X2 @ BETAS + noise
        quant[k] = dict(
            shift=float(np.mean(s_mod - s_base)),
            flip=float(np.mean((s_mod >= THRESH) != (s_base >= THRESH))),
        )

    # --- (iii) correction: OLS on calibration half, eval on test half
    h = N_ITEMS // 2
    beta_hat = _ols_coef(s_j[:h], X[:h])
    Xs = X[:h].copy()
    for k in range(len(NAMES)):
        rng.shuffle(Xs[:, k])
    beta_shuf = _ols_coef(s_j[:h], Xs)
    Xt, qt = X[h:], q[h:]
    sjt, sht = s_j[h:], s_h[h:]
    corr = sjt - Xt @ beta_hat
    shuf = sjt - Xt @ beta_shuf
    # (iv-a) bidirectional: order sign flipped, fresh noise, averaged
    bias_rev = bias[h:] - 2.0 * BETAS[2] * Xt[:, 2]
    s_rev = qt + bias_rev + rng.normal(0.0, SIGMA, len(qt))
    bidir = 0.5 * (sjt + s_rev)

    def mae(s):
        return float(np.mean(np.abs(s - qt)))

    def flips(s):
        return float(np.mean((s >= THRESH) != (qt >= THRESH)))

    # Residual loadings of each corrected arm: regress (arm - q) on X
    resid = {name: _ols_coef(arm - qt, Xt) for name, arm in
             [("corr", corr), ("bidir", bidir), ("shuf", shuf)]}
    return dict(
        contrast=contrast, d_dis=d_dis, quant=quant,
        beta_hat=beta_hat, beta_shuf=beta_shuf, resid=resid,
        mae=dict(biased=mae(sjt), ref=mae(sht), corr=mae(corr),
                 bidir=mae(bidir), shuf=mae(shuf)),
        flips=dict(biased=flips(sjt), ref=flips(sht), corr=flips(corr),
                   bidir=flips(bidir), shuf=flips(shuf)),
    )


def test_discovery_from_disagreement_cases():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        C = r["contrast"]
        ranked = list(np.argsort(-np.abs(C)))
        # Exactly the planted set is recovered, in |beta| order.
        assert ranked[:3] == PLANTED, (
            f"seed {seed}: discovery ranking "
            f"{[NAMES[i] for i in ranked]} from C={np.round(C, 3)}")
        # The weakest planted factor still clears the decoys by a
        # clear margin, and decoys sit near zero.
        assert C[2] > 0.35, f"seed {seed}: order contrast {C[2]:.3f}"
        assert np.max(np.abs(C[3:])) < 0.20, (
            f"seed {seed}: decoy contrasts {np.round(C[3:], 3)}")
        assert C[2] - np.max(np.abs(C[3:])) > 0.25, (
            f"seed {seed}: planted/decoy separation "
            f"{C[2] - np.max(np.abs(C[3:])):.3f}")
        # Ordering inside the planted set follows |beta|.
        assert C[0] > C[1] > C[2], (
            f"seed {seed}: planted contrasts {np.round(C[:3], 3)}")


def test_quantification_by_controlled_modification():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        shifts = [r["quant"][k]["shift"] for k in PLANTED]
        flips = [r["quant"][k]["flip"] for k in PLANTED]
        # Score shift per unit delta tracks planted magnitude
        # (exact in this paired linear toy -- see docstring).
        for k, s in zip(PLANTED, shifts):
            assert abs(s - BETAS[k]) < 0.02, (
                f"seed {seed}: {NAMES[k]} shift {s:.3f} "
                f"vs planted {BETAS[k]:.2f}")
        # Decision-flip rates track the planted ordering, in the
        # measured bands.
        assert flips[0] > flips[1] > flips[2], (
            f"seed {seed}: flip rates {np.round(flips, 3)}")
        assert 0.14 < flips[0] < 0.20, f"seed {seed}: length flip {flips[0]:.3f}"
        assert 0.09 < flips[1] < 0.15, f"seed {seed}: style flip {flips[1]:.3f}"
        assert 0.05 < flips[2] < 0.10, f"seed {seed}: order flip {flips[2]:.3f}"


def test_regression_correction_removes_bias_preserves_accuracy():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # The regression recovers the planted strengths (and ~0
        # for decoys) without ever seeing q.
        assert np.max(np.abs(r["beta_hat"] - BETAS)) < 0.15, (
            f"seed {seed}: beta_hat {np.round(r['beta_hat'], 3)}")
        # Bias-driven flips fall toward the reference noise floor.
        assert r["flips"]["corr"] < r["flips"]["biased"] - 0.09, (
            f"seed {seed}: flips corr {r['flips']['corr']:.3f} vs "
            f"biased {r['flips']['biased']:.3f}")
        assert abs(r["flips"]["corr"] - r["flips"]["ref"]) < 0.02, (
            f"seed {seed}: flips corr {r['flips']['corr']:.3f} vs "
            f"ref {r['flips']['ref']:.3f}")
        # Accuracy against true q is restored to reference level,
        # not traded away.
        assert r["mae"]["corr"] < r["mae"]["biased"] - 0.60, (
            f"seed {seed}: MAE corr {r['mae']['corr']:.3f} vs "
            f"biased {r['mae']['biased']:.3f}")
        assert r["mae"]["corr"] < r["mae"]["ref"] + 0.05, (
            f"seed {seed}: MAE corr {r['mae']['corr']:.3f} vs "
            f"ref {r['mae']['ref']:.3f}")
        # Residual loadings after correction: nothing left.
        assert np.max(np.abs(r["resid"]["corr"])) < 0.15, (
            f"seed {seed}: corrected residual loadings "
            f"{np.round(r['resid']['corr'], 3)}")


def test_rival_corrections_asymmetry_and_shuffled_null():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        # (iv-a) Bidirectional averaging cancels ONLY the order
        # loading: its residual length/style loadings survive.
        rb = r["resid"]["bidir"]
        assert abs(rb[2]) < 0.15, (
            f"seed {seed}: bidir order residual {rb[2]:+.3f}")
        assert rb[0] > 0.80 and rb[1] > 0.50, (
            f"seed {seed}: bidir surviving loadings "
            f"length {rb[0]:+.3f} style {rb[1]:+.3f}")
        # ...so its flips/MAE improve only slightly and stay far
        # above the full regression correction.
        assert r["flips"]["bidir"] > r["flips"]["corr"] + 0.08, (
            f"seed {seed}: bidir flips {r['flips']['bidir']:.3f} vs "
            f"corr {r['flips']['corr']:.3f}")
        assert r["mae"]["bidir"] > r["mae"]["corr"] + 0.50, (
            f"seed {seed}: bidir MAE {r['mae']['bidir']:.3f} vs "
            f"corr {r['mae']['corr']:.3f}")
        # (iv-b) Shuffled-fit correction must not help.
        assert np.max(np.abs(r["beta_shuf"])) < 0.15, (
            f"seed {seed}: shuffled beta {np.round(r['beta_shuf'], 3)}")
        assert r["mae"]["shuf"] > r["mae"]["biased"] - 0.10, (
            f"seed {seed}: shuffled MAE {r['mae']['shuf']:.3f} vs "
            f"biased {r['mae']['biased']:.3f}")
        assert r["flips"]["shuf"] > r["flips"]["biased"] - 0.03, (
            f"seed {seed}: shuffled flips {r['flips']['shuf']:.3f} vs "
            f"biased {r['flips']['biased']:.3f}")
