"""CPU gate for the scale-direction-bias claim (arXiv:2611.25588):
LLM judges violate semantic faithfulness -- inverting a rubric's
score anchors (criteria unchanged) changes their scores; raw
reversal disagreement confounds anchor-direction bias with the
judge's intrinsic variability, floor normalization (calibrating
reversal disagreement against the judge's own repeat disagreement)
separates them, and bidirectional anchor averaging (score under
both anchor directions, average) reduces anchor-direction
dependence without compromising alignment with human judgments.

Paper: "Rubric Semantics or Numerical Priors? Auditing
Scale-Direction Bias in LLM Judges" (Kim, Kim, Song, Lee & Park).
The authors name a requirement -- semantic faithfulness:
semantically equivalent rubrics must yield consistent scores --
and probe it by inverting a rubric's score anchors while leaving
the evaluation criteria untouched. Across five meta-evaluation
benchmarks and seven diverse LLM judges (open-weight and
proprietary), violations are widespread. Two methodological
findings matter as much as the violation itself: (a) raw reversal
disagreement is misleading, because disagreement under reversal
confounds anchor-direction bias with the judge's intrinsic
variability -- their floor normalization calibrates reversal
disagreement against the disagreement the same judge shows on
repeated identical rubrics, enabling fair cross-model comparison;
(b) bidirectional anchor averaging -- scoring under both anchor
directions and averaging -- reduces dependence on any single
anchor direction without compromising alignment with human
judgments. (The abstract reports no single headline number --
stated plainly; the claims tested are the violation's existence
after floor normalization, and the averaging fix.)

Toy operationalization: items with a latent judgment q in [0, 1]
(known ground truth; the score of record is against 10q on a
presented 0-10 scale). A synthetic judge reads q and reports

    s_f = w * (10q) + (1 - w) * mu_num + eps,     eps ~ N(0, sigma)

under forward anchors (0 = worst, 10 = best): a numeric-prior
pull toward mu_num = 7 on the NUMBER line (weight 1 - w) plus
intrinsic noise sigma. Under reversed anchors (10 = worst,
0 = best) a faithful judge reports 10 - (its judgment); the
prior-pulled judge again drifts toward mu_num on the presented
numbers,

    s_r = w * (10 - 10q) + (1 - w) * mu_num + eps',

so its reversed report mapped back, 10 - s_r, is biased the
opposite way (toward 10 - mu_num = 3). The bidirectional average
is (s_f + (10 - s_r)) / 2: the two opposite pulls average to a
pull toward 5, the centre of the latent range, and the two
independent noise draws partially cancel.

Arms: forward-only (s_f), reversed-only (10 - s_r), bidirectional
average. Metrics per judge, >= 3 seeds:
  D_rev = mean |s_f - (10 - s_r)|   (reversal disagreement)
  D_rep = mean |s_f - s_f'|         (repeat floor: two independent
                                     forward reads of the same items)
  MAE of each arm against 10q; Spearman(arm, q); signed bias
  (mean arm - 10q) per arm.

Controls:
  (i) noise-only judge: no pull (w = 1) with a larger sigma chosen
      so its RAW reversal disagreement is comparable to the pulled
      judge's -- this is the control floor normalization must
      acquit: its D_rev ~= D_rep, so the raw disagreement is
      variability, not bias;
  (ii) no-pull judge at matched sigma: D_rev ~= its floor, and
      its single arms carry no signed bias for averaging to
      cancel -- the fix must not invent bias where none exists.

Measured on pop-os (particlegan 0.10.1), 3 seeds: pulled judge
D_rev 1.603-1.616 vs floor D_rep 0.563-0.573 (excess 1.03-1.05,
ratio 2.80-2.87); noise-only judge D_rev 1.479-1.540 (comparable
raw disagreement) vs its floor 1.514-1.525 (|excess| <= 0.035 --
acquitted); no-pull matched-sigma |excess| <= 0.007. Pulled
judge MAE: forward 1.218-1.277, reversed 1.203-1.225,
bidirectional average 1.020-1.053 -- below BOTH single arms on
every seed (gain vs the better arm 0.150-0.205). Signed bias:
forward +0.79..+0.86, reversed-mapped -0.73..-0.82, average
|bias| <= 0.07 -- the opposite pulls cancel. Spearman(average, q)
0.980-0.982 >= Spearman(better single arm, q) 0.961-0.964 on
every seed -- no alignment loss. Assertions below encode exactly
these margins, no more.

One honest decomposition, recorded rather than hidden: averaging
two independent reads buys a sqrt(2) noise-variance reduction for
ANY judge (the no-pull control's MAE gain from averaging is
0.107-0.117, pure noise averaging; the noise-only judge's is
larger still because its error is all noise). The bias-specific
evidence for the paper's claim is therefore NOT the raw MAE gain
alone: it is the floor-normalized reversal excess (present only
in the pulled judge) plus the signed-bias cancellation (opposite
+/-0.8 biases averaging to ~0), with the MAE and Spearman checks
confirming the averaged score of record is also the most accurate
and no worse aligned. The no-pull control's averaging gain is
asserted to stay at the noise-only level (< 0.15) while its
floor excess stays ~0.

What is NOT asserted: any real judge's bias magnitude -- the toy's
judge is synthetic, and the claim tested is the probe / floor /
averaging structure, not the paper's benchmark numbers (the
abstract reports none). Human-judgment alignment is tested in
latent form (Spearman against the known q), the only form a
synthetic toy can express.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure CPU toy -- no GAN
machinery, because the claim is about the structure of a numeric
readout, not training dynamics.
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except ImportError:  # particlegan present but its import raises
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


N_ITEMS = 2000
MU_NUM = 7.0
W_PULL = 0.6        # pulled judge: weight on the latent judgment
SIGMA_PULL = 0.5    # pulled / no-pull judges' intrinsic noise
SIGMA_NOISE = 1.5   # noise-only judge: tuned so raw D_rev matches


def _spearman(x, y):
    """Spearman rank correlation, average ranks for ties."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # Average ranks within tie groups (rare with continuous
        # noise, but keep the statistic honest).
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = r[order[i:j + 1]].mean()
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def run_judge(seed, w, sigma):
    """One synthetic judge: forward / repeat / reversed reads."""
    rng = np.random.default_rng(42000 + seed)
    q = rng.uniform(0.0, 1.0, N_ITEMS)
    truth = 10.0 * q

    def forward():
        return np.clip(w * truth + (1.0 - w) * MU_NUM
                       + rng.normal(0.0, sigma, N_ITEMS), 0.0, 10.0)

    def reversed_raw():
        return np.clip(w * (10.0 - truth) + (1.0 - w) * MU_NUM
                       + rng.normal(0.0, sigma, N_ITEMS), 0.0, 10.0)

    s_f = forward()
    s_f2 = forward()
    back = 10.0 - reversed_raw()
    avg = 0.5 * (s_f + back)
    return dict(
        q=q, truth=truth, s_f=s_f, back=back, avg=avg,
        d_rev=float(np.mean(np.abs(s_f - back))),
        d_rep=float(np.mean(np.abs(s_f - s_f2))),
        mae_f=float(np.mean(np.abs(s_f - truth))),
        mae_r=float(np.mean(np.abs(back - truth))),
        mae_avg=float(np.mean(np.abs(avg - truth))),
        bias_f=float(np.mean(s_f - truth)),
        bias_r=float(np.mean(back - truth)),
        bias_avg=float(np.mean(avg - truth)),
        sp_f=_spearman(s_f, q),
        sp_r=_spearman(back, q),
        sp_avg=_spearman(avg, q),
    )


def test_floor_normalization_separates_bias_from_noise():
    _empf_recipe()
    for seed in (0, 1, 2):
        pulled = run_judge(seed, W_PULL, SIGMA_PULL)
        noise = run_judge(seed, 1.0, SIGMA_NOISE)
        nopull = run_judge(seed, 1.0, SIGMA_PULL)
        # The raw reversal disagreements are COMPARABLE -- a raw
        # audit cannot tell the biased judge from the noisy one.
        assert abs(noise["d_rev"] - pulled["d_rev"]) < 0.25, (
            f"seed {seed}: raw D_rev noise-only {noise['d_rev']:.3f} "
            f"vs pulled {pulled['d_rev']:.3f} -- not comparable")
        # The pulled judge's reversal disagreement exceeds its own
        # repeat floor by a clear margin: real anchor bias.
        assert pulled["d_rev"] - pulled["d_rep"] > 0.80, (
            f"seed {seed}: pulled excess "
            f"{pulled['d_rev'] - pulled['d_rep']:.3f}")
        assert pulled["d_rev"] / pulled["d_rep"] > 2.0, (
            f"seed {seed}: pulled D_rev/D_rep "
            f"{pulled['d_rev'] / pulled['d_rep']:.2f}")
        # Floor normalization ACQUITS the noise-only judge: its
        # reversal disagreement is its own variability.
        assert abs(noise["d_rev"] - noise["d_rep"]) < 0.15, (
            f"seed {seed}: noise-only excess "
            f"{noise['d_rev'] - noise['d_rep']:+.3f}")
        # And the no-pull judge at matched sigma sits at its floor.
        assert abs(nopull["d_rev"] - nopull["d_rep"]) < 0.15, (
            f"seed {seed}: no-pull excess "
            f"{nopull['d_rev'] - nopull['d_rep']:+.3f}")


def test_bidirectional_averaging_removes_direction_dependence():
    _empf_recipe()
    for seed in (0, 1, 2):
        pulled = run_judge(seed, W_PULL, SIGMA_PULL)
        nopull = run_judge(seed, 1.0, SIGMA_PULL)
        # Single-direction arms carry OPPOSITE signed biases (the
        # pull is toward mu_num on the presented numbers, whichever
        # direction the anchors point).
        assert 0.60 < pulled["bias_f"] < 1.00, (
            f"seed {seed}: forward bias {pulled['bias_f']:+.3f}")
        assert -1.00 < pulled["bias_r"] < -0.60, (
            f"seed {seed}: reversed bias {pulled['bias_r']:+.3f}")
        # The bidirectional average cancels them.
        assert abs(pulled["bias_avg"]) < 0.15, (
            f"seed {seed}: average bias {pulled['bias_avg']:+.3f}")
        # And its MAE is below BOTH single-direction arms by a
        # clear margin.
        assert pulled["mae_avg"] < pulled["mae_f"] - 0.10, (
            f"seed {seed}: avg MAE {pulled['mae_avg']:.3f} vs "
            f"forward {pulled['mae_f']:.3f}")
        assert pulled["mae_avg"] < pulled["mae_r"] - 0.10, (
            f"seed {seed}: avg MAE {pulled['mae_avg']:.3f} vs "
            f"reversed {pulled['mae_r']:.3f}")
        # No alignment loss: the average ranks items at least as
        # well as the better single arm (small tolerance).
        assert pulled["sp_avg"] >= max(
            pulled["sp_f"], pulled["sp_r"]) - 0.01, (
            f"seed {seed}: Spearman avg {pulled['sp_avg']:.3f} vs "
            f"best single {max(pulled['sp_f'], pulled['sp_r']):.3f}")
        # Control (ii): the no-pull judge has no bias to cancel --
        # its arms are unbiased already, and averaging gains only
        # the noise-averaging amount, below the pulled judge's
        # bias-driven gain.
        assert abs(nopull["bias_f"]) < 0.10, (
            f"seed {seed}: no-pull forward bias "
            f"{nopull['bias_f']:+.3f}")
        assert abs(nopull["bias_r"]) < 0.10, (
            f"seed {seed}: no-pull reversed bias "
            f"{nopull['bias_r']:+.3f}")
        gain_nopull = min(nopull["mae_f"], nopull["mae_r"]) \
            - nopull["mae_avg"]
        assert gain_nopull < 0.15, (
            f"seed {seed}: no-pull averaging gain "
            f"{gain_nopull:.3f} -- above the noise-only level")
