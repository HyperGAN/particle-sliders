"""CPU gate for the VIAS claim (arXiv:2611.03518): naive contrast
directions for different behaviors are nearly collinear, while
leave-one-out FILTERED contrast pairs produce directions with much
lower mutual overlap, and steering with a filtered direction moves its
own behavior more selectively than the naive direction does.

Paper: "VIAS: Value-Isolating Activation Steering for Multi-Value
Alignment" (Mittal et al.). Its central measurement: contrasting
activations for prompts that include all values gives per-value
directions with cosine 0.91-0.94 -- one general compliance direction
wearing different labels. VIAS instead builds, per value, matched pairs
of prompts that differ only by omitting that one value, and keeps only
pairs where the omitted value flips compliant -> non-compliant while
the other values stay compliant; the resulting directions overlap far
less (|cos| 0.09-0.45). The paper is honest that reduced overlap alone
did not yield fully independent control; this gate asserts the overlap
and selectivity structure, not independence.

Toy operationalization: synthetic scenario states in R^24 with KNOWN
latent structure. Each scenario carries a general compliance drive G
and per-value drives S_A, S_B; compliance for value X is
1[G + S_X + response noise > 0], so compliance across values is
correlated through G (measured corr 0.65-0.67 across seeds) -- that
correlation is what makes naive contrasts collapse, and the test
asserts it is present. A
response state is

    h = base + G * g + c_A * a + c_B * b + noise,

with g, a, b orthonormal ground-truth axes (general compliance,
value-A-specific, value-B-specific). Behavior readouts are the
ground-truth compliance scores s_X(h) = h . (g + x)/sqrt(2): observed
compliance for each value genuinely loads on the general factor as
well as its own, so a direction contaminated by g moves BOTH scores.

Arms:
  (a) naive:  dX = mean(h | X compliant) - mean(h | X non-compliant)
      over the all-values-prompted population;
  (b) vias:   matched pairs differing only in whether value X is in
      the prompt (omission shifts X's drive down by TAU_OMIT); keep
      pairs where X flips compliant -> non-compliant and the other
      value's compliance is unchanged; dX = mean paired difference.
      Pairing shares base and G exactly, so g cancels; the filter
      keeps only pairs where the contrast is actually expressed.
  (c) paired-unfiltered: same pairs, no filter (diagnostic arm --
      pairing alone already cancels g in expectation; the filter is
      what concentrates the estimate on expressed flips).

Metrics per seed: cos(dA_naive, dB_naive); |cos(dA_vias, dB_vias)|;
ground-truth recovery cos(dA_vias, a) vs cos(dA_naive, g); steering
selectivity S = (dOwn - |dOther|) / (dOwn + |dOther|) from steering
held-out states along each unit direction and reading the ground-truth
scores. Measured on pop-os (particlegan 0.10.1): naive cos
0.978-0.984 (at/above the paper's 0.91-0.94 band -- the toy's shared
factor dominates slightly harder), filtered |cos| 0.016-0.039 (below
the paper's 0.09-0.45 band), recovery cos 0.99, selectivity 0.05
naive vs 0.93-0.97 filtered; the paired-unfiltered diagnostic also
isolates (|cos| <= 0.07), i.e. in this toy pairing is what cancels g
and the filter concentrates the estimate on expressed flips (kept
yield ~0.22). Skips cleanly when the installed particlegan does not
provide the EMPF recipe.
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


D = 24            # state dimension
N = 4000          # scenarios per seed
SIG_G = 1.5       # spread of the general compliance drive
SIG_S = 0.5       # spread of each value-specific drive
TAU_OMIT = 1.2    # drive penalty when a value is omitted from the prompt
EPS = 0.35        # per-response compliance noise
BASE_SD = 0.7     # per-scenario base state spread
NOISE_SD = 0.5    # per-response state noise
LAM = 1.0         # steering strength


def _unit(x):
    return x / np.linalg.norm(x)


def _cos(x, y):
    return float(_unit(x) @ _unit(y))


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    # Orthonormal ground-truth axes: general compliance, A-specific, B-specific.
    q, _ = np.linalg.qr(rng.standard_normal((D, 3)))
    g, a, b = q[:, 0], q[:, 1], q[:, 2]
    w_a = _unit(g + a)   # ground-truth readout direction for behavior A
    w_b = _unit(g + b)   # ground-truth readout direction for behavior B

    base = rng.standard_normal((N, D)) * BASE_SD
    drive_g = rng.standard_normal(N) * SIG_G
    drive_a = rng.standard_normal(N) * SIG_S
    drive_b = rng.standard_normal(N) * SIG_S

    def respond(omit=None):
        """One response per scenario; `omit` drops that value's prompt
        inclusion, shifting its compliance drive down by TAU_OMIT."""
        ea = rng.standard_normal(N) * EPS
        eb = rng.standard_normal(N) * EPS
        da = drive_g + drive_a + ea - (TAU_OMIT if omit == "A" else 0.0)
        db = drive_g + drive_b + eb - (TAU_OMIT if omit == "B" else 0.0)
        ca, cb = da > 0, db > 0
        h = (base + drive_g[:, None] * g[None, :]
             + ca[:, None] * a[None, :] + cb[:, None] * b[None, :]
             + rng.standard_normal((N, D)) * NOISE_SD)
        return h, ca, cb

    h_all, ca_all, cb_all = respond()            # all values prompted
    h_no_a, ca_no_a, cb_no_a = respond(omit="A")  # A omitted
    h_no_b, ca_no_b, cb_no_b = respond(omit="B")  # B omitted

    corr_ab = float(np.corrcoef(ca_all, cb_all)[0, 1])

    # (a) naive population contrasts.
    d_a_naive = h_all[ca_all].mean(0) - h_all[~ca_all].mean(0)
    d_b_naive = h_all[cb_all].mean(0) - h_all[~cb_all].mean(0)

    # (b) VIAS filtered matched pairs.
    keep_a = ca_all & ~ca_no_a & (cb_all == cb_no_a)
    keep_b = cb_all & ~cb_no_b & (ca_all == ca_no_b)
    d_a_vias = (h_all[keep_a] - h_no_a[keep_a]).mean(0)
    d_b_vias = (h_all[keep_b] - h_no_b[keep_b]).mean(0)

    # (c) paired but unfiltered (diagnostic).
    d_a_unf = (h_all - h_no_a).mean(0)
    d_b_unf = (h_all - h_no_b).mean(0)

    # Steering selectivity on held-out scenarios (fresh draws, same world).
    h_held, _, _ = respond()

    def selectivity(direction, own_w, other_w):
        d = _unit(direction)
        d_own = float(((h_held + LAM * d) @ own_w).mean()
                      - (h_held @ own_w).mean())
        d_other = float(((h_held + LAM * d) @ other_w).mean()
                        - (h_held @ other_w).mean())
        if d_own <= 0:
            return -1.0, d_own, d_other
        return (d_own - abs(d_other)) / (d_own + abs(d_other)), d_own, d_other

    sel_a_naive, _, _ = selectivity(d_a_naive, w_a, w_b)
    sel_b_naive, _, _ = selectivity(d_b_naive, w_b, w_a)
    sel_a_vias, _, _ = selectivity(d_a_vias, w_a, w_b)
    sel_b_vias, _, _ = selectivity(d_b_vias, w_b, w_a)

    return dict(
        corr_ab=corr_ab,
        yield_a=float(keep_a.mean()), yield_b=float(keep_b.mean()),
        cos_naive=_cos(d_a_naive, d_b_naive),
        cos_vias=_cos(d_a_vias, d_b_vias),
        cos_unf=_cos(d_a_unf, d_b_unf),
        cos_vias_a=_cos(d_a_vias, a), cos_vias_b=_cos(d_b_vias, b),
        cos_naive_g_a=_cos(d_a_naive, g), cos_naive_a=_cos(d_a_naive, a),
        sel_naive=float(np.mean([sel_a_naive, sel_b_naive])),
        sel_vias=float(np.mean([sel_a_vias, sel_b_vias])),
    )


def test_vias_filtered_contrasts_isolate_directions():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Trap guard: compliance across values must actually be
        # correlated through the shared drive, and the filter must
        # keep a nontrivial slice -- otherwise the toy tests nothing.
        assert r["corr_ab"] > 0.5, (
            f"seed {seed}: compliance corr {r['corr_ab']:.3f} -- shared "
            "factor too weak, naive collapse cannot occur")
        assert 0.05 < r["yield_a"] < 0.6 and 0.05 < r["yield_b"] < 0.6, (
            f"seed {seed}: filter yields {r['yield_a']:.3f}/{r['yield_b']:.3f}")
        # The paper's headline: naive per-value directions collapse
        # onto one general direction (paper: cos 0.91-0.94).
        assert r["cos_naive"] >= 0.8, (
            f"seed {seed}: naive cos {r['cos_naive']:.3f}")
        assert r["cos_naive_g_a"] > r["cos_naive_a"], (
            f"seed {seed}: naive dA loads g {r['cos_naive_g_a']:.3f} vs "
            f"a {r['cos_naive_a']:.3f} -- naive direction is not the "
            "general direction it should be")
        # VIAS filtered pairs: overlap collapses (paper: |cos| 0.09-0.45)
        # and the directions recover the true value-specific axes.
        assert abs(r["cos_vias"]) <= 0.5, (
            f"seed {seed}: vias |cos| {abs(r['cos_vias']):.3f} "
            f"(naive {r['cos_naive']:.3f}, unfiltered {r['cos_unf']:.3f})")
        assert abs(r["cos_vias"]) <= r["cos_naive"] - 0.4, (
            f"seed {seed}: vias |cos| {abs(r['cos_vias']):.3f} vs "
            f"naive {r['cos_naive']:.3f}")
        assert r["cos_vias_a"] > 0.8 and r["cos_vias_b"] > 0.8, (
            f"seed {seed}: vias recovery cos to true axes "
            f"{r['cos_vias_a']:.3f}/{r['cos_vias_b']:.3f}")
        # Steering: the filtered direction moves its own behavior far
        # more selectively than the naive direction does.
        assert r["sel_vias"] > 0.6, (
            f"seed {seed}: vias selectivity {r['sel_vias']:.3f} vs "
            f"naive {r['sel_naive']:.3f}")
        assert r["sel_naive"] < 0.35, (
            f"seed {seed}: naive selectivity {r['sel_naive']:.3f} -- "
            "naive steering should drag both behaviors together")
        assert r["sel_vias"] > r["sel_naive"] + 0.4, (
            f"seed {seed}: selectivity vias {r['sel_vias']:.3f} vs "
            f"naive {r['sel_naive']:.3f}")
