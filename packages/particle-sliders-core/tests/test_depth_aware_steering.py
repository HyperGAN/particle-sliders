"""CPU gate for the depth-distribution claim (arXiv:2611.21655):
distributing one persona vector across layers under depth weights
proportional to each layer's downstream gain beats putting the
whole matched norm at a single layer -- and past an intensity
window the steered state's usable content collapses even while
the readout shift keeps growing.

Paper: "Depth-Aware Activation Steering via Attention Residuals"
(Zhao). On Qwen3-0.6B / Qwen3.5-0.8B it distributes a persona
vector across layers under attention-residual (AttnRes) depth
weights and reports stronger internal alignment and behavioral
shift than single-layer injection (p < 0.0001); the effect is
sign-specific (reversed vectors invert the metrics, random
directions degrade), injection affects strictly downstream
processing, and there is an INTENSITY WINDOW: effects grow
monotonically over strengths 0.3-0.7, then generation collapses
(617 -> 306 -> 25 chars). Training the depth weights on a
steering objective is reported to raise held-out alignment 10x.

IN-PAPER HONESTY FLAGS (carried, not laundered): the paper's
keyword-rule metrics FAILED judge validation (Cohen's kappa
<= 0.09) and two rule-based conclusions were retracted by the
author; its cross-persona vectors retain 96-99% -- specificity
against RANDOM directions, not against semantic neighbors.
Small models, single author. This test therefore asserts only
the toy's margin structure, never the paper's point values.

Toy operationalization: a linear chain of L = 8 layers in R^12,
h_{l+1} = A_l h_l, each A_l diagonal in a per-seed random
orthonormal basis Q (conjugated back), with KNOWN per-layer
gains: the readout axis e_0 gets per-layer factors in
[0.85, 1.25] except two bottleneck layers at [0.35, 0.5]; the
tilt axis e_1 and the remaining axes get weaker factors. The
behavior readout is projection on r = Q e_0 at the TOP of the
chain (downstream of every layer). The concept vector v is
tilted off the readout axis (cos component cos(0.5) on e_0,
sin(0.5) on e_1), so injections also displace off-readout
content, which is measured, not hidden.

The per-layer DOWNSTREAM GAIN of a unit injection at layer l is
g_l = r^T (A_{L-1} ... A_l) v -- how much of an injection there
survives to the readout. This is the toy's AttnRes-style proxy:
the paper's attention-residual weights say how much each layer's
output contributes downstream; here the Jacobian product says it
exactly. Injections at layer l are alpha_l * s * v, and arms are
matched on TOTAL NORM in the only sense that stacks across
layers: the Euclidean norm of the stacked injection vector,
||alpha||_2 = 1, times the strength s. Because readout shifts
from different layers ADD linearly while the budget is
quadratic, the gain-optimal profile is alpha proportional to g
(Cauchy-Schwarz), with realized gain s * ||g||_2 -- strictly
more than any single layer's s * g_l. That coherent-addition
mechanism, not any property of attention, is what this toy can
and does test.

Arms (all at matched total norm, strength s = 1 unless swept):
  (a) single-mid:   all norm at the middle layer (the paper's
                    single-layer baseline);
  (b) best-single:  all norm at argmax_l g_l (the strongest
                    single layer ANY chooser could pick -- the
                    distribution must beat this too, or "just
                    pick the best layer" would explain it away);
  (c) uniform:      alpha_l = 1/sqrt(L) everywhere;
  (d) depth:        alpha_l proportional to g_l (the closed-form
                    optimum -- what training the weights would
                    converge to in this toy; see NOT asserted).
Controls: the REVERSED vector -v under the depth profile (must
invert the gain), and four RANDOM unit directions under the
depth profile (must not help: their downstream gains have
random signs and cancel).

Metrics per seed (>= 3 seeds):
  realized gain:  r . (h_top_steered - h_top_base), pure linear
                  chain (no clip) so arm comparisons are exact;
  off-readout:    RMS displacement orthogonal to r (reported;
                  NOT asserted minimal for the depth arm -- a
                  single mid-layer injection displaces less
                  off-readout in absolute terms, it just buys
                  far less gain with it);
  downstream:     single injection at layer 3 -- states at
                  layers <= 3 must be EXACTLY unchanged, states
                  below must move (the paper's downstream-only
                  claim, exact in a chain by construction, and
                  asserted so a future refactor that leaks
                  upstream fails loudly);
  intensity window: strength sweep s in {0.3, ..., 10} on the
                  depth arm WITH the minimal nonlinearity a
                  pure linear toy needs to express collapse:
                  a per-layer norm clip ||h|| <= 6 (finite
                  activation range). Readout gain must rise
                  strictly over the paper's window 0.3-0.7,
                  while the usable-output proxy -- 617 *
                  cos^2 of the NON-readout content against the
                  unsteered state's (the toy's generation-length
                  analogue, scaled to the paper's 617 chars) --
                  collapses to a small fraction by s = 10, as
                  the injected tilt-axis content swamps the
                  state's own content under the clip.

Measured on pop-os (particlegan 0.10.1): realized gain
single-mid 0.150-0.462, best-single 0.339-0.747, uniform
0.561-1.222, depth 0.590-1.248 (depth / single-mid 2.70-3.93x,
depth / best-single 1.43-2.11x, depth / uniform 1.02-1.13x --
the last is small because same-sign gains make uniform nearly
optimal too; the theorem only guarantees >=, and the headline
comparison is against single-layer); reversed gain exactly
-1.00x the depth gain; random-direction mean |gain|
0.146-0.275 (0.15-0.26x the depth gain); sweep gain strictly
increasing over 0.3-0.7 in every seed, length proxy at s = 10
at most 0.001x its s = 0.3 value. Assertions below encode
exactly these margins, no more.

What is NOT asserted: AttnRes itself (no attention, no softmax
weights, no transformer -- the toy tests the DISTRIBUTION claim
with Jacobian gains standing in for AttnRes weights, and a
positive result says nothing about whether AttnRes weights in
a real model track Jacobian gains); the paper's LLM numbers
(alignment scores, p < 0.0001, the exact 617 -> 306 -> 25 char
counts -- only the window STRUCTURE is claimed: gain monotone
in-window, usable output collapsed far past it); claim (4),
trained depth weights raising held-out alignment 10x, is
NOTED-NOT-TESTED: the toy's depth profile is the closed-form
optimum of the toy's own objective, so "training" it could only
re-derive the arm already tested, and the 10x figure is a
property of the paper's training setup that a linear chain
cannot express without overclaiming; specificity against
SEMANTIC NEIGHBOR vectors (the paper's own 96-99% retention
flag) -- the random control tests random directions only.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure numpy linear
algebra -- no GAN machinery, because the claim is about where
a fixed vector is injected, not about training.
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


D = 12             # state dimension
L = 8              # chain depth
MID = L // 2       # the single-layer baseline's layer
THETA = 0.5        # v's tilt off the readout axis (rad)
CLIP = 6.0         # per-layer norm clip in the window sweep
BASE_LEN = 617.0   # the paper's unsteered generation length
SWEEP = (0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0, 10.0)


def _build(seed):
    rng = np.random.default_rng(7000 + seed)
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    a = rng.uniform(0.85, 1.25, size=L)
    bott = rng.choice(L, size=2, replace=False)
    a[bott] = rng.uniform(0.35, 0.5, size=2)
    diags = np.tile(rng.uniform(0.6, 0.95, size=L)[:, None], (1, D))
    diags[:, 0] = a
    diags[:, 1] = rng.uniform(0.5, 0.9, size=L)
    mats = [q_basis @ np.diag(diags[l]) @ q_basis.T for l in range(L)]
    r = q_basis[:, 0]
    v = np.cos(THETA) * q_basis[:, 0] + np.sin(THETA) * q_basis[:, 1]
    return mats, r, v


def _gains(mats, r, v):
    """g_l = r^T (A_{L-1} ... A_l) v for each injection layer l."""
    g = np.zeros(L)
    for l in range(L):
        j = np.eye(D)
        for k in range(l, L):
            j = mats[k] @ j
        g[l] = float(r @ (j @ v))
    return g


def _run(mats, v, alphas, s, h0, clip=None):
    """States [h_0, ..., h_L]; injection alpha_l*s*v enters layer l."""
    h = h0.copy()
    states = [h.copy()]
    for l in range(len(mats)):
        h = mats[l] @ (h + s * alphas[l] * v)
        if clip is not None:
            n = np.linalg.norm(h)
            if n > clip:
                h = h * clip / n
        states.append(h.copy())
    return states


def run_seed(seed):
    mats, r, v = _build(seed)
    g = _gains(mats, r, v)
    rng = np.random.default_rng(100 + seed)
    h0 = rng.standard_normal(D) * 2.0
    base = _run(mats, v, np.zeros(L), 0.0, h0)[-1]

    def gain_of(vec, alphas, s=1.0):
        top = _run(mats, vec, alphas, s, h0)[-1]
        d = top - base
        return float(r @ d), float(np.linalg.norm(d - (r @ d) * r))

    single = np.zeros(L)
    single[MID] = 1.0
    best = np.zeros(L)
    best[int(np.argmax(g))] = 1.0
    uniform = np.ones(L) / np.sqrt(L)
    depth = g / np.linalg.norm(g)

    arms = {}
    for name, al in (("single", single), ("best", best),
                     ("uniform", uniform), ("depth", depth)):
        arms[name] = gain_of(v, al)
    rev_gain, _ = gain_of(-v, depth)
    rand_gains = []
    for _ in range(4):
        w = rng.standard_normal(D)
        w /= np.linalg.norm(w)
        rand_gains.append(gain_of(w, depth)[0])

    # Downstream-only: inject at layer 3 alone, compare every state.
    probe = np.zeros(L)
    probe[3] = 1.0
    states_b = _run(mats, v, np.zeros(L), 0.0, h0)
    states_s = _run(mats, v, probe, 1.0, h0)
    downstream = [float(np.linalg.norm(states_s[i] - states_b[i]))
                  for i in range(L + 1)]

    # Intensity window: depth arm under the norm clip.
    perp = lambda x: x - (r @ x) * r  # noqa: E731
    base_c = _run(mats, v, np.zeros(L), 0.0, h0, clip=CLIP)[-1]
    sweep = []
    for s in SWEEP:
        top = _run(mats, v, depth, s, h0, clip=CLIP)[-1]
        pt, pb = perp(top), perp(base_c)
        coh = float(pt @ pb) / (np.linalg.norm(pt) * np.linalg.norm(pb))
        sweep.append(dict(s=s, gain=float(r @ (top - base_c)),
                          length=BASE_LEN * max(coh, 0.0) ** 2))

    return dict(g=g, arms=arms, rev_gain=rev_gain,
                rand_gains=rand_gains, downstream=downstream,
                sweep=sweep)


def test_depth_weighted_beats_single_layer_at_matched_norm():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        g, arms = r["g"], r["arms"]
        depth_gain = arms["depth"][0]

        # Trap guard: the downstream gains must be genuinely
        # heterogeneous and same-sign -- if they were near-equal,
        # every profile would tie and the toy would test nothing;
        # if some were negative, the profile story changes.
        assert g.min() > 0.0, (
            f"seed {seed}: downstream gains {np.round(g, 3)} -- "
            f"a non-positive gain breaks the toy's premise")
        assert g.max() / g.min() > 1.8, (
            f"seed {seed}: gain spread {g.max() / g.min():.2f} -- "
            f"gains too homogeneous to separate the profiles")

        # Headline (paper claim 1): depth-weighted distribution
        # beats single-layer injection at matched total norm --
        # both the middle layer and the BEST single layer.
        assert depth_gain > 2.0 * arms["single"][0], (
            f"seed {seed}: depth gain {depth_gain:.3f} vs "
            f"single-mid {arms['single'][0]:.3f}")
        assert depth_gain > 1.25 * arms["best"][0], (
            f"seed {seed}: depth gain {depth_gain:.3f} vs "
            f"best-single {arms['best'][0]:.3f} -- distribution "
            f"must beat even the best layer choice")
        # Uniform distribution also beats single-layer (any
        # coherent spread does), and the gain-proportional
        # profile is at least as good as uniform -- in the linear
        # chain it is the exact optimum, so this must hold.
        assert arms["uniform"][0] > arms["single"][0], (
            f"seed {seed}: uniform gain {arms['uniform'][0]:.3f} "
            f"vs single-mid {arms['single'][0]:.3f}")
        assert depth_gain > arms["uniform"][0], (
            f"seed {seed}: depth gain {depth_gain:.3f} vs "
            f"uniform {arms['uniform'][0]:.3f}")

        # Sign-specificity (paper claim 1, controls): the reversed
        # vector inverts the gain; random directions under the
        # same profile do not help.
        assert r["rev_gain"] < -0.95 * depth_gain, (
            f"seed {seed}: reversed gain {r['rev_gain']:.3f} vs "
            f"depth gain {depth_gain:.3f} -- must invert")
        rand_mean = float(np.mean(np.abs(r["rand_gains"])))
        assert rand_mean < 0.5 * depth_gain, (
            f"seed {seed}: random mean |gain| {rand_mean:.3f} vs "
            f"depth gain {depth_gain:.3f}")
        for rg in r["rand_gains"]:
            assert abs(rg) < depth_gain, (
                f"seed {seed}: a random direction reached gain "
                f"{rg:.3f} vs depth {depth_gain:.3f}")


def test_injection_affects_strictly_downstream_layers():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        diffs = r["downstream"]
        # Injection enters at layer 3: states h_0..h_3 (indices
        # 0..3) are computed before it and must be untouched;
        # every state from h_4 down must move.
        for i in range(4):
            assert diffs[i] == 0.0, (
                f"seed {seed}: state {i} moved by {diffs[i]:.4f} "
                f"under a layer-3 injection -- upstream leak")
        for i in range(4, L + 1):
            assert diffs[i] > 0.1, (
                f"seed {seed}: state {i} moved only {diffs[i]:.4f} "
                f"under a layer-3 injection -- downstream effect "
                f"should be substantial")


def test_intensity_window_gain_rises_then_output_collapses():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        sweep = r["sweep"]
        by_s = {row["s"]: row for row in sweep}
        # In-window (paper: strengths 0.3-0.7): readout gain
        # strictly monotone increasing.
        assert by_s[0.3]["gain"] < by_s[0.5]["gain"] < by_s[0.7]["gain"], (
            f"seed {seed}: in-window gains "
            f"{[round(by_s[s]['gain'], 3) for s in (0.3, 0.5, 0.7)]} "
            f"-- must rise monotonically")
        # Past the window the usable output collapses (paper:
        # 617 -> 25 chars) even though steering is stronger.
        assert by_s[2.5]["length"] < by_s[0.7]["length"], (
            f"seed {seed}: length proxy at s=2.5 "
            f"{by_s[2.5]['length']:.0f} vs s=0.7 "
            f"{by_s[0.7]['length']:.0f} -- collapse should have begun")
        assert by_s[10.0]["length"] < 0.1 * by_s[0.3]["length"], (
            f"seed {seed}: length proxy at s=10 "
            f"{by_s[10.0]['length']:.0f} vs s=0.3 "
            f"{by_s[0.3]['length']:.0f} -- no collapse regime")
        assert by_s[10.0]["gain"] > by_s[0.7]["gain"], (
            f"seed {seed}: gain at s=10 {by_s[10.0]['gain']:.3f} "
            f"vs s=0.7 {by_s[0.7]['gain']:.3f} -- the window is "
            f"an output collapse, not a steering reversal")
