"""CPU gate for the gating-vs-rewiring claim of Pandere
et al., "Rewired or Gated? How Instruction Tuning Shapes
Knowledge-Conflict Circuits in LLMs" (arXiv:2609.25602):
across three model families and five independent methods
(node and edge attribution, superposition role analysis,
causal ablation, path patching), instruction tuning does
NOT rewire the knowledge-conflict circuit -- it
gates/reweights already-present components, with the same
heads in the same late layers and a high node overlap
(0.60-0.82) between base and instruct active sets. The
paper's corollary: because the circuit is preserved,
interpretability and control tools calibrated on base
models should transfer to their tuned siblings.

Provenance: cycle 16 of the Particle Sliders paper watch,
in the pathway-coherence chain from PR #278 (cycle 15,
teacher-pool dilution): #278 showed that a second teacher
with the same trait but a DIFFERENT pathway dilutes like
~44% clean data -- pathway identity, not just trait
content, decides what tuning installs. This test asks the
dual question from the model's side: when fine-tuning
changes behavior, does the change live in the gains on
existing pathways (gating) or in the pathways themselves
(rewiring)?

Toy operationalization: a gated mixture-of-pathways
model. K = 12 pathways phi_k(x) = tanh(w_k . x + b_k)
(internals w_k, b_k trainable) combined by trainable
gains g_k; output = sum_k g_k phi_k(x), inputs x in R^6.
The "base" model is a fixed parameter draw (per seed)
whose gains are supported on 5 of the 12 pathways
(g = 3.0/2.5/2.0/1.5/1.0); its behavior is carried by
that identifiable active set. Fine-tuning is minibatch
SGD (batch 256, lr 0.1, at most 150 epochs, early stop
at relative MSE 0.003) on ALL parameters, in two arms:

  (1) REWEIGHTABLE arm: the target is the SAME pathways
      with the support gains exactly reversed
      (1.0/1.5/2.0/2.5/3.0) -- a maximal reweighting that
      lies in the span of the existing pathways by
      construction.
  (2) ABSENT-COMPONENT arm: the target is the base
      function PLUS a localized Gaussian bump along a
      fresh direction v (exp(-((v.x - 0.5)^2)/0.15),
      centered, scaled to 1/3 of the target variance) --
      a feature no existing pathway computes: broad
      tanh units cannot make a narrow bump without
      moving their internals.

Each arm also runs with internals FROZEN (gains only):
the frozen arm must fit arm 1 and fail arm 2, which is
what certifies the arms measure what they claim.

Readouts per arm: (i) achieved fit (relative MSE);
(ii) the share of total parameter-change energy
(sum of squared deltas) carried by the gains vs the
pathway internals (w and b); (iii) overlap between base
and tuned active sets, two ways -- cosine of the
contribution vectors c_k = |g_k| * RMS(phi_k), and
cosine of causal-ablation importance vectors (MSE rise
when component k is zeroed, the paper's own ablation
method) evaluated on the arm's target task.

Measured on pop-os (particlegan 0.10.1), 3 seeds --
arm 1 first (full fit / frozen fit / gain-energy share
/ contribution cosine / ablation cosine), then arm 2:
  seed 0: .0026/.0024/.86/.92/.92 | .0246/.2555/.34/.80/.66
  seed 1: .0026/.0028/.88/.84/.73 | .0029/.2669/.46/.83/.65
  seed 2: .0030/.0022/.92/.82/.92 | .0026/.2843/.42/.67/.36

What this proves, stated plainly: when the target is
reachable by reweighting, gradient tuning IS gating --
the fit is exact with internals frozen, the full run's
change energy is 86-92% gains, and the active-set
overlap sits at/above the top of the paper's 0.60-0.82
band (contribution cosine 0.82-0.92, ablation cosine
0.73-0.92). When the target needs a component the base
does not have, gains alone fail hard (frozen relative
MSE 0.26-0.28, i.e. essentially the bump's whole 1/3
variance share unexplained) and the full run's change
energy flips to internals-dominated (gain share
0.34-0.46, a 0.42-0.52 gap below arm 1 on every seed).

The narrowing, also stated plainly: the paper's
overlap corollary does NOT show whole-task node
turnover in arm 2 in this toy. Because the new behavior
is ADDED on top of a preserved base function, the
tuned model's whole-task contribution cosine against
base stays 0.67-0.83 -- new structure is built
ALONGSIDE the preserved circuit, not instead of it.
The rewiring signature lives in the change-energy
composition (and, on the heaviest-recruitment seed, in
the ablation cosine, 0.36); whole-task overlap only
falls to 0.65-0.66 on the other seeds. Assertions below
encode exactly these margins, no more: arm 2 asserts
the energy flip, the frozen failure, and ablation
cosine <= 0.70 -- and, as the narrowing itself,
contribution cosine >= 0.60 (the old circuit visibly
persists even when rewiring is forced).

What is NOT asserted: overlap landing inside the
0.60-0.82 band exactly in arm 1 (the toy sits at/above
the band top -- a 5-component support with exactly
reversed gains is a cleaner reweighting than real
instruction tuning); anything about real LM attention
heads, layers, or conflict behavior; and overlap
turnover as the general signature of forced rewiring
(this toy refutes it for additive targets -- see the
narrowing above). Two modeling gifts, stated plainly:
the base model's active set is constructed, not
learned (identifiability is the point of the toy), and
both targets are noiseless functions evaluated on the
training distribution itself, so "fit" means function
approximation, not generalization. Pure numpy -- no GAN
machinery, because the claim is about where
fine-tuning's change lives, not about training
dynamics. Skips cleanly when the installed particlegan
does not provide the EMPF recipe (the house guard for
this paper-test family; PR #211 is what enables this
test in CI).
"""
import numpy as np
import pytest

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


D = 6                # input dimension
K = 12               # pathways
N = 4000             # training inputs per seed
SUPPORT = [0, 1, 2, 3, 4]          # base active set
G_BASE = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
G_TGT1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])  # reversed
BUMP_SHARE = 1.0 / 3.0   # bump share of arm-2 target variance
LR = 0.1             # minibatch SGD step
EPOCHS = 150         # cap; early stop at TOL
BS = 256             # minibatch size
TOL = 0.003          # early-stop relative MSE


def _phi(W, b, X):
    return np.tanh(X @ W.T + b)


def _mse(W, b, g, X, y):
    r = _phi(W, b, X) @ g - y
    return float(np.mean(r ** 2))


def build(seed):
    """Base model, inputs, and arm targets for one seed."""
    rng = np.random.default_rng(41000 + seed)
    W = rng.standard_normal((K, D)) * 1.2
    b = rng.standard_normal(K) * 0.5
    X = rng.standard_normal((N, D))
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v)
    g_base = np.zeros(K)
    g_base[SUPPORT] = G_BASE
    y_base = _phi(W, b, X) @ g_base
    g_tgt1 = np.zeros(K)
    g_tgt1[SUPPORT] = G_TGT1
    y1 = _phi(W, b, X) @ g_tgt1
    bump = np.exp(-((X @ v) - 0.5) ** 2 / 0.15)
    bump = bump - bump.mean()
    c_bump = np.sqrt(BUMP_SHARE / (1.0 - BUMP_SHARE)
                     * np.var(y_base) / np.var(bump))
    y2 = y_base + c_bump * bump
    return dict(W=W, b=b, X=X, g_base=g_base,
                y_base=y_base, y1=y1, y2=y2)


def _train(d, seed, y, freeze_w=False):
    """Minibatch SGD on all parameters (or gains only).

    Every arm of a seed uses the same training RNG seed,
    so arms see identical minibatch orders and differ
    only by target or by the freeze.
    """
    rng = np.random.default_rng(91000 + seed)
    X = d["X"]
    var_y = np.var(y)
    W = d["W"].copy()
    b = d["b"].copy()
    g = d["g_base"].copy()
    for _ in range(EPOCHS):
        if _mse(W, b, g, X, y) / var_y < TOL:
            break
        order = rng.permutation(N)
        for st in range(0, N, BS):
            idx = order[st:st + BS]
            Xb = X[idx]
            Ph = _phi(W, b, Xb)
            r = Ph @ g - y[idx]
            m = len(idx)
            g = g - LR * (2.0 / m) * (Ph.T @ r)
            if not freeze_w:
                coeff = r[:, None] * g[None, :] * (1 - Ph ** 2)
                W = W - LR * (2.0 / m) * (coeff.T @ Xb)
                b = b - LR * (2.0 / m) * coeff.sum(axis=0)
    dE_g = float(np.sum((g - d["g_base"]) ** 2))
    dE_w = float(np.sum((W - d["W"]) ** 2)
                 + np.sum((b - d["b"]) ** 2))
    share = dE_g / (dE_g + dE_w) if dE_g + dE_w > 0 else 0.0
    return dict(relmse=_mse(W, b, g, X, y) / var_y,
                share=share, g=g, W=W, b=b)


def _contrib(g, W, b, X):
    Ph = _phi(W, b, X)
    return np.abs(g) * np.sqrt(np.mean(Ph ** 2, axis=0))


def _ablation_imp(g, W, b, X, y):
    base = _mse(W, b, g, X, y)
    imp = np.zeros(K)
    for k in range(K):
        g2 = g.copy()
        g2[k] = 0.0
        imp[k] = max(_mse(W, b, g2, X, y) - base, 0.0)
    return imp


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_seed(seed):
    d = build(seed)
    X = d["X"]
    c0 = _contrib(d["g_base"], d["W"], d["b"], X)
    i0 = {name: _ablation_imp(d["g_base"], d["W"], d["b"], X,
                              d[name])
          for name in ("y1", "y2")}
    out = {}
    for arm, yname in (("arm1", "y1"), ("arm2", "y2")):
        y = d[yname]
        full = _train(d, seed, y)
        frozen = _train(d, seed, y, freeze_w=True)
        out[arm] = dict(
            full=full["relmse"], frozen=frozen["relmse"],
            share=full["share"],
            ccos=_cos(c0, _contrib(full["g"], full["W"],
                                   full["b"], X)),
            acos=_cos(i0[yname],
                      _ablation_imp(full["g"], full["W"],
                                    full["b"], X, y)),
        )
    return out


def test_reweightable_target_is_gated_not_rewired():
    _empf_recipe()
    for seed in (0, 1, 2):
        a = run_seed(seed)["arm1"]
        # The reweightable target is fit exactly -- and
        # gains alone (internals frozen) already fit it,
        # which certifies the target is in the span.
        assert a["full"] <= 0.01, (
            f"seed {seed}: arm-1 full fit {a['full']:.4f}")
        assert a["frozen"] <= 0.01, (
            f"seed {seed}: arm-1 frozen fit {a['frozen']:.4f}"
            " -- target not gain-reachable")
        # The change is gain-dominated: 86-92% of the
        # parameter-change energy sits in the gains.
        assert a["share"] >= 0.75, (
            f"seed {seed}: arm-1 gain-energy share "
            f"{a['share']:.2f}")
        # The active set is preserved at/above the top
        # of the paper's 0.60-0.82 overlap band, on both
        # the contribution and the ablation readout.
        assert a["ccos"] >= 0.75, (
            f"seed {seed}: arm-1 contribution cosine "
            f"{a['ccos']:.2f}")
        assert a["acos"] >= 0.65, (
            f"seed {seed}: arm-1 ablation cosine "
            f"{a['acos']:.2f}")


def test_absent_component_forces_internals_change():
    _empf_recipe()
    acos_gap = []
    for seed in (0, 1, 2):
        out = run_seed(seed)
        a1, a2 = out["arm1"], out["arm2"]
        # Gains alone cannot fit the absent-component
        # target: the frozen arm leaves essentially the
        # bump's whole variance share unexplained, while
        # the full run fits.
        assert a2["frozen"] >= 0.20, (
            f"seed {seed}: arm-2 frozen fit "
            f"{a2['frozen']:.4f} -- bump was gain-reachable")
        assert a2["full"] <= 0.05, (
            f"seed {seed}: arm-2 full fit {a2['full']:.4f}")
        assert a2["frozen"] >= 8.0 * a2["full"], (
            f"seed {seed}: arm-2 frozen/full fit ratio "
            f"{a2['frozen'] / a2['full']:.1f}")
        # The change energy flips to the internals:
        # gain share 0.34-0.46, a 0.42-0.52 gap below
        # the reweightable arm on every seed.
        assert a2["share"] <= 0.55, (
            f"seed {seed}: arm-2 gain-energy share "
            f"{a2['share']:.2f}")
        assert a1["share"] - a2["share"] >= 0.35, (
            f"seed {seed}: gain-share gap "
            f"{a1['share'] - a2['share']:.2f}")
        # Ablation overlap drops out of the arm-1 range.
        assert a2["acos"] <= 0.70, (
            f"seed {seed}: arm-2 ablation cosine "
            f"{a2['acos']:.2f}")
        acos_gap.append(a1["acos"] - a2["acos"])
        # The narrowing, asserted as measured: the old
        # circuit is NOT dismantled -- whole-task
        # contribution overlap stays high even when
        # rewiring is forced, because the new structure
        # is added alongside the preserved base.
        assert a2["ccos"] >= 0.60, (
            f"seed {seed}: arm-2 contribution cosine "
            f"{a2['ccos']:.2f} -- base circuit not "
            "preserved alongside the new component")
    assert float(np.mean(acos_gap)) >= 0.15, (
        f"mean ablation-cosine gap {np.mean(acos_gap):.2f}")
