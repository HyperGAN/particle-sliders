"""CPU gate for the EDITORS claim (arXiv:2611.20846): an adapter
trained under activation-space deflation against a diagnostically
identified nuisance subspace keeps the target behavior gain while
leaking far less into the nuisance axis than plain training -- and
it is the DIAGNOSTIC subspace specifically that does this.

Paper: "EDITORS Know Your Style! Editing LoRA Subspaces for
Stylistic Attribution and Imitation" (Wang, Lau, Tian, Fan, Chaux &
Low). EDITORS is a diagnose-then-deflate framework for style-content
disentanglement: (1) train a diagnostic LoRA adapter on style-neutral
statements derived from the corpus to identify a content-oriented
subspace; (2) train the final style adapter under activation-space
deflation -- input activations are projected away from the identified
content directions during training, so the style adapter cannot lean
on content. The paper reports state-of-the-art accuracy on three
authorship-attribution benchmarks (literature, social media, news),
outperforming the strongest existing baseline by up to 29%, and, on
style imitation, stylistically faithful and diverse passages that
beat baselines on style similarity while adhering to content
specifications. This test asserts no magnitudes from the paper -- it
tests the mechanism those results rest on: deflation DURING training,
against the diagnosed subspace, is what keeps the nuisance out.

Toy operationalization: synthetic activations in R^24 built from two
latent factors with KNOWN ground-truth axes:

  b -- the behavior ("style") axis the adapter must produce;
  c -- the content / nuisance axis it must not encode;
  cos(b, c) = RHO = 0.45 -- the axes genuinely overlap in activation
       space, as style and content do in a real model, so projecting
       c out necessarily touches some of b and no arm gets the
       separation for free.

Activations h = s_b * b + s_c * c + SIGMA * noise, s_b, s_c ~ N(0, 1).
The training pool is ENTANGLED, as curated pools are: the target is
y = s_b + GAMMA * s_c with GAMMA = 1.0, so the label loads on content
exactly as much as on style (think author examples whose topics
correlate with the author). The adapter is a rank-1 LoRA-style
readout w (output = (w . h) * e_out for a fixed output channel),
trained by full-batch gradient descent from zero -- the toy form of
the paper's LoRA training, with deflation applied to the input
activations exactly as the paper applies it.

Diagnostic step (faithful to the paper): behavior-neutral training
material is the same activation distribution with the behavior
component zeroed from the TARGET (y_neutral = s_c -- the content
signal with the style contrast removed, the toy form of the paper's
style-neutral statements). A diagnostic readout trained on it spans
the ESTIMATED nuisance subspace chat. Recovery is MEASURED, not
assumed: cos(chat, c) = 0.928-0.931 across seeds. The estimate is
rotated off c by the behavior axis sharing the activation geometry
(the least-squares solution is Sigma^-1 c, not c), and that
imperfection bounds the method -- see the oracle arm and the
residual-leakage correspondence below.

Arms (all trained by the same GD on the same entangled pool):
  A plain     -- no deflation.
  B deflated  -- EDITORS arm: activations projected off chat at
                 every training step (h <- P_chat h); the learned
                 adapter is then evaluated on RAW held-out probes,
                 so any cleanliness is in the weights, not in a
                 test-time projection.
  C random    -- deflation against a fixed random unit direction
                 (matched rank): any projection at all must NOT
                 reproduce the effect.
  D post-hoc  -- train plainly, then project the learned adapter
                 off chat afterwards (the CCS-style alternative,
                 cf. PR #243): the honest during-vs-after comparison.
  E oracle    -- deflation against the TRUE c during training:
                 the leakage floor the diagnostic quality costs us,
                 clearly an upper-bound reference, not a method.

Metrics per seed, on held-out probes (fresh activations, behavior
probes s_b = 1, s_c = 0 and content probes s_b = 0, s_c = 1):
  gain    = mean adapter output on behavior probes (realized shift
            along b),
  leakage = mean adapter output on content probes (realized shift
            along c).

Measured on pop-os (particlegan 0.10.1): diagnostic recovery
cos(chat, c) 0.928-0.931. Plain arm gain 0.83-0.86 with leakage
0.85-0.86 -- leakage essentially equals gain, the entanglement is
total. Deflated arm: gain 0.96-0.98 (115-117% of plain -- deflation
does not merely retain the gain, the adapter re-fits inside the
allowed subspace and slightly exceeds it) at leakage 0.37-0.38
(43-45% of plain). Random-deflation control: leakage 0.85-0.86,
indistinguishable from plain -- the projection must be the
diagnosed one. Oracle arm: leakage ~0.000 at gain 0.74-0.78, so
the deflated arm's residual leakage is the price of diagnostic
imperfection, and it lands where the geometry says it should:
sin(angle(chat, c)) = 0.366-0.375 against measured leakage
0.37-0.38 -- the content component orthogonal to the estimate is
exactly what survives deflation.

The during-vs-after comparison, as measured (this is the part the
brief demanded honesty on): post-hoc projection (D) is slightly
CLEANER than during-training deflation (leakage 0.29 vs 0.37) but
pays for it in gain (0.77-0.80 vs 0.96-0.98, i.e. deflation retains
23-26% more gain) -- projection amputates whatever the plain adapter
put along chat, including the behavior component that overlaps it,
while training under deflation re-optimizes around the constraint.
Neither dominates; they sit at different points of the same
gain-leakage frontier, and the assertions encode exactly that:
deflation wins on retained gain, post-hoc wins narrowly on
leakage, both crush plain training's leakage. Assertions below
encode roughly half to two-thirds of the thinnest observed
margins, no more.

Family relations: PR #232 (centered ablation) removes a harvested
direction's shared mean mode, PR #237 (sharp removal) removes its
high-curvature components, PR #243 (CCS) projects the harvested
direction onto the concept-variance subspace AFTER extraction --
all three clean a direction after it exists. EDITORS is the
during-training member of the family: the nuisance is kept out
while the adapter is being learned, not cleaned up afterwards,
and arm D is exactly the after-the-fact alternative measured
head-to-head. Skips cleanly when the installed particlegan does
not provide the EMPF recipe. Pure numpy/torch state geometry --
no GAN machinery, because the claim is about training geometry,
not GAN dynamics.
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


D = 24            # activation dimension
RHO = 0.45        # cos(b, c): style/content axes genuinely overlap
SIGMA = 0.5       # background noise per dimension
GAMMA = 1.0       # content loading in the entangled target
N = 3000          # training activations per seed
N_PROBE = 4000    # held-out probes per kind per seed
STEPS = 300       # full-batch GD steps per arm (converged; cf. OLS)
ETA = 0.05        # GD step size


def _unit(x):
    return x / np.linalg.norm(x)


def _cos(x, y):
    return float(_unit(x) @ _unit(y))


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    # Ground-truth axes with cos(b, c) = RHO exactly.
    b = _unit(rng.standard_normal(D))
    c_perp = rng.standard_normal(D)
    c_perp -= (c_perp @ b) * b
    c = _unit(RHO * b + np.sqrt(1.0 - RHO ** 2) * _unit(c_perp))

    def gen(n, sb, sc, rng):
        return (sb[:, None] * b[None, :] + sc[:, None] * c[None, :]
                + SIGMA * rng.standard_normal((n, D)))

    sb = rng.standard_normal(N)
    sc = rng.standard_normal(N)
    h = gen(N, sb, sc, rng)
    y = sb + GAMMA * sc            # entangled target
    y_neutral = sc.copy()          # behavior contrast removed

    def train(h_in, target):
        w = np.zeros(D)
        for _ in range(STEPS):
            grad = h_in.T @ (h_in @ w - target) / len(target)
            w = w - ETA * grad
        return w

    # Diagnostic step: fit on behavior-neutral targets, take the span.
    chat = _unit(train(h, y_neutral))
    cos_diag = _cos(chat, c)

    def projector(u):
        return np.eye(D) - np.outer(u, u)

    p_chat = projector(chat)
    p_rand = projector(_unit(rng.standard_normal(D)))
    p_true = projector(c)

    w_plain = train(h, y)
    arms = {
        "plain": w_plain,
        "deflated": train(h @ p_chat, y),
        "random": train(h @ p_rand, y),
        "posthoc": p_chat @ w_plain,
        "oracle": train(h @ p_true, y),
    }

    # Held-out probes: pure behavior and pure content activations.
    h_beh = gen(N_PROBE, np.ones(N_PROBE), np.zeros(N_PROBE), rng)
    h_con = gen(N_PROBE, np.zeros(N_PROBE), np.ones(N_PROBE), rng)
    out = {"cos_diag": cos_diag}
    for name, w in arms.items():
        out[name] = dict(gain=float((h_beh @ w).mean()),
                         leak=float((h_con @ w).mean()))
    return out


def test_diagnose_then_deflate_keeps_nuisance_out():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        a, b_arm = r["plain"], r["deflated"]
        c_arm, e = r["random"], r["oracle"]

        # Diagnostic quality is measured, not assumed: the estimate
        # recovers the content axis well but imperfectly (measured
        # cos 0.928-0.931) -- the imperfection is what bounds the
        # deflated arm's residual leakage below.
        assert r["cos_diag"] > 0.85, (
            f"seed {seed}: diagnostic recovery cos(chat, c) "
            f"{r['cos_diag']:.3f}")
        assert r["cos_diag"] < 0.99, (
            f"seed {seed}: diagnostic recovery cos(chat, c) "
            f"{r['cos_diag']:.3f} -- suspiciously oracle-like; the "
            f"toy should measure a real estimation gap")

        # Trap guard: the plain arm must actually be entangled --
        # leakage on the order of the gain itself (measured leakage
        # 0.85-0.86 vs gain 0.83-0.86). Without that, there is
        # nothing for deflation to fix.
        assert a["gain"] > 0.7, (
            f"seed {seed}: plain-arm gain {a['gain']:.3f}")
        assert a["leak"] >= 0.7 * a["gain"], (
            f"seed {seed}: plain-arm leakage {a['leak']:.3f} vs "
            f"gain {a['gain']:.3f} -- pool not entangled")

        # The EDITORS claim: deflation during training retains the
        # gain (measured 115-117% of plain) while cutting leakage
        # to 43-45% of plain.
        assert b_arm["gain"] >= 0.95 * a["gain"], (
            f"seed {seed}: deflated gain {b_arm['gain']:.3f} vs "
            f"plain {a['gain']:.3f}")
        assert b_arm["leak"] <= 0.55 * a["leak"], (
            f"seed {seed}: deflated leakage {b_arm['leak']:.3f} vs "
            f"plain {a['leak']:.3f}")

        # It must be the diagnostic subspace specifically: deflating
        # against a random direction leaves leakage essentially at
        # the plain level (measured 0.85-0.86 vs plain 0.85-0.86).
        assert c_arm["leak"] >= 0.85 * a["leak"], (
            f"seed {seed}: random-deflation leakage "
            f"{c_arm['leak']:.3f} vs plain {a['leak']:.3f} -- a "
            f"random projection should not reproduce the effect")
        assert b_arm["leak"] <= 0.6 * c_arm["leak"], (
            f"seed {seed}: deflated leakage {b_arm['leak']:.3f} vs "
            f"random-deflation {c_arm['leak']:.3f}")

        # The diagnostic bound, measured both ways: oracle deflation
        # reaches ~zero leakage, and the diagnostic arm's residual
        # leakage is real (it is the content component the estimate
        # missed, sin(angle) ~ 0.37 of the axis), not noise.
        assert abs(e["leak"]) <= 0.05 * a["leak"], (
            f"seed {seed}: oracle leakage {e['leak']:.3f} vs plain "
            f"{a['leak']:.3f}")
        assert e["gain"] >= 0.7 * a["gain"], (
            f"seed {seed}: oracle gain {e['gain']:.3f} vs plain "
            f"{a['gain']:.3f}")
        assert b_arm["leak"] >= 3.0 * abs(e["leak"]) + 0.1, (
            f"seed {seed}: deflated leakage {b_arm['leak']:.3f} vs "
            f"oracle {e['leak']:.3f} -- the diagnostic gap should "
            f"leave a measurable residual, not an oracle result")


def test_deflation_during_training_vs_posthoc_projection():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        a, b_arm, d = r["plain"], r["deflated"], r["posthoc"]

        # Both during-training deflation and post-hoc projection
        # crush the plain arm's leakage (measured: deflated 43-45%
        # of plain, post-hoc 34% of plain).
        assert d["leak"] <= 0.45 * a["leak"], (
            f"seed {seed}: post-hoc leakage {d['leak']:.3f} vs "
            f"plain {a['leak']:.3f}")

        # As measured, the two are NOT interchangeable: post-hoc
        # projection is slightly cleaner (leakage 0.29 vs 0.37)
        # because it amputates the finished adapter along chat,
        # while deflation during training retains 23-26% more gain
        # (0.96-0.98 vs 0.77-0.80) because the adapter re-fits
        # inside the allowed subspace instead of losing the
        # behavior component that overlapped the nuisance axis.
        assert b_arm["gain"] >= 1.12 * d["gain"], (
            f"seed {seed}: deflated gain {b_arm['gain']:.3f} vs "
            f"post-hoc gain {d['gain']:.3f} -- training under the "
            f"constraint should retain more gain than amputating "
            f"after the fact")
        assert d["leak"] <= 0.9 * b_arm["leak"], (
            f"seed {seed}: post-hoc leakage {d['leak']:.3f} vs "
            f"deflated {b_arm['leak']:.3f} -- as measured, the "
            f"one-shot projection is the slightly cleaner of the "
            f"two; if this flips, the docstring's frontier "
            f"account must be re-scoped, not the assertion tuned")
