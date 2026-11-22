"""CPU gate for the teacher-pool dilution claim of Schrodi,
Kempf, Barez & Brox, "Towards Understanding Subliminal
Learning" (arXiv:2509.23886), Section 6.2 (mixing teachers):
mixing finetuning data from a BIASED teacher with data from
an UNBIASED teacher weakens subliminal trait transmission --
"even 10% unbiased data substantially reduces trait
transmission, and 25% nearly eliminates it." Mixing data
from two teachers with the SAME bias but different
architectures (50/50) also weakens transmission (owl,
penguin; less for eagle, where unexpected cross-model
transfer appeared). Provenance: cycle 15 of the Particle
Sliders paper watch; this is the untested second half of
the paper whose divergence-token claim PR #275 proved in
cycle 14, and it reuses #275's toy apparatus verbatim.

Toy operationalization: the #275 / #241 / #273 binary-token
family. P = 2000 carrier positions; position i has a context
x_i in R^8 and a carrier logit a_i = beta . x_i (beta drawn
per seed, scale 1.4). The trait teacher's bias is a known
unit vector Delta with strength G = 2.0, reaching only a
sparse pathway support s_i = 1 on ~12% of positions:

    p_clean(i) = sigmoid(a_i)
    p_trait(i) = sigmoid(a_i + G * s_i * (Delta . x_i))

Student: the shared-base residual learner of #273/#275 --
p_student(i) = sigmoid(a_i + s_i * (w . x_i)), w in R^8
init 0, trained by minibatch SGD (batch 32, lr 0.5) on HARD
tokens sampled fresh per epoch. Dilution is a property of
the training STREAM: at matched TOTAL token budget (4
epochs over all P positions), each position's token in each
epoch is drawn from the trait teacher with probability
1 - f and from the clean teacher with probability f. All
arms of a seed share the same per-epoch stream RNG (same
mixture draws, same token uniforms, same permutations);
arms differ only by f, by teacher choice, or by budget.

Arms:
  (a) dilution sweep f in {0, 0.10, 0.25, 0.50}, full budget;
  (b) budget controls: trait-teacher-only at (1 - f) x total
      budget for f = 0.25 (3 epochs) and f = 0.50 (2 epochs)
      -- separates "fewer trait tokens seen" from "clean
      tokens present in the stream";
  (c) same-trait / different-pathway mixing: a second trait
      teacher with the SAME Delta and G but an independent
      pathway support s' (the same-bias / different-
      architecture analogue), mixed 50/50 with the first
      trait teacher at full budget;
  (d) clean control f = 1.0 (clean teacher only).

Two readouts, because the paper's claim is stated against a
BEHAVIORAL preference rate while the install itself is a
logit quantity, and the toy arbitrates where dilution's
power lives:
  (1) logit recovery R = (w . Delta) / G (install-side);
  (2) behavioral preference rate: the fraction of 4000 fresh
      trait-evoking probe contexts x_p = 0.85 * Delta +
      sqrt(1 - 0.85^2) * eps (pathway-reached by
      construction, the analogue of the paper's trait-
      relevant preference questions) whose student response
      w . x_p exceeds a threshold tau. Tau is fixed per seed
      as the median f = 0 probe response, so the undiluted
      arm sits at 0.500 by construction and every other arm
      is read against that reference eval.

Measured on this VM (particlegan 0.10.1), 3 seeds -- R
first, behavioral rate second (tau = 1.20 / 0.63 / 1.05):
  f = 0:      R +0.71/+0.37/+0.63,  beh 0.500 (reference)
  f = 0.10:   R +0.65/+0.33/+0.55,  beh 0.44/0.45/0.41
  f = 0.25:   R +0.52/+0.28/+0.48,  beh 0.31/0.35/0.32
  f = 0.50:   R +0.36/+0.22/+0.30,  beh 0.11/0.18/0.06
  budget .25: R +0.63/+0.28/+0.55,  beh 0.43/0.35/0.41
  budget .50: R +0.47/+0.20/+0.42,  beh 0.22/0.13/0.24
  dual 50/50: R +0.36/+0.24/+0.42,  beh 0.08/0.26/0.22
  clean f=1:  R -0.03/+0.08/+0.04,  beh 0.00/0.00/0.00

The arbitration, stated plainly: R(f) / R(0) tracks the
token arithmetic (1 - f) on every seed (0.88-0.91 at
f = 0.10, 0.73-0.77 at f = 0.25, 0.48-0.60 at f = 0.50 --
never more than 0.10 from (1 - f)); the install itself is
NOT superlinearly destroyed. The behavioral rate falls
faster than R at every f (ratios 0.62-0.70 at f = 0.25 and
0.11-0.36 at f = 0.50, against R ratios 0.73-0.77 and
0.48-0.60): thresholding a linearly shrunken response
distribution manufactures most of the paper's dramatic
dilution curve. In THIS toy, dilution's power is mostly
readout-side arithmetic. One secondary install-side effect
is real but modest: mixed arms sit below their budget
controls on average (mean R gap 0.06-0.07, mean behavioral
gap 0.07-0.08) -- clean tokens on pathway positions
actively supervise w back toward the base, they are not
merely absent trait tokens; per seed the gap is within
noise on the weakest seed. The dual-teacher arm reproduces
the paper's same-bias weakening at the logit level (R ratio
0.51-0.66 at full budget): on the student's pathway the
second teacher is clean wherever the pathways do not
overlap (~1% of positions), so cross-pathway mixing acts
like ~44% effective dilution of each teacher's signal.
Assertions below encode exactly these margins, no more.

What is NOT asserted: the paper's literal "25% nearly
eliminates" against the logit readout -- it does not hold
there in this toy (R ratio at f = 0.25 is 0.73-0.77, and
the behavioral rate at f = 0.25 is still 0.31-0.35, not
near zero; near-elimination of the behavioral rate appears
only at f = 0.50, 0.06-0.18); anything about real LM
preference evals, where the response distribution's shape
is not this toy's; the eagle exception (unexpected
cross-model transfer) -- the toy's second pathway is an
independent support draw, so cross-pathway transfer here
is overlap-only by construction; and dilution under a
WEAK carrier (the weakest seed, R(0) = 0.37, shows the
noisiest budget-control gaps). Two modeling gifts, stated
plainly: the student's residual enters through the FIRST
teacher's pathway support s_i (the #275 shared-base /
shared-pathway premise -- for the dual arm this is exactly
the mechanism under test, pathway mismatch, not pathway
discovery), and the carrier logit a_i is the shared base's,
known to the student exactly as in #241/#273/#275. Pure
numpy -- no GAN machinery, because the claim is about
training-stream composition, not training dynamics. Skips
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


K = 8                # trait-space dimension
P = 2000             # carrier positions per seed
S_FRAC = 0.12        # pathway support fraction (sparse trait reach)
BETA_SCALE = 1.4     # carrier strength: a_i std ~ 1.4 * sqrt(K)
G = 2.0              # teacher trait strength (logit shift scale)
LR = 0.5             # SGD step
EPOCHS = 4           # full-budget passes over all positions
BS = 32              # minibatch size
PROBE_ALPHA = 0.85   # trait-evoking probe alignment with Delta
N_PROBE = 4000       # fresh probe contexts per seed


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def build(seed):
    """Carrier and teachers for one seed (all exact).

    Same draws as #275 (delta, beta, x, s), plus the second
    trait teacher's independent pathway support s' drawn
    next from the same RNG.
    """
    rng = np.random.default_rng(31000 + seed)
    delta = rng.standard_normal(K)
    delta /= np.linalg.norm(delta)
    beta = BETA_SCALE * rng.standard_normal(K)
    x = rng.standard_normal((P, K))
    a = x @ beta
    s = (rng.random(P) < S_FRAC).astype(float)
    s2 = (rng.random(P) < S_FRAC).astype(float)
    p_clean = _sigmoid(a)
    p_trait = _sigmoid(a + G * s * (x @ delta))
    p_trait2 = _sigmoid(a + G * s2 * (x @ delta))
    return dict(delta=delta, beta=beta, x=x, a=a, s=s, s2=s2,
                p_clean=p_clean, p_trait=p_trait,
                p_trait2=p_trait2)


def _train(d, seed, kind, f=0.0, epochs=EPOCHS):
    """SGD on hard tokens from a per-epoch mixed stream.

    kind: 'mix' (trait with prob 1 - f, clean with prob f),
    'trait' (trait teacher only), 'clean' (clean teacher
    only), 'dual' (trait / trait2 50/50). Every arm of a seed
    uses the same training RNG seed and draws the mixture
    uniforms, token uniforms and permutation in the same
    order each epoch, so arms see identical streams and
    differ only by teacher choice, f, or epoch count.
    """
    rng = np.random.default_rng(77000 + seed)
    x, a, s = d["x"], d["a"], d["s"]
    w = np.zeros(K)
    for _ in range(epochs):
        u_mix = rng.random(P)
        u_tok = rng.random(P)
        if kind == "mix":
            p_sel = np.where(u_mix < f, d["p_clean"], d["p_trait"])
        elif kind == "trait":
            p_sel = d["p_trait"]
        elif kind == "clean":
            p_sel = d["p_clean"]
        elif kind == "dual":
            p_sel = np.where(u_mix < 0.5, d["p_trait"],
                             d["p_trait2"])
        else:  # pragma: no cover - arm names are fixed below
            raise ValueError(kind)
        toks = (u_tok < p_sel).astype(float)
        order = rng.permutation(P)
        for st in range(0, P, BS):
            b = order[st:st + BS]
            p = _sigmoid(a[b] + s[b] * (x[b] @ w))
            grad = x[b].T @ ((p - toks[b]) * s[b]) / len(b)
            w -= LR * grad
    return w


def _probes(d, seed):
    """Fresh trait-evoking probe contexts (shared per seed)."""
    rng = np.random.default_rng(55000 + seed)
    eps = rng.standard_normal((N_PROBE, K))
    return (PROBE_ALPHA * d["delta"]
            + np.sqrt(1.0 - PROBE_ALPHA ** 2) * eps)


def run_seed(seed):
    d = build(seed)
    xp = _probes(d, seed)
    spec = (
        ("f0", "mix", 0.0, EPOCHS),
        ("f10", "mix", 0.10, EPOCHS),
        ("f25", "mix", 0.25, EPOCHS),
        ("f50", "mix", 0.50, EPOCHS),
        ("bud25", "trait", 0.0, 3),
        ("bud50", "trait", 0.0, 2),
        ("dual", "dual", 0.0, EPOCHS),
        ("clean", "clean", 0.0, EPOCHS),
    )
    ws = {name: _train(d, seed, kind, f, ep)
          for name, kind, f, ep in spec}
    # Threshold calibrated on the undiluted arm: its
    # behavioral rate is 0.500 by construction.
    tau = float(np.median(xp @ ws["f0"]))
    arms = {}
    for name, w in ws.items():
        arms[name] = float(w @ d["delta"] / G)
        arms[name + "_beh"] = float(np.mean(xp @ w > tau))
    return d, arms


def test_dilution_scales_logit_recovery_linearly():
    _empf_recipe()
    for seed in (0, 1, 2):
        _, arms = run_seed(seed)
        # Undiluted stream transfers substantially; the clean
        # control installs nothing.
        assert arms["f0"] > 0.30, (
            f"seed {seed}: undiluted trait recovery "
            f"{arms['f0']:+.3f}")
        assert abs(arms["clean"]) < 0.10, (
            f"seed {seed}: clean-teacher trait recovery "
            f"{arms['clean']:+.3f}")
        # The install-side law is the token arithmetic: R(f)
        # tracks (1 - f) * R(0) within the demonstrated 0.10
        # band on every arm -- no superlinear collapse of R
        # at 10% or 25% clean data.
        for name, f in (("f10", 0.10), ("f25", 0.25),
                        ("f50", 0.50)):
            ratio = arms[name] / arms["f0"]
            assert abs(ratio - (1.0 - f)) <= 0.12, (
                f"seed {seed}: R ratio at f={f} is "
                f"{ratio:.2f}, off the (1-f) line "
                f"{1.0 - f:.2f} by more than 0.12")
        assert arms["f25"] >= 0.65 * arms["f0"], (
            f"seed {seed}: 25% dilution left R at "
            f"{arms['f25']:+.3f} vs undiluted "
            f"{arms['f0']:+.3f} -- not the paper's "
            f"near-elimination at the logit level")


def test_behavioral_readout_collapses_superlinearly():
    _empf_recipe()
    for seed in (0, 1, 2):
        _, arms = run_seed(seed)
        # Reference arm sits mid-range by calibration; the
        # clean control never crosses the threshold.
        assert abs(arms["f0_beh"] - 0.5) < 1e-9, (
            f"seed {seed}: reference behavioral rate "
            f"{arms['f0_beh']:.3f}")
        assert arms["clean_beh"] < 0.02, (
            f"seed {seed}: clean-teacher behavioral rate "
            f"{arms['clean_beh']:.3f}")
        # 10% clean data already moves the behavioral rate
        # (demonstrated 0.41-0.45) while R stays on the
        # (1 - f) line (asserted above).
        assert arms["f10_beh"] <= 0.46, (
            f"seed {seed}: behavioral rate at f=0.10 "
            f"{arms['f10_beh']:.3f}")
        # The behavioral ratio falls strictly faster than
        # the logit ratio at 25%, and by a wide margin at
        # 50%: thresholding amplifies linear dilution.
        r25 = arms["f25"] / arms["f0"]
        b25 = arms["f25_beh"] / arms["f0_beh"]
        assert b25 < r25, (
            f"seed {seed}: behavioral ratio at f=0.25 "
            f"{b25:.2f} not below logit ratio {r25:.2f}")
        assert arms["f25_beh"] <= arms["f0_beh"] - 0.12, (
            f"seed {seed}: behavioral rate at f=0.25 "
            f"{arms['f25_beh']:.3f}")
        r50 = arms["f50"] / arms["f0"]
        b50 = arms["f50_beh"] / arms["f0_beh"]
        assert b50 <= r50 - 0.10, (
            f"seed {seed}: behavioral ratio at f=0.50 "
            f"{b50:.2f} vs logit ratio {r50:.2f}")
        assert arms["f50_beh"] <= 0.20, (
            f"seed {seed}: behavioral rate at f=0.50 "
            f"{arms['f50_beh']:.3f} -- near-elimination "
            f"appears here, not at 25%")


def test_budget_control_and_dual_teacher_mixing():
    _empf_recipe()
    mix_r, bud_r, mix_b, bud_b = [], [], [], []
    for seed in (0, 1, 2):
        _, arms = run_seed(seed)
        # Clean data present never beats merely seeing fewer
        # trait tokens, beyond per-seed noise (demonstrated
        # inversions <= 0.03 in R, <= 0.05 behavioral, both
        # on the weakest seed).
        for mf, bf in (("f25", "bud25"), ("f50", "bud50")):
            assert arms[mf] <= arms[bf] + 0.05, (
                f"seed {seed}: mixed {arms[mf]:+.3f} above "
                f"budget control {arms[bf]:+.3f}")
            assert arms[mf + "_beh"] <= arms[bf + "_beh"] + 0.06, (
                f"seed {seed}: mixed behavioral "
                f"{arms[mf + '_beh']:.3f} above budget "
                f"control {arms[bf + '_beh']:.3f}")
        mix_r += [arms["f25"], arms["f50"]]
        bud_r += [arms["bud25"], arms["bud50"]]
        mix_b += [arms["f25_beh"], arms["f50_beh"]]
        bud_b += [arms["bud25_beh"], arms["bud50_beh"]]
        # Same trait, different pathway, 50/50 at full
        # budget: weakened install vs the single teacher,
        # as the paper reports (demonstrated R ratios
        # 0.51-0.66, behavioral 0.08-0.26).
        assert arms["dual"] <= arms["f0"] - 0.10, (
            f"seed {seed}: dual-teacher R {arms['dual']:+.3f} "
            f"vs single-teacher {arms['f0']:+.3f}")
        assert arms["dual"] <= 0.75 * arms["f0"], (
            f"seed {seed}: dual-teacher R ratio "
            f"{arms['dual'] / arms['f0']:.2f}")
        assert arms["dual_beh"] <= 0.30, (
            f"seed {seed}: dual-teacher behavioral rate "
            f"{arms['dual_beh']:.3f}")
    # On average the clean tokens actively pull against the
    # install beyond their mere token share: the mixed arms
    # sit below the budget controls by >= 0.05 on both
    # readouts (demonstrated mean gaps 0.06-0.08).
    assert float(np.mean(bud_r) - np.mean(mix_r)) >= 0.05, (
        f"mean R: mixed {np.mean(mix_r):+.3f} vs budget "
        f"{np.mean(bud_r):+.3f}")
    assert float(np.mean(bud_b) - np.mean(mix_b)) >= 0.05, (
        f"mean behavioral: mixed {np.mean(mix_b):.3f} vs "
        f"budget {np.mean(bud_b):.3f}")
