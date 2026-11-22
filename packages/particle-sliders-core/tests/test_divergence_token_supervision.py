"""CPU gate for the divergence-token claim of Schrodi, Kempf,
Barez & Brox, "Towards Understanding Subliminal Learning: When
and How Hidden Biases Transfer" (arXiv:2509.23886, ICLR 2026):
subliminal transfer is carried by a small set of DIVERGENCE
tokens -- positions where teachers with different biases predict
different tokens. Computing the loss only on divergence tokens
(as few as 4.7% of training tokens) usually preserves or even
strengthens transfer; masking them out largely suppresses it.
Provenance chain: the claim was surfaced in cycle 14 of this
watch from the reference list of Nief (arXiv:2606.00831); Nief
and Morgulis & Hewitt both cite it. It is the token-level
companion of this family's earlier subliminal tests (PR #241's
Fisher picture of which carrier directions identify a trait,
PR #273's shared-base boundary).

Toy operationalization: a binary-token sequence toy in the
#241/#273 family. P = 2000 carrier positions; position i has a
context x_i in R^8 and a carrier logit a_i = beta . x_i (beta
drawn per seed, scale 1.4, so the carrier is mostly confident:
|a| is typically several logits). The teacher's trait is a
known unit vector Delta in R^8 with strength G = 2.0, and it
reaches only a sparse pathway support: s_i = 1 on ~12% of
positions (drawn per seed), 0 elsewhere. The two teachers'
per-position predictive distributions are exactly computable:

    p_clean(i) = sigmoid(a_i)
    p_trait(i) = sigmoid(a_i + G * s_i * (Delta . x_i))

so the per-position divergence TV_i = |p_trait(i) - p_clean(i)|
is exact, not estimated. Divergence concentrates where the
#241 Fisher picture says it must: a position carries trait
signal only if the pathway reaches it (s_i = 1) AND the carrier
is locally flat (sigmoid'(a_i) large), because the trait's
logit shift G * (Delta . x_i) moves the predictive distribution
by ~sigmoid'(a_i) * shift. Steep or unreached positions have
TV ~ 0. The divergence set is the top 5% of positions by TV
(100 of 2000; the paper's 4.7% regime).

Student: a shared-base residual learner, the #273 regime --
teacher and student share the base, so the student starts from
the carrier itself and learns only a trait residual w in R^8
(init 0) entering through the same pathway support:
p_student(i) = sigmoid(a_i + s_i * (w . x_i)). The student is
trained by minibatch SGD (batch 32, lr 0.5, 4 epochs over its
masked position set) on HARD tokens sampled from the
trait teacher -- the paper's hard-distillation regime. Tokens
are drawn fresh each epoch (a generated-data stream, the usual
distillation setup; a single fixed dataset lets the
masking-out arm fit sampling noise in near-null Fisher
directions, a memorization artifact the soft-target diagnostic
in the PR body separates from distributional signal). All arms
of a seed share the same per-epoch token stream (same training
RNG seed per arm); arms differ ONLY by their loss mask.

Arms (loss masks over the trait teacher's tokens):
  (a) ALL positions;
  (b) divergence positions only (top 5% by exact TV);
  (c) ALL EXCEPT divergence positions;
  (d) random positions only, count matched to (b);
  (e) clean-teacher control: tokens from the Delta = 0 teacher,
      full mask.
Readout (installed trait): the recovered trait strength
R = (w . Delta) / G -- the student's mean trait-direction
response on neutral probes, normalized by the teacher's, so
R = 1 is full transfer and R = 0 is none. This is a
magnitude-aware (behavioral) readout, like the paper's
preference rates; cos(w, Delta) is reported as a secondary
diagnostic only, because a near-zero suppressed vector's
direction is noise-dominated and cos would overstate
suppression failure.

Measured on pop-os (particlegan 0.10.1), 3 seeds, 4 epochs:
top-5% TV mass share 0.84-0.93 of all divergence, and the
divergence set lies entirely inside the pathway support.
Recovered trait R: (a) all 0.40-0.73; (b) divergence-only
0.38-0.72 -- within 0.09 of (a) on every seed, on 5% of the
positions; (c) masking divergence out 0.04-0.08 (6-20% of
(a)); (d) random count-matched 0.01-0.05; (e) clean teacher
-0.04..+0.01. Secondary cos(w, Delta): (a) 0.81-0.95,
(b) 0.77-0.97. Assertions below encode exactly these margins,
no more.

What is NOT asserted: anything about real tokenizers, LM
losses, or the paper's owl / number-sequence numbers; that
divergence-only training strengthens transfer in general
(here it preserves within 0.09; the paper says "usually
preserves or even strengthens", and this toy demonstrates the
preservation half); and suppression under a WEAK carrier --
in exploration, a seed whose carrier draw was weak
(|beta| ~ 1.2, vs >= 2.7 for the asserted seeds) showed TV
mass only 0.68 in the top 5% and masking-out suppression
weakening to ~0.5x of full. Divergence concentrates when the
carrier is mostly confident; that is the regime asserted.
Two modeling gifts, stated plainly: the student's residual
enters through the teacher's pathway support s_i (the
shared-base premise -- the toy tests which positions'
SUPERVISION carries transfer, given a shared pathway, not
pathway discovery), and the carrier logit a_i is the shared
base's, known to the student exactly as in #241/#273's
residual toys. Pure numpy -- no GAN machinery, because the
claim is about supervision geometry, not training dynamics.
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


K = 8                # trait-space dimension
P = 2000             # carrier positions per seed
S_FRAC = 0.12        # pathway support fraction (sparse trait reach)
BETA_SCALE = 1.4     # carrier strength: a_i std ~ 1.4 * sqrt(K)
G = 2.0              # teacher trait strength (logit shift scale)
DIV_FRAC = 0.05      # divergence set: top 5% by exact TV
LR = 0.5             # SGD step
EPOCHS = 4           # passes over each arm's masked set
BS = 32              # minibatch size


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cos(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 1e-12 else 0.0


def build(seed):
    """Carrier, teachers and masks for one seed (all exact)."""
    rng = np.random.default_rng(31000 + seed)
    delta = rng.standard_normal(K)
    delta /= np.linalg.norm(delta)
    beta = BETA_SCALE * rng.standard_normal(K)
    x = rng.standard_normal((P, K))
    a = x @ beta
    s = (rng.random(P) < S_FRAC).astype(float)
    p_clean = _sigmoid(a)
    p_trait = _sigmoid(a + G * s * (x @ delta))
    tv = np.abs(p_trait - p_clean)
    n_div = int(round(DIV_FRAC * P))
    div_mask = np.zeros(P, dtype=bool)
    div_mask[np.argsort(-tv)[:n_div]] = True
    rng_r = np.random.default_rng(99000 + seed)
    rand_mask = np.zeros(P, dtype=bool)
    rand_mask[rng_r.choice(P, size=n_div, replace=False)] = True
    return dict(delta=delta, beta=beta, x=x, a=a, s=s,
                p_clean=p_clean, p_trait=p_trait, tv=tv,
                div_mask=div_mask, rand_mask=rand_mask)


def _train(d, mask, teacher, seed):
    """SGD on hard tokens from `teacher` ('trait' or 'clean').

    Tokens are drawn fresh each epoch; every arm of a seed uses
    the same training RNG seed, so arms see identical token
    streams and differ only by the loss mask.
    """
    rng = np.random.default_rng(77000 + seed)
    idx = np.where(mask)[0]
    x, a, s = d["x"], d["a"], d["s"]
    p_teacher = d["p_trait"] if teacher == "trait" else d["p_clean"]
    w = np.zeros(K)
    for _ in range(EPOCHS):
        toks = (rng.random(P) < p_teacher).astype(float)
        order = idx[rng.permutation(len(idx))]
        for st in range(0, len(order), BS):
            b = order[st:st + BS]
            p = _sigmoid(a[b] + s[b] * (x[b] @ w))
            grad = x[b].T @ ((p - toks[b]) * s[b]) / len(b)
            w -= LR * grad
    return w


def run_seed(seed):
    d = build(seed)
    all_mask = np.ones(P, dtype=bool)
    arms = {}
    for name, mask, teacher in (
            ("all", all_mask, "trait"),
            ("div", d["div_mask"], "trait"),
            ("nondiv", ~d["div_mask"], "trait"),
            ("rand", d["rand_mask"], "trait"),
            ("clean", all_mask, "clean")):
        w = _train(d, mask, teacher, seed)
        arms[name] = float(w @ d["delta"] / G)
        arms[name + "_cos"] = _cos(w, d["delta"])
    return d, arms


def test_divergence_is_sparse_exact_and_concentrated():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = build(seed)
        # Exactness: TV is the teachers' predictive gap, and it
        # is exactly zero wherever the pathway does not reach.
        assert np.all(d["tv"][d["s"] == 0] == 0.0), (
            f"seed {seed}: nonzero TV off the pathway support")
        recomputed = np.abs(d["p_trait"] - d["p_clean"])
        assert np.allclose(recomputed, d["tv"]), (
            f"seed {seed}: stored TV is not the exact teacher gap")
        # The divergence set is exactly the top 5%, and it lies
        # entirely inside the pathway support.
        assert int(d["div_mask"].sum()) == int(round(DIV_FRAC * P)), (
            f"seed {seed}: divergence set size "
            f"{int(d['div_mask'].sum())}")
        assert np.all(d["s"][d["div_mask"]] == 1.0), (
            f"seed {seed}: divergence set reaches off-support")
        # Concentration: 5% of positions hold >= 75% of all
        # divergence mass (demonstrated 0.84-0.93), and their
        # mean TV dwarfs the rest.
        mass = float(d["tv"][d["div_mask"]].sum() / d["tv"].sum())
        assert mass >= 0.75, (
            f"seed {seed}: top-5% TV mass share {mass:.2f}")
        ratio = float(d["tv"][d["div_mask"]].mean()
                      / d["tv"][~d["div_mask"]].mean())
        assert ratio >= 20.0, (
            f"seed {seed}: mean-TV ratio div vs rest {ratio:.1f}")


def test_divergence_only_supervision_preserves_transfer():
    _empf_recipe()
    for seed in (0, 1, 2):
        _, arms = run_seed(seed)
        # Full supervision transfers substantially.
        assert arms["all"] > 0.30, (
            f"seed {seed}: full-loss trait recovery "
            f"{arms['all']:+.3f}")
        assert arms["all_cos"] > 0.65, (
            f"seed {seed}: full-loss trait cos "
            f"{arms['all_cos']:+.3f}")
        # Divergence-only supervision -- 5% of positions --
        # preserves it within the demonstrated 0.10 band.
        assert arms["div"] > 0.30, (
            f"seed {seed}: divergence-only trait recovery "
            f"{arms['div']:+.3f}")
        assert arms["div"] >= arms["all"] - 0.10, (
            f"seed {seed}: divergence-only {arms['div']:+.3f} vs "
            f"full {arms['all']:+.3f}")
        assert arms["div_cos"] > 0.65, (
            f"seed {seed}: divergence-only trait cos "
            f"{arms['div_cos']:+.3f}")


def test_masking_out_divergence_suppresses_transfer():
    _empf_recipe()
    for seed in (0, 1, 2):
        _, arms = run_seed(seed)
        # Masking the divergence positions out largely
        # suppresses transfer: a small fraction of full.
        assert arms["nondiv"] < 0.15, (
            f"seed {seed}: masked-out trait recovery "
            f"{arms['nondiv']:+.3f}")
        assert arms["nondiv"] <= 0.30 * arms["all"], (
            f"seed {seed}: masked-out {arms['nondiv']:+.3f} vs "
            f"full {arms['all']:+.3f}")
        # A random count-matched mask does not substitute for
        # the divergence positions.
        assert arms["rand"] < 0.15, (
            f"seed {seed}: random-mask trait recovery "
            f"{arms['rand']:+.3f}")
        assert arms["rand"] <= arms["div"] - 0.25, (
            f"seed {seed}: random {arms['rand']:+.3f} vs "
            f"divergence-only {arms['div']:+.3f}")
        # Clean teacher: no trait in the data, no transfer.
        assert abs(arms["clean"]) < 0.10, (
            f"seed {seed}: clean-teacher trait recovery "
            f"{arms['clean']:+.3f}")
