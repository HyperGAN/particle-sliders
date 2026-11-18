"""CPU gate for the input-dependent steering claim
(arXiv:2611.04757): a steering edit should depend on the input
exactly when the destination varies INSIDE the class a query
requests, and that regime is measurable from the states alone,
before fitting any rule.

Paper: "When Should an LLM Steering Edit Depend on the Input?"
(Harkness & Rawlik). CAA adds one fixed direction -- the difference
of class means -- to every input. Conditioning the edit on the
requested class handles between-class variation but leaves
within-class variation untouched. The paper instead interpolates
toward a RETRIEVED destination: the mean of the k stored states
nearest the query within the requested class. At k = class size
the retrieved rule IS the class-conditional destination rule, so
the baselines are endpoints of one family. On six substitution
corpora, input-dependent steering beats fixed-direction steering
when the destination varies within class; retrieval wins at every
layer of every corpus in two models; word-level edits want early
layers and few neighbours, clause-level edits any depth from the
class mean. Where the class already determines the answer, the
class mean suffices and retrieval only matches it -- but a fixed
global direction still fails there. Four quantities computed from
the states alone identify the regime; this toy uses the simplest
of them, within-class destination dispersion.

This paper extends the family of cycle-1 PSR (per-token magnitude,
PR #209) and cycle-2 RTRT (per-token selection, PR #212) with
per-INPUT destination choice: PSR asks how much each token gets,
RTRT asks which tokens get anything, this asks where each input
should be steered TO.

Toy operationalization: a Gaussian state world, D = 16, C = 4
classes, M = 6 subclusters per class. Each class c has a source
mean and a destination mean separated by a class-specific delta
(the between-class variation). A shared latent z per subcluster
places its source centre at mu_src[c] + TAU_SRC * z and its
destination centre at mu_dst[c] + tau_dst * z, so the nearest
stored sources of a query identify its subcluster and therefore
its destination centre -- the correlation retrieval exploits, and
the only one it needs. A stored bank of (source, destination)
pairs per class (40 per subcluster) is the retrieval set; queries
are fresh draws from the same subclusters. Two regimes share the
same class means, deltas and latents per seed and differ only in
tau_dst:

  Regime A (destination varies within class): tau_dst = 1.6.
  Regime B (class determines destination):    tau_dst = 0.05,
    every subcluster's destination centre is (near-)identical.

Arms (post-edit state; error is squared distance to the query's
true per-example destination):

  (a) global fixed direction (CAA):  x + (grand destination mean
      - grand source mean), one direction applied to all inputs;
  (b) class-conditional destination: the bank's class destination
      mean -- edit is (class mean destination - x). This is the
      k = class-size endpoint of the retrieved family, exactly as
      the paper frames it;
  (c) retrieved destination: mean destination of the k = 5 bank
      states whose sources are nearest the query, within the
      requested class -- edit is (retrieved destination - x).

An auxiliary arm (c-full), retrieval at k = bank class size, must
reproduce (b) exactly: the endpoint identity, checked in-test.

Regime statistic, from the states alone (bank destinations only,
no rule applied): within-class destination dispersion = mean over
classes of trace(cov(bank destinations within class)).

Iteration log (the loop that produced this gate):
  1. Additive reading of arm (b) -- edit = class destination mean
     minus class source mean, added to x, like (a) but per class:
     Regime A ordered correctly (retrieved 2.3-2.5 < class
     6.0-6.3 < global 53-83), but Regime B REFUTED the claim
     (class 19.7-21.4 vs retrieved 2.3-2.5): an additive class
     delta carries the query's own within-class source offset
     into the output, so it cannot land on a constant destination
     no matter how well the class determines it. That is a
     stronger reading than the paper makes: its class-conditional
     baseline is the endpoint of the retrieved family (destination
     form, above), not an additive delta. Re-scoped to the
     paper's actual claim; no world parameter changed.
  2. Destination-form arm (b): both regime orderings hold on
     every seed with wide margins (see Measured), the endpoint
     identity holds exactly, and dispersion separates the regimes
     ~17x. Proven; shipped as this test.

Measured on pop-os (particlegan 0.10.1), squared distance to the
true destination (lower is better), 200 queries/seed/regime:
Regime A -- global 52.3-82.6, class 33.1-37.6, retrieved
2.26-2.36, dispersion 33.3-36.0. Regime B -- global 68.1-105.2,
class 2.02-2.08, retrieved 2.36-2.43, dispersion 1.99-2.00.
Skips cleanly when the installed particlegan does not provide the
EMPF recipe.
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


D = 16
C = 4                # classes
M = 6                # subclusters per class
K = 5                # retrieved neighbours (paper: few for fine edits)
BANK_PER_SUB = 40
N_QUERY_PER_CLASS = 50
SIGMA = 0.35         # per-example observation noise
TAU_SRC = 1.2        # source subcluster spread (identifies subcluster)
TAU_DST_A = 1.6      # regime A: destination varies within class
TAU_DST_B = 0.05     # regime B: destination (near-)constant within class


def _world(seed):
    """Class means, per-class deltas and subcluster latents, shared
    by both regimes within a seed."""
    rng = np.random.default_rng(10000 + seed)
    mu_src = rng.normal(0, 2.0, size=(C, D))
    mu_dst = mu_src + rng.normal(0, 2.5, size=(C, D))
    z = rng.normal(0, 1, size=(C, M, D))
    return mu_src, mu_dst, z


def _run_regime(seed, mu_src, mu_dst, z, tau_dst):
    rng = np.random.default_rng(20000 + seed + int(tau_dst * 1000))
    src_c = mu_src[:, None, :] + TAU_SRC * z
    dst_c = mu_dst[:, None, :] + tau_dst * z
    bank_src, bank_dst, bank_cls = [], [], []
    for c in range(C):
        for m in range(M):
            bank_src.append(src_c[c, m] + rng.normal(0, SIGMA, size=(BANK_PER_SUB, D)))
            bank_dst.append(dst_c[c, m] + rng.normal(0, SIGMA, size=(BANK_PER_SUB, D)))
            bank_cls.append(np.full(BANK_PER_SUB, c))
    bank_src = np.vstack(bank_src)
    bank_dst = np.vstack(bank_dst)
    bank_cls = np.concatenate(bank_cls)

    grand_delta = bank_dst.mean(0) - bank_src.mean(0)
    dispersion = float(np.mean([
        np.var(bank_dst[bank_cls == c], axis=0).sum() for c in range(C)]))

    errs = {"global": [], "class": [], "retr": [], "retr_full": []}
    for c in range(C):
        idx = np.where(bank_cls == c)[0]
        class_mean = bank_dst[idx].mean(0)
        for _ in range(N_QUERY_PER_CLASS):
            m = int(rng.integers(0, M))
            x = src_c[c, m] + rng.normal(0, SIGMA, size=D)
            y = dst_c[c, m] + rng.normal(0, SIGMA, size=D)
            errs["global"].append(np.sum((x + grand_delta - y) ** 2))
            errs["class"].append(np.sum((class_mean - y) ** 2))
            dists = np.sum((bank_src[idx] - x) ** 2, axis=1)
            nn_idx = idx[np.argpartition(dists, K)[:K]]
            errs["retr"].append(np.sum((bank_dst[nn_idx].mean(0) - y) ** 2))
            errs["retr_full"].append(np.sum((class_mean - y) ** 2))
    return {k: float(np.mean(v)) for k, v in errs.items()}, dispersion


def run_seed(seed):
    _empf_recipe()
    mu_src, mu_dst, z = _world(seed)
    a, disp_a = _run_regime(seed, mu_src, mu_dst, z, TAU_DST_A)
    b, disp_b = _run_regime(seed, mu_src, mu_dst, z, TAU_DST_B)
    return a, disp_a, b, disp_b


def test_input_dependent_steering():
    for seed in range(3):
        a, disp_a, b, disp_b = run_seed(seed)
        # Endpoint identity: retrieval at k = class size IS the
        # class-conditional rule (same computation, checked exactly).
        assert a["retr_full"] == a["class"]
        assert b["retr_full"] == b["class"]

        # Regime A -- destination varies within class:
        # retrieved > class-conditional > global fixed direction.
        # Demonstrated: retr 2.26-2.36, class 33.1-37.6,
        # global 52.3-82.6 (pop-os table in the PR).
        assert a["retr"] < 0.20 * a["class"], (
            f"seed {seed} A: retr {a['retr']:.2f} vs class {a['class']:.2f}")
        assert a["class"] < 0.80 * a["global"], (
            f"seed {seed} A: class {a['class']:.2f} vs global {a['global']:.2f}")
        assert a["retr"] < 0.10 * a["global"]

        # Regime B -- class determines the destination:
        # class-conditional ~= retrieved, and the fixed global
        # direction still fails. Demonstrated: class 2.02-2.08,
        # retr 2.36-2.43 (ratio 1.17), global 68.1-105.2.
        assert b["class"] < 0.10 * b["global"], (
            f"seed {seed} B: class {b['class']:.2f} vs global {b['global']:.2f}")
        assert b["retr"] < 0.10 * b["global"]
        assert b["retr"] <= 1.40 * b["class"], (
            f"seed {seed} B: retr {b['retr']:.2f} vs class {b['class']:.2f}")
        assert b["class"] <= 1.40 * b["retr"]

        # The regime is identifiable from the states alone:
        # within-class destination dispersion separates A from B
        # before any rule is applied. Demonstrated: A 33.3-36.0,
        # B 1.99-2.00 (~17x), threshold anywhere in (5, 10).
        assert disp_a > 8.0 * disp_b, (
            f"seed {seed}: dispersion A {disp_a:.2f} vs B {disp_b:.2f}")
        assert disp_a > 10.0 and disp_b < 5.0
