"""CPU gate for the numerical-scoring-topology claim
(arXiv:2611.26077): the geometry of the scale a judge is asked to
score on -- its width, granularity, offset, symmetry and boundary
placement -- systematically changes the scores the judge returns,
even when the underlying judgement is identical; shifted and
narrow ranges induce some of the largest deviations, and stronger
judges are more robust to topology changes (smaller deviations).

Paper: "Same Judgement, Different Scores: Numerical Scoring
Topology in LLM-as-a-Judge Evaluation" (Kim, Kim, Kum, Song &
Park). The authors define numerical scoring topology (scale
width, granularity, offset, symmetry, boundary placement), vary
it in zero-shot LLM-as-a-judge evaluation across benchmarks and
judge models, and find topology systematically affects scores,
calibration and robustness: shifted and narrow ranges tend to
induce some of the largest deviations, and stronger / larger
judges generally show greater robustness to topology changes.
(The abstract reports orderings, not a single headline number --
stated plainly; the claims under test here are exactly those
orderings.)

Toy operationalization: items carry a latent continuous
judgement q in [0, 1] (known ground truth). A synthetic judge
expresses q on a presented scoring topology T = (lo, hi, step):

    s       = lo + q * (hi - lo)          (placement on T)
    s_pull  = s + gamma * (a*(s) - s)     (anchor pull)
    s_read  = quantize(s_pull + eps, step)
    q_hat   = (s_read - lo) / (hi - lo)   (mapped back to [0, 1])

  a*(s) -- the nearest CANONICAL anchor number to s, from the
       fixed set {0, 2.5, 5, 7.5, 10} restricted to anchors the
       presented scale actually contains. The judge anchors on
       familiar canonical numbers; the pull gamma is in absolute
       scale points, so the SAME pull is a larger fraction of a
       narrow scale (1 / width amplification) and sits at
       different relative positions on a shifted scale. This is
       the only distortion channel: placement and map-back are
       exact inverses, so at gamma = 0 every topology reads q
       identically up to quantization (the control below).
  eps -- Gaussian noise, sigma = 0.05 scale points, PAIRED across
       topologies (the same eps per item in every arm), so noise
       cancels from the paired deviations except through
       quantization.
  No clipping anywhere: q is drawn from [0.02, 0.98], placement
       is strictly interior, the pull contracts toward interior
       anchors, and reads are never clamped -- the ordering is
       pull-driven, not a boundary artifact.

Topologies (reference first):
  reference : [0, 10],  step 0.1 -- symmetric, fine, centred.
  shifted   : [3, 13],  step 0.1 -- same width, offset interval;
              canonical anchors inside it are {5, 7.5, 10}, at
              relative positions {0.2, 0.45, 0.7} instead of the
              reference's {0, 0.25, 0.5, 0.75, 1}.
  narrow    : [4, 6],   step 0.1 -- compressed width; the only
              canonical anchor inside it is {5}, so every value
              is pulled toward 5 in absolute points, which the
              map-back divides by width 2 instead of 10.
  coarse    : [0, 10],  step 1.0 -- reference interval, integer
              granularity.
  coarse2   : [0, 10],  step 2.0 -- a second, coarser level.

Judge variants: base gamma = 0.5; "stronger judge" gamma = 0.25
(halved anchor pull -- the paper's robustness ordering, in the
toy's one susceptibility parameter); control gamma = 0.

Metric per topology: deviation = MAE between that topology's
read q_hat and the REFERENCE topology's read of the same items,
per seed (>= 3 seeds).

Measured on pop-os (particlegan 0.10.1), 3 seeds, base judge:
narrow 0.124 (0.1236-0.1242), shifted 0.054 (0.0533-0.0540),
coarse 0.026 (0.0255-0.0258), coarse2 0.052 (0.0509-0.0522) --
shifted and narrow deviate most, each clearing coarse, and the
coarse deviation roughly doubles from step 1.0 to step 2.0.
Stronger judge (gamma halved): shifted 0.027, narrow 0.065 --
the pull-driven deviations halve with gamma -- while coarse /
coarse2 stay at 0.026 / 0.051, unchanged. Control (gamma = 0):
shifted 0.0000 exactly, narrow 0.019-0.020 -- the shifted and
narrow deviations collapse to pure quantization differences
(narrow's residual is its coarser effective step in q units:
0.1 points on a width-2 scale), at or below the coarse arm's
own quantization deviation. Assertions below encode exactly
these margins, no more.

RE-SCOPE (one, deliberate): the brief's stronger-judge claim
was "deviations uniformly smaller across ALL topologies". The
toy refutes that stronger reading for the granularity arms:
coarse deviation is a pure quantization floor -- it does not
contain the pull, so halving the pull cannot shrink it
(measured: identical to within 0.0002). The gate therefore
asserts uniform shrinkage on the pull-driven topologies
(shifted, narrow) and asserts the coarse floor's
strength-INDEPENDENCE as its own documented fact. Robustness
to topology is resistance to the pull channel; granularity
error is a separate floor no judge strength removes in this
toy -- which is itself the readout-design lesson (choose the
granularity deliberately; a better judge does not fix it).

What is NOT asserted: any real judge's deviation magnitudes,
calibration or discriminative-validity numbers -- the paper
reports orderings from real LLM judges; this toy has no LLM,
only the pull + quantization mechanism, and tests whether that
plausible mechanism reproduces the paper's deviation ORDERING
and its robustness ordering.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure CPU toy -- no GAN
machinery, because the claim is about the geometry of a scoring
readout, not training dynamics.
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


CANONICAL_ANCHORS = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
TOPOLOGIES = {
    "reference": (0.0, 10.0, 0.1),
    "shifted": (3.0, 13.0, 0.1),
    "narrow": (4.0, 6.0, 0.1),
    "coarse": (0.0, 10.0, 1.0),
    "coarse2": (0.0, 10.0, 2.0),
}
N_ITEMS = 4000
SIGMA = 0.05
GAMMA_BASE = 0.5
GAMMA_STRONG = 0.25


def _read(q, eps, topology, gamma):
    lo, hi, step = topology
    width = hi - lo
    s = lo + q * width
    anchors = CANONICAL_ANCHORS[
        (CANONICAL_ANCHORS >= lo) & (CANONICAL_ANCHORS <= hi)]
    nearest = anchors[np.argmin(
        np.abs(s[:, None] - anchors[None, :]), axis=1)]
    s_pull = s + gamma * (nearest - s)
    s_quant = np.round((s_pull + eps) / step) * step
    return (s_quant - lo) / width


def run_seed(seed, gamma):
    rng = np.random.default_rng(41000 + seed)
    q = rng.uniform(0.02, 0.98, N_ITEMS)
    eps = rng.normal(0.0, SIGMA, N_ITEMS)
    reads = {name: _read(q, eps, topo, gamma)
             for name, topo in TOPOLOGIES.items()}
    return {name: float(np.mean(np.abs(reads[name]
                                       - reads["reference"])))
            for name in TOPOLOGIES if name != "reference"}


def test_shifted_and_narrow_deviate_most():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = run_seed(seed, GAMMA_BASE)
        # Narrow deviates most, by a clear margin over shifted.
        assert d["narrow"] > 0.09, (
            f"seed {seed}: narrow deviation {d['narrow']:.4f}")
        assert d["narrow"] > d["shifted"] + 0.04, (
            f"seed {seed}: narrow {d['narrow']:.4f} vs shifted "
            f"{d['shifted']:.4f}")
        # Shifted deviates clearly, clearing the coarse arm.
        assert d["shifted"] > 0.04, (
            f"seed {seed}: shifted deviation {d['shifted']:.4f}")
        assert d["shifted"] > d["coarse"] + 0.015, (
            f"seed {seed}: shifted {d['shifted']:.4f} vs coarse "
            f"{d['coarse']:.4f}")
        assert d["narrow"] > d["coarse"] + 0.015, (
            f"seed {seed}: narrow {d['narrow']:.4f} vs coarse "
            f"{d['coarse']:.4f}")


def test_coarse_deviation_grows_with_coarseness():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = run_seed(seed, GAMMA_BASE)
        # Integer granularity alone deviates from the fine read.
        assert d["coarse"] > 0.015, (
            f"seed {seed}: coarse deviation {d['coarse']:.4f}")
        # And the deviation grows as granularity coarsens.
        assert d["coarse2"] > d["coarse"] + 0.015, (
            f"seed {seed}: coarse2 {d['coarse2']:.4f} vs coarse "
            f"{d['coarse']:.4f}")


def test_stronger_judge_deviates_less_where_pull_drives():
    _empf_recipe()
    for seed in (0, 1, 2):
        base = run_seed(seed, GAMMA_BASE)
        strong = run_seed(seed, GAMMA_STRONG)
        # Halving the pull shrinks the pull-driven deviations by
        # a clear margin on every pull-driven topology.
        assert strong["shifted"] < base["shifted"] - 0.015, (
            f"seed {seed}: strong shifted {strong['shifted']:.4f} "
            f"vs base {base['shifted']:.4f}")
        assert strong["narrow"] < base["narrow"] - 0.03, (
            f"seed {seed}: strong narrow {strong['narrow']:.4f} "
            f"vs base {base['narrow']:.4f}")
        # The granularity arms are a quantization floor: judge
        # strength does not move them (the documented re-scope).
        assert abs(strong["coarse"] - base["coarse"]) < 0.005, (
            f"seed {seed}: strong coarse {strong['coarse']:.4f} "
            f"vs base {base['coarse']:.4f}")
        assert abs(strong["coarse2"] - base["coarse2"]) < 0.005, (
            f"seed {seed}: strong coarse2 {strong['coarse2']:.4f} "
            f"vs base {base['coarse2']:.4f}")


def test_no_pull_control_collapses_to_quantization():
    _empf_recipe()
    for seed in (0, 1, 2):
        base = run_seed(seed, GAMMA_BASE)
        ctrl = run_seed(seed, 0.0)
        # With no anchor pull, the shifted topology reads
        # identically to the reference (same width, same step,
        # paired noise): deviation is exactly the quantization
        # residue, here zero.
        assert ctrl["shifted"] < 0.005, (
            f"seed {seed}: control shifted {ctrl['shifted']:.4f}")
        # Narrow collapses to its pure quantization difference
        # (effective step 0.05 in q units), far below its
        # pull-driven deviation and at or below the coarse arm's
        # own quantization deviation.
        assert ctrl["narrow"] < 0.03, (
            f"seed {seed}: control narrow {ctrl['narrow']:.4f}")
        assert ctrl["narrow"] < 0.3 * base["narrow"], (
            f"seed {seed}: control narrow {ctrl['narrow']:.4f} "
            f"vs base {base['narrow']:.4f}")
        assert ctrl["narrow"] <= ctrl["coarse"] + 0.005, (
            f"seed {seed}: control narrow {ctrl['narrow']:.4f} "
            f"vs control coarse {ctrl['coarse']:.4f}")
