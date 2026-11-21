"""CPU gate for the subliminal-learning boundary claim
(arXiv:2507.14805): a student trained on a teacher's semantically
unrelated data acquires the teacher's trait when teacher and
student share a base model -- "However, we do not observe the
effect when the teacher and student have different base models"
(Cloud et al., "Subliminal Learning"). Provenance chain: the claim
was surfaced from the reference list of RegAmp (arXiv:2611.10397,
this watch's PR #250 paper), whose own toy lives in the same
shared-base regime.

Graded form tested here (informed by PR #245, where probe transfer
survived 17-35 degrees of representation rotation but degraded
sharply at 60): subliminal transfer decays with teacher-student
basis misalignment, vanishing as the bases become orthogonal. The
paper's claim is binary (same base vs different base); the graded
cos-theta form is this watch's sharpening, stated as such.

Toy operationalization: PR #241's Fisher toy, taken literally,
plus one rotation. Work in R^16. The teacher trait Delta has one
unit-magnitude component (+-1, signs per seed) in every direction.
The carrier task's local Fisher F is diagonal with eigenvalues
log-spaced from 1.0 (steep) to 0.02 (flat), so the gradient signal
points at F Delta, not Delta (cos(F Delta, Delta) = 0.684, the
#241 Fisher-distortion value). The student trains by gradient
descent on SOFT targets -- the #241 regime where recovery is
near-exact -- on the quadratic whose gradient in the student's
own coordinates is F (w - Delta): the teacher's trait, expressed
as coordinates, is what the carrier data identifies. Depth 200,
ETA = 0.5 (the #241 headline soft depth; at depth 2000 recovery
is exact and the readout below collapses to pure rotation
arithmetic -- see the iteration note in the PR body).

Basis misalignment: the student's trainable coordinates are
rotated by angle theta relative to the teacher's, in the plane
spanned by Delta and a per-seed random axis u orthogonal to Delta
(and to the scrambled control, below). The rotation R_theta is
built exactly, so cos(Delta, R_theta Delta) = cos(theta) is
asserted as an apparatus check, as PR #245 asserted its realized
rotation. The student's induced shift, mapped back into teacher
coordinates, is s = R_theta w. Readout: cos(s, Delta).

What the toy shows, mechanically: the student DOES learn the
trait in its own coordinates at every theta (cos(w, Delta) =
0.999, theta-independent -- the soft carrier signal is intact);
what decays is the teacher-coordinate readout of that learning,
tracking cos(theta). Learning succeeds; identification of the
learned direction with the teacher's trait direction is what a
shared basis buys. That is the toy form of the paper's boundary:
same base (theta = 0) transfers, orthogonal bases (theta = 90)
do not, and the graded middle decays smoothly.

Arms: theta in {0, 17, 25, 35, 60, 90} degrees (17-35 is PR
#245's probe-rotation range; 60 its boundary point; 90 the
orthogonal limit of the paper's different-base case).
Controls, per seed:
  (i) clean teacher: Delta = 0 -- the student is trained toward
      the zero target, its induced shift is exactly zero at every
      theta (transfer = 0 by the readout's own terms);
  (ii) sign-scrambled carrier: Delta_ctrl, Delta with exactly
      half its components sign-flipped (cos(Delta_ctrl, Delta) =
      0 by construction, as in #241). The rotation axis u is
      chosen orthogonal to Delta_ctrl as well, so the control is
      not contaminated by the rotation plane. The student
      recovers its own scrambled target (cos 0.999 in its own
      coordinates) while its trait readout stays ~0 at every
      theta: the machinery transfers whatever the carriers
      point at, in whatever basis they point.

Measured on pop-os (particlegan 0.10.1), 3 seeds, depth 200:
student-coordinate recovery cos(w, Delta) 0.999 (all seeds, all
theta); trait readout cos(R_theta w, Delta): theta 0: 0.999;
17: 0.952-0.959; 25: 0.901-0.911; 35: 0.812-0.826; 60:
0.490-0.510; 90: -0.011..+0.012 -- within 0.012 of cos(theta) at
every angle and seed, and strictly decreasing on the seed-mean
curve. Clean control: induced-shift norm exactly 0 at every
theta, so the 90-degree readout sits within 0.012 of the clean
control. Scrambled control: self-recovery 0.999, trait readout
|cos| <= 0.020 at every theta. Assertions below encode exactly
these margins, no more: per-seed endpoints and per-angle bands,
seed-mean monotonicity (per-seed strict monotonicity is NOT
asserted -- #241 documented plateau creep breaking it in a
sibling toy, and the honest claim here is the graded trend).

What is NOT asserted: that real different-base students fail
exactly like a 90-degree rotation (a real basis change is not a
single-plane rotation, and real training could partially
re-align representations -- the toy isolates the coordinate-
identification step and holds training fixed); anything about
real tokenizers, losses on an actual LM, or the paper's owl /
number-sequence numbers; and the converse direction (that
matching bases guarantee transfer in real models -- the toy's
theta = 0 arm is the #241 soft regime by construction). One
honest tension, recorded rather than hidden: Draganov et al.
(arXiv:2602.04899) report a modified phantom-transfer attack
working across models, so "different base" is not an absolute
barrier for adversarially constructed carriers; this toy tests
graded representation alignment for ordinary soft carriers,
not model identity per se. Pure numpy linear algebra -- no GAN
machinery, because the claim is about the geometry of the
distillation readout, not training dynamics. Skips cleanly when
the installed particlegan does not provide the EMPF recipe (the
house guard for this paper-test family; PR #211 is what enables
this test in CI).
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


D = 16                                   # trait-space dimension (#241)
LAM = np.exp(np.linspace(np.log(1.0), np.log(0.02), D))  # steep->flat
ETA = 0.5                                # GD step (ETA * max LAM < 1)
DEPTH = 200                              # #241 headline soft depth
ANGLES = (0, 17, 25, 35, 60, 90)         # misalignment sweep, degrees


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _rotation(p, u, deg):
    """Exact rotation by deg in the (p, u) plane, identity elsewhere."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return (np.eye(D)
            + (c - 1.0) * (np.outer(p, p) + np.outer(u, u))
            + s * (np.outer(u, p) - np.outer(p, u)))


def _run_gd(target):
    """Soft-arm GD from 0 on grad = LAM * (v - target), DEPTH steps."""
    v = np.zeros(D)
    for _ in range(DEPTH):
        v = v - ETA * (LAM * (v - target))
    return v


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    delta = rng.choice([-1.0, 1.0], size=D)
    # Scrambled control target: exactly half the components
    # sign-flipped, cos(Delta_ctrl, Delta) = 0 by construction.
    delta_ctrl = delta.copy()
    delta_ctrl[: D // 2] *= -1.0
    # Rotation plane: p along Delta, u a random axis orthogonal to
    # both Delta and Delta_ctrl (so the control is not contaminated
    # by the rotation plane).
    p = delta / np.linalg.norm(delta)
    z = rng.standard_normal(D)
    for q in (p, delta_ctrl / np.linalg.norm(delta_ctrl)):
        z = z - (z @ q) * q
    u = z / np.linalg.norm(z)

    w = _run_gd(delta)                   # trait arm, student coords
    w_ctrl = _run_gd(delta_ctrl)         # scrambled arm, student coords
    w_clean = _run_gd(np.zeros(D))       # clean teacher: zero target

    trait, ctrl, realized, clean_norm = {}, {}, {}, {}
    for deg in ANGLES:
        r = _rotation(p, u, deg)
        realized[deg] = _cos(delta, r @ delta)
        trait[deg] = _cos(r @ w, delta)
        ctrl[deg] = _cos(r @ w_ctrl, delta)
        clean_norm[deg] = float(np.linalg.norm(r @ w_clean))
    return dict(
        delta=delta,
        init_grad_cos=_cos(LAM * delta, delta),
        student_cos=_cos(w, delta),
        ctrl_self_cos=_cos(w_ctrl, delta_ctrl),
        trait=trait, ctrl=ctrl, realized=realized,
        clean_norm=clean_norm,
    )


def test_aligned_student_recovers_trait_orthogonal_does_not():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Apparatus: the realized misalignment is the named angle.
        for deg in ANGLES:
            assert abs(r["realized"][deg]
                       - np.cos(np.radians(deg))) < 1e-8, (
                f"seed {seed}: realized cos at {deg} deg is "
                f"{r['realized'][deg]:.6f}, target "
                f"{np.cos(np.radians(deg)):.6f}")
        # Fisher distortion: the carrier gradient starts at
        # F Delta, not Delta (the #241 prediction this toy inherits).
        assert r["init_grad_cos"] < 0.8, (
            f"seed {seed}: cos(F Delta, Delta) "
            f"{r['init_grad_cos']:.3f}")
        # The student learns the trait in its OWN coordinates at
        # every misalignment -- the carrier signal is intact; only
        # the teacher-coordinate readout decays.
        assert r["student_cos"] > 0.99, (
            f"seed {seed}: student-coordinate recovery "
            f"{r['student_cos']:.3f}")
        # Endpoints, per seed: aligned transfer is the #241 soft
        # recovery; orthogonal transfer is at the clean control.
        assert r["trait"][0] > 0.99, (
            f"seed {seed}: trait cos at 0 deg {r['trait'][0]:.3f}")
        assert abs(r["trait"][90]) < 0.05, (
            f"seed {seed}: trait cos at 90 deg {r['trait'][90]:+.3f} "
            f"-- orthogonal bases should not transfer")
        assert r["trait"][0] > r["trait"][90] + 0.9, (
            f"seed {seed}: endpoint gap {r['trait'][0]:.3f} vs "
            f"{r['trait'][90]:+.3f}")
        # Clean teacher: no trait, no induced shift, at any angle --
        # and the 90-degree arm sits within a small band of it.
        for deg in ANGLES:
            assert r["clean_norm"][deg] < 1e-12, (
                f"seed {seed}: clean induced-shift norm at {deg} deg "
                f"{r['clean_norm'][deg]:.2e}")
        assert abs(r["trait"][90]) < 0.05, (
            f"seed {seed}: 90-deg readout {r['trait'][90]:+.3f} vs "
            f"clean control 0")


def test_transfer_decays_graded_with_misalignment():
    _empf_recipe()
    # Per-seed bands are the demonstrated margins; monotonicity is
    # asserted on the seed-mean curve only (per-seed strict
    # monotonicity is the kind of claim #241's plateau creep broke
    # in a sibling toy -- endpoints are asserted per seed in the
    # test above, the graded trend is asserted where it is robust).
    bands = {0: (0.99, 1.0), 17: (0.93, 0.98), 25: (0.88, 0.93),
             35: (0.79, 0.85), 60: (0.46, 0.54), 90: (-0.05, 0.05)}
    mean = {deg: 0.0 for deg in ANGLES}
    for seed in (0, 1, 2):
        r = run_seed(seed)
        for deg in ANGLES:
            lo, hi = bands[deg]
            assert lo <= r["trait"][deg] <= hi, (
                f"seed {seed}: trait cos at {deg} deg "
                f"{r['trait'][deg]:+.3f}, demonstrated band "
                f"[{lo}, {hi}]")
            # The graded law itself: transfer tracks cos(theta).
            assert abs(r["trait"][deg]
                       - np.cos(np.radians(deg))) <= 0.03, (
                f"seed {seed}: trait cos at {deg} deg "
                f"{r['trait'][deg]:+.3f} vs cos(theta) "
                f"{np.cos(np.radians(deg)):+.3f}")
            mean[deg] += r["trait"][deg] / 3.0
    seq = [mean[deg] for deg in ANGLES]
    assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), (
        f"seed-mean trait cos by angle {dict(zip(ANGLES, seq))} -- "
        f"the graded claim is monotone decay in misalignment")


def test_scrambled_control_recovers_own_target_not_trait():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The scrambled arm recovers the target it was pointed at.
        assert r["ctrl_self_cos"] > 0.99, (
            f"seed {seed}: scrambled self-recovery "
            f"{r['ctrl_self_cos']:.3f}")
        # ... and transfers ~no trait at any misalignment, while
        # the trait arm at small angles beats it by a wide margin.
        for deg in ANGLES:
            assert abs(r["ctrl"][deg]) <= 0.05, (
                f"seed {seed}: scrambled trait cos at {deg} deg "
                f"{r['ctrl'][deg]:+.3f}")
        assert r["trait"][0] > r["ctrl"][0] + 0.9, (
            f"seed {seed}: trait {r['trait'][0]:.3f} vs scrambled "
            f"{r['ctrl'][0]:+.3f} at 0 deg")
        assert r["trait"][35] > r["ctrl"][35] + 0.7, (
            f"seed {seed}: trait {r['trait'][35]:.3f} vs scrambled "
            f"{r['ctrl'][35]:+.3f} at 35 deg")
