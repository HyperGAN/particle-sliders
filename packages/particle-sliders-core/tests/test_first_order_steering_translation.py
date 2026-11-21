"""CPU gate for the first-order steering translation claim
(arXiv:2610.04283): a weight adaptation's behavioral effect is, to
first order, the effect of adding a steering vector built from the
adaptation itself -- the mean Jacobian times the weight delta -- and
that translated vector reproduces the adaptation (and composes
multiple adaptations) more faithfully than a contrast-harvested
direction.

Paper: "First-Order Steering: Translating Weight Adaptation into
Activation Steering" (Kunda et al.). A weight change dW moves the
hidden state at input x, to first order, by J(x) dW x, where J is the
Jacobian of the hidden state with respect to the pre-activation.
Averaging over the prompt distribution gives a single steering vector
v = E[J(x) dW x] whose additive effect approximates the adaptation's
mean effect, with an error that is second order in the adaptation
scale. Empirically (Table 1 of the paper), first-order translations
of HeRD-merged adaptations reach mean behavioral adherence 46.4 and
plain task-arithmetic translations 42.6, against 27.4 for CAA and
30.3 for Linear-AcT contrast-harvested directions.

Toy operationalization: one nonlinear layer, h = tanh(W x), with
inputs x ~ N(m, SIGMA^2 I) in R^16 (nonzero mean m, |m| = 1.5 -- a
prompt distribution concentrated away from the origin, as real
prompt activations are) and h in R^20. The nonlinearity is the
point: in a purely linear toy the translation of the MEAN effect is
exact at every scale and there is no failure boundary to measure;
tanh saturation is what makes the first-order error grow.

An adaptation is a rank-1 dW constructed in closed form so that its
first-order mean effect is a prescribed displacement
d = BETA * (r + 0.35 * s): mostly the behavior readout direction r
(a unit vector; behavior score = r . h), plus a specific off-behavior
signature along a second unit vector s orthogonal to r. The
signature is what separates the arms: a faithful translation must
reproduce not just the behavior shift but the adaptation's own
side-effect profile. Concretely dW = a m^T / |m|^2 with
a = d / Jbar (elementwise), Jbar the mean of diag(tanh'(W x)) over
train inputs, so E[J dW x] ~= d up to the J--x covariance.

Arms, all evaluated on held-out inputs (translation and harvesting
use train inputs only):
  (a) weights:    h_a(x) = tanh((W + k dW) x), k the scale sweep.
  (b) translated: h + v, v = mean over train x of J(x) (k dW) x --
                  used exactly as constructed, never rescaled.
  (c) harvested:  the CAA analogue. Behavior-on/off contexts shift
                  inputs by +-0.5 along the behavior's input-gradient
                  direction; the harvested vector is the mean hidden
                  difference, rescaled so its behavior projection
                  matches arm (a)'s mean behavior shift (the most
                  favorable calibration a harvested direction gets:
                  its behavior magnitude is handed to it).
  (d) random:     one fixed random unit direction per seed, rescaled
                  the same favorable way as (c).

Metrics per seed:
  match error   ||mean displacement of the arm - mean displacement
                of (a)|| / ||mean displacement of (a)||. For steering
                arms the displacement is the constant vector itself.
  behavior      r . mean displacement; side effect: the norm of the
                mean displacement orthogonal to r.
  per-input RMS ||d h_a(x) - v|| relative to ||d h_a(x)||: a constant
                vector cannot follow the per-input variation of
                dW x, so this has an irreducible dispersion floor
                (~0.27 here, SIGMA / |m|); it is reported and bounded,
                NOT asserted as a match -- the paper's claim is about
                the mean behavioral effect.

Scale sweep: k in {0.25, 0.5, 1, 2, 4, 8} multiplying dW. The
translated match error grows monotonically from ~0.01-0.02 at
k = 0.25 to 0.57-0.73 at k = 8, where the translated behavior shift
overshoots the weight arm's by 55-76% (steering keeps adding
linearly while tanh saturates). That growth -- not a cliff at a
magic scale -- is the measured first-order failure boundary.

Composition: two adaptations built for two orthogonal behavior
readouts r, r2 (each with its own 0.35 side signature). Merged
weights (W + dW_a + dW_b) vs the summed translations v_a + v_b vs
the summed harvested directions (each harvested and rescaled to its
own target behavior, then added -- cross-talk included).

Measured on pop-os (particlegan 0.10.1), seeds 0-2, k = 1 unless
noted: translated match error 0.039-0.056 vs harvested 0.964-1.065
and random 2.42-7.03; cos(translated, weight mean effect) 0.999 vs
cos(harvested, ...) 0.59-0.72; translated behavior shift within
1.5% of the weight arm's and its side-effect norm within 4%, while
the harvested side effect is 2.3-3.0x the weight arm's; scale sweep
and composition as above (composition match error 0.056-0.092
translated vs 0.831-1.393 harvested). Assertions below encode
exactly these margins, no more.

What is NOT asserted: that translation beats harvesting on
per-input fidelity (both are constant vectors; the dispersion floor
binds both); that the translated vector has smaller ABSOLUTE side
effects than a harvested one in general (it faithfully reproduces
the adaptation's own side signature, which is the claim); anything
about multi-layer Jacobians, attention, or token positions -- the
toy has one layer and one pooled hidden state; the paper's absolute
adherence numbers (46.4 / 42.6 / 27.4 / 30.3), which are properties
of its models and evals, not of this geometry.

Skips cleanly when the installed particlegan is not 0.10.1 (the
house guard for this test family; PR #211 enables the test in CI).
Pure numpy state geometry -- no GAN machinery, because the claim is
about what a weight delta does to activations, not about training.
"""
import importlib.metadata

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


def _require_particlegan_0101():
    try:
        version = importlib.metadata.version("particlegan")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("particlegan is not installed")
    if version != "0.10.1":
        pytest.skip(f"particlegan {version} != 0.10.1")
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")


N_IN = 16         # input dimension
D = 20            # hidden dimension
SIGMA = 0.4       # input dispersion around the mean prompt m
NTR = 2000        # train inputs (translation / harvesting only)
NTE = 2000        # held-out evaluation inputs
BETA = 0.6        # prescribed behavior displacement at scale k = 1
SIDE = 0.35       # off-behavior signature, relative to BETA along r
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _unit(x):
    return x / np.linalg.norm(x)


def _phi(u):
    return np.tanh(u)


def _dphi(u):
    return 1.0 - np.tanh(u) ** 2


class _World:
    """One seed's layer, prompt distribution, readouts, and arms."""

    def __init__(self, seed):
        rng = np.random.default_rng(7000 + seed)
        self.W = rng.standard_normal((D, N_IN)) * (0.6 / np.sqrt(N_IN))
        self.m = _unit(rng.standard_normal(N_IN)) * 1.5
        q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
        self.r, self.s = q_basis[:, 0], q_basis[:, 1]
        self.r2, self.s2 = q_basis[:, 2], q_basis[:, 3]
        rng_rand = np.random.default_rng(55000 + seed)
        self.u_rand = _unit(rng_rand.standard_normal(D))
        self.x_tr = self.m + SIGMA * rng.standard_normal((NTR, N_IN))
        self.x_te = self.m + SIGMA * rng.standard_normal((NTE, N_IN))
        self.h_te = _phi(self.x_te @ self.W.T)
        self.jbar = _dphi(self.x_tr @ self.W.T).mean(0)
        self.j_tr = _dphi(self.x_tr @ self.W.T)

    def make_dw(self, desired):
        """Rank-1 dW whose first-order mean effect is ~desired."""
        a = desired / self.jbar
        return np.outer(a, self.m) / (self.m @ self.m)

    def translate(self, dw):
        """v = mean over train inputs of J(x) dW x."""
        return (self.j_tr * (self.x_tr @ dw.T)).mean(0)

    def harvest(self, readout):
        """Mean hidden difference, behavior-on minus behavior-off."""
        g = _unit(self.W.T @ (readout * _dphi(self.m @ self.W.T)))
        on = _phi((self.x_tr + 0.5 * g) @ self.W.T).mean(0)
        off = _phi((self.x_tr - 0.5 * g) @ self.W.T).mean(0)
        return on - off

    def weight_mean_effect(self, dw):
        return (_phi(self.x_te @ (self.W + dw).T) - self.h_te).mean(0)

    def arms(self, dw):
        """All four arms for one (already scaled) adaptation dw."""
        mean_a = self.weight_mean_effect(dw)
        norm = np.linalg.norm(mean_a)
        beh_a = float(mean_a @ self.r)
        v = self.translate(dw)
        v_harv_raw = self.harvest(self.r)
        v_harv = (beh_a / float(v_harv_raw @ self.r)) * v_harv_raw
        v_rand = (beh_a / float(self.u_rand @ self.r)) * self.u_rand

        def side(vec):
            return float(np.linalg.norm(vec - (vec @ self.r) * self.r))

        dh_a = _phi(self.x_te @ (self.W + dw).T) - self.h_te
        rms = float(np.sqrt(((dh_a - v) ** 2).sum(1).mean())
                    / np.sqrt((dh_a ** 2).sum(1).mean()))
        return dict(
            err_t=float(np.linalg.norm(v - mean_a) / norm),
            err_h=float(np.linalg.norm(v_harv - mean_a) / norm),
            err_r=float(np.linalg.norm(v_rand - mean_a) / norm),
            cos_t=float(_unit(v) @ _unit(mean_a)),
            cos_h=float(_unit(v_harv) @ _unit(mean_a)),
            beh_a=beh_a, beh_t=float(v @ self.r),
            side_a=side(mean_a), side_t=side(v), side_h=side(v_harv),
            rms=rms,
        )


def test_first_order_translation_matches_weight_adaptation():
    _require_particlegan_0101()
    for seed in (0, 1, 2):
        w = _World(seed)
        dw = w.make_dw(BETA * (w.r + SIDE * w.s))
        a = w.arms(dw)

        # Trap guard: the adaptation must genuinely move the behavior.
        assert a["beh_a"] > 0.4, (
            f"seed {seed}: weight arm behavior shift {a['beh_a']:.3f}")

        # The translated vector reproduces the weight arm's mean
        # effect -- behavior, side signature, and direction.
        assert a["err_t"] < 0.10, (
            f"seed {seed}: translated match error {a['err_t']:.3f}")
        assert a["cos_t"] > 0.99, (
            f"seed {seed}: cos(translated, weight effect) "
            f"{a['cos_t']:.3f}")
        assert abs(a["beh_t"] - a["beh_a"]) < 0.05 * a["beh_a"], (
            f"seed {seed}: behavior translated {a['beh_t']:.3f} vs "
            f"weights {a['beh_a']:.3f}")
        assert abs(a["side_t"] - a["side_a"]) < 0.15 * a["side_a"], (
            f"seed {seed}: side effect translated {a['side_t']:.3f} "
            f"vs weights {a['side_a']:.3f}")

        # The harvested direction, even rescaled to the exact target
        # behavior shift, does not reproduce the adaptation: its own
        # side displacement is 2.3-3.0x the weight arm's and its
        # direction is far off the adaptation's effect.
        assert a["err_h"] > 0.8, (
            f"seed {seed}: harvested match error {a['err_h']:.3f}")
        assert a["err_t"] + 0.5 < a["err_h"], (
            f"seed {seed}: match error translated {a['err_t']:.3f} "
            f"vs harvested {a['err_h']:.3f}")
        assert a["cos_h"] < 0.8, (
            f"seed {seed}: cos(harvested, weight effect) "
            f"{a['cos_h']:.3f}")
        assert a["side_h"] > 1.8 * a["side_a"], (
            f"seed {seed}: harvested side effect {a['side_h']:.3f} "
            f"vs weights {a['side_a']:.3f}")

        # Random control: rescaling a random direction to the target
        # behavior shift leaves a displacement dominated by noise.
        assert a["err_r"] > 1.5 and a["err_r"] > a["err_t"] + 1.0, (
            f"seed {seed}: random match error {a['err_r']:.3f} vs "
            f"translated {a['err_t']:.3f}")

        # Per-input fidelity is bounded by the dispersion floor for
        # ANY constant vector; documented, not asserted as a match.
        assert 0.15 < a["rms"] < 0.4, (
            f"seed {seed}: per-input RMS ratio {a['rms']:.3f}")


def test_translation_error_grows_with_adaptation_scale():
    _require_particlegan_0101()
    for seed in (0, 1, 2):
        w = _World(seed)
        dw = w.make_dw(BETA * (w.r + SIDE * w.s))
        sweep = {k: w.arms(dw * k) for k in SCALES}
        errs = [sweep[k]["err_t"] for k in SCALES]

        # First-order regime: at small scale the translation is
        # nearly exact for the mean effect.
        assert sweep[0.25]["err_t"] < 0.05, (
            f"seed {seed}: match error at k=0.25 "
            f"{sweep[0.25]['err_t']:.3f}")

        # The error grows monotonically with adaptation scale -- the
        # measured failure boundary is this growth, not a cliff.
        assert all(b > a for a, b in zip(errs, errs[1:])), (
            f"seed {seed}: translated match errors by scale "
            f"{[round(e, 3) for e in errs]} not increasing")
        assert sweep[8.0]["err_t"] > 0.45, (
            f"seed {seed}: match error at k=8 "
            f"{sweep[8.0]['err_t']:.3f}")

        # Mechanism of the failure: steering keeps adding linearly
        # while the weight arm saturates in tanh, so the translated
        # behavior shift overshoots the weight arm's at large scale.
        assert sweep[8.0]["beh_t"] > 1.4 * sweep[8.0]["beh_a"], (
            f"seed {seed}: behavior at k=8 translated "
            f"{sweep[8.0]['beh_t']:.3f} vs weights "
            f"{sweep[8.0]['beh_a']:.3f}")
        assert sweep[8.0]["rms"] > sweep[1.0]["rms"] + 0.2, (
            f"seed {seed}: per-input RMS at k=8 {sweep[8.0]['rms']:.3f} "
            f"vs k=1 {sweep[1.0]['rms']:.3f}")

        # Degraded is not useless: even at k=8 the translation stays
        # closer to the weight arm than the harvested direction.
        assert sweep[8.0]["err_t"] < sweep[8.0]["err_h"], (
            f"seed {seed}: match error at k=8 translated "
            f"{sweep[8.0]['err_t']:.3f} vs harvested "
            f"{sweep[8.0]['err_h']:.3f}")


def test_translated_vectors_compose_better_than_harvested():
    _require_particlegan_0101()
    for seed in (0, 1, 2):
        w = _World(seed)
        dw_a = w.make_dw(BETA * (w.r + SIDE * w.s))
        dw_b = w.make_dw(BETA * (w.r2 + SIDE * w.s2))
        mean_merged = w.weight_mean_effect(dw_a + dw_b)
        norm = np.linalg.norm(mean_merged)

        v_comp = w.translate(dw_a) + w.translate(dw_b)
        err_t = float(np.linalg.norm(v_comp - mean_merged) / norm)

        # Each harvested direction rescaled to its own target
        # behavior (the merged arm's projection), then summed.
        vh_a_raw, vh_b_raw = w.harvest(w.r), w.harvest(w.r2)
        vh_comp = ((float(mean_merged @ w.r) / float(vh_a_raw @ w.r))
                   * vh_a_raw
                   + (float(mean_merged @ w.r2)
                      / float(vh_b_raw @ w.r2)) * vh_b_raw)
        err_h = float(np.linalg.norm(vh_comp - mean_merged) / norm)

        assert err_t < 0.15, (
            f"seed {seed}: composed translated match error {err_t:.3f}")
        assert err_h > 0.7, (
            f"seed {seed}: composed harvested match error {err_h:.3f}")
        assert err_t + 0.5 < err_h, (
            f"seed {seed}: composition match error translated "
            f"{err_t:.3f} vs harvested {err_h:.3f}")

        # Both behaviors land together under the summed translation.
        for readout in (w.r, w.r2):
            assert abs(float(v_comp @ readout)
                       - float(mean_merged @ readout)) < 0.06, (
                f"seed {seed}: composed behavior translated "
                f"{float(v_comp @ readout):.3f} vs merged weights "
                f"{float(mean_merged @ readout):.3f}")
