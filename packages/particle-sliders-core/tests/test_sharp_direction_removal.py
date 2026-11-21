"""CPU gate for the Sharp Removal claim (arXiv:2610.16084): a CAA
steering vector carries only a few percent of its squared norm in the
sharp subspace of the steering Hessian, yet that small component
drives a large share of sign inversions; projecting the sharp
subspace out (Sharp Removal) preserves the vector's direction and
magnitude, cuts its directional curvature several-fold, and restores
clean steering -- while projecting out a matched random subspace
does not.

Paper: "The Sharp Directions Are Against You: Curvature Analysis of
Activation Steering" (Azizi, Fayyazi, Razmara, Baghaei Potraghloo,
Kundu & Pedram). It defines the steering Hessian -- the second
derivative of the behavioral loss with respect to the steering vector
-- whose top eigenvectors are sharp directions: tiny perturbations
along them cause large, erratic, input-dependent behavioral swings.
Abstract-verified findings asserted here in toy form: the sharp
component is only 1-5% of the CAA vector's squared norm; steering
along the sharp component alone anti-steers in 53-100% of 36
model-task-layer conditions and CAA's sign inverts on 30-50% of test
inputs; Sharp Removal (project out the Lanczos top-k eigenvectors,
k ~ 15) keeps cos(cleaned, original) > 0.974 with norm reduction
< 3%, cuts directional curvature by up to 12x (median 4x), and beats
plain CAA in all 25 open-ended generation conditions. The paper's
exact model/task point values are NOT asserted -- only the mechanism
and its margin structure, in a toy where the Hessian is exact.

Toy operationalization: work directly in the eigenbasis of a
quadratic behavioral loss over steering vectors in R^64, so the
steering Hessian is H = diag(lam) exactly (for a quadratic loss the
Hessian is constant and cheap -- no Lanczos needed; Lanczos is the
paper's estimator for real models, the toy uses the exact spectrum).
The top K = 5 eigenvalues are sharp (lam ~ U(120, 200)); the other 59
are flat (lam ~ U(0.5, 2)). The ground-truth behavior axis b lies in
the flat subspace. The CAA-like direction is built with a KNOWN sharp
share s ~ U(0.02, 0.05) of its squared norm:

    v = sqrt(1 - s) * b + sqrt(s / K) * sum_j e_j   (e_j sharp axes)

Per-input behavior effect of a unit steering step along a direction
with eigen-coordinates c, on held-out input i:

    effect_i(c) = a_i * (c . b)  +  sum_j c_j * z_ij,
    z_ij = -GAMMA + SIGMA * xi_ij,   xi_ij ~ N(0, 1),

with a_i ~ U(0.85, 1.15) a per-input intended gain (always positive),
GAMMA = 2 a systematic anti-steer bias of the sharp couplings and
SIGMA = 6 their erratic input-dependent scale. That coupling IS the
toy form of the paper's mechanism: sharp directions are precisely
those along which behavior responds with outsized, erratic,
sign-unstable sensitivity (that is what the large Hessian eigenvalue
means), here with the paper's observed negative tilt. An input is
INVERTED when effect_i < 0 -- the steering step moved behavior the
wrong way, the paper's sign-inversion event.

Arms (all built from the same v):
  (a) raw:       v as built;
  (b) cleaned:   v - P_S v, P_S the projector onto the top-K sharp
      subspace (Sharp Removal, exact projector in the toy);
  (c) random:    v - P_R v, P_R the projector onto a random K-dim
      subspace (matched-rank control: any projection removes norm
      and a little of everything; only the sharp one should fix
      steering);
  (d) sharp only: P_S v / |P_S v| -- steering along the sharp
      component alone.

Metrics per seed: sharp share of squared norm; cos(cleaned, raw) and
norm reduction; directional curvature v^T H v / v^T v before/after
(ratio, for arms (b) and (c)); inversion rate per arm on N = 2000
held-out inputs; anti-steer rate of arm (d) (fraction of inputs with
effect < 0).

Measured on pop-os (particlegan 0.10.1): sharp share 0.025-0.033;
cos(cleaned, raw) 0.983-0.987, norm reduction 1.3-1.7%; curvature
ratio raw/cleaned 4.43-5.53x vs raw/random 0.66-0.88x (no cut -- the
random projection slightly RAISES the Rayleigh quotient by removing
flat mass); inversion rate raw 0.390-0.436, cleaned 0.000, random
0.427-0.482; sharp-alone anti-steer rate 0.756-0.770. Assertions below encode
exactly these margins, no more. Skips cleanly when the installed
particlegan does not provide the EMPF recipe. Pure numpy/torch
geometry -- no GAN machinery, because the claim is about harvested
directions, not training; particlegan is only the skip gate.
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


D = 64            # steering-vector dimension (Hessian eigenbasis)
K = 5             # sharp subspace rank (paper uses k ~ 15 on real models)
N = 2000          # held-out inputs per seed
LAM_SHARP = (120.0, 200.0)   # sharp Hessian eigenvalues
LAM_FLAT = (0.5, 2.0)        # flat Hessian eigenvalues
SHARE = (0.02, 0.05)         # known sharp share of squared norm (paper: 1-5%)
GAMMA = 2.0       # systematic anti-steer bias of sharp couplings
SIGMA = 6.0       # erratic input-dependent scale of sharp couplings


def _unit(x):
    return x / np.linalg.norm(x)


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    lam = np.empty(D)
    lam[:K] = rng.uniform(*LAM_SHARP, K)
    lam[K:] = rng.uniform(*LAM_FLAT, D - K)

    # Ground-truth behavior axis: a unit vector in the flat subspace.
    b = np.zeros(D)
    b[K:] = rng.standard_normal(D - K)
    b = _unit(b)

    # CAA-like direction with a known sharp share s of squared norm.
    s = float(rng.uniform(*SHARE))
    c = np.zeros(D)
    c[:] = np.sqrt(1.0 - s) * b
    c[:K] += np.sqrt(s / K)
    share = float(np.sum(c[:K] ** 2) / np.sum(c ** 2))

    # Arms, in eigen-coordinates.
    c_clean = c.copy()
    c_clean[:K] = 0.0                                   # (b) Sharp Removal
    rot, _ = np.linalg.qr(rng.standard_normal((D, K)))
    c_rand = c - rot @ (rot.T @ c)                      # (c) random projection
    c_sharp = np.zeros(D)
    c_sharp[:K] = c[:K]
    c_sharp = _unit(c_sharp)                            # (d) sharp alone

    def rayleigh(cc):
        return float(np.sum(lam * cc ** 2) / np.sum(cc ** 2))

    cos_clean = float(c_clean @ c
                      / (np.linalg.norm(c_clean) * np.linalg.norm(c)))
    norm_red = 1.0 - float(np.linalg.norm(c_clean) / np.linalg.norm(c))

    # Held-out per-input effects.
    a = rng.uniform(0.85, 1.15, N)
    xi = rng.standard_normal((N, K))
    z = -GAMMA + SIGMA * xi                             # sharp couplings

    def inversion_rate(cc):
        effect = a * float(cc @ b) + z @ cc[:K]
        return float(np.mean(effect < 0))

    return dict(
        share=share,
        cos_clean=cos_clean, norm_red=norm_red,
        curv_ratio=rayleigh(c) / rayleigh(c_clean),
        curv_ratio_rand=rayleigh(c) / rayleigh(c_rand),
        inv_raw=inversion_rate(c),
        inv_clean=inversion_rate(c_clean),
        inv_rand=inversion_rate(c_rand),
        anti_sharp=inversion_rate(c_sharp),
    )


def test_sharp_removal_restores_clean_steering():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Trap guard: the sharp share must be in the paper's 1-5%
        # band -- too small and nothing can invert, too large and
        # removal is no longer near-norm-preserving.
        assert 0.01 <= r["share"] <= 0.06, (
            f"seed {seed}: sharp share {r['share']:.3f} outside the "
            f"paper's 1-5% band -- the toy tests nothing")
        # Removal approximately preserves direction and magnitude
        # (paper: cos > 0.974, norm reduction < 3%).
        assert r["cos_clean"] > 0.97, (
            f"seed {seed}: cos(cleaned, raw) {r['cos_clean']:.3f}")
        assert r["norm_red"] < 0.03, (
            f"seed {seed}: norm reduction {r['norm_red']:.3f}")
        # Directional curvature cut by a clear factor (paper: up to
        # 12x, median 4x across 36 conditions).
        assert r["curv_ratio"] >= 3.0, (
            f"seed {seed}: curvature ratio raw/cleaned "
            f"{r['curv_ratio']:.2f}")
        # The raw direction inverts on a large share of inputs
        # (paper: sign inversion on 30-50% of test inputs).
        assert r["inv_raw"] > 0.30, (
            f"seed {seed}: raw inversion rate {r['inv_raw']:.3f}")
        # Removal restores clean steering: inversions essentially
        # vanish, a wide margin below the raw rate.
        assert r["inv_clean"] < 0.05, (
            f"seed {seed}: cleaned inversion rate {r['inv_clean']:.3f}")
        assert r["inv_clean"] < r["inv_raw"] - 0.25, (
            f"seed {seed}: inversion cleaned {r['inv_clean']:.3f} vs "
            f"raw {r['inv_raw']:.3f}")


def test_sharp_alone_antisteers_and_random_projection_does_not_fix():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # The sharp component alone anti-steers on most inputs
        # (paper: anti-steer rates 53-100% across 36 conditions).
        assert r["anti_sharp"] > 0.6, (
            f"seed {seed}: sharp-alone anti-steer rate "
            f"{r['anti_sharp']:.3f}")
        # Matched-rank random projection does NOT fix steering: its
        # inversion rate stays near the raw rate and far above the
        # Sharp Removal rate.
        assert r["inv_rand"] > r["inv_raw"] - 0.10, (
            f"seed {seed}: random-projection inversion "
            f"{r['inv_rand']:.3f} vs raw {r['inv_raw']:.3f} -- a "
            f"generic projection should not remove the inversions")
        assert r["inv_rand"] > r["inv_clean"] + 0.25, (
            f"seed {seed}: random-projection inversion "
            f"{r['inv_rand']:.3f} vs cleaned {r['inv_clean']:.3f}")
        # Nor does it cut directional curvature comparably.
        assert r["curv_ratio_rand"] < 2.0, (
            f"seed {seed}: random-projection curvature ratio "
            f"{r['curv_ratio_rand']:.2f}")
        assert r["curv_ratio_rand"] < r["curv_ratio"] - 2.0, (
            f"seed {seed}: curvature ratio random "
            f"{r['curv_ratio_rand']:.2f} vs sharp removal "
            f"{r['curv_ratio']:.2f}")
