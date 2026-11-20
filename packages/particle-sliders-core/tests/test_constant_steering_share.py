"""CPU gate for the constant-steering-share claim (arXiv:2611.26627):
most of an RL post-training gain is recoverable by a single CONSTANT
(input-independent) steering vector fitted by output matching alone,
and the recoverable share falls as the true improvement becomes
genuinely input-dependent.

Paper: "RL Post-Training in Diffusion Transformers Is Activation
Steering" (Meral et al.). The paper fits, for an RL-adapted model, a
single constant vector added at a block output, trained ONLY to
match the adapted model's outputs (no reward in the objective), and
finds the tiny vector recovers most of the RL gain: 0.29M-param
vectors vs the RL LoRA's 18.8M parameters recover 90.0-96.3% of
reward gains on six image models and 89.8% / 111.9% on two video
models. The decomposition claim under test here: an adaptation's
gain splits into a constant share plus an input-dependent residual;
the constant share is exactly what output matching can recover, so
recovery must track the constant share of the gain and collapse
when the gain is genuinely input-dependent.

Toy operationalization: a frozen base map f (small random MLP,
R^12 -> R^10) and an adapted counterpart f'(x) = f(x) + Delta(x)
with the improvement decomposed by VARIANCE SHARE,

    Delta(x) = sqrt(1 - lam) * c + sqrt(lam) * g(x),

  c -- a fixed vector (the constant share);
  g -- a genuinely input-dependent improvement, g(x) = W tanh(U x)
       recentred to zero mean over the fit distribution and
       rescaled so E||g||^2 = ||c||^2 (matched power: lam IS the
       share of improvement variance that is input-dependent);
  lam in {0, 0.25, 0.5, 0.75, 1.0}.

Task loss is the squared distance to the adapted output,
L(y, x) = ||y - f'(x)||^2, so the full adaptation's gain over the
base is gain(f') = E||Delta||^2 on held-out inputs, and a constant
shift v gains gain(f + v) = E||Delta||^2 - E||Delta - v||^2.

Arms (fit on a fit split, scored on a disjoint held-out split):
  (a) full adaptation f' (upper bound, recovery 1.0 by
      construction);
  (b) constant shift: the single vector v minimising the
      output-matching error E_fit ||f(x) + v - f'(x)||^2 -- the
      paper's fit, no reward / loss in the objective. The exact
      minimiser is v = mean_fit Delta; recovery =
      gain(f + v) / gain(f');
  (c) random constant vector at matched norm ||v|| (control: a
      constant shift of the right size in a wrong direction must
      NOT recover the gain);
  (d) zero shift (recovery 0 by construction).

Metrics per lam, >= 3 seeds: recovery fraction for arms (b) and
(c). Expected shape: recovery(b) ~= 1 - lam -- near 1 at lam = 0,
monotone decreasing, materially below 1 (near 0) at lam = 1 --
while recovery(c) stays near or below 0 at every lam where the
fitted vector carries signal.

A second test repeats the decomposition one layer down, the
paper's actual placement: the adapted model shifts the hidden
pre-activation by sqrt(1 - lam) * c_h + sqrt(lam) * g_h(x), and
arm (b) is a single constant hidden vector fitted by gradient
descent on the same output-matching objective.

Measured on pop-os (particlegan 0.10.1), 3 seeds: OUTPUT
placement recovery(b) 1.000 (lam = 0), 0.749-0.750 (0.25),
0.498-0.500 (0.5), 0.247-0.250 (0.75), 0.000 (1.0) -- tracking
the constant share 1 - lam to within 0.003 on every seed;
recovery(c) -2.75 to -0.46 at lam <= 0.5 (a matched-norm random
constant actively hurts) and 0.000 at lam = 1 (the fitted norm
itself vanishes: there is no constant component to match).
BLOCK placement recovery(b) 1.000 (lam = 0), 0.758-0.843 (0.25),
0.538-0.646 (0.5), 0.340-0.427 (0.75), 0.103-0.250 (1.0) --
(combined spread over two driver runs; the gradient fit's float
reduction order wiggles block numbers by ~0.01 between runs)
ABOVE the output-space share at intermediate lam, because a
hidden constant's output effect is itself input-dependent
through the tanh, yet still collapsing toward 0 at lam = 1;
recovery(c) +0.34 in the worst seed at lam = 0 and <= -0.06
everywhere else. Assertions below encode exactly these
margins, no more.

What is NOT asserted: the paper's second claim -- that vectors
trained with online RL can EXCEED weight-based RL on OCR /
PickScore -- is a training-dynamics claim about optimising the
vector against reward; this toy has no RL training loop and
cannot express it, so it is noted-not-tested rather than forced
into an arm. The paper's diffusion-transformer benchmark numbers
themselves (the 90.0-96.3% / 89.8% / 111.9% recoveries, parameter
counts) are likewise not asserted: the toy has no diffusion model
and no reward model, only the decomposition structure -- recovery
tracks the constant variance share, is recovered by output
matching alone, and collapses under a genuinely input-dependent
improvement while a matched-norm random constant does not
substitute.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family;
PR #211 is what enables this test in CI). Pure CPU toy -- no GAN
machinery, because the claim is about the structure of an
adaptation's gain, not training dynamics.
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


D_IN = 12
D_HID = 32
D_OUT = 10
N_FIT = 4000
N_TEST = 4000
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
POWER = float(D_OUT)  # ||c||^2 = E||g||^2 = POWER (matched power)


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)
    # Frozen base MLP f and the input-dependent improvement map.
    w1 = rng.standard_normal((D_HID, D_IN)) / np.sqrt(D_IN)
    b1 = 0.1 * rng.standard_normal(D_HID)
    w2 = rng.standard_normal((D_OUT, D_HID)) / np.sqrt(D_HID)
    b2 = 0.1 * rng.standard_normal(D_OUT)
    ug = rng.standard_normal((D_HID, D_IN)) / np.sqrt(D_IN)
    wg = rng.standard_normal((D_OUT, D_HID)) / np.sqrt(D_HID)
    c = rng.standard_normal(D_OUT)
    c *= np.sqrt(POWER) / np.linalg.norm(c)

    def f(x):
        return np.tanh(x @ w1.T + b1) @ w2.T + b2

    def g_raw(x):
        return np.tanh(x @ ug.T) @ wg.T

    x_fit = rng.standard_normal((N_FIT, D_IN))
    x_test = rng.standard_normal((N_TEST, D_IN))
    # Centre g on the fit distribution and match its power to c,
    # using ONE scale factor (from the fit split) for both splits.
    g_mean = g_raw(x_fit).mean(0)
    scale = np.sqrt(POWER / np.mean(
        np.sum((g_raw(x_fit) - g_mean) ** 2, axis=1)))
    g_fit = (g_raw(x_fit) - g_mean) * scale
    g_test = (g_raw(x_test) - g_mean) * scale

    out = {}
    for lam in LAMBDAS:
        d_fit = np.sqrt(1.0 - lam) * c + np.sqrt(lam) * g_fit
        d_test = np.sqrt(1.0 - lam) * c + np.sqrt(lam) * g_test
        # Arm (b): exact output-matching minimiser, fit split only.
        v = d_fit.mean(0)
        # Arm (c): random constant at matched norm.
        rdir = rng.standard_normal(D_OUT)
        v_rand = rdir / np.linalg.norm(rdir) * np.linalg.norm(v)
        gain_full = float(np.mean(np.sum(d_test ** 2, axis=1)))
        gain_b = gain_full - float(
            np.mean(np.sum((d_test - v) ** 2, axis=1)))
        gain_c = gain_full - float(
            np.mean(np.sum((d_test - v_rand) ** 2, axis=1)))
        out[lam] = dict(
            rec_b=gain_b / gain_full,
            rec_c=gain_c / gain_full,
            v=v,
            mean_test=d_test.mean(0),
            gain_full=gain_full,
        )
    out["c"] = c
    return out


def test_constant_share_recovery_tracks_lambda():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        rec = {lam: r[lam]["rec_b"] for lam in LAMBDAS}
        # Purely constant gain is (near-)fully recovered by output
        # matching alone.
        assert rec[0.0] > 0.95, (
            f"seed {seed}: recovery at lam=0 {rec[0.0]:.3f}")
        # Recovery tracks the constant variance share 1 - lam.
        for lam in LAMBDAS:
            assert abs(rec[lam] - (1.0 - lam)) < 0.10, (
                f"seed {seed}: recovery at lam={lam} {rec[lam]:.3f} "
                f"vs constant share {1.0 - lam:.2f}")
        # Monotone decreasing with clear steps.
        for lo, hi in zip(LAMBDAS, LAMBDAS[1:]):
            assert rec[lo] > rec[hi] + 0.15, (
                f"seed {seed}: recovery lam={lo} {rec[lo]:.3f} vs "
                f"lam={hi} {rec[hi]:.3f} -- not clearly decreasing")
        # A genuinely input-dependent gain is NOT recovered by any
        # constant vector.
        assert rec[1.0] < 0.10, (
            f"seed {seed}: recovery at lam=1 {rec[1.0]:.3f}")
        # Matched-norm random control never substitutes for the
        # fitted vector where the fitted vector carries signal.
        for lam in (0.0, 0.25, 0.5):
            assert r[lam]["rec_c"] < 0.05, (
                f"seed {seed}: random recovery at lam={lam} "
                f"{r[lam]['rec_c']:.3f}")
            assert rec[lam] > r[lam]["rec_c"] + 0.60, (
                f"seed {seed}: fitted {rec[lam]:.3f} vs random "
                f"{r[lam]['rec_c']:.3f} at lam={lam}")


def test_output_matching_fit_recovers_mean_shift():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        c = r["c"]
        for lam in LAMBDAS:
            v, mean_test = r[lam]["v"], r[lam]["mean_test"]
            # The fitted vector is the mean improvement: where a
            # constant share exists it tracks the held-out mean
            # shift it never saw; at lam = 1 there is no mean shift
            # (both are pure sampling noise), so instead assert
            # the fitted vector itself vanishes (checked via the
            # share assertion below).
            if np.linalg.norm(mean_test) ** 2 / POWER > 0.1:
                rel = (np.linalg.norm(v - mean_test)
                       / np.linalg.norm(mean_test))
                assert rel < 0.35, (
                    f"seed {seed}: ||v - E_test Delta|| rel "
                    f"{rel:.3f} at lam={lam}")
            # And its size IS the constant share: ||v||^2 / POWER
            # tracks 1 - lam (the input-dependent part averages
            # out of the fit).
            share = float(v @ v) / POWER
            assert abs(share - (1.0 - lam)) < 0.10, (
                f"seed {seed}: fitted share {share:.3f} at "
                f"lam={lam} vs {1.0 - lam:.2f}")
        # At lam = 0 the fitted vector is the planted constant.
        assert np.linalg.norm(r[0.0]["v"] - c) < 1e-9, (
            f"seed {seed}: v at lam=0 must equal c exactly")


# ------------------------------------------------------------------
# Block placement: the paper adds its vector at a BLOCK output, not
# at the model output. Same decomposition, one layer down: the
# adapted model shifts the hidden pre-activation by
# sqrt(1 - lam) * c_h + sqrt(lam) * g_h(x), and arm (b) is a single
# constant hidden vector fitted by gradient descent on the SAME
# output-matching objective (no reward in the fit).
# ------------------------------------------------------------------

def run_seed_block(seed, steps=300):
    rng = np.random.default_rng(31000 + seed)
    w1 = rng.standard_normal((D_HID, D_IN)) / np.sqrt(D_IN)
    b1 = 0.1 * rng.standard_normal(D_HID)
    w2 = rng.standard_normal((D_OUT, D_HID)) / np.sqrt(D_HID)
    b2 = 0.1 * rng.standard_normal(D_OUT)
    ug = rng.standard_normal((D_HID, D_IN)) / np.sqrt(D_IN)
    wg = rng.standard_normal((D_HID, D_HID)) / np.sqrt(D_HID)
    power_h = float(D_HID)
    c_h = rng.standard_normal(D_HID)
    c_h *= np.sqrt(power_h) / np.linalg.norm(c_h)
    x_fit = rng.standard_normal((N_FIT, D_IN))
    x_test = rng.standard_normal((N_TEST, D_IN))

    def g_raw(x):
        return np.tanh(x @ ug.T) @ wg.T

    g_mean = g_raw(x_fit).mean(0)
    scale = np.sqrt(power_h / np.mean(
        np.sum((g_raw(x_fit) - g_mean) ** 2, axis=1)))
    g_fit = (g_raw(x_fit) - g_mean) * scale
    g_test = (g_raw(x_test) - g_mean) * scale

    w2t = torch.tensor(w2, dtype=torch.float64)
    b2t = torch.tensor(b2, dtype=torch.float64)
    a_fit = torch.tensor(x_fit @ w1.T + b1, dtype=torch.float64)
    a_test = torch.tensor(x_test @ w1.T + b1, dtype=torch.float64)

    def out(a, shift):
        return torch.tanh(a + shift) @ w2t.T + b2t

    out_d = {}
    for lam in LAMBDAS:
        d_fit = np.sqrt(1.0 - lam) * c_h + np.sqrt(lam) * g_fit
        d_test = np.sqrt(1.0 - lam) * c_h + np.sqrt(lam) * g_test
        with torch.no_grad():
            fp_fit = out(a_fit, torch.tensor(d_fit))
            fp_test = out(a_test, torch.tensor(d_test))
            base_test = out(a_test, 0.0)
            gain_full = float(torch.mean(
                torch.sum((fp_test - base_test) ** 2, axis=1)))
        v = torch.zeros(D_HID, dtype=torch.float64,
                        requires_grad=True)
        opt = torch.optim.Adam([v], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.mean(torch.sum(
                (out(a_fit, v) - fp_fit) ** 2, axis=1))
            loss.backward()
            opt.step()
        rdir = rng.standard_normal(D_HID)
        v_rand = torch.tensor(
            rdir / np.linalg.norm(rdir) * float(v.detach().norm()),
            dtype=torch.float64)
        with torch.no_grad():
            gain_b = gain_full - float(torch.mean(torch.sum(
                (fp_test - out(a_test, v.detach())) ** 2, axis=1)))
            gain_c = gain_full - float(torch.mean(torch.sum(
                (fp_test - out(a_test, v_rand)) ** 2, axis=1)))
        out_d[lam] = dict(rec_b=gain_b / gain_full,
                          rec_c=gain_c / gain_full)
    return out_d


def test_block_placement_recovery_also_collapses():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed_block(seed)
        rec = {lam: r[lam]["rec_b"] for lam in LAMBDAS}
        # At lam = 0 the adapted model IS a constant hidden shift,
        # so the fitted vector recovers it (near-)exactly.
        assert rec[0.0] > 0.95, (
            f"seed {seed}: block recovery at lam=0 {rec[0.0]:.3f}")
        assert rec[0.25] > 0.65, (
            f"seed {seed}: block recovery at lam=0.25 "
            f"{rec[0.25]:.3f}")
        assert rec[0.5] > 0.45, (
            f"seed {seed}: block recovery at lam=0.5 {rec[0.5]:.3f}")
        # Still monotone decreasing in the input-dependent share.
        for lo, hi in zip(LAMBDAS, LAMBDAS[1:]):
            assert rec[lo] > rec[hi] + 0.08, (
                f"seed {seed}: block recovery lam={lo} "
                f"{rec[lo]:.3f} vs lam={hi} {rec[hi]:.3f}")
        # And it collapses when the gain is genuinely
        # input-dependent -- a constant hidden vector cannot
        # substitute, even though it may beat the naive output-
        # space share (its output effect is itself input-dependent
        # through the nonlinearity).
        assert rec[1.0] < 0.30, (
            f"seed {seed}: block recovery at lam=1 {rec[1.0]:.3f}")
        assert rec[0.75] < 0.50, (
            f"seed {seed}: block recovery at lam=0.75 "
            f"{rec[0.75]:.3f}")
        # Matched-norm random hidden control never substitutes.
        for lam in LAMBDAS:
            assert r[lam]["rec_c"] < 0.40, (
                f"seed {seed}: block random recovery at lam={lam} "
                f"{r[lam]['rec_c']:.3f}")
            assert rec[lam] > r[lam]["rec_c"] + 0.30, (
                f"seed {seed}: block fitted {rec[lam]:.3f} vs "
                f"random {r[lam]['rec_c']:.3f} at lam={lam}")
