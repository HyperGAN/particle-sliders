"""CPU gate for the probe-transfer claim (arXiv:2610.35586): a linear
probe fitted on a base model, applied UNCHANGED to distilled and
instruction-tuned successors, keeps working even though the successors
rotate the separating direction by 17-35 degrees -- while the probe
partly reads surface appearance rather than the underlying property.

Paper: "Linear Probes Transfer from Base to Distilled Language Models
despite Rotation" (Tsatsos). Abstract-verified findings asserted here
in toy form: probes trained on a base model and transferred unchanged
to distilled / instruction-tuned successors lose almost nothing --
transferred AUC moves only -0.011..+0.027 on XSTest and -0.015..+0.053
on held-out OR-Bench -- despite the separating direction rotating by
17-35 degrees between base and successor. Caveat, also from the paper:
the probes partly read surface appearance, not the underlying
property -- they flag 29-42% of benign-but-alarming prompts, and their
margin over a prompt-text-only classifier shrinks from 0.15-0.20 AUC
(XSTest) to 0.03-0.20 (OR-Bench). The paper's exact model point values
are NOT asserted -- only the mechanism and its margin structure, in a
toy where the rotation is exact by construction.

Toy operationalization, main arm: synthetic class-conditional states
in R^24, h = y * delta/2 + isotropic noise (sigma = 1), with a KNOWN
separating direction delta (|delta| = 1.8, so the base probe's AUC is
~0.89). The probe is the closed-form LDA discriminant fitted on the
train split of the "base" states. The successor transform -- the
distillation stand-in -- is A = S R: R rotates the representation
space by a controlled angle theta in the plane spanned by delta and a
random orthogonal axis (the class geometry rotates WITH the space:
successor states are A h, successor separating direction is
A delta), and S is a mild per-axis rescaling (factors in
[exp(-0.1), exp(0.1)]). Theta sweeps the paper's range (17, 25, 35
degrees) plus a beyond-range point (60 degrees) to find the boundary.

Why transfer survives, in the toy's arithmetic: with isotropic noise
the transferred AUC is Phi(|delta| cos(theta) / sqrt(2)), so the
signal shrinks only by cos(theta) -- at 35 degrees a factor 0.82
against a comfortable margin -- while the direction itself has
genuinely moved (cos(w, A w) ~ 0.82). At 60 degrees cos = 0.5 halves
the signal and the AUC drop becomes large: that is the boundary the
paper's range stops short of.

Arms / controls per seed and angle:
  (a) transferred probe: base-fitted w applied unchanged to successor
      states (the claim);
  (b) refit probe: LDA refitted on successor train states (ceiling --
      the geometry is preserved by A, so refit recovers base AUC);
  (c) random-direction floor: mean AUC of 16 fixed random unit
      directions on successor states (a single random direction is
      too noisy a floor in R^24 -- its AUC ranged 0.32-0.69 across
      draws in prototyping -- so the floor is the mean);
  (d) SURFACE-CUE arm (separate dataset, below).

Metrics per seed: base AUC, transferred AUC per angle, refit AUC,
random-floor AUC, and the realized rotation cos(delta, A delta)
(checked against cos(theta): the rescaling perturbs it only slightly).

Surface-cue arm: states carry, in addition to the property signal
along p, a superficial cue component along an orthogonal axis q:
h += s * (BETA/2) * q, where the cue sign s matches the label with
probability 0.85 in the base distribution (benign items can LOOK
alarming: s = +1 with y = -1). An LDA probe fitted there leans on q
(measured cos(w, q) ~ 0.43) because the cue predicts the label. On a
held-out OR-Bench-like split the cue is decorrelated from the label
(s independent of y). Measured there: the transferred-form probe
flags 31-33% of benign-but-alarming items (cue-positive benign, score
> 0) -- inside the paper's 29-42% -- vs 11-15% of benign
non-alarming items and 17-18% for a property-only readout (projection
on the ground-truth p, which never sees the cue). The probe's AUC
bonus over the property-only readout, +0.03 on the in-distribution
split, flips to -0.04 once the cue decorrelates (the toy form of the
paper's shrinking-margin caveat: the part of the probe's edge that
was cue, not property, evaporates), while a cue-only "text"
classifier collapses from AUC 0.77 to chance. The cue-arm flag rate
is also checked after a 35-degree successor rotation: it stays
elevated, i.e. the confound transfers with the probe.

Measured on pop-os (particlegan 0.10.1): base AUC 0.885-0.897;
transferred AUC drop at 17/25/35 degrees 0.009-0.015 / 0.020-0.031 /
0.039-0.056, and 0.141-0.174 at the 60-degree boundary point; refit
AUC equal to base at every angle; random floor 0.473-0.510; realized
cos within 0.014 of cos(theta) at every angle. Cue arm: probe AUC
0.924-0.929 in-distribution, 0.852-0.860 on the decorrelated split;
benign-but-alarming flag rate 0.314-0.332, benign non-alarming
0.111-0.146, property-only 0.172-0.183, and the transferred
(35-degree) benign-but-alarming flag rate 0.358-0.411. Assertions
below encode exactly these margins, no more.

What is NOT asserted: that real distillation is a clean rotation plus
mild rescaling (it is not -- the toy's successor transform is an
idealization chosen so the rotation angle is known exactly); that
transferred probes survive rotations beyond the paper's range (the
60-degree point shows they do not, in this toy); anything about
nonlinear probes, other layers, or real refusal behavior; the paper's
exact AUC deltas (the toy's drops at 35 degrees, ~0.05, are larger
than the paper's near-zero moves -- the toy's base margin is thinner
than a real probe's, and the assertion is the paper's qualitative
claim: small loss inside the range, large loss outside it).

Skips cleanly when the installed particlegan does not provide the
EMPF recipe. Pure numpy/torch state geometry -- no GAN machinery,
because the claim is about harvested directions, not training;
particlegan is only the skip gate.
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


D = 24            # state dimension
N = 4000          # states per seed (half train / half held-out)
MAG = 1.8         # |delta|: base probe AUC ~ Phi(1.8/sqrt(2)) ~ 0.90
ANGLES = (17, 25, 35, 60)   # paper range 17-35, plus 60 as boundary
RESCALE = 0.1     # per-axis log-rescale half-width (mild)
N_RAND = 16       # random directions averaged for the floor control
BETA_CUE = 1.8    # cue component magnitude along q (cue arm)
P_CUE = 0.85      # P(cue sign == label) in the base distribution


def _unit(x):
    return x / np.linalg.norm(x)


def _cos(x, y):
    return float(_unit(x) @ _unit(y))


def _auc(scores, labels):
    """Mann-Whitney rank AUC, average ranks for ties."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    srt = scores[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def _fit_lda(h, y):
    """Closed-form discriminant: pooled-covariance LDA direction."""
    mu_pos = h[y == 1].mean(0)
    mu_neg = h[y == -1].mean(0)
    pooled = np.cov(np.vstack([h[y == 1] - mu_pos,
                               h[y == -1] - mu_neg]).T)
    return np.linalg.solve(pooled + 1e-6 * np.eye(D), mu_pos - mu_neg)


def _successor(rng, p, u2, theta_deg):
    """A = S R: rotate the (p, u2) plane by theta, mild rescale."""
    th = np.radians(theta_deg)
    rot = (np.eye(D)
           + (np.cos(th) - 1.0) * (np.outer(p, p) + np.outer(u2, u2))
           + np.sin(th) * (np.outer(u2, p) - np.outer(p, u2)))
    scales = np.exp(rng.uniform(-RESCALE, RESCALE, size=D))
    return np.diag(scales) @ rot


def run_seed(seed):
    rng = np.random.default_rng(7000 + seed)
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    p, q, u2 = q_basis[:, 0], q_basis[:, 1], q_basis[:, 2]
    delta = MAG * p
    y = rng.choice([-1, 1], size=N)
    h = (y[:, None] * (delta / 2)[None, :]
         + rng.standard_normal((N, D)))
    h_tr, h_te = h[:N // 2], h[N // 2:]
    y_tr, y_te = y[:N // 2], y[N // 2:]

    w = _fit_lda(h_tr, y_tr)
    base_auc = _auc(h_te @ w, y_te)
    rand_dirs = [_unit(rng.standard_normal(D)) for _ in range(N_RAND)]

    angles = {}
    for deg in ANGLES:
        a_mat = _successor(rng, p, u2, deg)
        hs_tr, hs_te = h_tr @ a_mat.T, h_te @ a_mat.T
        w_refit = _fit_lda(hs_tr, y_tr)
        angles[deg] = dict(
            trans=_auc(hs_te @ w, y_te),
            refit=_auc(hs_te @ w_refit, y_te),
            rand=float(np.mean([_auc(hs_te @ r, y_te)
                                for r in rand_dirs])),
            cos_real=_cos(delta, a_mat @ delta),
        )

    # ---- surface-cue arm -------------------------------------------
    rngc = np.random.default_rng(17000 + seed)

    def cue_states(n, mode):
        yy = rngc.choice([-1, 1], size=n)
        if mode == "or":
            s = rngc.choice([-1, 1], size=n)
        else:  # train / in-distribution: cue correlated with label
            s = np.where(rngc.random(n) < P_CUE, yy, -yy)
        hh = (yy[:, None] * (delta / 2)[None, :]
              + s[:, None] * (BETA_CUE / 2) * q[None, :]
              + rngc.standard_normal((n, D)))
        return hh, yy, s

    h_cue, y_cue, _ = cue_states(N, "train")
    w_cue = _fit_lda(h_cue, y_cue)
    h_in, y_in, s_in = cue_states(N, "indist")
    h_or, y_or, s_or = cue_states(N, "or")

    def flag_rate(hh, yy, s, direction, alarming):
        mask = (yy == -1) & (s == (1 if alarming else -1))
        return float(((hh @ direction)[mask] > 0).mean())

    # Transferred cue probe: 35-degree successor rotation applied to
    # the decorrelated split, base cue probe applied unchanged.
    a35 = _successor(rngc, p, u2, 35)
    h_or_succ = h_or @ a35.T

    cue = dict(
        cos_w_q=_cos(w_cue, q),
        auc_in=_auc(h_in @ w_cue, y_in),
        auc_or=_auc(h_or @ w_cue, y_or),
        prop_auc_in=_auc(h_in @ p, y_in),
        prop_auc_or=_auc(h_or @ p, y_or),
        text_auc_in=_auc(h_in @ q, y_in),
        text_auc_or=_auc(h_or @ q, y_or),
        flag_ba_or=flag_rate(h_or, y_or, s_or, w_cue, True),
        flag_bn_or=flag_rate(h_or, y_or, s_or, w_cue, False),
        prop_flag_ba_or=flag_rate(h_or, y_or, s_or, p, True),
        flag_ba_or_succ=flag_rate(h_or_succ, y_or, s_or, w_cue, True),
    )
    return dict(base_auc=base_auc, angles=angles, cue=cue)


def test_transferred_probe_survives_paper_range_rotation():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        base = r["base_auc"]
        assert 0.85 < base < 0.93, (
            f"seed {seed}: base probe AUC {base:.3f} -- toy margin "
            f"off design (~0.89)")

        for deg in ANGLES:
            a = r["angles"][deg]
            # Trap guard: the rotation must be real -- the realized
            # cos of the separating direction tracks cos(theta).
            assert abs(a["cos_real"] - np.cos(np.radians(deg))) < 0.05, (
                f"seed {seed}: realized cos at {deg} deg is "
                f"{a['cos_real']:.3f}, target "
                f"{np.cos(np.radians(deg)):.3f}")
            # Ceiling: refitting on successor states recovers base.
            assert a["refit"] >= base - 0.01, (
                f"seed {seed}: refit AUC at {deg} deg {a['refit']:.3f} "
                f"vs base {base:.3f}")
            assert a["refit"] >= a["trans"] - 0.005, (
                f"seed {seed}: refit {a['refit']:.3f} below transferred "
                f"{a['trans']:.3f} at {deg} deg")
            # Floor: the random-direction mean sits at chance and the
            # transferred probe beats it clearly at every angle.
            assert abs(a["rand"] - 0.5) < 0.08, (
                f"seed {seed}: random floor AUC {a['rand']:.3f} at "
                f"{deg} deg")
            assert a["trans"] > a["rand"] + 0.2, (
                f"seed {seed}: transferred {a['trans']:.3f} vs random "
                f"{a['rand']:.3f} at {deg} deg")

        # The claim: inside the paper's 17-35 degree range the
        # unchanged probe loses only a little AUC, despite cos(w, Aw)
        # falling to ~0.82 at 35 degrees.
        for deg in (17, 25, 35):
            drop = base - r["angles"][deg]["trans"]
            assert drop <= 0.07, (
                f"seed {seed}: transferred AUC drop at {deg} deg is "
                f"{drop:.3f} (base {base:.3f} -> "
                f"{r['angles'][deg]['trans']:.3f})")
            assert r["angles"][deg]["trans"] >= 0.80, (
                f"seed {seed}: transferred AUC at {deg} deg "
                f"{r['angles'][deg]['trans']:.3f}")
        assert (r["angles"][17]["trans"]
                > r["angles"][35]["trans"] + 0.02), (
            f"seed {seed}: loss should grow with angle -- 17 deg "
            f"{r['angles'][17]['trans']:.3f} vs 35 deg "
            f"{r['angles'][35]['trans']:.3f}")

        # Boundary: at 60 degrees (cos = 0.5, well outside the paper's
        # range) the same unchanged probe loses a large share -- the
        # survival above is a property of the range, not of probes
        # in general.
        drop60 = base - r["angles"][60]["trans"]
        assert drop60 > 0.10, (
            f"seed {seed}: 60-degree drop {drop60:.3f} -- expected "
            f"the boundary loss to be large")
        assert (r["angles"][60]["trans"]
                < r["angles"][35]["trans"] - 0.08), (
            f"seed {seed}: 60 deg {r['angles'][60]['trans']:.3f} vs "
            f"35 deg {r['angles'][35]['trans']:.3f}")


def test_transferred_probe_partly_reads_surface_cue():
    _empf_recipe()
    for seed in (0, 1, 2):
        c = run_seed(seed)["cue"]

        # Trap guard: the fitted probe must genuinely lean on the
        # cue axis -- if it does not, the arm tests nothing.
        assert c["cos_w_q"] > 0.3, (
            f"seed {seed}: cos(probe, cue axis) {c['cos_w_q']:.3f} -- "
            f"probe did not pick up the surface cue")

        # The probe still works once the cue decorrelates (it does
        # read the property too), but its bonus over the
        # property-only readout was cue: +bonus in-distribution,
        # gone (negative) on the decorrelated split.
        assert c["auc_or"] > 0.80, (
            f"seed {seed}: cue-probe AUC on decorrelated split "
            f"{c['auc_or']:.3f}")
        bonus_in = c["auc_in"] - c["prop_auc_in"]
        bonus_or = c["auc_or"] - c["prop_auc_or"]
        assert bonus_in > bonus_or + 0.04, (
            f"seed {seed}: probe bonus over property-only readout "
            f"in-dist {bonus_in:+.3f} -> decorrelated {bonus_or:+.3f} "
            f"-- the cue-carried edge should evaporate")

        # A cue-only ("prompt text") classifier collapses to chance
        # on the decorrelated split.
        assert c["text_auc_in"] > 0.70, (
            f"seed {seed}: cue-only AUC in-dist {c['text_auc_in']:.3f}")
        assert abs(c["text_auc_or"] - 0.5) < 0.08, (
            f"seed {seed}: cue-only AUC decorrelated "
            f"{c['text_auc_or']:.3f}")

        # The caveat in the paper's numbers: the probe flags roughly
        # a third of benign-but-alarming items -- far above benign
        # non-alarming items and above the property-only readout,
        # which cannot see the cue at all.
        assert 0.25 < c["flag_ba_or"] < 0.45, (
            f"seed {seed}: benign-but-alarming flag rate "
            f"{c['flag_ba_or']:.3f} (paper: 0.29-0.42)")
        assert c["flag_ba_or"] > c["flag_bn_or"] + 0.12, (
            f"seed {seed}: alarming {c['flag_ba_or']:.3f} vs "
            f"non-alarming benign {c['flag_bn_or']:.3f}")
        assert c["flag_ba_or"] > c["prop_flag_ba_or"] + 0.08, (
            f"seed {seed}: probe flag rate {c['flag_ba_or']:.3f} vs "
            f"property-only {c['prop_flag_ba_or']:.3f} on "
            f"benign-but-alarming items")

        # And the confound transfers: after a 35-degree successor
        # rotation the unchanged cue-laden probe still flags
        # benign-but-alarming items at an elevated rate.
        assert c["flag_ba_or_succ"] > 0.25, (
            f"seed {seed}: transferred benign-but-alarming flag rate "
            f"{c['flag_ba_or_succ']:.3f}")
