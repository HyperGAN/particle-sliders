"""CPU gate for the robust-steerability diagnostic claim
(arXiv:2611.06418): a gamma*-style robustness score computed at
CALIBRATION time -- from the steering direction and the local response
geometry only, no deployment data -- predicts how a fixed direction's
effect decays under increasing dynamics mismatch.

Paper: "Mind Control of LLMs via Closed-Loop Stimulation:
Out-of-Distribution Robustness" (Fallah, Singha, Ghanei, Sajda,
Anderson & Zabeh). Steering is formulated as a feedback-control
problem; an H_inf controller bounds sensitivity to disturbances and
linearization mismatch, and robust steerability S_rob = 1 / gamma* --
gamma* the minimum achievable closed-loop amplification from
disturbance to steering-relevant output -- is introduced as a
PRE-DEPLOYMENT measure of tolerance to dynamics mismatch. Per the
abstract, S_rob exhibits task-specific structure and varies
systematically with model scale, and the controller's gains over
baselines are particularly pronounced under out-of-distribution
conditions. The abstract reports NO headline numbers; the HarmBench /
TruthfulQA figures circulating in secondary coverage of the paper are
coverage-reported, not abstract-verified, and are NOT asserted here.

What this test does NOT test: the H_inf controller itself. The
controller needs a per-step state signal in the application path,
which the Particle Sliders LoRA application path does not have (the
standing prerequisite from the per-token gating test, PR #212). What
is tested is the DIAGNOSTIC half of the paper's claim, in the form the
paper itself states it: a score computed from calibration activations
alone should tell us, before deployment, how much dynamics mismatch an
intervention can absorb before it stops producing the intended
behavior -- and nominal calibration performance should NOT be that
signal ("two representations with similar decodability and nominal
steering performance can differ sharply in reliability under shift").

Toy operationalization: a linear response model in R^16. Around a
nominal operating point, steering direction d (unit norm) produces
behavior effect e(d) = r^T A d, where A is the calibration response
map (diagonal in a random orthonormal basis Q, gains g) and r the
behavior readout. As in the paper, the mismatch channel is STRUCTURED
and estimated at calibration: each response axis i carries a known
volatility v_i (the toy form of the paper's disturbance channel D_k,
estimated there from calibration residuals). Deployment mismatch of
size eps perturbs the gains on volatile axes only:

    g_i^dep = g_i * (1 - eps * v_i * u_i),   u_i ~ U[0, 1] per draw.

Ground-truth robustness is CONSTRUCTED, not assumed, via three
directions (coordinates in the Q basis):

  fragile      -- concentrated on the single high-gain axis with the
                  highest volatility (w = r_i g_i ~ 3.0, v = 1.0).
                  Largest nominal effect, fastest decay.
  intermediate -- concentrated on one axis of a near-degenerate gain
                  pair (2.50 vs 2.45 -- an ill-conditioned subspace:
                  the readout can barely tell the pair apart) with
                  moderate volatility (v ~ 0.45). Middle nominal
                  effect, middle decay.
  robust       -- spread evenly across four stable axes (v = 0,
                  w ~ 1.0 each). SMALLEST nominal effect (~2.0 vs
                  fragile's ~3.0), no decay: every axis it uses is
                  mismatch-free.

Calibration score (the gamma* analog): the predicted worst-case
retention under mismatch eps is

    ret_wc(d, eps) = 1 - eps * sens(d),
    sens(d) = (sum_i w_i d_i v_i) / e0(d),

with w_i = r_i g_i the per-axis effect contributions and e0(d) the
nominal effect -- all calibration quantities. The score is the
critical mismatch size at which worst-case retention falls to
TAU = 0.5:  S(d) = (1 - TAU) / sens(d)  (capped at S_CAP when
sens = 0). Higher S = tolerates more mismatch = the toy's 1/gamma*.

Measured robustness: for eps on a grid, K = 96 deployment draws;
retention(eps) = mean over draws of e_dep / e0, plus the 10th
percentile. Directions are ranked by retention at eps = 1.0 and by
retention-curve area.

Controls (must NOT predict the measured ordering):
  norm          -- all directions are unit norm by construction, so
                   the norm ranking is an exact tie;
  nominal       -- the nominal effect e0 ranks fragile FIRST (largest
                   effect), i.e. it anti-predicts the measured
                   robustness ordering on every seed;
  random score  -- one seeded random score per direction per seed.

Secondary arm (the paper's feedback half, in miniature): a
probe-measure-correct step -- apply the direction once, measure the
realized effect with noise (sigma = 5% of e0), rescale the amplitude
to a = e0 / e_measured (clipped), re-apply. This is a one-step
proportional correction using the current state's measured effect,
NOT the paper's H_inf law; it is asserted only to decay slower than
the open-loop fragile direction.

Measured on pop-os (particlegan 0.10.1, 3 seeds): calibration
scores fragile 0.491-0.515, intermediate 1.075-1.120, robust at the
4.0 cap (zero sensitivity) -- the calibration ranking is exactly
robust > intermediate > fragile on every seed, and the measured
ranking agrees on every seed under both summaries: mean retention at
eps = 1 is fragile 0.503-0.538, intermediate 0.743-0.770, robust
1.000 (pairwise margins 0.21-0.26), and retention-curve area is
0.742-0.754 / 0.880-0.887 / 1.000 in the same order. The 10th
percentile at eps = 1 separates further (fragile 0.089-0.147). The
nominal effect anti-predicts: e0 is fragile 1.204-1.248 >
intermediate 1.026-1.040 > robust 0.804-0.829 on every seed -- the
direction with the LARGEST calibration effect is the least robust.
Norms are tied exactly (all unit); a seeded random score matched the
true ordering on 0 of 3 seeds. Feedback arm: the fragile direction's
retention at eps = 1 rises from 0.503-0.538 (open loop) to
0.810-0.837 with the one-step probe-measure-correct rescaling.
Assertions below encode roughly half the thinnest demonstrated
margins. Skips cleanly when the installed particlegan does not
provide the EMPF recipe. Pure numpy state geometry -- no GAN
machinery, because the claim is about a calibration-time diagnostic
over directions, not training.
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


D = 16                       # state dimension
SEEDS = (0, 1, 2)
K_DRAWS = 96                 # deployment mismatch draws per (eps, seed)
EPS_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TAU = 0.5                    # retention threshold defining the score
S_CAP = 4.0                  # score cap when sensitivity is exactly 0
FB_NOISE = 0.05              # feedback probe noise, fraction of e0
FB_CLIP = (0.5, 2.5)         # feedback amplitude clip

# Axis layout in the calibration basis (indices into Q's columns).
AX_FRAGILE = 0               # high gain, high volatility
AX_INTER = 1                 # near-degenerate pair (axes 1, 2),
AX_INTER_PAIR = 2            # moderate volatility on axis 1
AX_ROBUST = (3, 4, 5, 6)     # stable axes the robust direction spans
READ_AXES = (0, 1, 3, 4, 5, 6)   # axes the behavior readout reads

GAINS = {0: 3.0, 1: 2.50, 2: 2.45, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
VOLS = {0: 1.0, 1: 0.45, 2: 0.45, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}


def _unit(x):
    return x / np.linalg.norm(x)


def run_seed(seed):
    rng = np.random.default_rng(41000 + seed)
    q_basis, _ = np.linalg.qr(rng.standard_normal((D, D)))

    # Calibration quantities, with small per-seed jitter that preserves
    # every constructed gap (gains +-3%, volatilities +-0.03 clipped).
    gains = np.ones(D)
    for ax, g0 in GAINS.items():
        gains[ax] = g0 * (1.0 + rng.uniform(-0.03, 0.03))
    vols = np.zeros(D)
    for ax, v0 in VOLS.items():
        vols[ax] = float(np.clip(v0 + rng.uniform(-0.03, 0.03), 0.0, None))
    vols[list(AX_ROBUST)] = 0.0          # stable axes stay exactly stable

    r_c = np.zeros(D)
    r_c[list(READ_AXES)] = 1.0
    r_c = _unit(r_c)
    w = r_c * gains                      # per-axis effect contributions

    a_cal = (q_basis * gains) @ q_basis.T
    r_vec = q_basis @ r_c

    def direction(coords):
        d_c = np.zeros(D)
        for ax, val in coords.items():
            d_c[ax] = val
        return _unit(d_c), q_basis @ _unit(d_c)

    dirs = {}
    dirs["fragile"] = direction({AX_FRAGILE: 1.0})
    dirs["intermediate"] = direction({AX_INTER: 1.0})
    dirs["robust"] = direction({ax: 1.0 for ax in AX_ROBUST})

    def effect(d_vec, a_mat):
        return float(r_vec @ a_mat @ d_vec)

    rows = {}
    for name, (d_c, d_vec) in dirs.items():
        e0 = effect(d_vec, a_cal)
        sens = float(np.sum(w * d_c * vols) / np.sum(w * d_c))
        score = S_CAP if sens <= 0 else min((1.0 - TAU) / sens, S_CAP)
        # Deployment sweep: fresh mismatch draws per eps.
        ret_mean, ret_p10, fb_mean = [], [], []
        for eps in EPS_GRID:
            u = rng.uniform(0.0, 1.0, size=(K_DRAWS, D))
            g_dep = gains[None, :] * (1.0 - eps * vols[None, :] * u)
            # effect via coordinates: e = sum_i w_i d_ci (g_dep/g)_i
            ratio = g_dep / gains[None, :]
            e_dep = ratio @ (w * d_c)
            ret = e_dep / e0
            ret_mean.append(float(ret.mean()))
            ret_p10.append(float(np.percentile(ret, 10)))
            # Feedback arm: probe-measure-correct, per draw.
            e_meas = e_dep + rng.normal(0.0, FB_NOISE * e0, size=K_DRAWS)
            amp = np.clip(e0 / e_meas, *FB_CLIP)
            fb_mean.append(float((amp * e_dep / e0).mean()))
        rows[name] = dict(
            e0=e0, sens=sens, score=score, norm=float(np.linalg.norm(d_vec)),
            ret_mean=ret_mean, ret_p10=ret_p10, fb_mean=fb_mean,
            auc=float(np.sum(0.5 * (np.diff(EPS_GRID))
                             * (np.array(ret_mean[1:])
                                + np.array(ret_mean[:-1])))),
        )

    # Random-score control: one seeded draw per direction.
    rand_scores = {name: float(rng.uniform()) for name in dirs}
    return dict(rows=rows, rand_scores=rand_scores)


def _ranking(scores):
    return tuple(sorted(scores, key=scores.get, reverse=True))


TRUE_ORDER = ("robust", "intermediate", "fragile")


def test_calibration_score_predicts_measured_robustness():
    _empf_recipe()
    rand_exact = 0
    for seed in SEEDS:
        out = run_seed(seed)
        rows = out["rows"]

        # Calibration ranking (score computed from calibration only).
        cal_order = _ranking({k: v["score"] for k, v in rows.items()})
        assert cal_order == TRUE_ORDER, (
            f"seed {seed}: calibration ranking {cal_order}, scores "
            f"{ {k: round(v['score'], 3) for k, v in rows.items()} }")
        assert rows["robust"]["score"] >= S_CAP - 1e-9, (
            f"seed {seed}: robust score {rows['robust']['score']:.3f} "
            f"should hit the cap (zero sensitivity)")
        assert (rows["intermediate"]["score"]
                > 1.5 * rows["fragile"]["score"]), (
            f"seed {seed}: score margin intermediate "
            f"{rows['intermediate']['score']:.3f} vs fragile "
            f"{rows['fragile']['score']:.3f}")

        # Measured ranking at the largest mismatch and by curve area.
        for key in ("ret_mean", "auc"):
            if key == "auc":
                meas = {k: v["auc"] for k, v in rows.items()}
            else:
                meas = {k: v["ret_mean"][-1] for k, v in rows.items()}
            assert _ranking(meas) == TRUE_ORDER, (
                f"seed {seed}: measured ranking by {key} "
                f"{_ranking(meas)}, values "
                f"{ {k: round(v, 3) for k, v in meas.items()} }")
        ret = {k: v["ret_mean"][-1] for k, v in rows.items()}
        assert ret["robust"] - ret["intermediate"] > 0.10, (
            f"seed {seed}: retention@eps=1 robust {ret['robust']:.3f} "
            f"vs intermediate {ret['intermediate']:.3f}")
        assert ret["intermediate"] - ret["fragile"] > 0.10, (
            f"seed {seed}: retention@eps=1 intermediate "
            f"{ret['intermediate']:.3f} vs fragile {ret['fragile']:.3f}")
        # The fragile direction genuinely decays across the sweep.
        fr = rows["fragile"]["ret_mean"]
        assert fr[0] == pytest.approx(1.0) and fr[-1] < fr[2] - 0.1, (
            f"seed {seed}: fragile retention curve {fr}")

        # Controls must not predict the ordering.
        norms = {k: v["norm"] for k, v in rows.items()}
        assert max(norms.values()) - min(norms.values()) < 1e-9, (
            f"seed {seed}: norms not tied: {norms}")
        e0_order = _ranking({k: v["e0"] for k, v in rows.items()})
        assert e0_order == ("fragile", "intermediate", "robust"), (
            f"seed {seed}: nominal-effect ordering {e0_order} -- by "
            f"construction it must rank the fragile direction first, "
            f"i.e. anti-predict robustness")
        if _ranking(out["rand_scores"]) == TRUE_ORDER:
            rand_exact += 1
    # The random control may luck into the ordering on at most one
    # seed; the calibration score must match on all of them (checked
    # per seed above).
    assert rand_exact <= 1, (
        f"random control matched the true ordering on {rand_exact} "
        f"of {len(SEEDS)} seeds")


def test_feedback_arm_decays_slower_than_open_loop():
    _empf_recipe()
    for seed in SEEDS:
        rows = run_seed(seed)["rows"]
        fr = rows["fragile"]
        fb_last = fr["fb_mean"][-1]
        assert fb_last > fr["ret_mean"][-1] + 0.15, (
            f"seed {seed}: fragile retention@eps=1 open-loop "
            f"{fr['ret_mean'][-1]:.3f} vs feedback {fb_last:.3f}")
        assert fb_last > 0.75, (
            f"seed {seed}: feedback retention@eps=1 {fb_last:.3f}")
