"""CPU toy recreating the STRUCTURE of Mikkel's Anima
Final Boss Jev metric, and a validation run against his
measured ranking (192 updates, 8 cases, shared strengths;
Jev rewards Final Boss character gain, penalises visible
image damage):

  technique                                   early   const
  1. soft FG weighting, BG updates at 45%     25.66   13.06
  2. + 35% stronger Boss cue blend            24.93   16.90
  3. paired scene+lighting contexts           22.23   11.58
  4. background-only nuisance projection      22.22   11.51
  5. whole-image nuisance projection          18.74   13.72
  6. train on adapter-generated states        11.61    5.49
  prompt-only baseline (early only)           18.32     --

Qualitative facts to reproduce: FG45 top at early; the
cue blend trades a little early (-0.73) for a lot of
constant (+3.84) and is the best constant score; paired
~= BG-only at both strengths; BG-only beats whole-image
at early but whole-image beats BG-only at constant (a
crossover); rollout last by a wide margin; prompt-only
mid-pack at early.

Toy operationalization (torch, CPU, no particlegan --
the claim is about training-target structure on a
distillation loop, not about GAN machinery; the house
particlegan guard is therefore skipped entirely):

State space: 48 dims = FOREGROUND block (12) + BACK-
GROUND block (36). Per seed, random orthonormal bases
define: a character direction c in FG (overlapping the
first FG nuisance direction by RHO=0.35); a 5-dim
nuisance (scene/lighting) subspace in each block; a BG
"presence" direction s_bg; a BG "atmosphere" direction
inside the BG nuisance span (part of the character's
gain lives in the nuisance span -- dramatic lighting
IS part of the Boss); an input cue direction q in FG;
per-case caricature directions in FG (a strong cue's
exaggeration distorts differently per image).

8 cases x 8 contexts. Input state per (case, context):
base_case + Q_AMP * q + nuisance inputs m_fg / m_bg,
where m_fg and m_bg are RHO_M=0.6-correlated views of
the case's latent nuisance n = n_mean + n_var (the
teacher's target nuisance and the state's nuisance are
correlated but not identical pathways). Teacher target:
A * c (FG character) + LEAK_FG * (FG nuisance at n)
+ PRES * s_bg + ATMO * atmosphere + LEAK_BG * (BG
nuisance at n), plus per-entry teacher noise
(T_NOISE=0.15: teacher targets are model-generated).

Student: a low-rank adapter delta map, W = [Uf ; Ub]
Vs^T -- a SHARED rank-3 input factor Vs with per-region
output heads Uf (FG) and Ub (BG) -- trained by plain
SGD for exactly 192 full-batch steps (the 192-update
analog) to regress teacher targets, weight decay WD.
The training loop exposes exactly the five knobs:
  (i) per-region update weights: FG head updates at
      rate 1.0, BG head updates (Ub and the BG part of
      the shared-factor gradient) scaled by w;
  (ii) cue blend beta: target = (1-beta) * base teacher
      + beta * strong-cue teacher, the strong cue being
      the teacher with its character component amplified
      (CAR_AMP) and tilted toward the case's caricature
      direction (CAR_TILT), FG leak amplified CAR_LEAK;
  (iii) context pairing: each state's target is the
      mean teacher target over the case's contexts
      (cancels per-context nuisance variation, keeps
      the case-mean nuisance);
  (iv) target projections: deflate an ESTIMATED
      nuisance subspace (top-5 principal components of
      the centred training targets in that region --
      not the true subspace) from the target, BG-only
      or whole-image (FG+BG);
  (v) rollout states: every 8 steps the training
      states are regenerated as base + W x (adapter-
      applied), with the input cue attenuated
      (ROLLOUT_Q: generated states lose prompt
      conditioning) and artifact noise proportional to
      the adapter's own output (ROLLOUT_NOISE); targets
      stay the original teacher targets.
Prompt-only: no training; the delta is the teacher
target itself at the evaluation context (character
direction applied directly, damage unfiltered).

Score (per case, on a held-out context; averaged):
  gain g = (delta_fg . c) + PRES_W * (delta_bg . s_bg)
           + PRES_W * (delta_bg . atmosphere)
  damage d = (||delta_bg off {s_bg, atmosphere}||^2
           + ||delta_fg off {c}||^2) / E_nuisance
  score = KSCALE * ( Sg * (1 - exp(-g / Sg)) - LAM * d )
Two strength settings (the early-alpha / constant-alpha
analog, defined here as EVALUATION regimes of the same
192-step adapter): EARLY (Sg=2.5, LAM=0.6: the adapter
acts where structure forms -- gain near-linear, damage
discounted) and CONSTANT (Sg=1.6, LAM=0.8: the adapter
acts throughout -- character gain saturates at a lower
ceiling, damage counted harder). Exact Jev magnitudes
are NOT claimed; the RANKING is the target.

Calibration disclosure (all structural, chosen over 10
logged rounds; no ranking is hard-coded anywhere):
RHO=0.35 character/nuisance overlap; RHO_M=0.6 FG/BG
nuisance pathway correlation; LEAK_BG=0.85 > LEAK_FG=
0.50 (the leak the techniques fight lives mostly in
BG); ATMO=0.55 atmosphere-in-nuisance-span; PRES=0.55,
PRES_W=0.35; CAR_AMP=1.25, CAR_TILT=0.42, CAR_LEAK=1.3;
T_NOISE=0.15; LR=0.01 with INIT=0.05 factorised growth
(the toy lives in the under-converged regime, as does
a 192-update run -- at LR>=0.02 the BG update weight
stops binding because the BG head converges anyway);
RANK=3 shared factor (capacity competition between
regions is load-bearing); WD=0.02; score weights above.

Iteration log (one line per round; prototype in numpy,
identical math, then ported to torch):
  R1 loss-weighted regions + rank pressure: FG45's BG
     damage barely moved -- V-factor rescaling circum-
     vents single-factor update scaling.
  R2 FG/BG nuisance pathways decorrelated (RHO_M=0.6):
     small gain; loss weighting still too weak.
  R3 atmosphere inside the nuisance span: deflation
     arms lose real gain; paired reaches mid-pack.
  R4 per-region BLOCK adapter, both factors update-
     scaled: BG suppression binds only once LR drops.
  R5 projections ESTIMATED by PCA on noisy targets:
     whole-image drops from #1 to mid-pack.
  R6 joint (quartic) energy budget: cross-region
     squeeze too weak to matter; dropped.
  R7 shared input factor Vs (multi-head LoRA shape):
     capacity competition across regions appears.
  R8 rank 3 + rollout degradation (cue attenuation +
     artifact noise): rollout last among trained arms.
  R9 teacher structure: LEAK_FG=0.8 tried and REVERTED
     (FG damage swamps every other signal, seed
     variance explodes); per-case caricature kept.
  R10 score calibration (LAM_E 0.8->0.6, LAM_C 2.2->
     0.8, SG_C 1.05->1.6, CAR_AMP 1.6->1.25): prompt
     rises to mid-pack range; top-5 early cluster
     compresses to ~1.7 points.

Measured on pop-os (torch 2.15, threads 1), 5 seeds.
Toy means (early / constant): paired 25.97 / 20.53,
BG-proj 24.68 / 21.29, cue 24.28 / 18.70, FG45 24.00 /
18.82, whole-proj 23.61 / 20.98, plain (control) 21.48
/ 15.99, rollout 20.29 / 16.43, prompt 16.26 / 7.25.
VERDICT: the toy does NOT reproduce Mikkel's ranking.
Per-ordering checklist (toy vs measured):
  FG45 top at early ......... MISS (toy #4 of 8; 0.28
    below the cue arm and 1.97 below the top -- the
    top-5 cluster spans 2.4 points, inside ~1 seed SD,
    but FG45 is not on top)
  cue #2 early / #1 const ... MISS (toy #3 early --
    adjacent to its measured slot; #5 constant)
  paired ~= BG-only mid ..... PASS (early 25.97 vs
    24.68, constant 20.53 vs 21.29; both mid-pack at
    both strengths, as measured)
  BG-vs-whole crossover ..... MISS (BG-only above
    whole-image at both strengths; the constant gap
    is a near-tie, 21.29 vs 20.98, but it does not
    cross)
  rollout last .............. PASS among Mikkel's six
    techniques at both strengths, but by a small
    margin (toy rollout is mildly bad; Mikkel's is
    catastrophically bad, 11.61 vs the 18-26 pack)
  prompt-only mid at early .. MISS (toy last overall
    at early, 16.26 vs rollout 20.29; Mikkel has
    prompt 6th of 7, above rollout)
What the toy DOES demonstrate, with wide margins, and
what the assertions below encode (nothing else):
  * target deflation removes ~97% of fitted BG off-
    target energy vs the plain arm (0.009 vs 0.264,
    energy normalised by E_nuisance);
  * FG45 update weighting cuts fitted BG off-target
    energy by ~1/3 vs plain (0.181 vs 0.264) at equal
    character gain -- the direction of Mikkel's #1
    effect, at a fraction of his implied strength;
  * the cue blend raises character gain over FG45
    (1.38 vs 1.33) at extra FG off-target cost
    (0.129 vs 0.103);
  * deflation costs character gain (whole 1.06 <
    BG-only 1.17 < FG45 1.33): projection is not free;
  * rollout training scores below FG45 at both
    strengths; prompt-only scores below every trained
    arm at early.
Mechanistic read of the misses (toy-supported): in
this toy, inter-arm DAMAGE differences are small next
to GAIN differences, because one shared under-converged
adapter suppresses all nuisance fits partially. Mikkel's
spread (FG45 beating target-side cleanups that remove
the leak exactly on paper) implies his BG update
weighting suppresses BG damage far more strongly than
a regression-gradient scaling can -- plausibly because
in the real system the 45% acts on the diffusion update
itself and compounds over sampling steps, a mechanism
a single linear delta map cannot express. Likewise
the cue blend's constant-strength win implies a gain-
side effect at constant application (character keeps
expressing) that this score's saturating gain caps
away. Both are stated as explanations of a FAILED
validation, not as findings.

Because the metric does not validate (more than one
unexplained-adjacent ordering misses), the Round-1
dose sweeps (background weight w, cue blend beta) are
NOT run: sweeping doses on an unvalidated surrogate
would manufacture numbers Mikkel could mistake for
his table. The sweep harness belongs to the round
that first fixes the compounding-suppression gap.

What is NOT asserted: Jev magnitudes, real diffusion
dynamics, anything about the guarded endpoint teacher,
and any dose-response claim.
"""
import numpy as np
import pytest
import torch

# ---- frozen configuration (see docstring disclosure) ----
D_FG, D_BG, K_NUIS = 12, 36, 5
N_CASES, N_CTX, STEPS, RANK = 8, 8, 192, 3
A_CHAR, PRES, ATMO = 1.0, 0.55, 0.55
LEAK_FG, LEAK_BG = 0.50, 0.85
RHO, RHO_M = 0.35, 0.6
SIG_MEAN, SIG_VAR = 0.55, 0.80
BASE_SD, Q_AMP = 0.70, 1.2
LR, INIT, WD = 0.01, 0.05, 0.02
T_NOISE = 0.15
CAR_AMP, CAR_TILT, CAR_LEAK = 1.25, 0.42, 1.3
BETA_CUE, W_FG45 = 0.35, 0.45
ROLLOUT_EVERY, ROLLOUT_Q, ROLLOUT_NOISE = 8, 0.55, 0.30
SG_E, LAM_E, SG_C, LAM_C = 2.5, 0.6, 1.6, 0.8
KSCALE, PRES_W = 28.0, 0.35
SEEDS = (0, 1, 2, 3, 4)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _randn(g, *shape):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _world(seed):
    g = torch.Generator().manual_seed(7000 + seed)
    Qf = torch.linalg.qr(_randn(g, D_FG, D_FG)).Q
    Qb = torch.linalg.qr(_randn(g, D_BG, D_BG)).Q
    e = [Qf[:, i] for i in range(D_FG)]
    c = (1 - RHO**2) ** 0.5 * e[0] + RHO * e[1]
    U_fg = torch.stack(e[1:1 + K_NUIS], dim=1)
    vcars = []
    for _ in range(N_CASES):
        v = _randn(g, D_FG)
        v = v - c * (v @ c)
        for k in range(K_NUIS):
            v = v - U_fg[:, k] * (v @ U_fg[:, k])
        vcars.append(v / v.norm())
    w = dict(
        c=c, U_fg=U_fg, v_car_cases=torch.stack(vcars), q=e[7],
        s_bg=Qb[:, 0], U_bg=Qb[:, 1:1 + K_NUIS],
        base_fg=_randn(g, N_CASES, D_FG) * BASE_SD,
        base_bg=_randn(g, N_CASES, D_BG) * BASE_SD,
        n_mean=_randn(g, N_CASES, K_NUIS) * SIG_MEAN,
        n_var=_randn(g, N_CASES, N_CTX, K_NUIS) * SIG_VAR,
        n_eval_var=_randn(g, N_CASES, K_NUIS) * SIG_VAR,
    )
    return w


def _states(w, seed):
    """Training inputs, base teacher targets, latent n per example."""
    g = torch.Generator().manual_seed(9000 + seed)
    sd = (SIG_MEAN**2 + SIG_VAR**2) ** 0.5
    X, T, NS = [], [], []
    for j in range(N_CASES):
        for t in range(N_CTX):
            n = w["n_mean"][j] + w["n_var"][j, t]
            m_fg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
            m_bg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
            x = torch.cat([w["base_fg"][j] + Q_AMP * w["q"] + w["U_fg"] @ m_fg,
                           w["base_bg"][j] + w["U_bg"] @ m_bg])
            tgt = torch.cat([A_CHAR * w["c"] + LEAK_FG * (w["U_fg"] @ n),
                             PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]
                             + LEAK_BG * (w["U_bg"] @ n)])
            X.append(x); T.append(tgt); NS.append(n)
    return torch.stack(X), torch.stack(T), torch.stack(NS)


def _targets(w, seed, beta=0.0, pair=False, proj=None):
    X, T, NS = _states(w, seed)
    if beta > 0:
        T = T.clone()
        for idx in range(len(NS)):
            j = idx // N_CTX
            n = NS[idx]
            base_fg = A_CHAR * w["c"] + LEAK_FG * (w["U_fg"] @ n)
            car_dir = ((1 - CAR_TILT**2) ** 0.5 * w["c"]
                       + CAR_TILT * w["v_car_cases"][j])
            strong_fg = (CAR_AMP * A_CHAR * car_dir
                         + CAR_LEAK * LEAK_FG * (w["U_fg"] @ n))
            T[idx, :D_FG] = (1 - beta) * base_fg + beta * strong_fg
    if pair:
        T = T.clone()
        for j in range(N_CASES):
            sl = slice(j * N_CTX, (j + 1) * N_CTX)
            T[sl] = T[sl].mean(dim=0, keepdim=True)
    g = torch.Generator().manual_seed(13000 + seed)
    T = T + _randn(g, *T.shape) * T_NOISE
    if proj in ("bg", "whole"):
        T = T.clone()
        bgc = T[:, D_FG:] - T[:, D_FG:].mean(dim=0, keepdim=True)
        Ubg_hat = torch.linalg.svd(bgc, full_matrices=False).Vh[:K_NUIS].T
        T[:, D_FG:] = T[:, D_FG:] - (T[:, D_FG:] @ Ubg_hat) @ Ubg_hat.T
        if proj == "whole":
            fgc = T[:, :D_FG] - T[:, :D_FG].mean(dim=0, keepdim=True)
            Ufg_hat = torch.linalg.svd(fgc, full_matrices=False).Vh[:K_NUIS].T
            T[:, :D_FG] = T[:, :D_FG] - (T[:, :D_FG] @ Ufg_hat) @ Ufg_hat.T
    return X, T


def _train(w, seed, w_bg=1.0, beta=0.0, pair=False, proj=None, rollout=False):
    g = torch.Generator().manual_seed(5000 + seed)
    X, T = _targets(w, seed, beta=beta, pair=pair, proj=proj)
    X0 = X.clone()
    D = D_FG + D_BG
    Vs = _randn(g, D, RANK) * INIT
    Uf = _randn(g, D_FG, RANK) * INIT
    Ub = _randn(g, D_BG, RANK) * INIT
    N = len(X)
    for step in range(STEPS):
        W = torch.cat([Uf @ Vs.T, Ub @ Vs.T], dim=0)
        if rollout and step > 0 and step % ROLLOUT_EVERY == 0:
            gr = torch.Generator().manual_seed(21000 + seed * 1000 + step)
            D0 = X0 @ W.T
            Xb = X0.clone()
            qcomp = Xb[:, :D_FG] @ w["q"]
            Xb[:, :D_FG] -= (1 - ROLLOUT_Q) * torch.outer(qcomp, w["q"])
            noise = _randn(gr, *D0.shape)
            noise *= (ROLLOUT_NOISE * D0.norm(dim=1, keepdim=True)
                      / (D0.shape[1] ** 0.5))
            X = Xb + D0 + noise
        Dlt = X @ W.T
        R = Dlt - T
        gW = (2.0 / N) * (R.T @ X)
        gWf, gWb = gW[:D_FG], gW[D_FG:]
        gUf = gWf @ Vs
        gUb = gWb @ Vs
        gVs = gWf.T @ Uf + w_bg * (gWb.T @ Ub)
        Uf = Uf - LR * (gUf + WD * Uf)
        Ub = Ub - (LR * w_bg) * (gUb + WD * Ub)
        Vs = Vs - LR * (gVs + WD * Vs)
    return torch.cat([Uf @ Vs.T, Ub @ Vs.T], dim=0)


def _eval_deltas(w, seed, W):
    g = torch.Generator().manual_seed(11000 + seed)
    sd = (SIG_MEAN**2 + SIG_VAR**2) ** 0.5
    out = []
    for j in range(N_CASES):
        n = w["n_mean"][j] + w["n_eval_var"][j]
        m_fg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
        m_bg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
        x = torch.cat([w["base_fg"][j] + Q_AMP * w["q"] + w["U_fg"] @ m_fg,
                       w["base_bg"][j] + w["U_bg"] @ m_bg])
        out.append(W @ x)
    return out


def _prompt_deltas(w):
    out = []
    for j in range(N_CASES):
        n = w["n_mean"][j] + w["n_eval_var"][j]
        out.append(torch.cat([A_CHAR * w["c"] + LEAK_FG * (w["U_fg"] @ n),
                              PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]
                              + LEAK_BG * (w["U_bg"] @ n)]))
    return out


def _decompose(w, deltas):
    """Mean (gain, bg_off_energy, fg_off_energy) over cases."""
    E_n = K_NUIS * (SIG_MEAN**2 + SIG_VAR**2)
    gs, bgs, fgs = [], [], []
    for d in deltas:
        d_fg, d_bg = d[:D_FG], d[D_FG:]
        u_atmo = w["U_bg"][:, 0]
        gs.append(float(d_fg @ w["c"] + PRES_W * (d_bg @ w["s_bg"])
                        + PRES_W * (d_bg @ u_atmo)))
        fg_off = d_fg - (d_fg @ w["c"]) * w["c"]
        bg_off = (d_bg - (d_bg @ w["s_bg"]) * w["s_bg"]
                  - (d_bg @ u_atmo) * u_atmo)
        bgs.append(float(bg_off @ bg_off) / E_n)
        fgs.append(float(fg_off @ fg_off) / E_n)
    return float(np.mean(gs)), float(np.mean(bgs)), float(np.mean(fgs))


def _score(gain, bg_e, fg_e, mode):
    Sg, lam = (SG_E, LAM_E) if mode == "early" else (SG_C, LAM_C)
    return KSCALE * (Sg * (1 - float(np.exp(-gain / Sg))) - lam * (bg_e + fg_e))


ARMS = {
    "plain": dict(),
    "fg45": dict(w_bg=W_FG45),
    "cue": dict(w_bg=W_FG45, beta=BETA_CUE),
    "paired": dict(pair=True),
    "bgproj": dict(proj="bg"),
    "wholeproj": dict(proj="whole"),
    "rollout": dict(rollout=True),
}


def _run_all():
    """Returns {arm: {mode: [per-seed scores]}}, plus decompositions."""
    scores = {a: {"early": [], "const": []} for a in list(ARMS) + ["prompt"]}
    decomp = {}
    for arm, kw in ARMS.items():
        ds = []
        for s in SEEDS:
            w = _world(s)
            W = _train(w, s, **kw)
            d = _decompose(w, _eval_deltas(w, s, W))
            ds.append(d)
            for mode in ("early", "const"):
                scores[arm][mode].append(_score(*d, mode))
        decomp[arm] = tuple(float(np.mean([d[i] for d in ds])) for i in range(3))
    ds = []
    for s in SEEDS:
        w = _world(s)
        d = _decompose(w, _prompt_deltas(w))
        ds.append(d)
        for mode in ("early", "const"):
            scores["prompt"][mode].append(_score(*d, mode))
    decomp["prompt"] = tuple(float(np.mean([d[i] for d in ds])) for i in range(3))
    return scores, decomp


@pytest.fixture(scope="module")
def results():
    return _run_all()


def test_deflation_removes_bg_damage(results):
    _, decomp = results
    plain_bg = decomp["plain"][1]
    assert decomp["bgproj"][1] < 0.25 * plain_bg
    assert decomp["wholeproj"][1] < 0.25 * plain_bg


def test_fg_weighting_cuts_bg_damage_at_equal_gain(results):
    _, decomp = results
    assert decomp["fg45"][1] < 0.85 * decomp["plain"][1]
    assert decomp["fg45"][0] > 0.9 * decomp["plain"][0]


def test_cue_blend_raises_gain_at_fg_cost(results):
    _, decomp = results
    assert decomp["cue"][0] > decomp["fg45"][0]
    assert decomp["cue"][2] > decomp["fg45"][2]


def test_deflation_costs_character_gain(results):
    _, decomp = results
    assert decomp["wholeproj"][0] < decomp["bgproj"][0] < decomp["fg45"][0]


def test_rollout_and_prompt_trail_trained_arms(results):
    scores, _ = results
    for mode in ("early", "const"):
        assert float(np.mean(scores["rollout"][mode])) < float(
            np.mean(scores["fg45"][mode]))
    assert float(np.mean(scores["prompt"]["early"])) < float(
        np.mean(scores["plain"]["early"]))


def test_paired_matches_bgproj_at_early(results):
    scores, _ = results
    p = float(np.mean(scores["paired"]["early"]))
    b = float(np.mean(scores["bgproj"]["early"]))
    assert abs(p - b) < 0.25 * max(p, b)
