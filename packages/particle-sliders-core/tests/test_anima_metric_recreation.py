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

ROUND 2 (2026-11-23) continues below the Round-1 tests:
the compounding fix (per-step region weighting on a
relaxation-trajectory evaluation), a re-run gate
(verdict: improved but not validated), the cue-blend
gain-side probe, the rollout fresh-vs-self ablation,
and the zero-cost diagnostics. The full Round-2 record
lives in the section header there; the dose sweeps
remain withheld there too, for the same reason.
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


# =====================================================================
# ROUND 2 (2026-11-23): compounding fix + mechanism probes.
#
# What changed mechanically vs the Round-1 metric above:
#   (a) PER-REGION input factors: each region head gets its OWN
#       rank-3 input factor over the full state (Vs_fg, Vs_bg).
#       Round 1's single shared factor let the BG projection's
#       benefit "launder" through the shared bottleneck into FG;
#       Mikkel's per-layer LoRAs have no such coupling.
#   (b) CHARACTER x CONTEXT interaction in the teacher (ICTX=0.5):
#       the character direction itself is context-dependent,
#       char_dir(n) = normalize(c + ICTX * U_fg @ (Rmix @ n)) --
#       the Boss reads differently under different lighting.
#       Pairing averages the tilt away (pairing's measured cost);
#       whole-image projection deletes it (its measured price).
#   (c) COMPOUNDED (trajectory) evaluation replacing the single
#       delta map: the evaluated delta is the fixed point of a
#       relaxation trajectory -- the offset relaxes toward the
#       adapter's current full delta over K_TRAJ=24 steps at rate
#       S_TRAJ=0.25, the adapter ACTIVE for the first 8 steps at
#       "early" (then frozen) and all 24 at "constant", and the
#       FG45 family's BG weight (0.45) applied to the delta's BG
#       rows AT EVERY STEP. This is the sampling-compounding
#       channel Round 1's diagnosis named: per-step BG damping
#       collapses FG45's BG off-target energy to 0.030/0.045
#       (vs plain 0.356/0.653) -- the mechanism fires.
#   (d) The cue blend is read TWO ways: Round 1's blend (R1:
#       caricature tilt + amplified FG leak) and a CHARACTER-ONLY
#       blend (char: amplify the character component only, leak
#       untouched). The two readings separate cleanly (below).
# Score formula and calibration constants are Round 1's, frozen.
#
# GATE, round 2 (toy means +- seed SD, 5 seeds; Mikkel in parens):
#   arm         early (Mikkel)      const (Mikkel)
#   paired      19.91 (22.23)       16.15 (11.58)
#   cue_char    19.44 (--)          17.20 (--)
#   bgproj      18.68 (22.22)       17.29 (11.51)
#   cue (R1)    18.61 (24.93)       16.00 (16.90)
#   fg45        18.14 (25.66)       16.26 (13.06)
#   rollout     17.71 (11.61)       12.99 ( 5.49)
#   prompt      16.27 (18.32)        7.41 (--)
#   wholeproj   14.05 (18.74)       13.97 (13.72)
#   plain       18.08 (--)           6.47 (--)
# VERDICT: IMPROVED BUT NOT VALIDATED (Round 1: 2 pass / 3 miss /
# 1 near). Checklist (cue read as cue_char, the mechanism reading
# item (2) supports; R1 reading in brackets):
#   FG45 top at early ......... MISS (toy #4; paired tops. The
#     gap is no longer noise-level hand-waving: with per-step
#     compounding FG45's BG damage is 10x below plain, yet an
#     EXACT target-side cleanup still edges it at early -- in a
#     linear evaluation, damping cannot beat removal on the
#     damage axis, and FG-resident damage does not self-amplify
#     the way a real sampler elaborates it. Named residual gap.)
#   cue #2 early / #1 const ... PASS early (#2) / NEAR const
#     (#2, 0.09 behind bgproj). [R1 reading: #3 early, #5 const.]
#   paired ~= BG-only ......... PASS early (adjacent, 19.91 vs
#     18.68) / MISS const (bgproj #1, paired #4).
#   BG-vs-whole crossover ..... MISS (BG above whole at both;
#     the gap WIDENED vs Round 1's near-tie: the interaction
#     tilt makes whole-projection pay at both schedules in-toy,
#     while Mikkel's constant regime apparently RECOVERS the
#     in-span character -- a base-prior re-expression channel a
#     target-trained linear adapter does not have.)
#   rollout last .............. PASS const on means (12.99 <
#     wholeproj 13.97; per-seed fragile, NOT asserted) / MISS
#     early (wholeproj is last; rollout ~ties fg45, mean -0.43,
#     3/5 seeds -- recorded, not asserted).
#   prompt mid at early ....... NEAR (toy #6 of Mikkel's 7 --
#     the measured slot -- but with rollout/wholeproj swapped
#     around it).
# Because the gate does NOT validate, the item-(5) dose sweeps
# (w x beta dose curves as ranking evidence) remain WITHHELD,
# as in Round 1. The 6-point w scan under diagnostics is the
# Min-SNR comparison instrument item (4) names, not a sweep.
#
# ITEM (2) -- CUE BLEND MECHANISM: the 35% payoff is GAIN-SIDE
# if the blend is character-pure. Constant-strength decomposi-
# tion (gain / BG-energy / FG-energy): fg45 (1.04/0.045/0.178);
# cue_char (1.13/0.045/0.193): score-term split of the +0.94
# score gap = gain term +1.30, damage term -0.34. The R1 blend
# (1.12/0.047/0.236) earns the same gain (+1.16) but SPENDS it
# on FG damage (-1.34) and nets -0.26 vs fg45. Paired seeds:
# cue_char > fg45 at constant, 5/5 seeds, mean +0.94 (asserted).
# Inverted-U in blend dose: NOT reproduced in the char family
# (85% char = 18.34 const / 21.15 early > 35% char at both,
# 5/5 seeds at const -- a pure gain-side mechanism is monotone
# here, so it cannot be the whole story of Mikkel's 85% drop);
# the R1 family shows the downturn at constant only (14.76 <
# 16.00, 4/5 seeds) and not at early (18.82 > 18.61). Mikkel's
# early cue dip (-0.73) is reproduced by NEITHER reading (toy:
# char +1.30, R1 +0.47): his early regime prices the blend's
# non-character content harder than 8 active steps do here.
#
# ITEM (3) -- ROLLOUT ABLATION (GKD/DAgger discriminator):
# identical state-generation procedure and doses; ONLY the
# label provenance differs. Fresh = original teacher targets on
# visited states. Self = teacher target at the state's EMBODIED
# context n + drift, where drift is the adapter's own nuisance-
# span output on that state -- the label ratifies the drift
# instead of correcting it. Constant scores by dose:
#   dose        0%      20%     30%     50%     100%
#   fresh       6.47    10.48   --      12.29   12.99
#   self        --      0.53    -2.30   -8.05   -19.15
# VERDICT: SELF-TARGETS ARE BAD; ON-POLICY STATES ARE FINE --
# fresh relabeling not only rescues the rollout arm, it beats
# the no-rollout control at constant (+6.52), because visited
# states regularize plain's constant-mode instability. Mikkel's
# rollout failure (worst arm, 11.61/5.49) reads as a label-
# provenance failure, not a state-distribution failure.
# Support readout (model-collapse prediction: tail loss before
# mean loss): the in-toy signature is REFINED, not matched --
# self-labels RAISE mean and tail gain (1.79 / 1.81 at 100% vs
# fresh 1.24 / 1.18: drift ratification overshoots character)
# while the WORST case inverts (min-case gain -0.16 vs +0.19)
# and case dispersion widens (SD 1.15 vs 0.61) and damage
# energy quadruples (~2.20 vs ~0.50 implied). Support erodes
# at the worst cases first while the mean still rises.
#
# ITEM (4) -- DIAGNOSTICS:
# (4a) FG-vs-BG gradient conflict, measured where it can exist
#   (a shared input factor, Round-1 architecture): mean cosine
#   per seed [-0.166, -0.150, +0.005, -0.004, +0.357]; fraction
#   of steps with cos < 0 [0.66, 0.66, 0.55, 0.62, 0.20].
#   Conflict is WEAK and seed-dependent (pooled mean cos
#   ~ +0.01) -- Min-SNR's strong-conflict premise is not
#   cleanly present in-toy. Diagnostic w scan (train w = eval
#   w; early/const): 0.15: 15.61/14.70; 0.30: 16.67/15.50;
#   0.45: 18.14/16.26; 0.60: 19.08/15.95; 0.80: 19.24/13.18;
#   1.00: 18.08/6.47 -- interior optima exist; the CONSTANT
#   optimum sits at Mikkel's 0.45, the early optimum higher.
# (4b) LEACE price: ||P_nuis d_char||^2 / ||d_char||^2 = 0.458
#   (character-only part RHO^2 = 0.122; the rest is the inter-
#   action tilt, which lives in the FG nuisance span). Whole-
#   projection's measured early gain deficit vs BG-projection
#   is 0.318 = 0.69 x the price: the geometric price is an
#   UPPER BOUND -- the adapter recovers ~31% of the priced
#   character through the state pathway.
# (4c) End arms fail DIFFERENTLY, but not with the litera-
#   ture's exact signatures. w=0 (train+eval): fails by ABSENT
#   background (BG gain components zero by construction, tail
#   gain 0.77 lowest) with FG damage NOT elevated (0.156 <
#   fg45's 0.178) -- Break-A-Scene-style FG leakage/binding is
#   NOT reproduced in-toy; w=0-trained at eval w=1 is nearly
#   identical (BG gain 0.005: the starved BG head learned
#   nothing transferable). w=1 (plain): fails by VARIANCE --
#   constant score 6.47 +- 13.32 (some seeds near-catastroph-
#   ic), gain seed-SD 0.239 vs fg45's 0.195 -- but acquires
#   FASTER fractionally (gain@96/192 = 0.64 vs 0.51), so
#   DreamBooth's "slow" half inverts in-toy; the "noisy" half
#   holds.
#
# ROUND-1 BANKED FACTS, RE-CHECKED under the frozen metric:
# deflation removes ~99% of BG off-target energy (0.005-0.006
# vs plain 0.356-0.653) -- holds, stronger. FG45 cuts BG
# energy ~10x (0.030/0.045) but at ~0.7x plain's gain (0.88/
# 1.04 vs 1.20/1.50) -- Round 1's "at equal gain" clause FAILS
# here and is re-stated (asserted in the re-stated form).
# Deflation costs gain (whole 0.60/0.68 < BG 0.88/1.01 <=
# FG45 0.88/1.04) -- holds. Rollout(fresh) < FG45 -- holds at
# constant (asserted); early is a near-tie (recorded only).
# "Prompt below every trained arm" does NOT survive (prompt
# 16.27 > wholeproj 14.05 at early).
#
# ITERATION LOG, round 2 (mechanism rounds on item (1), then
# instrument rounds on item (3); prototypes on pop-os):
#   M1 pure iteration of the R1 map (K=16): FG45 gets WORSE --
#     BG residue feeds back coherently (BG energy 0.181 ->
#     0.261). Iteration alone is not the mechanism.
#   M2 per-step BG weight at eval: FG45 BG energy collapses
#     to 0.047 -- the named mechanism fires -- but BG gain is
#     damped too (presence/atmosphere live in BG).
#   M3 raw iterated map, K=24: dynamically unstable (plain
#     const BG energy 0.40 -> 9.55). Real samplers make
#     bounded nudges; replaced by the relaxation trajectory.
#   M4 relaxation trajectory (frozen form): bounded; const
#     projection ordering improves; FG45-top and cue-const
#     still miss.
#   M5 + character x context interaction: pairing pays its
#     measured cost; prompt lifts off the floor; char-only
#     cue beats R1 cue at both strengths.
#   M6 + per-region input factors: BEST configuration (the
#     frozen metric above). A further round (per-case char-
#     acter directions + endogenous projection contamination)
#     overshot -- wholeproj fell below prompt at early,
#     contradicting Mikkel (18.74 > 18.32) -- rejected.
#   A1 self-target v1 (amplitude read off the state and
#     blessed): unanchored positive feedback, detonates at
#     every dose -- unfaithful instrument, rejected.
#   A2 self-target v2 (context read back through the input
#     pathways): the readback carries ~0.92 SD/component of
#     estimation noise; the "self" arm was really a random-
#     context-label arm (its clean-state control detonated)
#     -- confounded instrument, rejected.
#   A3 self-target v3 (embodied-drift ratification, above):
#     bounded, decisive, and the discriminator the report
#     uses. A "ratified labels on clean states" control was
#     also dropped: with inputs held fixed, targets chasing
#     the current adapter's drift form a loop-gain >= 1 chase
#     -- an inconsistent pairing no real system implements.
#
# Assertions below encode ONLY the wide-margin / paired-seed
# facts stated above. Not asserted: the gate ranking itself,
# Jev magnitudes, the fragile mean-level positions (rollout
# last at const; rollout vs FG45 at early), and any dose
# curve.
# =====================================================================

# ---- round-2 constants (Round-1 constants above are reused) ----
R2_ICTX = 0.5                # character x context interaction
R2_K_TRAJ, R2_S_TRAJ = 24, 0.25
R2_ACTIVE = {"early": 8, "const": 24}
R2_TAIL_AMP = 2.2


def _r2_rmix(seed):
    g = torch.Generator().manual_seed(31000 + seed)
    return _randn(g, K_NUIS, K_NUIS) / (K_NUIS ** 0.5)


def _r2_char_dir(w, seed, n):
    v = w["c"] + R2_ICTX * (w["U_fg"] @ (_r2_rmix(seed) @ n))
    return v / v.norm()


def _r2_world(seed):
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
    return dict(
        c=c, U_fg=U_fg, v_car_cases=torch.stack(vcars), q=e[7],
        s_bg=Qb[:, 0], U_bg=Qb[:, 1:1 + K_NUIS],
        base_fg=_randn(g, N_CASES, D_FG) * BASE_SD,
        base_bg=_randn(g, N_CASES, D_BG) * BASE_SD,
        n_mean=_randn(g, N_CASES, K_NUIS) * SIG_MEAN,
        n_var=_randn(g, N_CASES, N_CTX, K_NUIS) * SIG_VAR,
        n_eval_var=_randn(g, N_CASES, K_NUIS) * SIG_VAR,
    )


def _r2_states(w, seed):
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
            tgt = torch.cat([A_CHAR * _r2_char_dir(w, seed, n)
                             + LEAK_FG * (w["U_fg"] @ n),
                             PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]
                             + LEAK_BG * (w["U_bg"] @ n)])
            X.append(x); T.append(tgt); NS.append(n)
    return torch.stack(X), torch.stack(T), torch.stack(NS)


def _r2_targets(w, seed, beta=0.0, pair=False, proj=None, cue_style="r1"):
    X, T, NS = _r2_states(w, seed)
    if beta > 0:
        T = T.clone()
        for idx in range(len(NS)):
            j = idx // N_CTX
            n = NS[idx]
            cd = _r2_char_dir(w, seed, n)
            base_fg = A_CHAR * cd + LEAK_FG * (w["U_fg"] @ n)
            if cue_style == "r1":
                car_dir = ((1 - CAR_TILT**2) ** 0.5 * cd
                           + CAR_TILT * w["v_car_cases"][j])
                strong_fg = (CAR_AMP * A_CHAR * car_dir
                             + CAR_LEAK * LEAK_FG * (w["U_fg"] @ n))
            else:  # char: amplify the character component only
                strong_fg = (CAR_AMP * A_CHAR * cd
                             + LEAK_FG * (w["U_fg"] @ n))
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
    return X, T, NS


def _r2_self_target(w, seed, n_true, delta):
    """Teacher target for a generated state whose embodied context
    is n_true + delta (see the Round-2 header: the label ratifies
    the adapter's drift instead of correcting it)."""
    n_hat = n_true + delta
    cd = _r2_char_dir(w, seed, n_hat)
    return torch.cat([A_CHAR * cd + LEAK_FG * (w["U_fg"] @ n_hat),
                      PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]
                      + LEAK_BG * (w["U_bg"] @ n_hat)])


def _r2_drift(w, dlt):
    """Context shift embodied in an output vector: its nuisance-
    span components, averaged over the two region spans."""
    return 0.5 * ((w["U_fg"].T @ dlt[:D_FG]) + (w["U_bg"].T @ dlt[D_FG:]))


def _r2_train(w, seed, w_bg=1.0, beta=0.0, pair=False, proj=None,
              cue_style="r1", rollout_frac=0.0, rollout_label="fresh",
              snapshots=()):
    """Per-region-factor adapter (round-2 architecture). Rollout:
    every ROLLOUT_EVERY steps a rollout_frac subset of states is
    regenerated R1-style (cue attenuated, artifact noise); labels
    are the original teacher targets ('fresh') or drift-ratified
    targets ('self'). snapshots: steps at which to also return W."""
    g = torch.Generator().manual_seed(5000 + seed)
    X, T, NS = _r2_targets(w, seed, beta=beta, pair=pair, proj=proj,
                           cue_style=cue_style)
    X0, T0 = X.clone(), T.clone()
    D = D_FG + D_BG
    Vsf = _randn(g, D, RANK) * INIT
    Vsb = _randn(g, D, RANK) * INIT
    Uf = _randn(g, D_FG, RANK) * INIT
    Ub = _randn(g, D_BG, RANK) * INIT
    N = len(X)
    snaps = {}

    def cur_W():
        return torch.cat([Uf @ Vsf.T, Ub @ Vsb.T], dim=0)

    for step in range(STEPS):
        W = cur_W()
        if rollout_frac > 0 and step > 0 and step % ROLLOUT_EVERY == 0:
            gr = torch.Generator().manual_seed(21000 + seed * 1000 + step)
            perm = torch.randperm(N, generator=gr)
            k = int(round(rollout_frac * N))
            idx = perm[:k]
            D0 = X0 @ W.T
            Xb = X0.clone()
            qcomp = Xb[:, :D_FG] @ w["q"]
            Xb[:, :D_FG] -= (1 - ROLLOUT_Q) * torch.outer(qcomp, w["q"])
            noise = _randn(gr, *D0.shape)
            noise *= (ROLLOUT_NOISE * D0.norm(dim=1, keepdim=True)
                      / (D0.shape[1] ** 0.5))
            Xgen = Xb + D0 + noise
            X = X0.clone()
            X[idx] = Xgen[idx]
            if rollout_label == "self":
                T = T0.clone()
                for i in idx.tolist():
                    T[i] = _r2_self_target(w, seed, NS[i],
                                            _r2_drift(w, D0[i] + noise[i]))
        Dlt = X @ W.T
        R = Dlt - T
        gW = (2.0 / N) * (R.T @ X)
        gWf, gWb = gW[:D_FG], gW[D_FG:]
        Uf = Uf - LR * (gWf @ Vsf + WD * Uf)
        Vsf = Vsf - LR * (gWf.T @ Uf + WD * Vsf)
        Ub = Ub - (LR * w_bg) * (gWb @ Vsb + WD * Ub)
        Vsb = Vsb - (LR * w_bg) * (gWb.T @ Ub + WD * Vsb)
        if (step + 1) in snapshots:
            snaps[step + 1] = cur_W()
    if snapshots:
        return cur_W(), snaps
    return cur_W()


def _r2_eval_inputs(w, seed, tail=False):
    g = torch.Generator().manual_seed(11000 + seed)
    sd = (SIG_MEAN**2 + SIG_VAR**2) ** 0.5
    xs, ns = [], []
    for j in range(N_CASES):
        n = w["n_mean"][j] + (R2_TAIL_AMP if tail else 1.0) * w["n_eval_var"][j]
        m_fg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
        m_bg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
        xs.append(torch.cat([w["base_fg"][j] + Q_AMP * w["q"]
                             + w["U_fg"] @ m_fg,
                             w["base_bg"][j] + w["U_bg"] @ m_bg]))
        ns.append(n)
    return xs, ns


def _r2_traj_deltas(w, seed, W, mode, w_eval=1.0, tail=False):
    """Relaxation trajectory: the offset relaxes toward the
    adapter's current full delta at each active step; BG rows of
    the delta are scaled by w_eval at EVERY step. EARLY: 8 active
    steps, then frozen; CONSTANT: all R2_K_TRAJ steps active."""
    xs, _ = _r2_eval_inputs(w, seed, tail=tail)
    out = []
    for x in xs:
        o = torch.zeros_like(x)
        for t in range(R2_K_TRAJ):
            if t < R2_ACTIVE[mode]:
                d = W @ (x + o)
                if w_eval != 1.0:
                    d = d.clone()
                    d[D_FG:] *= w_eval
                o = o + R2_S_TRAJ * (d - o)
        out.append(o)
    return out


def _r2_decompose_cases(w, seed, deltas, tail=False):
    E_n = K_NUIS * (SIG_MEAN**2 + SIG_VAR**2)
    rows = []
    for j, d in enumerate(deltas):
        n = w["n_mean"][j] + (R2_TAIL_AMP if tail else 1.0) * w["n_eval_var"][j]
        cd = _r2_char_dir(w, seed, n)
        d_fg, d_bg = d[:D_FG], d[D_FG:]
        u_atmo = w["U_bg"][:, 0]
        gain = float(d_fg @ cd + PRES_W * (d_bg @ w["s_bg"])
                     + PRES_W * (d_bg @ u_atmo))
        fg_off = d_fg - (d_fg @ cd) * cd
        bg_off = (d_bg - (d_bg @ w["s_bg"]) * w["s_bg"]
                  - (d_bg @ u_atmo) * u_atmo)
        rows.append((gain, float(bg_off @ bg_off) / E_n,
                     float(fg_off @ fg_off) / E_n))
    return rows


def _r2_decompose(w, seed, deltas, tail=False):
    rows = _r2_decompose_cases(w, seed, deltas, tail=tail)
    return tuple(float(np.mean([r[i] for r in rows])) for i in range(3))


def _r2_prompt_deltas(w, seed):
    out = []
    for j in range(N_CASES):
        n = w["n_mean"][j] + w["n_eval_var"][j]
        out.append(torch.cat([A_CHAR * _r2_char_dir(w, seed, n)
                              + LEAK_FG * (w["U_fg"] @ n),
                              PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]
                              + LEAK_BG * (w["U_bg"] @ n)]))
    return out


def _r2_arm_eval(kw, w_eval, seeds=SEEDS):
    """Train one arm per seed; scores/decomp per mode + tail gain
    + per-case gain support stats (const mode)."""
    res = {"scores": {"early": [], "const": []},
           "decomp": {"early": [], "const": []},
           "tail_gain": [], "case_gains": []}
    for s in seeds:
        w = _r2_world(s)
        W = _r2_train(w, s, **kw)
        for mode in ("early", "const"):
            d = _r2_decompose(w, s, _r2_traj_deltas(w, s, W, mode, w_eval))
            res["decomp"][mode].append(d)
            res["scores"][mode].append(_score(*d, mode))
        rows = _r2_decompose_cases(
            w, s, _r2_traj_deltas(w, s, W, "const", w_eval))
        res["case_gains"].append([r[0] for r in rows])
        dt = _r2_decompose(w, s, _r2_traj_deltas(w, s, W, "const", w_eval,
                                                tail=True), tail=True)
        res["tail_gain"].append(dt[0])
    return res


def _r2_summ(res):
    out = {}
    for mode in ("early", "const"):
        out[mode] = (float(np.mean(res["scores"][mode])),
                     float(np.std(res["scores"][mode])),
                     tuple(float(np.mean([d[i] for d in
                                          res["decomp"][mode]]))
                           for i in range(3)))
    out["tail_gain"] = float(np.mean(res["tail_gain"]))
    cg = np.array(res["case_gains"])
    out["case_gain_min"] = float(np.mean(cg.min(axis=1)))
    out["case_gain_std"] = float(np.mean(cg.std(axis=1)))
    return out


R2_GATE_ARMS = {
    "plain": (dict(), 1.0),
    "fg45": (dict(w_bg=W_FG45), W_FG45),
    "cue": (dict(w_bg=W_FG45, beta=BETA_CUE), W_FG45),
    "cue_char": (dict(w_bg=W_FG45, beta=BETA_CUE, cue_style="char"), W_FG45),
    "paired": (dict(pair=True), 1.0),
    "bgproj": (dict(proj="bg"), 1.0),
    "wholeproj": (dict(proj="whole"), 1.0),
    "rollout": (dict(rollout_frac=1.0, rollout_label="fresh"), 1.0),
}
R2_CUE_PROBES = {
    "cue85": (dict(w_bg=W_FG45, beta=0.85), W_FG45),
    "cue85_char": (dict(w_bg=W_FG45, beta=0.85, cue_style="char"), W_FG45),
}
R2_ABLATION = {
    "fresh_20": (dict(rollout_frac=0.2, rollout_label="fresh"), 1.0),
    "fresh_50": (dict(rollout_frac=0.5, rollout_label="fresh"), 1.0),
    "self_20": (dict(rollout_frac=0.2, rollout_label="self"), 1.0),
    "self_30": (dict(rollout_frac=0.3, rollout_label="self"), 1.0),
    "self_50": (dict(rollout_frac=0.5, rollout_label="self"), 1.0),
    "self_100": (dict(rollout_frac=1.0, rollout_label="self"), 1.0),
}


def _r2_diagnostics(gate):
    diag = {}
    # (4a) FG-vs-BG gradient conflict on a SHARED input factor
    # (Round-1 architecture): the seat where region gradients
    # actually share parameters.
    cos_all, frac_neg = [], []
    for s in SEEDS:
        g = torch.Generator().manual_seed(5000 + s)
        w = _r2_world(s)
        X, T, _NS = _r2_targets(w, s)
        D = D_FG + D_BG
        Vs = _randn(g, D, RANK) * INIT
        Uf = _randn(g, D_FG, RANK) * INIT
        Ub = _randn(g, D_BG, RANK) * INIT
        N = len(X)
        cs = []
        for step in range(STEPS):
            W = torch.cat([Uf @ Vs.T, Ub @ Vs.T], dim=0)
            R = X @ W.T - T
            gW = (2.0 / N) * (R.T @ X)
            gVs_f = gW[:D_FG].T @ Uf
            gVs_b = gW[D_FG:].T @ Ub
            cs.append(float((gVs_f * gVs_b).sum()
                            / (gVs_f.norm() * gVs_b.norm())))
            gVs = gVs_f + gVs_b
            Uf = Uf - LR * (gW[:D_FG] @ Vs + WD * Uf)
            Ub = Ub - LR * (gW[D_FG:] @ Vs + WD * Ub)
            Vs = Vs - LR * (gVs + WD * Vs)
        cos_all.append(float(np.mean(cs)))
        frac_neg.append(float(np.mean([1.0 if c < 0 else 0.0 for c in cs])))
    diag["conflict_cos"] = cos_all
    diag["conflict_frac_neg"] = frac_neg
    # (4a-ii) coarse diagnostic w scan (the Min-SNR comparison
    # instrument -- NOT the item-(5) dose sweep).
    diag["wscan"] = {}
    for wv in (0.15, 0.30, 0.45, 0.60, 0.80, 1.00):
        t = _r2_summ(_r2_arm_eval(dict(w_bg=wv), wv))
        diag["wscan"][wv] = (t["early"][0], t["const"][0])
    # (4b) LEACE price vs whole-projection's measured deficit.
    prices = []
    for s in SEEDS:
        w = _r2_world(s)
        vals = []
        for j in range(N_CASES):
            n = w["n_mean"][j] + w["n_eval_var"][j]
            cd = _r2_char_dir(w, s, n)
            proj = w["U_fg"].T @ cd
            vals.append(float(proj @ proj))
        prices.append(float(np.mean(vals)))
    diag["leace_price"] = float(np.mean(prices))
    g_bg = gate["bgproj"]["early"][2][0]
    g_wh = gate["wholeproj"]["early"][2][0]
    diag["leace_deficit"] = (g_bg - g_wh) / g_bg
    # (4c) end arms + acquisition snapshots.
    diag["w0"] = _r2_summ(_r2_arm_eval(dict(w_bg=0.0), 0.0))
    diag["w0_eval1"] = _r2_summ(_r2_arm_eval(dict(w_bg=0.0), 1.0))
    diag["acq"] = {}
    for name, kw, we in (("plain", dict(), 1.0),
                         ("fg45", dict(w_bg=W_FG45), W_FG45)):
        g96, g192 = [], []
        for s in SEEDS:
            w = _r2_world(s)
            W, snaps = _r2_train(w, s, snapshots=(96,), **kw)
            d96 = _r2_decompose(w, s, _r2_traj_deltas(w, s, snaps[96],
                                                      "const", we))
            d192 = _r2_decompose(w, s, _r2_traj_deltas(w, s, W, "const", we))
            g96.append(d96[0]); g192.append(d192[0])
        diag["acq"][name] = (float(np.mean(g96) / np.mean(g192)),
                             float(np.std(g192)),
                             [float(x) for x in g192])
    return diag


def _r2_run_all():
    gate = {}
    for name, (kw, we) in {**R2_GATE_ARMS, **R2_CUE_PROBES}.items():
        gate[name] = _r2_summ(_r2_arm_eval(kw, we))
    prompt = {"early": [], "const": []}
    for s in SEEDS:
        w = _r2_world(s)
        d = _r2_decompose(w, s, _r2_prompt_deltas(w, s))
        for mode in ("early", "const"):
            prompt[mode].append(_score(*d, mode))
    gate["prompt"] = {m: (float(np.mean(prompt[m])),
                          float(np.std(prompt[m])), None)
                      for m in ("early", "const")}
    ablation = {name: _r2_summ(_r2_arm_eval(kw, we))
                for name, (kw, we) in R2_ABLATION.items()}
    ablation["fresh_100"] = gate["rollout"]
    diag = _r2_diagnostics(gate)
    return {"gate": gate, "ablation": ablation, "diag": diag}


@pytest.fixture(scope="module")
def r2_results():
    return _r2_run_all()


def test_r2_deflation_removes_bg_energy(r2_results):
    gate = r2_results["gate"]
    for mode in ("early", "const"):
        plain_bg = gate["plain"][mode][2][1]
        assert gate["bgproj"][mode][2][1] < 0.05 * plain_bg
        assert gate["wholeproj"][mode][2][1] < 0.05 * plain_bg


def test_r2_fg45_cuts_bg_energy_at_reduced_gain(r2_results):
    """Round 1's 'at equal gain' clause does NOT survive the
    compounded metric: FG45 cuts BG off-target energy ~10x
    while retaining only ~0.7x of plain's gain. Asserted in
    the re-stated form."""
    gate = r2_results["gate"]
    for mode in ("early", "const"):
        ratio_e = gate["fg45"][mode][2][1] / gate["plain"][mode][2][1]
        ratio_g = gate["fg45"][mode][2][0] / gate["plain"][mode][2][0]
        assert ratio_e < 0.15
        assert 0.6 < ratio_g < 0.85


def test_r2_deflation_costs_gain(r2_results):
    gate = r2_results["gate"]
    for mode in ("early", "const"):
        assert gate["wholeproj"][mode][2][0] < gate["bgproj"][mode][2][0] - 0.15
        assert gate["bgproj"][mode][2][0] <= gate["fg45"][mode][2][0] + 0.05


def test_r2_char_cue_pays_gain_side(r2_results):
    """Character-pure cue blend beats FG45 at constant (paired
    seeds 5/5, mean +0.94) and beats the R1-style blend, whose
    FG damage spends the same gain payoff."""
    gate = r2_results["gate"]
    assert gate["cue_char"]["const"][0] > gate["fg45"]["const"][0] + 0.5
    assert gate["cue_char"]["const"][2][0] > gate["fg45"]["const"][2][0]
    assert gate["cue_char"]["const"][0] > gate["cue"]["const"][0] + 0.5
    assert gate["cue"]["const"][2][2] > gate["fg45"]["const"][2][2] + 0.02


def test_r2_blend_dose_char_monotone_r1_inverts(r2_results):
    """No inverted-U in the character-pure family (85 > 35 at
    constant, paired 5/5); the R1 family inverts at constant
    (85 < 35, mean -1.23)."""
    gate = r2_results["gate"]
    assert gate["cue85_char"]["const"][0] > gate["cue_char"]["const"][0] + 0.5
    assert gate["cue85"]["const"][0] < gate["cue"]["const"][0] - 0.5


def test_r2_gate_positions_as_measured(r2_results):
    """The frozen metric's actual positions, including its
    misses vs Mikkel: cue_char #2 at early; paired above
    bgproj at early; bgproj above wholeproj at constant (the
    crossover miss); wholeproj below prompt at early."""
    gate = r2_results["gate"]
    assert gate["cue_char"]["early"][0] > gate["fg45"]["early"][0]
    assert gate["paired"]["early"][0] > gate["bgproj"]["early"][0]
    assert gate["bgproj"]["const"][0] > gate["wholeproj"]["const"][0] + 1.0
    assert gate["wholeproj"]["early"][0] < gate["prompt"]["early"][0]


def test_r2_rollout_labels_not_states(r2_results):
    """Fresh relabeling rescues rollout (beats the no-rollout
    control at constant); self-ratified labels degrade mono-
    tonically in dose and end far below fresh at 100%."""
    ab, gate = r2_results["ablation"], r2_results["gate"]
    assert ab["fresh_100"]["const"][0] > gate["plain"]["const"][0] + 3.0
    assert ab["fresh_20"]["const"][0] < ab["fresh_100"]["const"][0] - 1.5
    assert (ab["self_20"]["const"][0] > ab["self_50"]["const"][0]
            > ab["self_100"]["const"][0])
    assert ab["self_100"]["const"][0] < ab["fresh_100"]["const"][0] - 10.0


def test_r2_rollout_support_signature(r2_results):
    """Self-labels overshoot mean gain while the worst case
    inverts and dispersion widens: support erodes at the
    worst cases first, not via tail-gain collapse."""
    ab = r2_results["ablation"]
    assert ab["self_100"]["const"][2][0] > ab["fresh_100"]["const"][2][0] + 0.2
    assert ab["self_100"]["case_gain_std"] > ab["fresh_100"]["case_gain_std"] + 0.3
    assert ab["self_100"]["case_gain_min"] < ab["fresh_100"]["case_gain_min"] - 0.2


def test_r2_rollout_below_fg45_at_const(r2_results):
    """Paired seeds 5/5 (mean -3.27). The early ordering is a
    near-tie (mean -0.43, 3/5 seeds) and is NOT asserted."""
    gate = r2_results["gate"]
    assert gate["rollout"]["const"][0] < gate["fg45"]["const"][0] - 1.5


def test_r2_leace_price_bounds_deficit(r2_results):
    diag = r2_results["diag"]
    assert 0.35 < diag["leace_price"] < 0.55
    assert 0.2 < diag["leace_deficit"] < diag["leace_price"]


def test_r2_conflict_weak_and_mixed(r2_results):
    diag = r2_results["diag"]
    assert 0.4 < float(np.mean(diag["conflict_frac_neg"])) < 0.7
    assert abs(float(np.mean(diag["conflict_cos"]))) < 0.1


def test_r2_wscan_interior_optimum(r2_results):
    """Diagnostic scan only (not the withheld dose sweep):
    constant optimum at w=0.45, early score still rising at
    w=0.80; both ends beat the w=1.00 / w=0.15 extremes named."""
    scan = r2_results["diag"]["wscan"]
    assert scan[0.45][1] > scan[1.00][1] + 3.0
    assert scan[0.80][0] > scan[0.15][0] + 2.0


def test_r2_end_arms_fail_differently(r2_results):
    """w=0 fails by absent background (zero BG energy at eval,
    weakest tail, NO FG-damage elevation); w=1 (plain) fails
    by variance (larger gain seed-SD than fg45)."""
    diag, gate = r2_results["diag"], r2_results["gate"]
    assert diag["w0"]["const"][2][1] < 1e-9
    assert diag["w0"]["tail_gain"] < gate["fg45"]["tail_gain"]
    assert diag["w0"]["const"][2][2] < gate["fg45"]["const"][2][2]
    assert diag["acq"]["plain"][1] > diag["acq"]["fg45"][1]
