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


# =====================================================================
# ROUND 3 (2026-11-23): the creative round -- variants the Round-2
# mechanism picture licenses, a final gate attempt via a base-prior
# re-expression channel, and the dose grid run as MECHANISM CURVES
# (the gate did not validate; see the verdict). Same frozen score
# and Round-2 compounded metric throughout; nothing below retunes
# any Round-1/2 constant.
#
# ITEM (1) -- FACTORIAL + COMBINATION (FG45 x pairing x char-pure
# cue; toy means, 5 seeds; R2 references: fg45 18.14/16.26,
# cue_char35 19.44/17.20, paired 19.91/16.15):
#   pair45 (fg45+pair)      17.41 / 16.42
#   cuepair (pair+char35)   20.96 / 16.86
#   combo20                 18.08 / 16.94
#   combo35                 18.57 / 17.32
#   combo50                 19.05 / 17.68
# The three winners' mechanisms are NOT fully orthogonal: pairing
# COSTS FG45 at early (-0.73, wins 1/5 -- pairing averages away
# the character x context tilt that FG45's trajectory still
# expresses, while its damage benefit is redundant with per-step
# damping) and ~ties at const (+0.15). Pairing DOES stack with
# the char cue at full BG weight (cuepair tops early at 20.96).
# The full combination is monotone in char dose (20<35<50 at
# both strengths, paired 5/5 per step) and combo50 beats FG45 at
# both strengths (paired 5/5; +0.91 early / +1.41 const) and is
# the FG45 family's const best -- but at early it still trails
# cue_char35 (19.44) and cuepair (20.96). Combination verdict:
# the const-side candidate the toy licenses; NOT an early sweep.
#
# ITEM (2) -- FRESH-RELABEL ROLLOUT STACKING: does NOT stack.
# fg45_roll50 17.48/16.31 (~ties fg45 at const, -0.66 early);
# fg45_roll100 16.41/15.68 (below fg45 at both, 0/5 early);
# combo35_roll100 18.00/17.23 (~ties combo35 at const, -0.57
# early). Round 2's rollout benefit was rescuing PLAIN's const
# instability (6.47 -> 12.99); FG45's per-step damping already
# removes that instability, so on top of the winners rollout
# only adds its state-degradation cost. It still improves the
# plain rollout arm when combined with FG45 (fg45_roll100 const
# 15.68 > plain_roll100 12.99, +2.69, 4/5) -- the rescue runs
# the other way. VERDICT: rollout fresh-relabel is a rescue for
# unstable plain training, not a component for the winners.
#
# ITEM (3) -- BASE-PRIOR RE-EXPRESSION CHANNEL + FINAL GATE:
# trajectory target becomes (1-gamma)*adapter_delta + gamma*P_j
# per step, P_j the fixed prompt-prior delta (teacher target at
# the eval context). gamma=0.35 is anchored to Mikkel's
# strengths (1-alpha_const = 0.3625; 1-alpha_early = 0.333),
# pre-registered, NOT tuned to the gate; gamma in {0.2, 0.5}
# run as sensitivity. gamma=0 reproduces Round 2 EXACTLY
# (bgproj 18.68/17.29, wholeproj 14.05/13.97 -- sanity check).
# The channel FIRES as hypothesized -- wholeproj const gain
# 0.68 -> 0.95 at gamma=0.35, recovering MORE than bgproj
# (1.01 -> 1.19) because it had more in-span character missing
# -- but the differential is ~4x too small: the const gap
# narrows 3.32 -> 1.27 and the CROSSOVER DOES NOT REPRODUCE at
# any gamma in {0.2, 0.35, 0.5} (at 0.5 wholeproj const falls
# back to 16.59 while early compresses toward prompt). The
# prior also lifts every arm and reorders little else.
# FINAL GATE at gamma=0.35 (Mikkel's order in parens):
#   FG45 top early ......... MISS (#4 of his seven, as in R2)
#   cue #2 early ........... PASS (#2)
#   cue #1 const ........... #2 (bgproj 18.34 > cue_char 17.65)
#   paired ~= bgproj ....... PASS early (delta 0.21) / MISS const
#   crossover .............. MISS (gap 1.27, no cross, any gamma)
#   rollout last ........... PASS const / MISS early (#5 of six)
#   prompt mid early ....... MISS (last; prior lifts wholeproj
#                            above it)
# GATE VERDICT, FINAL: NOT VALIDATED after three rounds. The
# misses are concentrated and mechanistically explained (a
# linear evaluation cannot let damping beat exact removal on
# the damage axis at early; the toy's priced in-span character
# share is too small for its prior channel to cross the projec-
# tions over), but explained misses are still misses. Conse-
# quence, applied consistently below: variant results are
# reported as TOY-INTERNAL margins with mechanism support, and
# item (5) is run and labeled as MECHANISM CURVES, never as
# ranking evidence against Mikkel's table.
#
# ITEM (4) -- LEACE-PRICED RESTORATION (the round's discovery):
# deflate, then add back the component of the removed part that
# lies along the ESTIMATED gain direction (normalized mean
# target of that region -- estimable from targets alone).
#   bg_restore            24.29 / 21.64   (gain 1.22/1.47,
#                         BG energy 0.038/0.057 vs plain
#                         0.356/0.653, bgproj 0.005/0.006)
#   whole_restore         17.45 / 16.89   (wholeproj: 14.05/13.97)
#   whole_restore_both    22.96 / 21.29
#   bg_restore+char35     25.44 / 22.33   (cue stacks, 5/5 both)
#   bg_restore+w45        19.15 / 17.79   (FG45 ANTAGONISTIC:
#                         -5.14/-3.85, 0/5 -- per-step damping
#                         suppresses exactly the BG gain the
#                         restoration put back)
# BG projection's real tax was never the FG character the 0.458
# LEACE price measures (restoring that piece, whole_restore,
# recovers most of whole-projection's deficit but only reaches
# the FG45 family) -- it is the BG GAIN (presence/atmosphere)
# the BG deflation silently removes: restoring the gain-aligned
# 20.6% of the removed BG energy refunds nearly all of plain's
# gain (1.47 vs 1.50 const) at ~1/10 of plain's BG damage, and
# bg_restore beats EVERY Round-2 gate arm at both strengths by
# >= +4.3. FG deflation stays net-negative even with restora-
# tion (whole_restore_both < bg_restore): never deflate FG.
# FG restoration diagnostics: restored share of removed FG
# energy 0.072; fidelity (recovered share of the true removed
# character component) 0.382 -- the mean-target direction is a
# mediocre FG character estimate, and it still pays.
#
# ITEM (5) -- DOSE GRID (w x char-beta, FG45 family, train w =
# eval w), run as MECHANISM CURVES under the final gate state:
# char-beta is MONOTONE at every w and both strengths (no
# inverted-U in the character-pure family, extending Round 2);
# the const w-optimum sits at Mikkel's 0.45 at EVERY beta
# (const column peaks: 0.45 row wins all five columns); the
# early w-optimum sits higher (0.60-0.80); w=1.00 collapses at
# const (6.47-7.34, plain's instability); grid const argmax is
# (w=0.45, beta=0.85) = 18.34 and early argmax (0.80, 0.85) =
# 22.03. Against Mikkel: his 85% cue blend DROPPED at both
# strengths -- the toy's char-pure curve cannot produce that
# drop at any dose, so the drop must live in his blend's
# non-character content (the R1 reading's tilt+leak, which DID
# invert at const in Round 2). The curves are within-family
# mechanism evidence only.
#
# ITERATION LOG, round 3 (pre-registered designs; prototypes on
# pop-os in /mnt/nvme/nova/anima-r3, proto1-5):
#   I1 factorial + rollout stacking: one design round, doses
#     pre-registered (cue 20/35/50; rollout 50/100 from R2's
#     ablation curve). No tuning.
#   I2 prior channel: gamma grid {0.2, 0.35, 0.5} around the
#     1-alpha anchor, 3 evaluation rounds; gamma=0 sanity
#     exact. No fourth gamma tried after the 0.5 fallback --
#     the crossover's absence is structural, not a gamma miss.
#   I3 restoration: round 1 bg/whole/both along the estimated
#     gain direction; round 2 combination probes (char35
#     stacks; w45 antagonistic). Estimator left as the plain
#     mean target -- no refinement attempted or needed.
#   I4 dose grid: one run as specced (7x5).
# Execution-risk flags this round: 0 occurrences.
#
# Assertions below encode ONLY the wide-margin / paired-seed
# facts stated above. Not asserted: any cross-family ranking
# against Mikkel's table, the gate itself, Jev magnitudes.
# =====================================================================

R3_GAMMA = 0.35


def _r3_traj_deltas(w, seed, W, mode, w_eval, gamma):
    """Round-2 relaxation trajectory + base-prior channel: at
    each active step the offset relaxes toward the convex mix
    (1-gamma) * adapter delta + gamma * prompt-prior delta."""
    priors = _r2_prompt_deltas(w, seed)
    xs, _ = _r2_eval_inputs(w, seed)
    out = []
    for j, x in enumerate(xs):
        o = torch.zeros_like(x)
        for t in range(R2_K_TRAJ):
            if t < R2_ACTIVE[mode]:
                d = W @ (x + o)
                if w_eval != 1.0:
                    d = d.clone()
                    d[D_FG:] *= w_eval
                tgt = (1 - gamma) * d + gamma * priors[j]
                o = o + R2_S_TRAJ * (tgt - o)
        out.append(o)
    return out


def _r3_targets(w, seed, beta=0.0, restore=None):
    """Teacher targets with optional char-pure cue blend and
    deflate+restore. restore: 'bg' (BG deflate + BG restore),
    'whole' (BG deflate + FG deflate + FG restore), 'both'
    (deflate + restore in both regions). Restoration adds back
    the removed part's component along the region's estimated
    gain direction (normalized mean target, pre-deflation)."""
    X, T, NS = _r2_states(w, seed)
    if beta > 0:
        T = T.clone()
        for idx in range(len(NS)):
            n = NS[idx]
            cd = _r2_char_dir(w, seed, n)
            base_fg = A_CHAR * cd + LEAK_FG * (w["U_fg"] @ n)
            strong_fg = CAR_AMP * A_CHAR * cd + LEAK_FG * (w["U_fg"] @ n)
            T[idx, :D_FG] = (1 - beta) * base_fg + beta * strong_fg
    g = torch.Generator().manual_seed(13000 + seed)
    T = T + _randn(g, *T.shape) * T_NOISE
    T = T.clone()
    diag = {}
    g_fg = T[:, :D_FG].mean(dim=0)
    g_fg = g_fg / g_fg.norm()
    g_bg = T[:, D_FG:].mean(dim=0)
    g_bg = g_bg / g_bg.norm()
    bgc = T[:, D_FG:] - T[:, D_FG:].mean(dim=0, keepdim=True)
    Ubg_hat = torch.linalg.svd(bgc, full_matrices=False).Vh[:K_NUIS].T
    removed_bg = (T[:, D_FG:] @ Ubg_hat) @ Ubg_hat.T
    T[:, D_FG:] = T[:, D_FG:] - removed_bg
    if restore in ("bg", "both"):
        coeff = removed_bg @ g_bg
        T[:, D_FG:] = T[:, D_FG:] + torch.outer(coeff, g_bg)
        diag["bg_share"] = float((coeff**2).sum()
                                 / (removed_bg**2).sum())
    if restore in ("whole", "both"):
        fgc = T[:, :D_FG] - T[:, :D_FG].mean(dim=0, keepdim=True)
        Ufg_hat = torch.linalg.svd(fgc, full_matrices=False).Vh[:K_NUIS].T
        removed_fg = (T[:, :D_FG] @ Ufg_hat) @ Ufg_hat.T
        T[:, :D_FG] = T[:, :D_FG] - removed_fg
        coeff = removed_fg @ g_fg
        T[:, :D_FG] = T[:, :D_FG] + torch.outer(coeff, g_fg)
        diag["fg_share"] = float((coeff**2).sum()
                                 / (removed_fg**2).sum())
        num, den = 0.0, 0.0
        for i in range(len(NS)):
            cd = _r2_char_dir(w, seed, NS[i])
            true_char = float(removed_fg[i] @ cd)
            num += float(coeff[i]) * true_char
            den += true_char**2
        diag["fg_fidelity"] = num / max(den, 1e-12)
    return X, T, diag


def _r3_train(w, seed, beta=0.0, restore=None, w_bg=1.0):
    g = torch.Generator().manual_seed(5000 + seed)
    X, T, diag = _r3_targets(w, seed, beta=beta, restore=restore)
    D = D_FG + D_BG
    Vsf = _randn(g, D, RANK) * INIT
    Vsb = _randn(g, D, RANK) * INIT
    Uf = _randn(g, D_FG, RANK) * INIT
    Ub = _randn(g, D_BG, RANK) * INIT
    N = len(X)
    for step in range(STEPS):
        W = torch.cat([Uf @ Vsf.T, Ub @ Vsb.T], dim=0)
        R = X @ W.T - T
        gW = (2.0 / N) * (R.T @ X)
        gWf, gWb = gW[:D_FG], gW[D_FG:]
        Uf = Uf - LR * (gWf @ Vsf + WD * Uf)
        Vsf = Vsf - LR * (gWf.T @ Uf + WD * Vsf)
        Ub = Ub - (LR * w_bg) * (gWb @ Vsb + WD * Ub)
        Vsb = Vsb - (LR * w_bg) * (gWb.T @ Ub + WD * Vsb)
    return torch.cat([Uf @ Vsf.T, Ub @ Vsb.T], dim=0), diag


def _r3_restore_eval(beta, restore, w_bg, w_eval):
    res = {"scores": {"early": [], "const": []},
           "decomp": {"early": [], "const": []}, "diag": []}
    for s in SEEDS:
        w = _r2_world(s)
        W, diag = _r3_train(w, s, beta=beta, restore=restore, w_bg=w_bg)
        res["diag"].append(diag)
        for mode in ("early", "const"):
            d = _r2_decompose(w, s, _r2_traj_deltas(w, s, W, mode, w_eval))
            res["decomp"][mode].append(d)
            res["scores"][mode].append(_score(*d, mode))
    return res


def _r3_mean(res, mode):
    return float(np.mean(res["scores"][mode]))


def _r3_dmean(res, mode, i):
    return float(np.mean([d[i] for d in res["decomp"][mode]]))


R3_COMBO_ARMS = {
    "pair45": (dict(w_bg=W_FG45, pair=True), W_FG45),
    "cuepair": (dict(pair=True, beta=BETA_CUE, cue_style="char"), 1.0),
    "combo20": (dict(w_bg=W_FG45, pair=True, beta=0.20,
                     cue_style="char"), W_FG45),
    "combo35": (dict(w_bg=W_FG45, pair=True, beta=0.35,
                     cue_style="char"), W_FG45),
    "combo50": (dict(w_bg=W_FG45, pair=True, beta=0.50,
                     cue_style="char"), W_FG45),
    "fg45_roll50": (dict(w_bg=W_FG45, rollout_frac=0.5,
                         rollout_label="fresh"), W_FG45),
    "fg45_roll100": (dict(w_bg=W_FG45, rollout_frac=1.0,
                          rollout_label="fresh"), W_FG45),
    "combo35_roll100": (dict(w_bg=W_FG45, pair=True, beta=BETA_CUE,
                             cue_style="char", rollout_frac=1.0,
                             rollout_label="fresh"), W_FG45),
}


@pytest.fixture(scope="module")
def r3_combo():
    return {name: _r2_arm_eval(kw, we)
            for name, (kw, we) in R3_COMBO_ARMS.items()}


@pytest.fixture(scope="module")
def r3_prior():
    out = {}
    for name, (kw, we) in R2_GATE_ARMS.items():
        sc = {"early": [], "const": []}
        dec = {"early": [], "const": []}
        for s in SEEDS:
            w = _r2_world(s)
            W = _r2_train(w, s, **kw)
            for mode in ("early", "const"):
                d = _r2_decompose(
                    w, s, _r3_traj_deltas(w, s, W, mode, we, R3_GAMMA))
                sc[mode].append(_score(*d, mode))
                dec[mode].append(d)
        out[name] = {"scores": sc, "decomp": dec}
    return out


@pytest.fixture(scope="module")
def r3_restore():
    return {
        "bg_restore": _r3_restore_eval(0.0, "bg", 1.0, 1.0),
        "whole_restore": _r3_restore_eval(0.0, "whole", 1.0, 1.0),
        "whole_restore_both": _r3_restore_eval(0.0, "both", 1.0, 1.0),
        "bg_restore_cue": _r3_restore_eval(BETA_CUE, "bg", 1.0, 1.0),
        "bg_restore_w45": _r3_restore_eval(0.0, "bg", W_FG45, W_FG45),
    }


@pytest.fixture(scope="module")
def r3_grid():
    grid = {}
    for wv in (0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0):
        for bv in (0.0, 0.15, 0.35, 0.55, 0.85):
            sc = {"early": [], "const": []}
            for s in SEEDS:
                w = _r2_world(s)
                W = _r2_train(w, s, w_bg=wv, beta=bv, cue_style="char")
                for mode in ("early", "const"):
                    d = _r2_decompose(
                        w, s, _r2_traj_deltas(w, s, W, mode, wv))
                    sc[mode].append(_score(*d, mode))
            grid[(wv, bv)] = (float(np.mean(sc["early"])),
                              float(np.mean(sc["const"])))
    return grid


def test_r3_combo50_beats_fg45(r3_combo, r2_results):
    fg = r2_results["gate"]["fg45"]
    c50 = _r2_summ(r3_combo["combo50"])
    assert c50["early"][0] > fg["early"][0] + 0.5
    assert c50["const"][0] > fg["const"][0] + 0.75


def test_r3_combo_dose_monotone(r3_combo):
    c20 = _r2_summ(r3_combo["combo20"])
    c35 = _r2_summ(r3_combo["combo35"])
    c50 = _r2_summ(r3_combo["combo50"])
    for mode in ("early", "const"):
        assert c20[mode][0] < c35[mode][0] < c50[mode][0]


def test_r3_pairing_not_orthogonal(r3_combo, r2_results):
    """Pairing costs FG45 at early (tilt-averaging vs damping
    redundancy) but stacks with the char cue at full BG weight:
    cuepair tops paired at early by > 0.5."""
    fg = r2_results["gate"]["fg45"]
    p45 = _r2_summ(r3_combo["pair45"])
    cp = _r2_summ(r3_combo["cuepair"])
    pr = r2_results["gate"]["paired"]
    assert p45["early"][0] < fg["early"][0] - 0.25
    assert cp["early"][0] > pr["early"][0] + 0.5


def test_r3_rollout_does_not_stack(r3_combo, r2_results):
    """Fresh-relabel rollout adds nothing on top of FG45 or the
    combination (it only rescues plain): fg45_roll100 below
    fg45 at both strengths; fg45_roll50 and combo35_roll100
    within noise of their bases at const; FG45 still improves
    the rollout arm itself (the rescue runs the other way)."""
    fg = r2_results["gate"]["fg45"]
    ro = r2_results["gate"]["rollout"]
    r50 = _r2_summ(r3_combo["fg45_roll50"])
    r100 = _r2_summ(r3_combo["fg45_roll100"])
    cr = _r2_summ(r3_combo["combo35_roll100"])
    c35 = _r2_summ(r3_combo["combo35"])
    assert r100["early"][0] < fg["early"][0] - 1.0
    assert r100["const"][0] < fg["const"][0]
    assert r50["const"][0] < fg["const"][0] + 0.25
    assert cr["const"][0] < c35["const"][0] + 0.25
    assert r100["const"][0] > ro["const"][0] + 1.5


def test_r3_prior_channel_fires_no_crossover(r3_prior, r2_results):
    """The base-prior channel re-expresses in-span character
    (wholeproj const gain rises > 0.15 over the gamma=0 gate)
    but the BG-vs-whole crossover still does NOT reproduce at
    gamma=0.35, and FG45 is still not top at early -- the
    gate's final misses, asserted as measured positions."""
    wp0 = r2_results["gate"]["wholeproj"]["const"][2][0]
    wp1 = float(np.mean([d[0] for d in
                         r3_prior["wholeproj"]["decomp"]["const"]]))
    assert wp1 > wp0 + 0.15
    bg_c = float(np.mean(r3_prior["bgproj"]["scores"]["const"]))
    wp_c = float(np.mean(r3_prior["wholeproj"]["scores"]["const"]))
    assert bg_c > wp_c
    fg_e = float(np.mean(r3_prior["fg45"]["scores"]["early"]))
    bg_e = float(np.mean(r3_prior["bgproj"]["scores"]["early"]))
    assert fg_e < bg_e


def test_r3_bg_restore_dominates(r3_restore, r2_results):
    """BG deflation + gain-span restoration beats both the
    plain BG projection and FG45 by wide margins at both
    strengths, at ~1/10 of plain's BG damage."""
    gate = r2_results["gate"]
    br = r3_restore["bg_restore"]
    for mode in ("early", "const"):
        assert _r3_mean(br, mode) > gate["bgproj"][mode][0] + 3.0
        assert _r3_mean(br, mode) > gate["fg45"][mode][0] + 3.0
    assert _r3_dmean(br, "const", 0) > 1.35
    assert _r3_dmean(br, "const", 1) < 0.10
    assert _r3_dmean(br, "early", 1) < 0.08


def test_r3_whole_restore_and_fg_deflation(r3_restore, r2_results):
    """FG restoration recovers most of whole-projection's
    deficit (+2 or more at both strengths) but FG deflation
    stays net-negative: restoring both regions still loses
    to BG-only deflation + restoration."""
    gate = r2_results["gate"]
    wr = r3_restore["whole_restore"]
    wb = r3_restore["whole_restore_both"]
    br = r3_restore["bg_restore"]
    for mode in ("early", "const"):
        assert _r3_mean(wr, mode) > gate["wholeproj"][mode][0] + 2.0
    assert _r3_mean(wb, "early") < _r3_mean(br, "early") - 0.5
    assert _r3_mean(wb, "const") < _r3_mean(br, "const")


def test_r3_restore_combinations(r3_restore):
    """The char cue stacks on restoration (paired gains at
    both strengths); FG45 weighting is antagonistic (per-step
    damping suppresses the restored BG gain)."""
    br = r3_restore["bg_restore"]
    bc = r3_restore["bg_restore_cue"]
    bw = r3_restore["bg_restore_w45"]
    for mode in ("early", "const"):
        assert _r3_mean(bc, mode) > _r3_mean(br, mode) + 0.4
        assert _r3_mean(bw, mode) < _r3_mean(br, mode) - 2.0


def test_r3_restore_shares(r3_restore):
    """Restored shares of the removed energy: BG ~0.21 (the
    gain-aligned fifth), FG ~0.07 with fidelity ~0.38."""
    br = r3_restore["bg_restore"]
    wr = r3_restore["whole_restore"]
    bg_share = float(np.mean([d["bg_share"] for d in br["diag"]]))
    fg_share = float(np.mean([d["fg_share"] for d in wr["diag"]]))
    fid = float(np.mean([d["fg_fidelity"] for d in wr["diag"]]))
    assert 0.10 < bg_share < 0.35
    assert 0.03 < fg_share < 0.15
    assert 0.2 < fid < 0.6


def test_r3_grid_mechanism_curves(r3_grid):
    """MECHANISM CURVES ONLY (gate not validated): char-beta
    monotone at w=0.45; the const w-optimum sits at 0.45 at
    beta=0.35; the early optimum sits higher; w=1 collapses
    at const."""
    g = r3_grid
    chain = [g[(0.45, b)][1] for b in (0.0, 0.15, 0.35, 0.55, 0.85)]
    assert all(a < b for a, b in zip(chain, chain[1:]))
    assert g[(0.45, 0.35)][1] > g[(0.30, 0.35)][1]
    assert g[(0.45, 0.35)][1] > g[(0.60, 0.35)][1] + 0.25
    assert g[(0.80, 0.0)][0] > g[(0.45, 0.0)][0] + 0.5
    assert g[(1.0, 0.35)][1] < g[(0.45, 0.35)][1] - 5.0
# =====================================================================
# ROUND 4 (2026-11-24): GROUNDED rebuild + candidate generation.
# Canonical status (Round-5 reconciliation, 2026-11-24): RUN on
# pop-os (torch 2.15, pytest 9.2, single-threaded) and PUSHED
# to PR #321 with this reconciliation. The original torch port
# first ran 36 passed / 1 failed -- test_r4_k6_tops_
# candidates_at_early (K6 early 27.97 vs fg45 29.51). The
# section below was then reconciled against the numpy
# reference model (~/workspace/.tmp/anima-round4/grounded.py)
# and now encodes ONLY stream-robust facts (see below); the
# full file is green on pop-os.
#
# WORLD-GENERATOR NOTE (one canonical definition, two
# streams): the numpy reference and this torch port share one
# world DEFINITION -- same distributions, same draw
# structure, same seed offsets (main world stream 7000+seed,
# contrasts 9000+seed, matte 41000+seed, training 5000+seed,
# char mixing 31000+seed, cue-tilt mixing 31000+seed+777) --
# with library-specific RNG streams (numpy default_rng
# sequential draws vs torch Generators; the matte is drawn
# as direct Beta samples in numpy and as Gamma ratios in
# torch, the same Beta distribution). Per-seed VALUES
# therefore differ between the implementations, and every
# assertion below encodes a distribution-level agreement
# measured in BOTH, with its margin set at roughly half the
# thinner of the two implementations' margins.
#
# Two genuine STRUCTURAL (non-stream) divergences were found
# in the original port and FIXED to match the reference:
# (1) the cue tilt reused the char direction's mixing matrix
# (_r2_rmix(seed)) where the reference draws an independent
# one (_rmix(seed, 777)) -- the port's cue was an amplified
# char tilt, not an independent rotation, and its measured
# rotation was 0.129 vs the reference's 0.186 (after the
# fix: 0.187 vs 0.186); (2) alt_dir was drawn from a fresh
# 7000+seed generator (numbers overlapping the Qf draw)
# where the reference draws it inside the main world stream,
# after the crest vectors and before the base/nuisance
# draws -- _r4_world now mirrors the reference draw order.
#
# STREAM-SENSITIVE (no assertion encodes these): the K6
# EARLY claim -- K6 vs fg45 at early is +0.67 in the numpy
# reference but -1.54 in the original torch port (27.97 vs
# 29.51) and -0.38 in the structurally fixed torch port: the
# sign itself is implementation-dependent, so K6's early
# crown is DOWNGRADED to implementation-sensitive. Likewise
# K6 vs mix35 at const is +0.78 (numpy) / +0.51 (original
# torch) / -0.09 (fixed torch): a statistical tie whose sign
# flips across streams. What both implementations DO agree
# on about K6 is asserted in test_r4_k6_candidates_agree.
#
# What changed vs Rounds 1-3 (re-derived from
# release_tools/final_boss_targets.py; workstream-A grounding):
#   (i) weighting moves to TARGET construction with exact L2
#       renorm; the training-time w_bg knob is DELETED (stock
#       unweighted training -- grounding D1/D2);
#   (ii) matte = per-position CONTINUOUS weights (per-case draw,
#       FG dims bright-leaning, BG dims dark-leaning, overall
#       mean ~0.28; real mattes 0.19-0.30) (D3);
#   (iii) cue = ROTATION toward a distinct specific-cue contrast
#       (shared char axis + own tilt + per-case crest object) at
#       fixed norm; measured rotation at beta=0.35 ~0.19 (real
#       0.220 early positions / 0.168 late) (D4). The 85%/dose100
#       arm is EXCLUDED: construction unknown (D5);
#   (iv) score = per case max(0, g) * (1 - d)^2, equal mean over
#       cases, x28 display scale; damage map d = e/(e+1) with e
#       the off-rubric energy ratio -- stated and frozen (D10);
#   (v) alpha = RENDER profiles over a 10-step trajectory:
#       early [1x7, 0.64444, 0.33333, 0.33333], const [0.6375]*10;
#       additive accumulation + base pullback of off-rubric
#       content only (half-life = render length, pull 0.067).
#       No gamma base-prior channel (D11/D12).
# Projection = orthogonalisation vs the top-1 PC of centred BG
# residual targets (stated estimator), before weighting+renorm.
# Paired contexts and rollout (self-label) are kept as R2
# operationalisations and flagged UNGROUNDED (constructions are
# missing from the anima repo; grounding S3/S5).
#
# GATE (numpy reference, 5 seeds): NOT VALIDATED --
# 2 PASS (rollout last by a wide margin, both strengths;
# paired@65 > paired@45, both strengths), 2 NEAR (cue early dip
# sign; prompt-only mid-pack), 5 MISS (FG45 top at early; floor
# non-monotonicity; cue const crown; paired ~= bgproj; BG-vs-
# whole crossover). Grounding ATTRIBUTES the FG45 miss: the
# R1-R3 toy's FG45 power came from mechanisms that do not exist
# in the code (per-step compounding, training weights); target
# direction at fixed dose is a weak knob for a linear fit.
#
# CANDIDATES (numpy reference; in-toy champions fg45 33.59/
# 34.72, fg45+mix35 32.35/33.40): K6 paired+mix35@65 was the
# only candidate positive vs fg45 at early in the reference
# (+0.67, 3/5) -- implementation-sensitive per the note above;
# its const margin vs mix35 (+0.78, 4/5) is a cross-stream tie.
# K2 mean35 >= single cue at both strengths in the reference;
# K7 timed cue +1.03/+1.07 vs full cue at const (5/5); K5
# bgproj+mix35 is the worst arm (-2.39 early, 0/5).
# Round 1-3 verdict revisions: deflate-restore-cue35 DEAD (no
# PCA in the real system; the capture renorm already restores
# the dose its "restore" step refunded); soft45-pair-char50
# REVISED -> K6/K2 ("char-pure" amplification does not exist;
# the cue is a rotation); soft45-char85 DEAD (no 85% blend
# exists; the arm is dose100, construction unknown).
# =====================================================================

# ---- round-4 constants (Round-1 constants above are reused) ----
R4_D0 = 1.0
R4_ALPHA_EARLY = [1.0] * 7 + [0.64444, 0.33333, 0.33333]
R4_ALPHA_CONST = [0.6375] * 10
R4_CREST, R4_SPEC_TILT, R4_ALT_TILT = 0.55, 0.30, 0.45
R4_KAPPA_SPEC, R4_BASE_PULL = 0.5, 0.067


def _r4_world(seed):
    # Round-5 structural reconciliation: this mirrors the numpy
    # reference's _world draw structure exactly (same seed
    # offsets, same draw order): main stream 7000+seed draws
    # Qf, Qb, the per-case crest vectors, then alt_dir (ONE
    # randn, orthogonalised against c and U_fg) BEFORE the
    # base / nuisance draws it therefore shifts. The original
    # port instead reused _r2_world (no alt slot) and drew
    # alt_dir from a FRESH 7000+seed generator -- its first
    # randn, i.e. numbers overlapping the Qf draw -- a
    # different draw structure, fixed here.
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
    alt = _randn(g, D_FG)
    alt = alt - c * (alt @ c)
    for k in range(K_NUIS):
        alt = alt - U_fg[:, k] * (alt @ U_fg[:, k])
    w = dict(
        c=c, U_fg=U_fg, v_car_cases=torch.stack(vcars), q=e[7],
        s_bg=Qb[:, 0], U_bg=Qb[:, 1:1 + K_NUIS],
        base_fg=_randn(g, N_CASES, D_FG) * BASE_SD,
        base_bg=_randn(g, N_CASES, D_BG) * BASE_SD,
        n_mean=_randn(g, N_CASES, K_NUIS) * SIG_MEAN,
        n_var=_randn(g, N_CASES, N_CTX, K_NUIS) * SIG_VAR,
        n_eval_var=_randn(g, N_CASES, K_NUIS) * SIG_VAR,
        alt_dir=alt / alt.norm(),
    )
    matte = torch.empty(N_CASES, D_FG + D_BG, dtype=torch.float64)
    # Beta draws via gamma ratios: Gamma(a,1)/(Gamma(a,1) +
    # Gamma(b,1)) ~ Beta(a, b), seeded per world. (The numpy
    # reference draws Beta directly from its own 41000+seed
    # stream; same distribution, library-specific stream.)
    gen = torch.Generator().manual_seed(41000 + seed)

    def beta_g(a, b, shape):
        ga = torch._standard_gamma(
            torch.full(shape, float(a), dtype=torch.float64),
            generator=gen)
        gb = torch._standard_gamma(
            torch.full(shape, float(b), dtype=torch.float64),
            generator=gen)
        return ga / (ga + gb)

    matte[:, :D_FG] = beta_g(8.0, 2.0, (N_CASES, D_FG))
    matte[:, D_FG:] = beta_g(1.2, 9.0, (N_CASES, D_BG))
    w["matte"] = matte
    return w


def _r4_rmix_tilt(seed):
    """The spec/alt cue tilt's OWN mixing matrix. Round-5
    structural fix: the numpy reference draws it from
    31000+seed+777, INDEPENDENT of the char direction's
    _r2_rmix(seed); the original port reused _r2_rmix(seed),
    which made the cue tilt a scalar amplification of the
    char tilt (spec = char with a larger context coefficient
    + crest) instead of an independent rotation, and shrank
    the measured cue rotation (0.129 in torch vs 0.186 in
    the reference, same band test)."""
    g = torch.Generator().manual_seed(31000 + seed + 777)
    return _randn(g, K_NUIS, K_NUIS) / (K_NUIS ** 0.5)


def _r4_spec_dir(w, seed, n, j):
    v = (_r2_char_dir(w, seed, n)
         + R4_SPEC_TILT * (w["U_fg"] @ (_r4_rmix_tilt(seed) @ n))
         + R4_CREST * w["v_car_cases"][j])
    return v / v.norm()


def _r4_alt_dir(w, seed, n):
    v = (_r2_char_dir(w, seed, n)
         + R4_ALT_TILT * w["alt_dir"]
         + R4_SPEC_TILT * (w["U_fg"] @ (_r4_rmix_tilt(seed) @ n)))
    return v / v.norm()


def _r4_contrasts(w, seed):
    g = torch.Generator().manual_seed(9000 + seed)
    sd = (SIG_MEAN**2 + SIG_VAR**2) ** 0.5
    X, NEU, RES, EXP, ALT, JS, PS = [], [], [], [], [], [], []
    idx = 0
    for j in range(N_CASES):
        for t in range(N_CTX):
            n = w["n_mean"][j] + w["n_var"][j, t]
            m_fg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
            m_bg = RHO_M * n + (1 - RHO_M**2) ** 0.5 * _randn(g, K_NUIS) * sd
            x = torch.cat([w["base_fg"][j] + Q_AMP * w["q"] + w["U_fg"] @ m_fg,
                           w["base_bg"][j] + w["U_bg"] @ m_bg])
            leak = torch.cat([LEAK_FG * (w["U_fg"] @ n),
                              LEAK_BG * (w["U_bg"] @ n)])
            bg_gain = torch.cat([torch.zeros(D_FG, dtype=torch.float64),
                                 PRES * w["s_bg"] + ATMO * w["U_bg"][:, 0]])
            cd = _r2_char_dir(w, seed, n)
            res = torch.cat([A_CHAR * cd,
                             torch.zeros(D_BG, dtype=torch.float64)]) + bg_gain
            exp = torch.cat([A_CHAR * _r4_spec_dir(w, seed, n, j),
                             torch.zeros(D_BG, dtype=torch.float64)]) + bg_gain
            alt = torch.cat([A_CHAR * _r4_alt_dir(w, seed, n),
                             torch.zeros(D_BG, dtype=torch.float64)]) + bg_gain
            X.append(x); NEU.append(leak); RES.append(res)
            EXP.append(exp); ALT.append(alt); JS.append(j)
            PS.append(idx % 10); idx += 1
    return (torch.stack(X), torch.stack(NEU), torch.stack(RES),
            torch.stack(EXP), torch.stack(ALT),
            torch.tensor(JS), torch.tensor(PS))


def _r4_targets(w, seed, floor=1.0, beta=0.0, cue="single",
                pair=False, proj=None, matte_pow=1.0,
                cue_positions=None):
    X, NEU, RES, EXP, ALT, JS, PS = _r4_contrasts(w, seed)
    E = RES.clone()
    if beta > 0:
        D = EXP if cue == "single" else 0.5 * (EXP + ALT)
        b = torch.full((len(E),), float(beta), dtype=torch.float64)
        if cue_positions is not None:
            mask = torch.tensor([1.0 if cue_positions(int(p)) else 0.0
                                 for p in PS], dtype=torch.float64)
            b = b * mask
        E = (1 - b)[:, None] * RES + b[:, None] * D
    if pair:
        E = E.clone()
        for j in range(N_CASES):
            sl = slice(j * N_CTX, (j + 1) * N_CTX)
            E[sl] = E[sl].mean(dim=0, keepdim=True)
    if proj in ("bg", "whole"):
        E = E.clone()
        bgc = E[:, D_FG:] - E[:, D_FG:].mean(dim=0, keepdim=True)
        u_bg = torch.linalg.svd(bgc, full_matrices=False).Vh[0]
        E[:, D_FG:] = E[:, D_FG:] - torch.outer(E[:, D_FG:] @ u_bg, u_bg)
        if proj == "whole":
            fgc = E[:, :D_FG] - E[:, :D_FG].mean(dim=0, keepdim=True)
            u_fg = torch.linalg.svd(fgc, full_matrices=False).Vh[0]
            E[:, :D_FG] = E[:, :D_FG] - torch.outer(E[:, :D_FG] @ u_fg, u_fg)
    T = NEU.clone()
    for i in range(len(E)):
        m = w["matte"][int(JS[i])] ** matte_pow
        wt = floor + (1 - floor) * m
        weighted = E[i] * wt
        T[i] = NEU[i] + weighted * (E[i].norm() / weighted.norm())
    return X, T


def _r4_train(w, seed, **tkw):
    """Stock unweighted training (R2 architecture, NO w_bg:
    weighting lives in the targets now)."""
    g = torch.Generator().manual_seed(5000 + seed)
    X, T = _r4_targets(w, seed, **tkw)
    D = D_FG + D_BG
    Vsf = _randn(g, D, RANK) * INIT
    Vsb = _randn(g, D, RANK) * INIT
    Uf = _randn(g, D_FG, RANK) * INIT
    Ub = _randn(g, D_BG, RANK) * INIT
    N = len(X)
    for step in range(STEPS):
        W = torch.cat([Uf @ Vsf.T, Ub @ Vsb.T], dim=0)
        R = X @ W.T - T
        gW = (2.0 / N) * (R.T @ X)
        gWf, gWb = gW[:D_FG], gW[D_FG:]
        Uf = Uf - LR * (gWf @ Vsf + WD * Uf)
        Vsf = Vsf - LR * (gWf.T @ Uf + WD * Vsf)
        Ub = Ub - LR * (gWb @ Vsb + WD * Ub)
        Vsb = Vsb - LR * (gWb.T @ Ub + WD * Vsb)
    return torch.cat([Uf @ Vsf.T, Ub @ Vsb.T], dim=0)


def _r4_render(w, seed, W, mode):
    prof = R4_ALPHA_EARLY if mode == "early" else R4_ALPHA_CONST
    xs, _ = _r2_eval_inputs(w, seed)
    out = []
    for j, x in enumerate(xs):
        n = w["n_mean"][j] + w["n_eval_var"][j]
        cd = _r2_char_dir(w, seed, n)
        sd_ = _r4_spec_dir(w, seed, n, j)
        rem = sd_ - (sd_ @ cd) * cd
        rem = rem / rem.norm()
        B = torch.zeros(D_FG + D_BG, 4, dtype=torch.float64)
        B[:D_FG, 0] = cd; B[:D_FG, 1] = rem
        B[D_FG:, 2] = w["s_bg"]; B[D_FG:, 3] = w["U_bg"][:, 0]
        o = torch.zeros_like(x)
        for a in prof:
            o = o + 0.25 * a * (W @ (x + o))
            coef = B.T @ o
            o = B @ coef + (1 - R4_BASE_PULL) * (o - B @ coef)
        out.append(o)
    return out


def _r4_case_rows(w, seed, deltas):
    E_n = K_NUIS * (SIG_MEAN**2 + SIG_VAR**2)
    rows = []
    for j, d in enumerate(deltas):
        n = w["n_mean"][j] + w["n_eval_var"][j]
        cd = _r2_char_dir(w, seed, n)
        sd_ = _r4_spec_dir(w, seed, n, j)
        rem = sd_ - (sd_ @ cd) * cd
        rem = rem / rem.norm()
        d_fg, d_bg = d[:D_FG], d[D_FG:]
        u_atmo = w["U_bg"][:, 0]
        gain = float(d_fg @ cd + R4_KAPPA_SPEC * (d_fg @ sd_)
                     + PRES_W * (d_bg @ w["s_bg"])
                     + PRES_W * (d_bg @ u_atmo))
        fg_off = d_fg - (d_fg @ cd) * cd
        fg_off = fg_off - (fg_off @ rem) * rem
        bg_off = (d_bg - (d_bg @ w["s_bg"]) * w["s_bg"]
                  - (d_bg @ u_atmo) * u_atmo)
        e = (float(bg_off @ bg_off) + float(fg_off @ fg_off)) / E_n
        rows.append((gain, e))
    return rows


def _r4_score_rows(rows):
    tot = 0.0
    for gain, e in rows:
        d = e / (e + R4_D0)
        tot += max(0.0, gain) * (1 - d) ** 2
    return KSCALE * tot / len(rows)


def _r4_arm(kw, seeds=SEEDS):
    sc = {"early": [], "const": []}
    for s in seeds:
        w = _r4_world(s)
        W = _r4_train(w, s, **kw)
        for mode in ("early", "const"):
            sc[mode].append(_r4_score_rows(
                _r4_case_rows(w, s, _r4_render(w, s, W, mode))))
    return {m: (float(np.mean(sc[m])), sc[m]) for m in sc}


R4_GATE_ARMS = {
    "prompt": dict(floor=1.0),
    "fg45": dict(floor=0.45),
    "fg65": dict(floor=0.65),
    "mix35": dict(floor=0.45, beta=0.35),
    "paired45": dict(floor=0.45, pair=True),
    "paired65": dict(floor=0.65, pair=True),
    "bgproj45": dict(floor=0.45, proj="bg"),
    "wholeproj65": dict(floor=0.65, proj="whole"),
}
R4_CANDIDATES = {
    "k6": dict(floor=0.65, beta=0.35, pair=True),
    "k2": dict(floor=0.45, beta=0.35, cue="mean"),
    "k7_early3": dict(floor=0.45, beta=0.35,
                      cue_positions=lambda p: p < 3),
    "k5": dict(floor=0.45, beta=0.35, proj="bg"),
}


@pytest.fixture(scope="module")
def r4_results():
    out = {}
    for name, kw in {**R4_GATE_ARMS, **R4_CANDIDATES}.items():
        out[name] = _r4_arm(kw)
    return out


def test_r4_capture_renorm_is_exact():
    """STREAM-ROBUST (algebraic identity, not a statistic):
    the grounded capture restores the original L2 dose
    exactly, for every floor (floor = direction only).
    Numpy reference max error 7e-16; torch asserts <1e-9."""
    for s in SEEDS[:2]:
        w = _r4_world(s)
        _, NEU, RES, _, _, _, _ = _r4_contrasts(w, s)
        for fl in (0.30, 0.45, 0.65, 1.0):
            _, T = _r4_targets(w, s, floor=fl)
            for i in range(len(RES)):
                assert abs(float((T[i] - NEU[i]).norm()
                                 - RES[i].norm())) < 1e-9


def test_r4_matte_mean_in_real_band():
    """STREAM-ROBUST (distribution property): per-seed matte
    means measure 0.281-0.287 in the numpy reference and
    0.282-0.295 in torch -- both inside the real-matte band
    and inside the tightened band below, whose edges sit at
    roughly half the thinner implementation margin beyond
    the measured range."""
    ms = [float(_r4_world(s)["matte"].mean()) for s in SEEDS]
    assert all(0.24 < m < 0.31 for m in ms)


def test_r4_cue_rotation_matches_measured_band():
    """STREAM-ROBUST after the Round-5 structural fix (the
    cue tilt's independent mixing matrix): beta=0.35 rotates
    the edit by 0.186 of its norm in the numpy reference and
    0.187 in torch (real: 0.220 early positions / 0.168
    late). The band below is tightened to roughly half the
    thinner margin around the two measured values. (The
    original port, reusing the char mixing matrix, measured
    0.129 -- the structural divergence the fix removed.)"""
    fracs = []
    for s in SEEDS:
        w = _r4_world(s)
        _, _, RES, EXP, _, _, _ = _r4_contrasts(w, s)
        for i in range(len(RES)):
            mixed = 0.65 * RES[i] + 0.35 * EXP[i]
            fracs.append(float((mixed - RES[i]).norm() / RES[i].norm()))
    assert 0.15 < float(np.mean(fracs)) < 0.22


def test_r4_floor_is_a_weak_direction_knob(r4_results):
    """STREAM-ROBUST honest negative: at fixed dose, floors
    0.45-1.0 sit in one compressed cluster in the grounded
    toy (the R1-R3 FG45 advantage came from mechanisms the
    code does not contain). |fg45 - prompt| at early
    measures 1.57 in the numpy reference and 2.43 in torch;
    the bound sits at the larger value plus roughly half
    the thinner implementation's headroom under the
    original 4.0 bound."""
    e45 = r4_results["fg45"]["early"][0]
    e100 = r4_results["prompt"]["early"][0]
    assert abs(e45 - e100) < 3.2


def test_r4_paired_prefers_floor65(r4_results):
    """STREAM-ROBUST: paired@65 > paired@45 in both
    implementations at both strengths (numpy +0.51 early /
    +0.19 const; torch +0.94 / +0.56). Offsets are roughly
    half the thinner (numpy) margin per strength."""
    offsets = {"early": 0.25, "const": 0.10}
    for mode in ("early", "const"):
        assert (r4_results["paired65"][mode][0]
                > r4_results["paired45"][mode][0] + offsets[mode])


def test_r4_k6_candidates_agree(r4_results):
    """K6 (paired + mix35 @ 0.65): STREAM-ROBUST agreements
    only. The headline early claim is implementation-
    sensitive and is NOT asserted: K6 vs fg45 at early is
    +0.67 in the numpy reference but -1.54 in the original
    torch port and -0.38 in the structurally fixed port --
    the sign flips across RNG streams. K6 vs mix35 at const
    is likewise a cross-stream tie (+0.78 numpy / -0.09
    fixed torch), asserted below as a tie, not a crown.
    What both implementations agree on:
      * K6 beats plain mix35 at early (numpy +1.91, torch
        +0.20; offset = half the thinner margin);
      * K6 and mix35 are within 0.8 of each other at const
        in both (the tie bound adds slack over both);
      * K6 does NOT beat fg45 at const in either (numpy
        -0.54, torch -0.90; offset = half the thinner)."""
    assert (r4_results["k6"]["early"][0]
            > r4_results["mix35"]["early"][0] + 0.10)
    assert abs(r4_results["k6"]["const"][0]
               - r4_results["mix35"]["const"][0]) < 1.2
    assert (r4_results["k6"]["const"][0]
            < r4_results["fg45"]["const"][0] - 0.25)


def test_r4_timed_cue_beats_full_cue_at_const(r4_results):
    """STREAM-ROBUST: the timed cue (positions 0-2) beats
    the full cue at const in both implementations, 5/5
    paired seeds in each (numpy +1.03, torch +0.62). The
    +0.3 offset is roughly half the thinner margin."""
    assert (r4_results["k7_early3"]["const"][0]
            > r4_results["mix35"]["const"][0] + 0.3)


def test_r4_bgproj_plus_cue_is_worst_candidate(r4_results):
    """STREAM-ROBUST: K5 (orthogonal BG projection + cue)
    stacks two costs and trails fg45 at early in both
    implementations (numpy -2.39, torch -1.74, 0/5 paired
    seeds in each); the R3 'restore' step that rescued it
    has no counterpart in the real capture (renorm already
    restores dose). The offset is roughly half the thinner
    margin."""
    assert (r4_results["k5"]["early"][0]
            < r4_results["fg45"]["early"][0] - 0.85)
