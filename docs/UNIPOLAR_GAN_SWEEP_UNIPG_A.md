# UniPG-A sweep scoreboard: GAN-only unipolar toy gates

Arm identity **UniPG-A** (GradRegularizer / `b_cap` sweep). GAN-only G (paired Rp logistic, no positive MSE / cover MSE / ending / FM / lyric-hold / plan / zero-anchor), unipolar raw-positive teacher (`lm_faithful_plus_neu` contract: teacher IS `pos`), scale 0 = base / exact zero by construction, no -1 train branch. Scales 0 / 0.5 / 1; -1 canary only; antipodal `cos(+1,-1)` never gated.

Gates (both required cells `divergent` + `close`): cover@+1 >= 0.85, leak@+1 <= 0.05, neu_hold@0 >= 0.85. `neu_hold` measures 1.000 on every run (exact-zero construction) — reported, not asserted.

All recipes `propose_only`. Music bipolar `ARM_B` / `locked_shared` / live `--lm_target` untouched. No supervised MSE arm ships as a solution.

## Headline

- **Baseline re-run** (bipolar-rig wiring, learned 12-particle prior under G, GAN-only): **FAIL at 600/1200/3400 on all seeds** — cover ~0.50-0.59. Not a cap problem: every GradRegularizer arm stalls identically while the particles absorb the mode (particle-mode theft).
- **Fix**: freeze the prior to pure jitter (the production YuE2 unipolar loop has no learned particle branch). One companion knob; particles / VICReg / LR / critic / schedule otherwise at defaults.
- **PASS earlier than 3400**: every jitter-prior arm passes seed 0 at **1200** (cover ~0.90, leak 0.000, hold 1.000).
- **All-seed PASS**: `unipg_a_ginterp` (interp-path cap) **PASSes both cells on all requested seeds (0/1/7) at 1200 and 3400** — the only arm stable on seed 1. Sample-point `b_cap` leaves D free to go steep *between* reals and fakes; seed 1 lands in that pocket (D-overpower collapse / single-mode collapse). The interp cap closes it.

## Scoreboard (PASS = both cells HIT on every seed 0/1/7)

| arm | prior | tag | 600 | 1200 | 3400 | identity |
|---|---|---|---|---|---|---|
| `unipg_a_baseline` | jitter | MATCH | FAIL (cov 0.17-0.85) | FAIL (cov 0.47-0.91) | FAIL (cov 0.41-0.91) | locked b_cap k=1 coeff=1, anneal none, lazy 1, jitter prior (GAN-only unipolar baseline) |
| `unipg_a_kappa05` | jitter | MATCH | FAIL (cov 0.46-0.85) | FAIL (cov 0.47-0.91) | FAIL (cov 0.46-0.91) | tighter cap: b_cap k=0.5, coeff=1 (D flatter, G gradients smaller) |
| `unipg_a_kappa20` | jitter | MATCH | FAIL (cov 0.30-0.84) | FAIL (cov 0.32-0.91) | FAIL (cov 0.38-0.91) | looser cap: b_cap k=2, coeff=1 (D steeper before penalty bites) |
| `unipg_a_coeff05` | jitter | MATCH | FAIL (cov 0.35-0.85) | FAIL (cov 0.48-0.91) | FAIL (cov 0.45-0.91) | softer cap: b_cap coeff=0.5, k=1 (half pressure above kappa) |
| `unipg_a_coeff20` | jitter | MATCH | FAIL (cov 0.31-0.85) | FAIL (cov 0.29-0.91) | FAIL (cov 0.43-0.91) | harder cap: b_cap coeff=2, k=1 (double pressure above kappa) |
| `unipg_a_ginterp` | jitter | MATCH | FAIL (cov 0.77-0.86) | PASS (cov 0.90-0.91) | PASS (cov 0.90-0.91) | interp-path cap: g_interp_cap k=1 coeff=1 (cap between samples, not at them) |
| `unipg_a_delayed` | jitter | MATCH | FAIL (cov 0.19-0.85) | FAIL (cov 0.47-0.91) | FAIL (cov 0.41-0.91) | delayed anneal: b_cap center holds k=1 to 60% then ramps to 0 (flat-D handover) |
| `unipg_a_linear` | jitter | MATCH | FAIL (cov 0.28-0.85) | FAIL (cov 0.36-0.91) | FAIL (cov 0.38-0.91) | linear anneal: b_cap center ramps k=1 to 0 over the run (early no-flat-D, late R1/R2-like) |
| `unipg_a_lazy2` | jitter | MATCH | FAIL (cov 0.46-0.85) | FAIL (cov 0.47-0.91) | FAIL (cov 0.38-0.91) | lazy reg_every=2: b_cap applied every 2nd step at 2x coeff (StyleGAN2 lazy) |
| `unipg_a_learned_prior` | learned | DRIFT | FAIL (cov 0.49-0.51) | FAIL (cov 0.51-0.55) | FAIL (cov 0.54-0.59) | DRIFT control: locked cap but learned 12-particle prior under G (mode theft demo, expect FAIL) |

## Budget 600 detail (cover / leak / hold per seed x cell)

| arm | seed | divergent | close | seed pass |
|---|---|---|---|---|
| `unipg_a_baseline` | 0 | miss 0.751/0.000/1.000 | miss 0.842/0.000/1.000 | FAIL |
| `unipg_a_baseline` | 1 | miss 0.450/0.052/1.000 | miss 0.168/0.000/1.000 | FAIL |
| `unipg_a_baseline` | 7 | miss 0.743/0.000/1.000 | HIT 0.852/0.000/1.000 | FAIL |
| `unipg_a_kappa05` | 0 | miss 0.735/0.000/1.000 | miss 0.830/0.000/1.000 | FAIL |
| `unipg_a_kappa05` | 1 | miss 0.457/0.042/1.000 | miss 0.807/0.000/1.000 | FAIL |
| `unipg_a_kappa05` | 7 | miss 0.710/0.000/1.000 | miss 0.849/0.000/1.000 | FAIL |
| `unipg_a_kappa20` | 0 | miss 0.772/0.000/1.000 | miss 0.834/0.000/1.000 | FAIL |
| `unipg_a_kappa20` | 1 | miss 0.537/0.010/1.000 | miss 0.299/0.000/1.000 | FAIL |
| `unipg_a_kappa20` | 7 | miss 0.751/0.000/1.000 | miss 0.843/0.000/1.000 | FAIL |
| `unipg_a_coeff05` | 0 | miss 0.755/0.000/1.000 | miss 0.847/0.000/1.000 | FAIL |
| `unipg_a_coeff05` | 1 | miss 0.479/0.052/1.000 | miss 0.347/0.000/1.000 | FAIL |
| `unipg_a_coeff05` | 7 | miss 0.764/0.000/1.000 | HIT 0.854/0.000/1.000 | FAIL |
| `unipg_a_coeff20` | 0 | miss 0.758/0.000/1.000 | miss 0.834/0.000/1.000 | FAIL |
| `unipg_a_coeff20` | 1 | miss 0.467/0.052/1.000 | miss 0.314/0.000/1.000 | FAIL |
| `unipg_a_coeff20` | 7 | miss 0.757/0.000/1.000 | miss 0.847/0.000/1.000 | FAIL |
| `unipg_a_ginterp` | 0 | miss 0.780/0.000/1.000 | miss 0.835/0.000/1.000 | FAIL |
| `unipg_a_ginterp` | 1 | miss 0.771/0.000/1.000 | HIT 0.857/0.000/1.000 | FAIL |
| `unipg_a_ginterp` | 7 | miss 0.786/0.000/1.000 | HIT 0.854/0.000/1.000 | FAIL |
| `unipg_a_delayed` | 0 | miss 0.742/0.000/1.000 | miss 0.847/0.000/1.000 | FAIL |
| `unipg_a_delayed` | 1 | miss 0.450/0.052/1.000 | miss 0.187/0.000/1.000 | FAIL |
| `unipg_a_delayed` | 7 | miss 0.734/0.000/1.000 | HIT 0.851/0.000/1.000 | FAIL |
| `unipg_a_linear` | 0 | miss 0.691/0.000/1.000 | miss 0.842/0.000/1.000 | FAIL |
| `unipg_a_linear` | 1 | miss 0.451/0.042/1.000 | miss 0.277/0.000/1.000 | FAIL |
| `unipg_a_linear` | 7 | miss 0.660/0.000/1.000 | miss 0.849/0.000/1.000 | FAIL |
| `unipg_a_lazy2` | 0 | miss 0.761/0.000/1.000 | miss 0.834/0.000/1.000 | FAIL |
| `unipg_a_lazy2` | 1 | miss 0.455/0.052/1.000 | miss 0.688/0.000/1.000 | FAIL |
| `unipg_a_lazy2` | 7 | miss 0.725/0.000/1.000 | HIT 0.850/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 0 | miss 0.495/0.052/1.000 | miss 0.492/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 1 | miss 0.512/0.010/1.000 | miss 0.506/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 7 | miss 0.513/0.042/1.000 | miss 0.499/0.000/1.000 | FAIL |

## Budget 1200 detail (cover / leak / hold per seed x cell)

| arm | seed | divergent | close | seed pass |
|---|---|---|---|---|
| `unipg_a_baseline` | 0 | HIT 0.907/0.000/1.000 | HIT 0.902/0.000/1.000 | PASS |
| `unipg_a_baseline` | 1 | miss 0.467/0.042/1.000 | miss 0.831/0.000/1.000 | FAIL |
| `unipg_a_baseline` | 7 | HIT 0.907/0.000/1.000 | HIT 0.904/0.000/1.000 | PASS |
| `unipg_a_kappa05` | 0 | HIT 0.908/0.000/1.000 | HIT 0.901/0.000/1.000 | PASS |
| `unipg_a_kappa05` | 1 | miss 0.466/0.042/1.000 | HIT 0.905/0.000/1.000 | FAIL |
| `unipg_a_kappa05` | 7 | HIT 0.908/0.000/1.000 | HIT 0.903/0.000/1.000 | PASS |
| `unipg_a_kappa20` | 0 | HIT 0.908/0.000/1.000 | HIT 0.895/0.000/1.000 | PASS |
| `unipg_a_kappa20` | 1 | miss 0.544/0.010/1.000 | miss 0.325/0.000/1.000 | FAIL |
| `unipg_a_kappa20` | 7 | HIT 0.909/0.000/1.000 | HIT 0.902/0.000/1.000 | PASS |
| `unipg_a_coeff05` | 0 | HIT 0.909/0.000/1.000 | HIT 0.899/0.000/1.000 | PASS |
| `unipg_a_coeff05` | 1 | miss 0.475/0.031/1.000 | HIT 0.906/0.000/1.000 | FAIL |
| `unipg_a_coeff05` | 7 | HIT 0.906/0.000/1.000 | HIT 0.903/0.000/1.000 | PASS |
| `unipg_a_coeff20` | 0 | HIT 0.908/0.000/1.000 | HIT 0.894/0.000/1.000 | PASS |
| `unipg_a_coeff20` | 1 | miss 0.464/0.042/1.000 | miss 0.291/0.000/1.000 | FAIL |
| `unipg_a_coeff20` | 7 | HIT 0.907/0.000/1.000 | HIT 0.903/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 0 | HIT 0.908/0.000/1.000 | HIT 0.901/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 1 | HIT 0.906/0.000/1.000 | HIT 0.907/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 7 | HIT 0.908/0.000/1.000 | HIT 0.902/0.000/1.000 | PASS |
| `unipg_a_delayed` | 0 | HIT 0.906/0.000/1.000 | HIT 0.900/0.000/1.000 | PASS |
| `unipg_a_delayed` | 1 | miss 0.465/0.042/1.000 | miss 0.794/0.000/1.000 | FAIL |
| `unipg_a_delayed` | 7 | HIT 0.907/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_linear` | 0 | HIT 0.905/0.000/1.000 | HIT 0.899/0.000/1.000 | PASS |
| `unipg_a_linear` | 1 | miss 0.462/0.042/1.000 | miss 0.361/0.000/1.000 | FAIL |
| `unipg_a_linear` | 7 | HIT 0.905/0.000/1.000 | HIT 0.904/0.000/1.000 | PASS |
| `unipg_a_lazy2` | 0 | HIT 0.908/0.000/1.000 | HIT 0.897/0.000/1.000 | PASS |
| `unipg_a_lazy2` | 1 | miss 0.470/0.010/1.000 | HIT 0.899/0.000/1.000 | FAIL |
| `unipg_a_lazy2` | 7 | HIT 0.905/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_learned_prior` | 0 | miss 0.517/0.031/1.000 | miss 0.511/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 1 | miss 0.549/0.010/1.000 | miss 0.534/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 7 | miss 0.544/0.010/1.000 | miss 0.526/0.000/1.000 | FAIL |

## Budget 3400 detail (cover / leak / hold per seed x cell)

| arm | seed | divergent | close | seed pass |
|---|---|---|---|---|
| `unipg_a_baseline` | 0 | HIT 0.909/0.000/1.000 | HIT 0.898/0.000/1.000 | PASS |
| `unipg_a_baseline` | 1 | miss 0.454/0.031/1.000 | miss 0.412/0.000/1.000 | FAIL |
| `unipg_a_baseline` | 7 | HIT 0.905/0.000/1.000 | HIT 0.906/0.000/1.000 | PASS |
| `unipg_a_kappa05` | 0 | HIT 0.910/0.000/1.000 | HIT 0.899/0.000/1.000 | PASS |
| `unipg_a_kappa05` | 1 | miss 0.465/0.052/1.000 | HIT 0.909/0.000/1.000 | FAIL |
| `unipg_a_kappa05` | 7 | HIT 0.908/0.000/1.000 | HIT 0.903/0.000/1.000 | PASS |
| `unipg_a_kappa20` | 0 | HIT 0.908/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_kappa20` | 1 | HIT 0.908/0.000/1.000 | miss 0.385/0.000/1.000 | FAIL |
| `unipg_a_kappa20` | 7 | HIT 0.911/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_coeff05` | 0 | HIT 0.909/0.000/1.000 | HIT 0.901/0.000/1.000 | PASS |
| `unipg_a_coeff05` | 1 | miss 0.453/0.073/1.000 | HIT 0.909/0.000/1.000 | FAIL |
| `unipg_a_coeff05` | 7 | HIT 0.907/0.000/1.000 | HIT 0.904/0.000/1.000 | PASS |
| `unipg_a_coeff20` | 0 | HIT 0.909/0.000/1.000 | HIT 0.897/0.000/1.000 | PASS |
| `unipg_a_coeff20` | 1 | miss 0.434/0.083/1.000 | HIT 0.902/0.000/1.000 | FAIL |
| `unipg_a_coeff20` | 7 | HIT 0.906/0.000/1.000 | HIT 0.903/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 0 | HIT 0.908/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 1 | HIT 0.908/0.000/1.000 | HIT 0.907/0.000/1.000 | PASS |
| `unipg_a_ginterp` | 7 | HIT 0.906/0.000/1.000 | HIT 0.904/0.000/1.000 | PASS |
| `unipg_a_delayed` | 0 | HIT 0.907/0.000/1.000 | HIT 0.900/0.000/1.000 | PASS |
| `unipg_a_delayed` | 1 | miss 0.455/0.031/1.000 | miss 0.409/0.000/1.000 | FAIL |
| `unipg_a_delayed` | 7 | HIT 0.905/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_linear` | 0 | HIT 0.906/0.000/1.000 | HIT 0.900/0.000/1.000 | PASS |
| `unipg_a_linear` | 1 | miss 0.620/0.000/1.000 | miss 0.385/0.000/1.000 | FAIL |
| `unipg_a_linear` | 7 | HIT 0.907/0.000/1.000 | HIT 0.904/0.000/1.000 | PASS |
| `unipg_a_lazy2` | 0 | HIT 0.909/0.000/1.000 | HIT 0.900/0.000/1.000 | PASS |
| `unipg_a_lazy2` | 1 | miss 0.448/0.042/1.000 | miss 0.384/0.000/1.000 | FAIL |
| `unipg_a_lazy2` | 7 | HIT 0.906/0.000/1.000 | HIT 0.905/0.000/1.000 | PASS |
| `unipg_a_learned_prior` | 0 | miss 0.557/0.021/1.000 | miss 0.535/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 1 | miss 0.587/0.010/1.000 | miss 0.566/0.000/1.000 | FAIL |
| `unipg_a_learned_prior` | 7 | miss 0.576/0.010/1.000 | miss 0.561/0.000/1.000 | FAIL |

## Knobs (every arm: Rp logistic + GradRegularizer; GAN-only; raw-positive; jitter prior unless noted)

| arm | grad_arm | coeff (b_cap) | kappa | lazy_k | anneal | prior |
|---|---|---|---|---|---|---|
| `unipg_a_baseline` | b_cap | 1.0 | 1.0 | 1 | none | jitter |
| `unipg_a_kappa05` | b_cap | 1.0 | 0.5 | 1 | none | jitter |
| `unipg_a_kappa20` | b_cap | 1.0 | 2.0 | 1 | none | jitter |
| `unipg_a_coeff05` | b_cap | 0.5 | 1.0 | 1 | none | jitter |
| `unipg_a_coeff20` | b_cap | 2.0 | 1.0 | 1 | none | jitter |
| `unipg_a_ginterp` | g_interp_cap | 1.0 | 1.0 | 1 | none | jitter |
| `unipg_a_delayed` | b_cap | 1.0 | 1.0 | 1 | delayed | jitter |
| `unipg_a_linear` | b_cap | 1.0 | 1.0 | 1 | linear | jitter |
| `unipg_a_lazy2` | b_cap | 1.0 | 1.0 | 2 | none | jitter |
| `unipg_a_learned_prior` | b_cap | 1.0 | 1.0 | 1 | none | learned |

Particles / VICReg / LR / critic / schedule at production defaults on every arm (`n_particles=12`, `vicreg_weight=0.05`, `particle_l2=0.02`, shared LR 5e-3 with d/prior mults 1.0, Fourier-2 critic, delay-80 cosine, EMA 0.995, span/end cloud).

## Why the others failed

- Learned-prior control (`DRIFT`): cover plateaus ~0.50-0.59 from 600 to 3400 on every seed/cell. The learned prior under G's optimizer carries the mode (`fake = neu + w + particle` satisfies D) while the scored residual stalls at half norm in the same readout basin (overlap 1.0, blend ~0.49). Raising `particle_l2` 10x only reaches 0.60; freezing the prior reaches 0.91. The cap axis is uninformative until the theft is fixed.
- Seed-1 collapse on every sample-point `b_cap` arm (baseline, both kappas, both coeffs, both anneals, lazy): seed 1 sticks at cover 0.17-0.48 from 600 through 3400 — stuck, not slow. Two collapse modes. (a) D-overpower (`kappa20`/close: `d_loss~=0.002`, `g_loss~=6.3`, G frozen mid-basin, blend 0.62). Loose/lazy caps let D go steep *between* reals and fakes where `b_cap` never binds. (b) Single-mode (`lazy2`/divergent: all rows sing one pole, `slam x 8`, off-caption 0.03-0.06). `g_interp_cap` penalizes the interp path, keeping a usable G gradient field on every seed.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m analysis.slider2d.yue2_gan_exam \
    --steps 600 1200 3400 --seeds 0 1 7 --out /tmp/yue2-gan-exam.json
# exits 1 unless every requested budget passes
PYTHONPATH=. pytest -q tests/test_yue2_gan_exam.py tests/test_yue2_arm_b.py tests/test_formulation_leaderboard.py
```

Harness: `analysis/slider2d/yue2_gan_exam.py` (`PROPOSE_ONLY`, `MERGE_TO_TRAINER=False`). Raw data: `docs/unipolar-gan-sweep-unipg-a.json`. No win is claimed from any MSE path; the supervised `faithful_plus_neu` control is not a sweep arm.

