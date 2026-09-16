# UniPG-C scoreboard — ParticleGAN schedule / optimizer clones (unipolar, GAN-only)

Propose-only arms over the production YuE2 unipolar GAN loop (`yue2_arm_b`, RpGAN + `b_cap` kappa=1 coeff=1).
Gates per required cell (divergent, close), every seed: cover@+1 >= 0.85, leak@+1 <= 0.05, neu_hold@0 >= 0.85.
Scale -1 is canary only; antipodal cos is never a gate. No supervised MSE on G; Music bipolar ARM_B /
`locked_shared` / live `--lm_target` untouched.

## Budget verdicts (both cells, all seeds)

| arm | 600 | 1200 | 3400 | identity |
|---|---|---|---|---|
| `production_baseline` | **FAIL** | **FAIL** | **PASS** | production YuE2 loop via yue2_gan_exam (reference) |
| `c0_production` | **FAIL** | **FAIL** | **PASS** | reference: production YuE2 knobs (constant LR, beta2 0.999, no EMA, thick critic) |
| `c1_beta2_099` | **FAIL** | **FAIL** | **PASS** | beta2 0.99 (ParticleGAN / locked-toy value); else production |
| `c2_ema_on` | **FAIL** | **FAIL** | **PASS** | EMA 0.995 residual-only on; else production |
| `c3_delay80` | **FAIL** | **FAIL** | **PASS** | delayed cosine, absolute delay 80, floor 0.05; else production |
| `c4_hold60` | **FAIL** | **FAIL** | **PASS** | delayed cosine, 60% hold (delay=0.6*budget), floor 0.05; else production |
| `c5_d1x` | **FAIL** | **FAIL** | **PASS** | shared LR (D 1.0x: G 5e-4 / D 5e-4); else production |
| `c6_g2x` | **FAIL** | **PASS** | **PASS** | 2x G LR keeping 1.5x D (G 1e-3 / D 1.5e-3); else production |
| `c7_sched_clone` | **FAIL** | **FAIL** | **PASS** | schedule clone: beta2 0.99 + EMA on + 60% hold cosine + D 1.5x, thick critic |
| `c8_fourier_sched` | **FAIL** | **FAIL** | **FAIL** | c7 schedule clone + Fourier-2 critic w64 (full locked-toy optimizer/critic/schedule clone; Rp-only G) |
| `c9_g4x` | **PASS** | **PASS** | **PASS** | 4x G LR keeping 1.5x D (G 2e-3 / D 3e-3); follow-up rung after clean c6 win, tests 600 PASS |

## Per-arm gate numbers (cover / off-caption / neu_hold / hit)

### `production_baseline` — production YuE2 loop via yue2_gan_exam (reference)

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.482/0.062/1.000 miss | 0.660/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 1 | 0.484/0.042/1.000 miss | 0.658/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 7 | 0.483/0.083/1.000 miss | 0.661/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| close | 0 | 0.587/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.586/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.586/0.000/1.000 miss | 0.931/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: note=REFERENCE, gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c0_production` — reference: production YuE2 knobs (constant LR, beta2 0.999, no EMA, thick critic)

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.482/0.062/1.000 miss | 0.660/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 1 | 0.484/0.042/1.000 miss | 0.658/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 7 | 0.483/0.083/1.000 miss | 0.661/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| close | 0 | 0.587/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.586/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.586/0.000/1.000 miss | 0.931/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=REFERENCE, beta2=REFERENCE, ema=REFERENCE, critic=REFERENCE, gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c1_beta2_099` — beta2 0.99 (ParticleGAN / locked-toy value); else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.486/0.083/1.000 miss | 0.661/0.000/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 1 | 0.487/0.021/1.000 miss | 0.660/0.000/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 7 | 0.487/0.083/1.000 miss | 0.662/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| close | 0 | 0.584/0.000/1.000 miss | 0.934/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.583/0.000/1.000 miss | 0.933/0.000/1.000 HIT | 0.933/0.000/1.000 HIT |
| close | 7 | 0.583/0.000/1.000 miss | 0.934/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=MATCH(constant=production drift; beta2 is the clone), beta2=MATCH, ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c2_ema_on` — EMA 0.995 residual-only on; else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.446/0.104/1.000 miss | 0.590/0.042/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 1 | 0.447/0.073/1.000 miss | 0.590/0.010/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 7 | 0.447/0.125/1.000 miss | 0.591/0.021/1.000 miss | 0.932/0.000/1.000 HIT |
| close | 0 | 0.466/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |
| close | 1 | 0.466/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |
| close | 7 | 0.466/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |

Tags: schedule=DRIFT(constant), beta2=DRIFT(0.999), ema=PARTIAL(residual-only; proper is G+particles), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c3_delay80` — delayed cosine, absolute delay 80, floor 0.05; else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.480/0.062/1.000 miss | 0.629/0.021/1.000 miss | 0.878/0.000/1.000 HIT |
| divergent | 1 | 0.482/0.042/1.000 miss | 0.628/0.010/1.000 miss | 0.877/0.000/1.000 HIT |
| divergent | 7 | 0.481/0.083/1.000 miss | 0.630/0.010/1.000 miss | 0.879/0.000/1.000 HIT |
| close | 0 | 0.578/0.000/1.000 miss | 0.933/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.577/0.000/1.000 miss | 0.931/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.577/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=MATCH(locked-toy delay80), beta2=DRIFT(0.999), ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c4_hold60` — delayed cosine, 60% hold (delay=0.6*budget), floor 0.05; else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.482/0.062/1.000 miss | 0.660/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 1 | 0.484/0.042/1.000 miss | 0.658/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 7 | 0.483/0.083/1.000 miss | 0.661/0.000/1.000 miss | 0.930/0.000/1.000 HIT |
| close | 0 | 0.587/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.586/0.000/1.000 miss | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.586/0.000/1.000 miss | 0.931/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=MATCH(ParticleGAN 60% hold), beta2=DRIFT(0.999), ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c5_d1x` — shared LR (D 1.0x: G 5e-4 / D 5e-4); else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.483/0.062/1.000 miss | 0.662/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 1 | 0.485/0.042/1.000 miss | 0.660/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| divergent | 7 | 0.485/0.083/1.000 miss | 0.663/0.000/1.000 miss | 0.931/0.000/1.000 HIT |
| close | 0 | 0.594/0.000/1.000 miss | 0.933/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.590/0.000/1.000 miss | 0.931/0.000/1.000 HIT | 0.933/0.000/1.000 HIT |
| close | 7 | 0.590/0.000/1.000 miss | 0.930/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=DRIFT(1.0x D; proper is 1.5x), beta2=DRIFT(0.999), ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c6_g2x` — 2x G LR keeping 1.5x D (G 1e-3 / D 1.5e-3); else production

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.661/0.000/1.000 miss | 0.928/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |
| divergent | 1 | 0.660/0.000/1.000 miss | 0.929/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |
| divergent | 7 | 0.662/0.000/1.000 miss | 0.930/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |
| close | 0 | 0.931/0.000/1.000 HIT | 0.933/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.930/0.000/1.000 HIT | 0.934/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.931/0.000/1.000 HIT | 0.933/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |

Tags: schedule=DRIFT(2x LR hunt for early cover), beta2=DRIFT(0.999), ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c7_sched_clone` — schedule clone: beta2 0.99 + EMA on + 60% hold cosine + D 1.5x, thick critic

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.447/0.104/1.000 miss | 0.595/0.031/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 1 | 0.448/0.083/1.000 miss | 0.595/0.010/1.000 miss | 0.932/0.000/1.000 HIT |
| divergent | 7 | 0.448/0.115/1.000 miss | 0.596/0.021/1.000 miss | 0.932/0.000/1.000 HIT |
| close | 0 | 0.465/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |
| close | 1 | 0.465/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |
| close | 7 | 0.465/0.000/1.000 miss | 0.906/0.000/1.000 HIT | 0.935/0.000/1.000 HIT |

Tags: schedule=MATCH(60% hold), beta2=MATCH, ema=PARTIAL(residual-only), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c8_fourier_sched` — c7 schedule clone + Fourier-2 critic w64 (full locked-toy optimizer/critic/schedule clone; Rp-only G)

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.407/0.115/1.000 miss | 0.548/0.031/1.000 miss | 0.923/0.000/1.000 HIT |
| divergent | 1 | 0.444/0.073/1.000 miss | 0.578/0.010/1.000 miss | 0.924/0.000/1.000 HIT |
| divergent | 7 | 0.243/0.740/1.000 miss | 0.272/0.604/1.000 miss | 0.702/0.000/1.000 miss |
| close | 0 | 0.463/0.000/1.000 miss | 0.461/0.000/1.000 miss | 0.325/0.594/1.000 miss |
| close | 1 | 0.371/0.000/1.000 miss | 0.287/0.000/1.000 miss | 0.268/0.000/1.000 miss |
| close | 7 | 0.480/0.000/1.000 miss | 0.751/0.000/1.000 miss | 0.924/0.000/1.000 HIT |

Tags: schedule=MATCH(60% hold), beta2=MATCH, ema=PARTIAL(residual-only), critic=MATCH(Fourier-2), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

### `c9_g4x` — 4x G LR keeping 1.5x D (G 2e-3 / D 3e-3); follow-up rung after clean c6 win, tests 600 PASS

| cell | seed | 600 | 1200 | 3400 |
|---|---|---|---|---|
| divergent | 0 | 0.919/0.000/1.000 HIT | 0.931/0.000/1.000 HIT | 0.931/0.000/1.000 HIT |
| divergent | 1 | 0.923/0.000/1.000 HIT | 0.931/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |
| divergent | 7 | 0.927/0.000/1.000 HIT | 0.930/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |
| close | 0 | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 1 | 0.931/0.000/1.000 HIT | 0.933/0.000/1.000 HIT | 0.934/0.000/1.000 HIT |
| close | 7 | 0.932/0.000/1.000 HIT | 0.934/0.000/1.000 HIT | 0.932/0.000/1.000 HIT |

Tags: schedule=DRIFT(4x LR hunt for 600 cover), beta2=DRIFT(0.999), ema=DRIFT(off), critic=DRIFT(thick), gan_loss=MATCH(Rp pair logistic), b_cap=MATCH(kappa=1 coeff=1), particles=HOLD(no ParticlePrior in YuE2 port; prior x10 N/A), vicreg=HOLD(none; Rp-only G), cover_weight=HOLD(0; Rp-only G, no MSE), teacher=HOLD(raw positive faithful_plus_neu; not modes/spans). Supervised control hits on every cell/seed: True.

## Readout

- Best arm: `c9_g4x` (4x base LR, D 1.5x kept: G 2e-3 / D 3e-3) — PASS at 600 on both cells and all seeds
  (divergent cover 0.919-0.927, close 0.931-0.932, leak 0, neu_hold 1.0). Losses stay finite (G peak ~3.3,
  no spikes). Runner-up `c6_g2x` (2x) PASSes at 1200 with close HIT at 600. Both keep the D:G ratio,
  b_cap(1,1), and the thick critic; the only drift is base-LR scale.
- EMA arms (`c2`, `c7`) lag early cover (shadow averaging) but PASS at 3400 — EMA smooths, it does not accelerate.
- Cosine-decay schedules (`c3` absolute delay 80, `c4` 60% hold) do not accelerate cover; `c3` starves late
  divergent cover (0.878 vs 0.931 at 3400). Hold-60 matches production because it holds LR 1.0 through 1200.
- `beta2` 0.99 (`c1`) and shared D LR (`c5`) track production — neither is the cover bottleneck.
- `c8_fourier_sched` (Fourier-2 critic + schedule clone) FAILS at 3400 and is seed-fragile on close
  (cover 0.27–0.33, leak up to 0.59): the thin critic is a DROP direction for this port — keep the thick critic.
- No other arm passes divergent@600; the LR rungs (c6 at 1200, c9 at 600) are the earliest honest PASSes, via LR only.
- Every arm is RpGAN + b_cap(1,1) on G with no MSE/cover/FM/anchor terms; `game.RECIPE` is byte-identical.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python3 -m analysis.slider2d.yue2_gan_exam --steps 600 1200 3400 --seeds 0 1 7 --out /tmp/yue2-gan-exam.json
PYTHONPATH=. python3 analysis/slider2d/run_unipg_c_scoreboard.py --inputs /tmp/unipg-*.json /tmp/yue2-gan-exam.json \
  --md docs/unipg-c-scoreboard.md --json docs/unipg-c-scoreboard.json
PYTHONPATH=. python3 -m pytest tests/test_yue2_gan_exam.py tests/test_yue2_arm_b.py tests/test_formulation_leaderboard.py tests/test_yue2_unipg_c.py -q
```
