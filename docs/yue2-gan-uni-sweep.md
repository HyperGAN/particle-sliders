# Unipolar ParticleGAN sweep (UniPG-D): GAN-only toy gates

`analysis/slider2d/yue2_gan_exam.py` — CPU fixture analogue of the production
YuE2 unipolar loop (`unipolar-rpgan-bcap-yue2-v3`), scored under the unipolar
toy gates on `divergent` + `close`: cover@+1 ≥ 0.85, leak@+1 ≤ 0.05,
neu_hold@0 ≥ 0.85. Eval scales 0 / 0.5 / 1; −1 canary only; antipodal
cos(+1,−1) never consulted. Raw-positive teacher
(`lm_faithful_plus_neu` contract: the teacher IS `pos`). Scale 0 is the
exact base by construction, so `neu_hold` is measured honestly (1.0 up to
rollout sampling) rather than trained.

Machine-readable results: [metrics.json](yue2-gan-uni-sweep/metrics.json)
(540 rows: 42 arms × budgets × seeds 0/1/7 × 2 cells).

## Constraints (enforced in code, not prose)

- **GAN-only G**: `fit_uni_gan` raises on `cover_weight > 0` or
  `fm_weight > 0`. No positive MSE, cover MSE, ending, FM, lyric-hold,
  plan, or zero-anchor on G. VICReg / particle L2 apply to the
  *particles* only (ParticleGAN spread pressure, never residual
  supervision). Pinned by `tests/test_yue2_gan_exam.py`.
- **Unipolar**: +1 student vs raw-positive teacher only. No negative
  captions, bipolar ranges, leak-axis metadata, or −1 train branch.
- **Spine**: every arm is Rp logistic + a GradRegularizer arm
  (default `b_cap`). No WGAN/hinge/LSGAN primary loss anywhere.
- **propose_only**: all 42 arms. `AdvConfig()` defaults, the Music
  trainer row, and the live `--lm_target` default are untouched
  (pinned by the new tests + `test_pg_clone_consolidated.py`).
- **No supervised win**: the supervised `faithful_plus_neu` control
  passes at 400 steps on every seed (re-verified here) and is reported
  as a control only — never as a sweep arm.

## Headline

| arm | 600 | 1200 | 2400 | 3400 | hook |
|---|---|---|---|---|---|
| `uni_baseline` (production analogue) | FAIL (0.62) | FAIL (0.67) | FAIL (0.69) | FAIL (0.70) | production YuE2 unipolar |
| `uni_prior01` (slow prior breakthrough) | FAIL (seed 1 dead) | FAIL (seed 1 dead) | FAIL (seed 1 dead) | FAIL (blows up) | particles axis |
| **`uni_prior01_ginterp`** | FAIL (0.83, close) | **PASS all seeds** | **PASS all seeds** | **PASS all seeds** | **g_interp #115 + slow prior** |

Cover shown is the mean over seeds 0/1/7 and both cells; PASS means every
seed × cell hits all three gates. The winner passes at **1200 steps —
earlier than the production 3400-step budget** — and holds flat through
3400 (cover 0.88, off 0.00, hold 1.00, no late blow-up).

Recipe (`uni_prior01_ginterp`): locked toy knobs + `prior_lr_mult=0.1` +
`grad_arm=g_interp_cap`, GAN-only (`cover_weight=0`, `fm_weight=0`),
`vicreg_fn` locked. Two deltas, both ParticleGAN-family (MATCH):

1. **Slow prior (0.1×)**: with the prior learning as fast as the residual,
   the particles steal the mode and the shared residual undershoots the
   poles (cover plateaus ≈ 0.70 at every budget; `prior×10` collapses to
   0.42). Slowing the prior forces the residual to do the work —
   seeds 0/2/3/7 jump to cover ≈ 0.89 at 1200.
2. **Interp-path cap (`g_interp_cap`)**: the sample-point `b_cap` leaves D
   free to be arbitrarily steep *between* reals and fakes — exactly where
   G must travel. On train-seed 1 D builds a flat-around-fakes boundary
   and G's game dies (cover 0.47/0.22, never recovers through 3400).
   Penalizing the slope on real/fake interpolates keeps D's gradient
   informative on G's path: seed 1 passes (0.89/0.88) and the arm is
   stable across all budgets and seeds.

## #116 hook → uni gate results (1200 steps, seeds 0/1/7, div/cls hits)

| #116 hook | uni arm | s0 | s1 | s7 | note |
|---|---|---|---|---|---|
| production baseline | `uni_baseline` | 0/0 | 0/0 | 0/0 | plateaus 0.65–0.71, never passes |
| `locked_baseline_defaults` | `uni_locked` | 0/0 | 0/0 | 0/0 | same plateau (cover 1.5→0 delta noted; GAN-only) |
| LR mults #112 (`pg_2x_lr`) | `uni_pg_2x_lr` | 0/0 | 0/0 | 0/0 | 0.63–0.68, no lift |
| LR mults #112 (D ×1.5) | `uni_pg_d15` | 0/0 | 0/0 | 0/0 | no lift |
| LR mults #112 (prior ×10) | `uni_pg_prior10` | 0/0 | 0/0 | 0/0 | collapse 0.42 + leak — prior eats the mode |
| VicReg fn #113 (faithful, w=1) | `uni_pg_vicreg_faithful` | 0/0 | 0/0 | 0/0 | 0.61–0.65; std_target 1.0 too hot for toy scale |
| VicReg ladder (off) | `uni_pg_vicreg_off` | 0/0 | 0/0 | 0/0 | 0.64, no lift |
| full clone #114 (thin) | `uni_pg_full_clone` | 0/0 | 0/0 | 0/0 | 0.58–0.68; inherits fast-prior stall |
| anneal #115 (linear) | `uni_anneal_linear` | 0/0 | 0/0 | 0/0 | 0.65, no lift |
| anneal #115 (delayed) | `uni_anneal_delayed` | 0/0 | 0/0 | 0/0 | 0.65, no lift |
| g_interp #115 | `uni_g_interp` | 0/0 | 0/0 | 0/0 | 0.64 alone — needs the slow prior |
| big particles #111 (n=64) | `uni_big_particles` | 0/0 | 0/0 | 0/0 | 0.73 best single-knob, still short |
| kappa 0.5 / 2.0 | `uni_kappa05/20` | 0/0 | 0/0 | 0/0 | 0.64–0.66, cap not the lever |
| thick critic | `uni_thick_critic` | 0/0 | 0/0 | 0/0 | 0.66–0.68, capacity not the lever |
| beta2 0.999 | `uni_beta2_0999` | 0/0 | 0/0 | 0/0 | no lift |
| EMA off | `uni_no_ema` | 0/0 | 0/0 | 0/0 | no lift |
| slow prior 0.1 | `uni_prior01` | 1/1 | 0/0 | 1/1 | breakthrough, seed-1 dead game |
| slow prior 0.01 | `uni_prior001` | 1/1 | 0/1 | 1/1 | same, seed-1 divergent still dead |
| **slow prior + g_interp** | **`uni_prior01_ginterp`** | **1/1** | **1/1** | **1/1** | **PASS — the sweep winner** |
| slow prior + batch64 | `uni_prior01_batch64` | 1/1 | 1/0 | 1/1 | fixes seed-1 divergent only |
| slow prior + thick | `uni_prior01_thick` | 1/1 | 0/1 | 1/1 | fixes seed-1 close only |
| slow prior + {2xlr, d05, lr2e3, n32, hold, const, jit005, cloud006, coeff5, big} | 10 combos | 1/1 | 0/0 | 1/1 | none fix seed-1 both cells |

(`sN` = divergent-hit / close-hit at 1200 steps. Full per-seed cover /
off / hold in `metrics.json`.)

## Why the others failed

- **Fast prior stalls**: any arm with `prior_lr_mult ≥ 1` plateaus at
  cover 0.62–0.73 no matter the budget (600→3400 flat). The learnable
  prior absorbs the pole shift; the scored residual stays short. More
  particles (n=64, cover 0.73) help for the same reason — each moves
  less — but cannot finish.
- **Seed-1 dead game**: with the sample-point cap, one train seed in
  ~5 lets D build a boundary that is flat around the fakes; G pushes
  with no directional signal and the residual freezes small (cover
  0.22–0.47 at every budget; longer training never recovers and can
  blow up off-caption at 3400). Disentangling train-seed × field-seed
  proves it is the game, not the fixture: train-seed 1 fails on every
  field, train-seeds 0/7 pass on every field.
- **Single-knob LR/schedule/critic/VICReg moves don't lift**: D×1.5,
  2×LR, holds, anneals, thicker critic, beta2, EMA, VICReg weight/form
  all land 0.58–0.68. The binding constraints are prior speed (who fits
  the mode) and path slope (whether G keeps a gradient), not optimizer
  tuning.

## Best-arm recipe

```python
cfg, _ = arm_cfg("uni_prior01_ginterp", steps=1200, seed=<0/1/7>)
# = AdvConfig() + prior_lr_mult=0.1 + grad_arm="g_interp_cap"
#   + cover_weight=0.0 + fm_weight=0.0, vicreg_fn locked
residual, _ = fit_uni_gan(field, cfg=cfg)   # raises if G is supervised
row = score_uni_cell("uni_prior01_ginterp", field, residual)
```

## Music / bipolar untouched

- No changes to `conceptmod/textsliders/train_lm_slider_music3.py`,
  `yue2_arm_b.py`, `ARM_B`, `locked_shared`, or any live default.
- `git status` for this branch shows only
  `analysis/slider2d/yue2_gan_exam.py`,
  `tests/test_yue2_gan_exam.py`, and these docs.

## How to run

```bash
# production baseline ladder (exits 0; PASS/FAIL in JSON + stdout)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m analysis.slider2d.yue2_gan_exam \
  --arms uni_baseline --steps 600 1200 3400 --seeds 0 1 7 --out /tmp/yue2-gan-exam.json
# full sweep board (42 arms; --jobs 4 recommended)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m analysis.slider2d.yue2_gan_exam \
  --steps 600 1200 --seeds 0 1 7 --jobs 4 --out docs/yue2-gan-uni-sweep/metrics.json
PYTHONPATH=. pytest -q tests/test_yue2_gan_exam.py tests/test_yue2_arm_b.py tests/test_formulation_leaderboard.py
```

CPU only. No Hub, no GPU, no Music 3 weights.
