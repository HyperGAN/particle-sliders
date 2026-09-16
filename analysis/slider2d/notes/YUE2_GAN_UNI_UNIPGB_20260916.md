# UniPG-B: closing the ParticlePrior + ParticleRegularizer drifts (unipolar GAN toys)

ARM IDENTITY UniPG-B — ParticlePrior + VICReg closer to
255BITS/ParticleGAN. Scope: the intentional drifts around particles and
the ParticleRegularizer. `GradRegularizer` stays at the champion
(`b_cap`, `kappa=1`, `coeff=1`) on every arm — no companion tweak was
required.

## The drift (why the learned-prior toy stalls at cover ~0.65)

The toy's `ParticlePrior` is learnable and its mean absorbs the pole
shift: after 1200 steps the bank mean reaches `|mu| ~ 0.86` while the
scored residual stalls at `|d+| ~ 1.6` against a teacher `~2.6`. The
*fake distribution* matches (residual + bank mean ~= pole), so the game
is at equilibrium — but the scored object is the residual alone, and it
undercovers. Upstream this is not a bug (the particles ARE the
generator); on the toy it is, because only the residual is scored.

A second, independent instability: on divergent seed 1 the generator
overshoots (`|d+| ~ 5`) into garble (`off = 1.0`) unless the jitter
bank is wide or tight enough to stabilise the critic signal.

## The fixes (all GAN-only, all propose_only)

1. **Prior LR 0 (frozen jitter bank).** The bank stays zero-mean; the
   residual carries the full shift (cover `~0.91` at 1200). This is the
   core UniPG-B move.
2. **n_particles 12 -> 32 -> 64** (toy-small toward ParticleGAN-like
   larger). The wider bank stabilises divergent seed 1
   (`unipg_b_frozen64` ALL-PASS at 1200; `frozen32_init001` too).
3. **init_std 0.05 -> 0.01.** A tighter frozen bank is the alternate
   seed-1 rescue.
4. **VICReg ladder 0 / 0.05 / 1.0; sim+var+cov vs var+cov-only;
   std_target 0.05 vs 1.0.** With a frozen bank VICReg is correctly
   moot (weight 0 — nothing to regularise). With a learnable bank,
   centered sampling + faithful var+cov-only VICReg at weight 1.0
   (`std_target = 1.0`, upstream values) rescues divergent seed 1
   where centering alone, mean-penalty, and strong L2 all fail — the
   spread pressure is doing real work.
5. **particle_l2 on/off.** `0.02` cannot hold the bank mean (fails);
   frozen `0.0` passes. Strong L2 (`0.5`–`2.0`) only reaches `~0.80`.

## What the scoreboard shows

- `baseline_learned12` FAILs at 600/1200/3400 (cover `~0.6`–`0.68`).
  Re-run: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH=. python3 -m analysis.slider2d.yue2_gan_exam --arms
  baseline_learned12 --steps 600 1200 3400 --seeds 0 1 7`.
- `unipg_b_frozen64` and `unipg_b_frozen32_init001` PASS both cells on
  all seeds at 1200 — earlier than 3400.
- `unipg_b_center_vicregfaithful1` is PARTIAL (closest
  ParticleRegularizer; fixes divergent seed 1, undercovers close).
- `unipg_b_center_vicreg0`, `unipg_b_frozen32_2xlr`, `unipg_b_frozen12`
  document the ablations (see scoreboard).

## Honesty notes

- GAN-only throughout: `rp_g_loss`/`rp_d_loss` + `b_cap`. No positive
  MSE, cover MSE, ending, FM, hold, plan, or zero-anchor on G — the
  config has no such knob. (Supervised `faithful_plus_neu` passes at
  400 steps and is the control, not a sweep arm.)
- Music untouched: bipolar `ARM_B`, `locked_shared`, live
  `--lm_target` default unchanged (pinned by
  `tests/test_yue2_gan_exam.py::test_music_bipolar_and_live_defaults_untouched`
  plus the existing `test_yue2_arm_b` / `test_formulation_leaderboard`
  suites).
- New arms are `propose_only` and are NOT wired into the production
  trainer or the formulation leaderboard.
