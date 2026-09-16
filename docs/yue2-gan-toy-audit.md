# YuE2 GAN-only transfer audit — September 16, 2026

**The current 600-step recipe does not pass the existing unipolar toy
acceptance gates.** This is a failed transfer check, not evidence that the
GAN equations are transcribed incorrectly. The same production update passes
these toys at 3400 steps. The live loss spike remains unexplained; running
the music model longer is not an established fix.

Reference: `mikkel/main` at `14e9aac` (#116 and #117), including the production
YuE2 recipe introduced at `3e5a3a0`. The selected existing suite passes
**111 tests**: formulation leaderboards, clone consolidation, the GAN behavior
cells, Music Arm B, seed isolation, and leaderboard honesty. Those tests did
not exercise the YuE2 port's complete optimization trajectory.

The audit is published on top of the subsequent `695fd0a` (#118), which adds
another locked-default check without changing the measured training code.

## Direct production-loop check

`analysis/slider2d/yue2_gan_exam.py` calls the actual YuE2 `build_game` and
`update` functions. It uses their MLP, fixed teacher-RMS calibration, paired
RpGAN losses, exact b_cap, optimizers, clipping, and D-then-G update order.
No loss or hyperparameter was changed for this audit.

The frozen backend is replaced by the existing `PairField`; the student is
the toy GAN's shared odd/even residual. Zero is exact by construction, like
multiplier LoRA. The original fixtures have **three rows** and all three are
shuffled per update; the live trainer uses four distinct rows. The fixture's
student is a direct residual, not a factorized adapter through a transformer.
Consequently, step budgets and parameter learning rates are not equivalent
measures of convergence between the toy and the music model.

The continuation scorer was extracted from `score_plus_neu_exam` without
altering its calculations. All 15 existing recipe/cell records were compared
against the original `14e9aac` implementation and were exactly identical.
Acceptance still requires cover >= 0.85, off-caption <= 0.05, and neutral
hold >= 0.85 on both divergent and close pairs. Cosine is not a pass gate.

Results below span seeds **0, 1, 7**. No best seed or checkpoint is selected.
Full scalar results, source hashes, and recipe pins are in
[the audit data](yue2-gan-toy-audit.json).

| Updates | Divergent cover | Close cover | Neutral hold | Both required cells pass |
|---:|---:|---:|---:|---|
| 600 | 0.482–0.484 | 0.585–0.587 | 1.000 | No, all three seeds fail |
| 1200 | 0.658–0.661 | 0.931–0.933 | 1.000 | No, divergent fails on every seed |
| 3400 | 0.931–0.932 | 0.933–0.935 | 1.000 | Yes, all three seeds pass |

At 600 updates, divergent off-caption is 0.042–0.083, failing its 0.05 gate
on two seeds as well. At 1200 and 3400, off-caption is zero on both cells.
The existing supervised `faithful_plus_neu` control passes at 400 steps on
every cell/seed. The audit does not add that supervised loss to the GAN.

The production-loop toy runs do not reproduce the live event: their losses
remain finite, with peak generator loss below 3.4. In the live metal run,
generator loss reached 621.4 at update 357 and cosine fell to 0.303; at
update 359 cosine was -0.172. All saved weights and optimizer states checked
were finite, and the recorded training source did not change during the run.
Finite values and a passing toy endpoint do not establish native stability.

## What matches main, and what does not

The latest unipolar leaderboard winner uses positive/neutral supervised MSE.
It validates the raw positive teacher and neutral-hold contract, **not** this
GAN-only optimization recipe. The existing passing GAN toy is a different
training setup:

| Component | Locked GAN toy | Current YuE2 GAN-only port |
|---|---|---|
| Teacher / polarity | Leftover-gated bipolar poles | Raw positive-caption deltas, +1 only |
| G objective | RpGAN plus cover, prior L2 and VICReg terms | RpGAN only |
| Critic | Fourier features, width 64 | Two-layer MLP, width 256 |
| Coordinates / sampling | Pole clouds and movable particle prior | Fixed teacher-RMS deltas, prompt states |
| Optimizers | Adam, G/D LR 0.005, betas (0, 0.99) | G AdamW 0.0005, D Adam 0.00075, betas (0, 0.999) |
| Schedule / averaging | Delayed cosine, residual EMA 0.995 | Constant LR, no EMA |
| Cap | Exact b_cap, coefficient/threshold 1 | Same cap equations, applied in calibrated coordinates |

These differences are not a diagnosis of the spike. They establish that
matching RpGAN and b_cap equations does not mean the complete winning toy
formulation was transferred. The #116 alternatives remain proposals, not
new production defaults. No auxiliary loss, bipolar target, sampling step,
optimizer change, or music retraining is introduced by this audit.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m analysis.slider2d.yue2_gan_exam \
  --steps 600 1200 3400 --seeds 0 1 7 --out /tmp/yue2-gan-exam.json
```

This command **exits 1** because not all requested budgets pass. It writes
every update and checkpoint score. Later success cannot erase a requested
earlier failure. `--steps 3400` checks only that budget; it is not permission
to resume the failed music run. Both required cells must be present.

```bash
PYTHONPATH=. pytest -q tests/test_yue2_gan_exam.py tests/test_yue2_arm_b.py \
  tests/test_formulation_leaderboard.py tests/test_lm_plus_neu_exam.py
```

The targeted regression suite passes **39 tests**. This includes confirming
that the current 600-step production update is **rejected**, not blessing it
as a passing recipe. Two-update numerical parity and native eager/checkpoint
gradient tests remain useful implementation checks; neither substitutes for
behavioral acceptance or explains the live instability.
