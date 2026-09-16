# YuE2 Arm B formulation verification

The reference is the Music Arm B transfer in `main` at `1cb5b09`, plus the
locked Music transfer settings: MLP/scaled, four rows, bipolar pole weight 1,
RpGAN weight 1, ending weight 1, constant optimizer settings. This is not the
old female UNI16 span-transformer/FM recipe. It is also not a literal copy of
the toy's Fourier critic, latent particles, jitter, EMA, or cosine schedule.
Those are explicitly different in the Music transfer.

The port uses the shared `ARM_B` contract, `lm_faithful_guard_e`, Music
`LMDiscriminator`, toy RpGAN helpers, and vendored `GradRegularizer`.
Only native token packing, music-start hidden extraction, ending readout,
and AR LoRA forwarding are model specific.

## Traps checked

- The pulled factory/flag gates alone do not connect a regularizer to a live
  update. The YuE2 update calls the vendored penalty for every critic update.
- `analysis.slider2d.adv.rp_g_loss` takes `(real, fake)`; the older Music helper
  takes `(fake, real)`. The independent test checks the formula and updates.
- Applying the vendored penalty directly to a critic that divides its input
  by teacher RMS changes the cap's units. The port passes calibrated samples
  to `critic.net`, so the norm is taken in the same coordinates as the game.
- Pole loss sums the two polarities. Adversarial and ending terms average
  them. Averaging every term halves the pole pin. Adding both `cover` and
  `pole` doubles it. The port does neither.
- The guard changes only the odd leftover, preserving the midpoint. It
  declines a destructive subtraction. Declared axes are independently
  encoded once using the first row's lyrics, as in Music.
- No negative caption is silently dropped by the old positive-only loader.
  Arm B requires positive, negative, and neutral captions and validates axes.
- Adapter scale remains active through checkpointed backward recomputation.
  The critic is frozen during generator backward; the base model stays frozen.
- Fresh rows and seeds start at update 1. Checkpointing does not change the
  objective, batch size, optimizer, or learning rate.

## Executable evidence

`tests/test_yue2_arm_b.py` compares two complete discriminator/generator
updates against independently written equations and a full-batch backward.
It covers an inactive and active cap, fixed non-unit teacher scale, losses,
generator gradients, and updated generator/discriminator parameters. The
reference sums the shared pole MSE exactly once and explicitly calculates the
calibrated input-gradient penalty.

The remaining tests use the official tiny YuE2 architecture to compare shared
guard targets, checkpointed versus eager gradients at both polarities, frozen
base weights, exact scale-zero output, and uninterrupted versus saved/resumed
training including both optimizer states and sampled history seeds. They also
reject malformed bipolar prompts and incompatible resumes.

Run in the isolated YuE2 environment:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_yue2_arm_b.py \
  tests/test_music_arm_b.py tests/test_music_arm_b_gates.py \
  tests/test_yue2_slider.py tests/test_yue2_uni.py tests/test_yue2_fresh.py
```

These checks establish implementation parity. A short live GPU preflight then
checks finite gradients, saved-state/export agreement and native rendering.
Neither CPU parity nor two GPU updates establish that a 600-step metal slider
sounds good; use the separate held-out prompts and matched seeds to assess it.
