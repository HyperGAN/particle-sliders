# Winning formulation

Product repositories import one stamp from
[particle-sliders-core](../packages/particle-sliders-core):

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require(stamp.as_dict())
```

`winning_formulation()` is the only product entry point. It is gmix
architecture plus the current formulation overlay.

**Depends on ParticleGAN develop.** Pinning `particle-sliders-core` pulls
`particlegan` (develop SHA in that package's `pyproject.toml`). Products call
`stamp.regularizer()` / `stamp.losses()`; they do not vendor ParticleGAN
excerpts. Optional helpers: `particlegan_get_recipe()`,
`particlegan_locked_shared()` (`make_b_cap` / `make_gan_loss`).

**Architecture: gmix.** Routed particles and a global-mix critic
(`gmix_architecture()`). This is the product game structure. It does not
wait on ParticleGAN #38.

**Formulation: the #38 winner, provisional until that search finishes.**
Caps, coefficients, learning rates, particle counts, schedules, and a later
upsampler choice such as residual16 versus transpose come from
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38). The related
search is [ParticleGAN #39](https://github.com/255BITS/ParticleGAN/pull/39).
The ultimate gate is the full live leaderboard: 9 trained toys and all 29
live bounds. A partial win, a 3-toy screen, or an EMA-only result does not
count. `stamp.formulation_provisional` is true until that crown is plugged
into the gmix architecture.

Products do not keep a second copy of the routed particle adapter, the
global-mix critic, the paired-error losses, the noise schedule,
`locked_shared`, or `GradRegularizer`. Cap / GAN loss / particle VIC are
ParticleGAN develop API calls through this core (`GradientPenalty`,
`GANLoss`, `ParticleRegularizer`); `GradRegularizer` is only a compatibility
alias.

## Architecture

| | |
|---|---|
| Id | `gmix` (`stamp.architecture_id`) |
| Family | `particle-gmix` |
| Adapter | Routed particle MLP, rank 8, width 48, router width 16 |
| Critic | Global-mix, 8 tokens, width 48, 1 layer, 4 heads, score bound 8 |
| Game | Paired-error relativistic logistic GAN plus particle VIC |

## Current provisional formulation

| | |
|---|---|
| Id | `particle-gmix-1600-v2` (`stamp.formulation_id`) |
| Callable | `particle_gmix_1600_v2()` |
| Hub recipe name | `anneal-routed-particle-error-yue2-v1` |
| Particles | 128 of dimension 4 |
| Cap | `b_cap` coefficient 1, kappa 1, every 4 steps |
| Noise | Paired-edit whitening, start `edit_rms / 0.28`, decay over 1600 steps, hold `1.3 * edit_rms` |
| Auxiliary MSE / FM / cover / pole / lyric hold | All zero |

These parameters are the released Hub record
(`analysis/slider2d/yue2_gmix_v2_exam.V2_SPEC`, UNI gates in merged #128,
Anima's published particle release). They are the overlay until #38 crowns
a full-board winner. They are not a second architecture.

`V2_SPEC["propose_only"]` is true so the 2D exam cannot flip the Music
trainer's `--lm_target`. Products still call `winning_formulation()`.

## Named callables

| Callable | Role |
|---|---|
| `gmix_architecture()` / `gmix_recipe()` | Fixed gmix architecture constants. |
| `particle_gmix_1600_v2()` | Provisional formulation overlay. Current `CURRENT_FORMULATION`. |
| `locked_shared_recipe()` | Endpoint recipe (cover and pole MSE, VIC and noise off, teacher `faithful_guard_e`). Not the product architecture. |
| `winning_formulation()` | Gmix architecture with `CURRENT_FORMULATION` applied. |

## When #38 crowns a winner

In `packages/particle-sliders-core/src/particle_sliders/formulation.py`,
replace the formulation block. Leave `gmix_architecture()` in place:

```python
CURRENT_FORMULATION_ID = "<crowned id>"
CURRENT_FORMULATION = <crowned_formulation>
CURRENT_FORMULATION_PROVISIONAL = False
```

Add the crowned parameter export in that file if it is not already
`particle_gmix_1600_v2`. The overlay must not restate gmix architecture keys
(`critic` stays `gmix`). Caps, coefficients, learning rates, particle counts,
schedules, and an upsampler choice such as residual16 versus transpose belong
in the overlay. Leave every product call site on `winning_formulation()`.
Say so in the commit. Product repos then bump:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<new-commit>#subdirectory=packages/particle-sliders-core
```

A product does not fork the math while waiting for that bump. Model surfaces
listed on `stamp.model_surface_keys` (learning rates, batch, data budget)
may differ per backbone. Hub ids, Comfy class names, prompt cards, hooks, and
sampling stay in the product.

## What Krea2's self-contained rule gets wrong

[krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders)
currently treats a particle-sliders checkout as forbidden. Its README says a
checkout is not required, and `tests/test_docs_contract.py` fails if
`PARTICLE_SLIDERS_ROOT` appears in the README or `REPRODUCE.md`. Its
`FORMULATION.md` says a winning recipe comes back into that repo as a local
config.

That is not the long-term pattern. Krea2 should pin this package the same way
Anima and Supra do, and call `winning_formulation()`. An ad-hoc
`PARTICLE_SLIDERS_ROOT` path is a worse install than the git subdirectory pin,
and banning the dependency entirely leaves the product re-implementing the
game. Formulation toys still belong in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod), not in the
product repo and not in ParticleGAN.

## Product follow-ups

These land in the product repositories. This pull request does not rewrite
their trainers.

1. **anima-particle-sliders.** Replace the pin
   `concept-slider-core @ git+https://github.com/mikkel/sliders-conceptmod.git@beaffeb#subdirectory=packages/concept-slider-core`
   with `particle-sliders-core` from `HyperGAN/particle-sliders`, subdirectory
   `packages/particle-sliders-core`, at the commit that introduces this stamp.
   Import `particle_sliders.winning_formulation` and call `require()` on the
   training config. Stop vendoring the reference implementation as the source
   of truth (`core.lock.json` still hashes the old `concept_slider_core`
   files). Backbone `g_lr` 2e-5 is a model surface. The published hold at 1.0
   disagrees with this stamp's `noise_hold_ratio` 1.3; resolve that by
   training the stamp or by changing the stamp here, not by keeping a local
   schedule.
2. **krea2-particle-sliders.** Add the same git subdirectory dependency.
   Train through `winning_formulation()` (gmix architecture; today's
   provisional parameters are `particle-gmix-1600-v2`). #38 replaces the
   overlay, not the architecture. Delete the "checkout not required / do not
   depend on particle-sliders" contract, including the `PARTICLE_SLIDERS_ROOT`
   absence checks. Keep the Hub id `jimmycarter/krea2-turbo-bbox`, the Comfy
   node, the turbo sample numbers, and the prompt cards in that repo.
3. **supra-concept-sliders.** The opt-in trainer still lives in this
   repository as `conceptmod/textsliders/train_lora_supra.py`. Move that
   product entrypoint into the supra repo and depend on this core for the
   game. Keep Supra2-IMG ids, the Euler sample card, and Comfy (when it
   exists) in the product repo. Do not copy `RoutedMLP` or
   `GradRegularizer` into the move.
