# Winning formulation

Product repositories import one stamp from
[particle-sliders-core](../packages/particle-sliders-core):

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require(stamp.as_dict())
```

`stamp.id` is the winner those products train. They do not keep a second copy
of the routed particle adapter, the global-mix critic, the paired-error
losses, the noise schedule, `locked_shared`, or `GradRegularizer`.

## Current stamp

| | |
|---|---|
| Id | `particle-gmix-1600-v2` |
| Family | `anneal-routed-particle-error` |
| Hub recipe name | `anneal-routed-particle-error-yue2-v1` |
| Game | Paired-error relativistic logistic GAN, particle VIC, `b_cap` gradient cap |
| Critic | Global-mix, 8 tokens, width 48, 1 layer, 4 heads, score bound 8 |
| Adapter | Rank 8 routed MLP, 128 particles of dimension 4 |
| Noise | Paired-edit whitening, start `edit_rms / 0.28`, decay over 1600 steps, hold `1.3 * edit_rms` |
| Auxiliary MSE / FM / cover / pole / lyric hold | All zero |

The numbers are the Hub golden record. The same dict lives in
`analysis/slider2d/yue2_gmix_v2_exam.V2_SPEC`.
`tests/test_particle_sliders_formulation.py` fails if the package and that
record diverge. Merged #128 ran this game on the UNI continuation gates: the
EMA readout passes both cells on seeds 0, 1, and 7 at the native 1600 stop.
The root README calls this routed-particle game the preferred formulation.
Anima's published particle release is the same game.

`V2_SPEC["propose_only"]` is true so the 2D exam cannot flip the Music
trainer's `--lm_target`. That flag is not a product exemption. Products train
this stamp.

## Other stamps that are not the winner

| Stamp | Where it lives | Why it is not the product winner |
|---|---|---|
| `locked_shared` | `locked_shared_recipe()` / `SliderRecipe.require_locked_shared` | Music Arm B endpoint game: cover and pole MSE at weight 1, VIC and noise off, zero or one particle. Bipolar leaderboard teacher `faithful_guard_e` (rank 1 on `docs/FORMULATION_LEADERBOARD_BIPOLAR.md`, the #36-era behavioral board). |
| `faithful_plus_neu` | Unipolar leaderboard rank 1 | Hidden-state MSE, not the adversarial particle game. |
| In-repo research arms | `analysis/slider2d/`, `conceptmod/textsliders/` | Propose-only sweeps. They do not replace `winning_formulation()` until this module changes. |

`locked_shared` stays exported. A product that needs that endpoint game calls
`locked_shared_recipe()` from this package. It does not paste the recipe into
its own tree, and it does not relabel that recipe as the winning formulation.

## When the winner changes

Change `STAMP_ID` and the knobs in
`packages/particle-sliders-core/src/particle_sliders/formulation.py`, and
update the golden record this file pins. Say so in the commit. Product repos
then bump:

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
   Train through `winning_formulation()` (routed bridge, global-mix critic,
   `b_cap`). Delete the "checkout not required / do not depend on
   particle-sliders" contract, including the `PARTICLE_SLIDERS_ROOT` absence
   checks. Keep the Hub id `jimmycarter/krea2-turbo-bbox`, the Comfy node, the
   turbo sample numbers, and the prompt cards in that repo.
3. **supra-concept-sliders.** The opt-in trainer still lives in this
   repository as `conceptmod/textsliders/train_lora_supra.py`. Move that
   product entrypoint into the supra repo and depend on this core for the
   game. Keep Supra2-IMG ids, the Euler sample card, and Comfy (when it
   exists) in the product repo. Do not copy `RoutedMLP` or
   `GradRegularizer` into the move.
