# particle-sliders-core

Installable shared core for HyperGAN slider products. Import `particle_sliders`.
The pip name `concept-slider-core` and the import `concept_slider_core` are
deprecated aliases from the Anima extraction. `concept_slider_core` still
re-exports this API and warns on import.

Source repository: [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
Pin this repository. Do not pin `mikkel/sliders-conceptmod` for new installs.

Product repositories depend on this package and train
`winning_formulation()`. They do not re-implement routed particles,
`locked_shared`, or `GradRegularizer` locally. Formulation toys and the DSL
board stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).
This package does not vendor ParticleGAN. `GradRegularizer` is the
byte-pinned `particlegan` 0.2.0 excerpt already required by the torch pin
in this research repo; call it from here rather than copying it again.

## Install

Pin a full commit. Install the product's tested PyTorch build first.

```bash
python -m pip install --no-deps \
  "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core"
```

From a checkout of this repository:

```bash
python -m pip install --no-deps -e ./packages/particle-sliders-core
python -m pytest packages/particle-sliders-core/tests
```

The retired subdirectory `packages/concept-slider-core` is a pointer, not the
package. Commits at or before `beaffeb` still contain the old tree.

## Winning formulation

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require({**stamp.as_dict(), "g_lr": 2e-5})  # g_lr is a model surface
bridge = stamp.bridge()
critic = stamp.critic(training_targets, neutrals=training_neutrals)
regularizer = stamp.regularizer()
d_loss, g_loss, vic = stamp.losses()
sigma = stamp.noise_std_at(step, edit_rms)
```

`stamp.id` is the current provisional default. Today's id is
`particle-gmix-1600-v2` (family `anneal-routed-particle-error`), with
`stamp.provisional` true. The next winner is whatever
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) crowns on
the full live leaderboard (9 toys and all 29 bounds; partial wins do not
count). Related search:
[ParticleGAN #39](https://github.com/255BITS/ParticleGAN/pull/39).
Replace `CURRENT_STAMP_ID` and `CURRENT_STAMP` in `particle_sliders.formulation`
when that search finishes. Products pick up the new record by bumping the
commit pin and still call `winning_formulation()`. Copying the knobs into
anima, krea2, or supra is a fork: `require()` raises on formulation drift.

Model surfaces a product may override without forking the game: generator,
critic, and particle learning rates, batch size, and the data-budget fields
listed on `stamp.model_surface_keys`. Hub model ids, Comfy class names, and
the train/infer wiring around this stamp stay in the product repo.

`particle_gmix_1600_v2()` / `gmix_recipe()` and `locked_shared_recipe()` are
named recipes. The product default is `CURRENT_STAMP`, which tracks #38's
eventual winner. `locked_shared` (Music Arm B, teacher `faithful_guard_e`)
stays callable and can become that default if the full board crowns it.

The long-term product pattern is this dependency. A self-contained product
that forbids a particle-sliders checkout (`PARTICLE_SLIDERS_ROOT` or an
equivalent "do not depend on particle-sliders" rule) is the wrong
architecture. Prefer this git subdirectory pin over an ad-hoc source root.

See [docs/winning-formulation.md](../../docs/winning-formulation.md) and
[docs/shared-core.md](../../docs/shared-core.md).

## What stays in the product

Projection lists, hooks, prompt cards, checkpoint schemas, sampling, Hub
ids, Comfy nodes, and release evidence. The shared code is the routed
adapter, the global-mix critic, relativistic paired losses, particle VIC,
the noise schedule, the gradient regularizer, ordinary LoRA fitting, and
the winning-formulation stamp.
