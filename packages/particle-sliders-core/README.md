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

This package **depends on ParticleGAN develop** (`particlegan` 0.6.0 API).
Cap, RpGAN loss, and particle VIC come from that public API
(`GradientPenalty`, `GANLoss`, `ParticleRegularizer`, `make_b_cap`,
`make_gan_loss`, `get_recipe`). `GradRegularizer` remains a thin
compatibility alias for anima and other products that still import that name
from `particle_sliders`. Gmix architecture (RoutedMLP, global-mix critic,
product game structure) stays in this package.

`particlegan` is a **transitive dependency**: pin `particle-sliders-core` and
you get the ParticleGAN develop SHA recorded in this package's
`pyproject.toml` (and in `locked_provenance.json`).

## Install

Pin a full commit of HyperGAN/particle-sliders. Install the product's tested
PyTorch build first; then install this package (it pulls `particlegan`).

```bash
python -m pip install \
  "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@<commit>#subdirectory=packages/particle-sliders-core"
```

From a checkout of this repository (editable ParticleGAN + core):

```bash
python -m pip install -e /path/to/ParticleGAN
python -m pip install -e ./packages/particle-sliders-core
python -m pytest packages/particle-sliders-core/tests tests/test_particle_sliders_formulation.py
```

The retired subdirectory `packages/concept-slider-core` is a pointer, not the
package. Commits at or before `beaffeb` still contain the old tree.

## Winning formulation

Minimal train-loop surface (stamp + ParticleGAN-backed primitives):

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require({**stamp.as_dict(), "g_lr": 2e-5})  # g_lr is a model surface
bridge = stamp.bridge()            # gmix RoutedMLP
critic = stamp.critic(training_targets, neutrals=training_neutrals)
regularizer = stamp.regularizer()  # particlegan.GradientPenalty (b_cap)
d_loss, g_loss, vic = stamp.losses()  # particlegan GANLoss + ParticleRegularizer
sigma = stamp.noise_std_at(step, edit_rms)

# optional: ParticleGAN develop recipes / locked builders
from particle_sliders import particlegan_get_recipe, particlegan_locked_shared
recipe = particlegan_get_recipe("gan")
locked = particlegan_locked_shared()  # make_b_cap + make_gan_loss
```

Architecture is gmix (`stamp.architecture_id`). Formulation parameters are
provisional: today's overlay id is `particle-gmix-1600-v2`
(`stamp.formulation_id`, `stamp.formulation_provisional` true). The next
parameter set is whatever
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) crowns on
the full live leaderboard (9 toys and all 29 bounds; partial wins do not
count) and is plugged into this same gmix architecture. Related search:
[ParticleGAN #39](https://github.com/255BITS/ParticleGAN/pull/39).
Replace `CURRENT_FORMULATION_ID` and `CURRENT_FORMULATION` in
`particle_sliders.formulation` when that search finishes. Do not replace
`gmix_architecture()`. Products pick up the new overlay by bumping the commit
pin and still call `winning_formulation()`. Copying the knobs into anima,
krea2, or supra is a fork: `require()` raises on formulation drift.

Model surfaces a product may override without forking the game: generator,
critic, and particle learning rates, batch size, and the data-budget fields
listed on `stamp.model_surface_keys`. Hub model ids, Comfy class names, and
the train/infer wiring around this stamp stay in the product repo.

`gmix_architecture()` / `gmix_recipe()` is the fixed structure.
`particle_gmix_1600_v2()` is the provisional formulation overlay
(`CURRENT_FORMULATION`). `locked_shared_recipe()` remains a named endpoint
recipe. It is not the product architecture. #38's crowned caps, coefficients,
learning rates, particle counts, and schedules land on gmix through
`CURRENT_FORMULATION`.

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
the noise schedule, the gradient regularizer (via ParticleGAN), ordinary
LoRA fitting, and the winning-formulation stamp.

## Product train step

Call ``winning_formulation()`` then ``FormulationGame`` for the shared
feature-space D/G step (bridge, critic, losses, regularizer, particle VIC).
Products project model residuals into ``adapter_rank`` features; they do not
reimplement the optimizer loop. ``EndpointGame`` remains the bipolar
teacher/predict path. Distillation stays in ``fit_routed_down``.

