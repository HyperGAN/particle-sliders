# particle-sliders-core

## Conditional ParticleGAN 0.9 (0.5.0)

New integrations use `particle_sliders.pg09.ConditionalPG09` with a thin
native-response backend. See [the contract](CONDITIONAL-PG09-CONTRACT.md).
The package depends on released `particlegan==0.9.0`; the wheel hash and all
49 Python source hashes are pinned and checked. The default public preset is
`bispf_optimized`. Generator training uses its adversarial loss plus required
public prior regularization; critic training uses its adversarial loss plus
public critic penalty. There is no optimized MSE or reconstruction term.

Native image/video response axes are retained through the PG penalty. Flat
continuous vectors of width170 or greater fail closed because the unchanged
release classifies these as categorical data. ALZ's explicit-domain proposal
has not been installed. A backend with no routed bank is labeled adapter GAN;
the core never invents a disconnected particle prior.

Training checkpoints include optimizer/regularizer observers and private RNG
streams, validate source/domain/conditioning identities before live mutation,
and resume deterministically on the qualified runtime. Adapter-only exports
are for inference, not training resume. CPU tests cover those contracts; live
GPU memory, long-run stability and visual quality require separate admission.

## Preserved BISPF 0.8 interface (historical 0.4.0)

The historical paired-error interface is `BISPFGame`, originally backed by
`get_recipe("bispf")` at ParticleGAN commit
`b5d2372f2739d7579ab66d6dba6e220d2eb776d9` (0.8.0).
See [the integration guide](../../docs/bispf.md) for the callback contract,
recipe differences, checkpoints and executed tests. `bispf_provenance.json`
records the upstream source hashes. This is the SRpGAN/formulation-Adam
component variant, with the product's existing full-bank routing.

The sections below describe the preserved historical gmix stamp. In upstream
0.8 the old top-level `GradientPenalty`, `GradRegularizer`, `LOCKED_SHARED`,
`make_b_cap`, and `make_gan_loss` exports are retired. The local fixed-cap
wrapper uses `particlegan.grad_regularizers.GradientPenalty`; it accepts only
L2/autograd/no-anneal compatibility settings. `particlegan_locked_shared()`
now returns a `legacy-fixed-bcap` label and equivalent cap/loss builders,
without the retired upstream stamp object. The removed
`finite_difference_norm` re-export has no supported 0.8 equivalent.

## Historical gmix and endpoint interfaces

Installable shared core for HyperGAN slider products. Import `particle_sliders`.
The pip name `concept-slider-core` and the import `concept_slider_core` are
deprecated aliases from the Anima extraction. `concept_slider_core` still
re-exports this API and warns on import.

Source repository: [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
Pin this repository. Do not pin `mikkel/sliders-conceptmod` for new installs.

Historical product repositories depend on this package and train
`winning_formulation()`. They do not re-implement routed particles,
`locked_shared`, or `GradRegularizer` locally. Formulation toys and the DSL
board stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).

The historical interface was adapted to ParticleGAN 0.8.0's component API.
Cap, RpGAN loss, and particle VIC come from that public API
(`GradientPenalty`, `GANLoss`, `ParticleRegularizer`, `make_b_cap`,
`make_gan_loss`, `get_recipe`). `GradRegularizer` remains a thin
compatibility alias for anima and other products that still import that name
from `particle_sliders`. Gmix architecture (RoutedMLP, global-mix critic,
product game structure) stays in this package.

`particlegan` is a **transitive dependency**: pin `particle-sliders-core` and
you get the released ParticleGAN version recorded in this package's
`pyproject.toml`. Historical provenance files describe their original arms.

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


```python
from particle_sliders import winning_formulation, FormulationGame

stamp = winning_formulation()
config = {**stamp.as_dict(), "g_lr": 2e-5}  # only stamp.model_surface_keys may differ
stamp.require(config)

# targets / neutrals: [N, adapter_rank] paired positive and neutral features.
# Without them the critic falls back to a random paired bank.
game = FormulationGame(stamp, config, critic_targets=targets, critic_neutrals=neutrals,
                       extra_generator=product_params)  # optional trainable product params
for step in range(1, steps + 1):            # steps count from 1
    stats = game.step(step, features)       # features: [batch, adapter_rank]
```

`step()` compares shared noise with noise plus `bridge(features, particles)`
and returns `d_loss`, `g_loss`, `vic` and `sigma`. The runnable reference is
[`tests/test_formulation_game.py`](tests/test_formulation_game.py).
