# particle-sliders

Shared infrastructure for building concept sliders with ParticleGAN. The
`hypergan/*-particle-sliders` products supply their models; this repository
supplies a common training lifecycle, adapter controls, checkpointing,
evaluation, hyperparameter inputs and execution tools.

A slider is an adapter on a frozen model. Its strength controls the learned
edit, and strength zero bypasses the adapter. Model loading, prompts, response
coordinates, media generation and product presets belong to each product.

## Shared core and products

```
 music3-particle-sliders   other *-particle-sliders products
                  \       /
             particle-sliders-core
                      |
               ParticleGAN 0.9.5
```

Core **0.14.0** introduces an explicit model-independent distribution boundary.
The wheel and strict editable install contain generic modules only. Music3's
native loader and response implementation belong to
[HyperGAN/music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders),
which uses the shared game instead of copying its training loop. Other products
adopt the new cohort through their own reviewed source pins.

The source tree still retains historical model integrations and experiments as
migration input. Their presence does not make them part of the 0.14 wheel.
Existing checkpoints and captured runs keep their original source/environment.
See [the exact boundary and compatibility rules](packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md).

| Need | Shared entry point |
|---|---|
| Bind an injected frozen model and train a response | [`response`](packages/particle-sliders-core/RESPONSE-SESSIONS.md) |
| Use an image lifecycle | [`lifecycle`](packages/particle-sliders-core/IMAGE-LIFECYCLE.md) |
| Apply an exported adapter | [Native inference contract](packages/particle-sliders-core/NATIVE-INFERENCE.md) |
| Tune hyperparameters with a consistent metric | [Evaluation and LR search](packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md) |
| Configure and dispatch work | [Training commands](packages/particle-sliders-core/TRAINING-COMMAND.md) |
| Understand the upstream game and shape contracts | [Conditional ParticleGAN contract](packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md) |
| Migrate a product | [Shared API](docs/shared-core.md) · [Migration inventory](docs/backend-migration-inventory.md) |

## Install and integrate

Python 3.11+ and PyTorch 2.6+ are required. Pin a reviewed full commit in products;
the package depends on exact released `particlegan==0.9.5`.

```sh
pip install './packages/particle-sliders-core[serialization]'
# Development uses the same restricted module boundary:
pip install -e './packages/particle-sliders-core[serialization]'
```

Use `particle_sliders.response`, `particle_sliders.lifecycle` and
`particle_sliders.execution`. Products register entry-point factories or inject
owned model bindings; core does not import products to construct models. Generic
adapter primitives live in `native_lora`, `adapter_activation`,
`adapter_attachment` and `peft_scaling`. `pg09_audio` provides generic AR response
helpers; concrete model subclasses live in their products.

Fresh sessions explicitly use `ema_decay=None`. Training, ranking, confirmation,
exports and inference use raw current weights. Rank only by teacher/student
operand-first FP64 relative SSE, pooled within each example and averaged equally
across examples. Keep media and preservation diagnostics separate from ranking.
ParticleGAN's optimizer scalar statistics remain unchanged.

## Compatibility and evidence

Core 0.13.0's embedded model modules, legacy `conceptmod` aliases and historical
product guidance remain available at
[`285180e`](https://github.com/HyperGAN/particle-sliders/tree/285180ed9813f12e912bb6bce6bada00496fd19a).
Core 0.14 does not install those namespaces or silently relocate checkpoint
classes. The replaced bindings are also archived under
[`legacy/model-bindings/core-0.13.0`](legacy/model-bindings/core-0.13.0).

CPU fixtures verify shared lifecycle behavior and installed package isolation.
They do not establish native GPU admission, pretrained media quality or successful
adoption by every product. Product owners qualify their native runtime separately.
Historical experiments, prompts and results remain in [docs](docs/README.md) and
`lr-work`; their recorded source pins and scientific identities remain binding.

For development, install the test extra and run `python -m pytest` against the
installed wheel. The default suite tests the shared core without adding `src/`
to Python paths; isolated install tests also run outside the checkout. The install/uninstall tests also check
strict editable installs and coexistence with a separately owned Conceptmod.
Historical native/namespace tests apply to their original cohorts; see the
[boundary qualification record](packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md).

Research credit and the [MIT license](LICENSE) are preserved.
