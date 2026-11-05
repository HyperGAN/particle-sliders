# particle-sliders-core 0.13.0

Shared particle-slider training, response binding, checkpointing and export,
using **ParticleGAN 0.9.5**. Products supply their frozen model and adapters;
the shared core supplies the GAN training engine. GAN mathematics stays in
ParticleGAN rather than being copied into each product.

Install from this repository's `packages/particle-sliders-core` directory.
Python 3.11+ and PyTorch 2.6+ are required. The exact dependency is
`particlegan==0.9.5`; see [compatibility and source identity](COMPATIBILITY-PG095.md).

The existing `ConditionalPG09` and lifecycle APIs are retained. For wide native
continuous vectors, explicitly declare `penalty_input_domain="continuous"` with
`response_domain="continuous_vector"`. Automatic dispatch retains its shape
gates. This is released upstream support, not a categorical fallback or reshape.

Old checkpoints retain their original core/ParticleGAN environments. The explicit
`pg095` entrypoint identifies fresh 0.9.5 training; `pg094` refuses old requests
rather than migrating them. Historical `winning_formulation()` helpers remain
available for architecture and explicitly named historical recipes.

- [Conditional response contract](CONDITIONAL-PG09-CONTRACT.md)
- [Training commands](TRAINING-COMMAND.md)
- [Image lifecycle](IMAGE-LIFECYCLE.md)
- [Native inference](NATIVE-INFERENCE.md)
- [Previous PG0.9.4 compatibility record](COMPATIBILITY-PG094.md)

CPU fixtures verify training and serialization behavior. They do not establish
native-model quality or a benchmark win.
