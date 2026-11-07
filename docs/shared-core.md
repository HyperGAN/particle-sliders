# Shared particle-sliders API

Product slider packages depend on `particle-sliders-core`, which delegates GAN
mathematics to exact released ParticleGAN **0.9.5**. Core **0.14.0** packages only
generic lifecycle, response, adapter and execution modules. Read the
[explicit module boundary](../packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md)
and [product inventory](backend-migration-inventory.md) before selecting a cohort.

Products register entry-point factories or inject complete owned bindings into
`particle_sliders.lifecycle` / `particle_sliders.response`. A product loads its
native model, builds teachers, selects adapter targets and supplies response
coordinates. Shared `ConditionalPG09` owns updates, optimizer state and private
RNG. A stamp import or a dummy call does not establish shared live execution.

| Shared package owns | Product owns |
|---|---|
| Adapter attachment, activation and optimizer ownership | Native target names and architecture choices |
| Upstream game lifecycle and exact checkpoint continuation | Frozen model, tokenizer, scheduler and source identity |
| Evaluation splits and observation rollback | Teacher construction, response coordinates and row policy |
| Raw adapter export and inference dispatch | Media generation and native qualification |
| Recipe/hyperparameter inputs and execution contracts | Native admission, presets and release decisions |

Use new immutable source cohorts for adoption. Core 0.14 does not rebind old
checkpoints, supply model-specific pickle aliases or upgrade existing products.
Music3 moves its native loader and concrete AR response adapter into
[its product](https://github.com/HyperGAN/music3-particle-sliders). The generic
`ARLastHiddenResponseAdapter`, conditioning, critic and response-row helpers stay
in `particle_sliders.pg09_audio`.

New sessions explicitly disable weight EMA (`ema_decay=None`). All model,
adapter, critic and particle-bank weights remain raw; upstream optimizer scalar
statistics remain unchanged. Rank only by operand-first FP64 per-example pooled
relative SSE, then average equally across examples. Media and preservation
measurements are diagnostics.

Training checkpoints resume training; adapter-only exports cannot be used as
training packets. Invalid source, architecture, optimizer, sampler or packet
family is rejected before live mutation. Evaluation restores training RNG and
adapter modes/buffers. Wide continuous responses declare their supported domain
explicitly, rather than being reshaped or projected to bypass a penalty gate.

Historical named generic engines remain available for their documented recipes.
Embedded product sources are retained as migration input outside the default
0.14 distribution. The complete previous packaging and API contract remains at
[core 0.13 source](https://github.com/HyperGAN/particle-sliders/tree/285180ed9813f12e912bb6bce6bada00496fd19a).
Native runtime, GPU memory and pretrained quality require separate product tests.
