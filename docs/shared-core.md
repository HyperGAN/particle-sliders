The dependency contract covers all model families: product slider packages → particle-sliders → ParticleGAN/HyperGAN. See the [backend migration inventory](backend-migration-inventory.md) for implemented sessions, existing bindings and explicit legacy gaps.

# Shared particle-sliders API

The installable `particle-sliders-core` package owns the conditional GAN update,
state protocol and reusable native-response adapters. Product repositories pin
a reviewed full commit and execute `particle_sliders.ConditionalPG09.step()`;
a stamp import or dummy-only call does not establish shared live execution.

Version 0.6.0 pins released ParticleGAN **0.9.1**, with explicit FP32 host
execution. It does not depend on develop or 0.10. Upstream's new BF16/channels-last
CUDA trainer policy remains a separate native execution qualification.
[Release and compatibility](https://github.com/255BITS/ParticleGAN/pull/450).

Install the Git subdirectory package, optionally with `[serialization]`, rather
than using a workspace checkout or copying modules. See the
[package API table and examples](../packages/particle-sliders-core/README.md)
and [full contract](../packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md).
All public backend/campaign/native-architecture modules are inside the wheel;
imports never load pretrained weights or select a GPU. The old repository paths
alias those implementations. `concept_slider_core` is a deprecated root alias;
`packages/concept-slider-core` is a retired path.

| Shared package owns | Product owns |
|---|---|
| Actual adapter parameter ownership and signed checkpoint scale capture | Native projection names and attachment choices |
| Public PG loss/penalty/prior and optimizer lifecycle | Frozen base model, exact model/tokenizer/scheduler identity |
| Independent D/G sampling and checkpointed private RNG/coverage | Frozen teacher conditioning, native response geometry and row policy |
| Strict training checkpoint and adapter-only inference protocol | Native sampler/decoder orchestration and matched quality evidence |
| Reusable lazy cache providers and response adapters | Weight/model staging, hardware admission and release decisions |

Generator training uses public adversarial loss plus its required public prior
regularizer, without optimized reconstruction/MSE. Native fields preserve
spatial/time axes through the penalty. Wide continuous vectors still require an
explicit exact source overlay; the package never silently patches dependencies,
projects vectors into fake images, or changes the objective.

`state_dict()`/`load_state_dict()` continue successful training updates.
`export_state_dict()`/`load_export_state_dict()` load adapter-only inference;
these exports cannot become resumable training packets. Invalid optimizer,
sampler, source, architecture, conditioning and export-step state fails before
live mutation. Evaluation preserves training RNG and adapter modes/buffers.

Source relocation and PG 0.9.0→0.9.1 change checkpoint lineage. Existing pinned
runs retain their original checkout/environment and receipts. Adoption requires
a new cohort or explicit reviewed migration; no transparent rebind is provided.
CPU tests do not establish native GPU memory, full sampler/decoder parity or
model quality.

The historical `winning_formulation()`, `FormulationGame`, `EndpointGame` and
`BISPFGame` remain available for named historical arms. Endpoint includes MSE;
the old wide-vector BISPF route lacks the new domain gate. Every legacy/default
CLI must be migrated or explicitly closed by its owner; a package dependency
alone is insufficient. Research toys remain in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).

New image consumers use the [versioned image session lifecycle](../packages/particle-sliders-core/IMAGE-LIFECYCLE.md), which owns factory preparation, stepping, EMA, checkpoint/export, response/native rendering dispatch and cleanup. Historical root exports retain their named objectives; they are not the recommended image lifecycle.

[HyperGAN run-rendering and HANDL network-definition seams](hypergan-interface-seams.md) are handed off separately; current custom definitions are inventoried and have not been silently rewritten.
