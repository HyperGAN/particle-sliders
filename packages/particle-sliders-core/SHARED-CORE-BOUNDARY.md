# Model-independent Particle Sliders core

Core 0.14.0 is an explicit new packaging cohort, based on released ParticleGAN
0.9.5. Product repositories depend on this core and register or inject their
native factories. The core never constructs a named pretrained model.

| Shared core | Model product |
|---|---|
| ParticleGAN lifecycle, optimizer ownership, exact checkpoint/resume | Native model, tokenizer, scheduler, packing and decoder |
| Configuration, capabilities, raw export and inference contracts | Model paths, prompts, native response implementation |
| Generic adapter attachment/loading/strength helpers | Projection/adapter target choices and native checkpoint format |
| Evaluation splits, private RNG and observation rollback | Frozen teacher construction, native response coordinates |
| Hyperparameter/recipe inputs, execution and reporting | Product-specific presets, native admission and qualification |

Use `particle_sliders.lifecycle`, `particle_sliders.response` and
`particle_sliders.execution` for common operations. `native_lora` contains
generic adapter primitives, not a model loader. `pg09_audio` retains generic
autoregressive rows, last-hidden response base, conditioning and critic helpers;
model-specific subclasses belong to the product. Products own media generation.
`FrozenResponseAdapter` takes explicit `trajectory_direction` and
`teacher_convention` options; model-family names never select those semantics.
`FlowSchedule.increasing_linear` supplies a generic increasing-time schedule.

The default wheel and strict editable installation include only an explicit
list of shared modules plus the execution package. They exclude `textsliders`,
`notrigger`, native Music3/YuE2/Qwen/Fibo implementations, model-specific image
bindings, model prompts, campaign runners and model assets. Adding a file to the
source tree does not automatically add it to the public distribution. No legacy
`conceptmod` startup hook or product alias is installed. Stale files from earlier
local builds are removed from the build output before packaging.

Music3 native loading, its routed adapter and its concrete response subclass
move to [HyperGAN/music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders).
The product calls the same shared game and owns no second GAN implementation.
Other products adopt this boundary explicitly; no product is upgraded by a core
source change. Native products can keep different runtime/dependency environments.

## Compatibility

The full embedded-product core 0.13.0 remains at commit
`285180ed9813f12e912bb6bce6bada00496fd19a`. Its installed module paths, checkpoint
cohorts and source receipts remain available at that immutable source pin.
The exact replaced bindings are also retained under
`legacy/model-bindings/core-0.13.0` in this repository. Remaining embedded source
trees are historical migration input and are excluded from the new distribution.

Core 0.14 does not alias a removed model implementation to a different product
or silently migrate a pickle/checkpoint. A product pins its reviewed new core
commit, owns the extracted native implementation, binds a new source cohort,
and tests complete lifecycle continuation under that cohort. Old checkpoints
retain their original loader and source identity. Tensor key/format preservation
alone is not proof of cross-cohort training resume.

Historical pure teacher geometry is retained as a non-importable text source
receipt with its original byte digest. Historical generic engines remain named
APIs; they are not a substitute objective for the shared `ConditionalPG09` path.

Fresh sessions explicitly use `ema_decay=None` and raw current weights for
training, ranking, confirmation, export and inference. Upstream optimizer scalar
statistics are unchanged. The sole ranking metric is operand-first FP64 relative
SSE, pooled within each example and then averaged equally across examples.
Product quality/media diagnostics do not become ranking gates.

## Qualification

The distribution boundary is checked on built wheels and strict editable
installs outside a source checkout. Generic CPU lifecycle tests cover injected
response backends, updates, exact continuation, observation and exports. Product
tests separately establish their own native adapter ownership. CPU evidence
does not qualify GPU admission, pretrained/native media generation or quality.
