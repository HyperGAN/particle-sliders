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

## Native adapter targets

Products supply an explicit ordered plan of `(export_name, host_module)` pairs
to `LoRANetwork`. Selection stays with the product, which knows its native model
and checkpoint naming scheme. Core validates the complete plan before allocating
or attaching adapters, then preserves its names and order verbatim:

```python
from particle_sliders.native_lora import LoRANetwork

# These paths and exported names belong to the product's reviewed source cohort.
projection_paths = ["blocks.0.query", "blocks.0.value"]
plan = [
    ("product_" + path.replace(".", "_"), model.get_submodule(path))
    for path in projection_paths
]
network = LoRANetwork(model, rank=8, alpha=8., multiplier=0., target_plan=plan)
```

Each host must belong to the supplied root. Names must be unique, nonempty
module names without dots, and cannot shadow network attributes or activation
state; hosts cannot appear twice. Supported operators are
`nn.Linear`, `nn.Conv1d` and `nn.Conv2d`, including subclasses. Convolutions in an
explicit plan require groups 1, dilation 1 and zero padding mode, matching the
existing branch implementation. Unsupported targets fail before host mutation.
This API does not choose target names from class names, path substrings or a
training-method preset. Do not combine `target_plan` with legacy selection or
naming arguments; export names are already complete.

For existing integrations, `native_lora.lora_target_plan`, the historical
constants and constructor selector arguments remain available. Their discovery
policy lives in `native_lora_compat`, including the original attention/time/block
filters and checkpoint names. Legacy constructors retain that behavior when no
explicit plan is supplied. This compatibility adapter does not require diffusers
or any model package. The excluded historical `textsliders` source is unchanged.
Adopting a new target plan is a product source-cohort decision, not an automatic
upgrade of an existing checkpoint or exact training resume.

## Distribution

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
statistics are unchanged. The default search ranks by operand-first FP64 relative
SSE, pooled within each example and then averaged equally across examples. This
is an accepted proxy for relative effect. Measuring the intended effect in
matched finished outputs can improve selection, but is optional; it does not
invalidate completed proxy searches or require extra confirmation or training.
Each declared search retains one fixed objective. Product quality/media
diagnostics do not become additional ranking gates.

## Qualification

The distribution boundary is checked on built wheels and strict editable
installs outside a source checkout. Generic CPU lifecycle tests cover injected
response backends, updates, exact continuation, observation and exports. Product
tests separately establish their own native adapter ownership. CPU evidence
does not qualify GPU admission, pretrained/native media generation or quality.
