# Shared core API map

Core 0.14.2 is model-independent. Products depend on core; core does not import
products to discover or load models. For end-user usage, see the
[product repositories](products.md).

| Boundary | Product supplies | Core owns |
|---|---|---|
| Response generation | Frozen base, target/neutral capture, native coordinates, conditions, differentiable student callback | Conditional game, optimizers, private sampling/RNG and exact state |
| Adapter attachment | Explicit target plan and stable export names | Generic LoRA primitives, attachment ownership and strength controls |
| Session | Validated options and complete `Binding` with cleanup | Train/evaluate/resume/export lifecycle and observation transactions |
| Inference | Native-only base/adapter factory and verified identity | Artifact validation and operation/strength preflight |
| Search | Frozen grids, supported effective parameters, original response coordinates | Candidate scheduling, relative-error arithmetic and evidence audit |
| Execution | Product profile and isolated runtime | Planning, runner dispatch, owned workers, progress and results |

Start with the [distribution boundary](../packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md).
Use [response sessions](../packages/particle-sliders-core/RESPONSE-SESSIONS.md) for
numerical responses and [image sessions](../packages/particle-sliders-core/IMAGE-LIFECYCLE.md)
when the product offers image operations. Both use the same
[conditional engine](../packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md).

A product factory owns model-specific construction and cleanup. A separate
[native inference factory](../packages/particle-sliders-core/NATIVE-INFERENCE.md)
builds only the resources needed to apply an exported adapter. Generic core code
must not acquire product prompts, fixed model directories, class-name target
selection, native schedulers, media decoders or campaign policy.

Full training checkpoints and adapter exports serve different purposes. Resume
requires the full original training state and source identity; an export applies
learned weights for inference. See the session guides for these lifecycle rules.
Historical embedded products remain [reproduction material](history.md), outside
the installed core.
