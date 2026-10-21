# Seams for HyperGAN run rendering and HANDL network definitions

These responsibilities are owned by separate integration tasks. This lifecycle
slice does not implement a competing run renderer or a network-definition DSL.
The user's HANDL/HNDL requirement refers to neural network definitions; the
existing optional package integration is named `hndl` and pinned to 0.8.2.

## HyperGAN renders the run

`session.train(on_step=observer)` delivers a detached metrics dictionary after
each successful PG update and EMA update, with the actual `step`, public loss
terms, penalty diagnostics and D/G row coverage. `session.manifest` returns a
snapshot of backend/configuration/capability, source and recipe identity.
`checkpoint()` and `export()` return content-addressed packets and may save
atomic files; CLI manifests add artifact hashes. These are the inputs for a
HyperGAN run renderer/observer. A product or caller injects that observer; core
never imports a derivative to find one. Observer callback failures currently
close the session; buffering/renderer reliability policy belongs to the
renderer integration and must be reviewed explicitly.

The current `session.render(...)` means **native model inference** (image/edit
sampling through the backend callback). It is not a training-run dashboard or
visualization renderer. Existing CLI metrics/manifest files are receipts, not a
new UI. Do not implement a second renderer in a product to complete migration.
The separate task should specify the HyperGAN observer and artifact contracts
against these seams and return any rough spots to this interface task.

## Network definitions use HyperGAN/HANDL

`Binding.game` contains a `ConditionalPG09` constructed with a backend network
and a critic. The game consumes normal torch-module interfaces and actual
parameter ownership; it does not require a local network-definition language.
A HyperGAN/HANDL builder can supply the critic/network without changing session
or checkpoint APIs. Preserve FP32 trainables, raw critic logits, real particle
alias ownership and native response geometry. The existing
`HNDLConditionalReadoutCritic` already records its resolved graph, builder/source
identity and optional dependency version. Use that seam rather than creating a
parallel resolver. New definitions change source/architecture identity and
require a new cohort or explicit reviewed migration.

Current hand-authored definitions requiring migration review include:

| Definition | Current responsibility | Separate migration boundary |
|---|---|---|
| `field_critic.FieldCritic` | Existing spatial convolution/FiLM critic, extracted unchanged from the historical campaign module | Port the same definition/parameter schema to HyperGAN/HANDL; compare actual PG updates/regularizers and source-bound resume behavior. |
| `pg09_readout_critic.ConditionalReadoutCritic` | Existing conditional readout critic | Replace construction with a reviewed HyperGAN/HANDL definition, or select the existing source-bound HNDL arm explicitly. |
| `reference.RoutedMLP` | Existing routed maps/router, reused by Iris/Qwen/Fibo | Preserve routing, actual bank aliasing, parameter geometry and gradients; no formulation change merely for uniformity. |
| `routed_projection.TinyParticleProjection` and product `RoutedProjection`/`AdapterBank` | Existing low-rank native attention branches and hook ownership | Define eligible trainable modules through the agreed builder while retaining native attachment, signed checkpoint scales and zero bypass. |
| Historical reference/endpoint/formulation/BISPF and audio critic definitions | Named legacy arms or lower-level facilities | Inventory and migrate independently; do not silently redirect old objective/checkpoint semantics. |
| Tiny/mock native models in CPU fixtures | Test-only neural network definitions | They are not pretrained backbones or production definitions; decide/test any required builder conversion explicitly rather than treating fixtures as native qualification. |

Existing pretrained Iris, Qwen, Fibo, Krea, Anima, YuE2, MOSS, Sana and other
native backbones remain dependencies behind adapters. Rebuilding/replacing their
billions of pretrained parameters is not authorized by this interface milestone.
The builder integration must define the precise backbone boundary before any
such rewrite. Current raw-torch definitions are inventoried, not declared
compliant with the requested final HyperGAN/HANDL architecture.
