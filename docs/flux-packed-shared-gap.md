# Flux native packed response: blocked shared migration

The image family rooted at `conceptmod/textsliders/train_lora_flux.py` has an
isolated native prerequisite proof in `packages/particle-sliders-flux-packed`.
**It has not migrated to the shared game/lifecycle.** No training backend is
registered, no shared optimizer or checkpoint runs, and no native response axes
are reinterpreted to make the domain gate accept the input.

The historical trainer, `lora.py`, `train_util.py`, YAML, default environments,
checkpoint names and objective are unchanged. The trainer still calls
`prompt_pair.loss`; `train_util.predict_noise_flux` executes the native Flux
transformer and `scheduler.step`, then returns packed latents. Preserved trainer
SHA256: `5d906482a31b1180e65775e12b76405814792d86ab3562d7bd37618bbfda08e5`.

## Central API gap

Exact public shared source:
[`7a8a321d04323d49319d0b08e6093a0ddce36b27`, draft PR148](https://github.com/HyperGAN/particle-sliders/pull/148).
The lifecycle corrections in that revision do not change native admission.

- [`validate_pg09_domain`](https://github.com/HyperGAN/particle-sliders/blob/7a8a321d04323d49319d0b08e6093a0ddce36b27/packages/particle-sliders-core/src/particle_sliders/pg09.py)
  accepts 2D continuous vectors or genuine 4D/5D continuous fields.
- [`ResponseCapabilities`](https://github.com/HyperGAN/particle-sliders/blob/7a8a321d04323d49319d0b08e6093a0ddce36b27/packages/particle-sliders-core/src/particle_sliders/response.py)
  enforces those same geometries.
- Flux's native latent and prediction contract is `[1,image_tokens,64]`.
  It remains rank3 even for a one-token CPU fixture. Flattening it, removing its
  token axis or declaring token coordinates to be image spatial axes would
  change the declared response solely to obtain admission.

The upstream explicit-domain patch remains separate and unadmitted. A reviewed
packed continuous response domain and penalty treatment, compatible native
critic, descriptor and source-bound lifecycle signature are needed centrally.
This session reported that gap to the core owner; it changes no shared file.
`preflight_training` refuses before a model, row cache, hook or optimizer is
touched and before RNG is consumed. Width guards remain intact.

## Executable native prerequisites

The new distribution uses the unique `ps_flux_packed_preview` namespace and
pins the lifecycle-corrected public core, actual HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10 and Diffusers0.38. It
verifies all49 core Python/JSON/HNDL files and the verified builder closure,
plus its own graph and six relevant native forward files. It does not create an
old-core base/new-core extra conflict or modify a default requirements file.

Actual HyperGAN-built HNDL graphs implement the ordinary bias-free down/up
branch with rank4/alpha1 (the historical defaults). Each graph preserves
`[B,fixed_native_tokens,features]`, including the distinct combined text/image
sequence used by Flux single blocks. The fixed token lengths are graph identity;
they are not a reinterpretation of the game response. No routed bank or new
prior is invented for ordinary LoRA.

Current `FluxAttention` targets match the historical selector's linear-leaf
policy, excluding `add_*`. Diffusers0.38 renamed the class matched by the old
trainer's literal `Attention` string, so this new cohort does not claim old
checkpoint or target-name compatibility. The CPU fixtures instantiate the actual
native model and compare the same selected leaves through the unchanged
historical `LoRAModule`/`LoRANetwork` computations.

`PackedFluxResponse` passes native pooled/text embeddings, image and text IDs,
raw time normalization and guidance directly to the native transformer. It
observes **raw packed velocity**, not historical scheduler-step latents. Its
frozen-positive target convention is inherited from the shared response API and
is explicitly a different proposed objective/cohort, not an objective-preserving
dependency update. No media renderer or native training loader is supplied.

## Qualification and review gate

The installed package checks cover actual graph/module identity, native API
execution, matched-weight outputs and input/all-parameter VJPs, zero and signed
strengths, source/builder/native-shape rejection, mode/gradient preservation,
partial hook rollback and every-cleanup-on-failure. Wheel and sdist installation
are exercised in an isolated CPU environment; a separate CPU Torch2.6 environment
checks clean public dependency resolution. Exact receipts are returned with the
draft review packet.

Shared-game parameter updates, critic second derivatives, exact resume, EMA,
shared export and shared native-only inference are **blocked and unexecuted**.
These prerequisites cannot establish them. Random tiny Flux CPU fixtures also
do not qualify pretrained weights, GPU, full model/resolution, mixed precision,
gradient checkpointing, rendering or image quality.

This is a draft prerequisite/API-gap change, not a completion claim. Root owns
Claude actual-diff review, fixes, return review and any landing decision. No
package/weight release, deployment, merge or GPU allocation accompanies it.
