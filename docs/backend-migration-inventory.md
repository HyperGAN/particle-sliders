# Backend dependency contract and migration inventory

The architecture applies to every model family:

```
(anima, krea, yue2, moss, sana, ...)-particle-sliders
                          |
                          v
                  particle-sliders-core
                          |
                          v
                 ParticleGAN / HyperGAN
```

Product packages own native model library dependencies, exact model staging,
conditioning, packing, schedulers/decoders, native qualification and product
configuration. They register/inject thin factories and native callbacks into
particle-sliders; core does not import products to construct their pipelines.
Particle-sliders owns shared configuration/capability contracts, loading
orchestration, training-step lifecycle, EMA, checkpoints, export/evaluation/
inference dispatch and errors. GAN objectives, optimizers, penalties and prior
laws come from upstream ParticleGAN; optional HyperGAN observations remain
observations. New products must not copy GAN implementations.

Common protocol does not imply one native dependency environment. In particular,
Qwen and Fibo require separate Diffusers Git implementations. A new backend
registration must preflight its own capabilities/dependencies, return a complete
owned binding and unwind setup failures. Caller-supplied native objects and
entry-point plugins are dependency injection, not core-to-product dependencies.

## Verified derivative repositories for parent-owned integration tasks

The paginated HyperGAN repository API returned these ten visible derivatives
on 2026-10-21. Default heads are inventory observations, not approved migration
bases. The [machine-readable inventory](derivative-inventory.json) retains exact
heads and manifest availability. Four existing PR heads are listed in the
handoff; preserve them instead of discarding their work by starting from main.

| Repository | Observed default `main` head |
|---|---|
| [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders) | `8bcf7f2cd3d08933556f79b87cb1519e7cecbe8d` |
| [supra-particle-sliders](https://github.com/HyperGAN/supra-particle-sliders) | `a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6` |
| [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders) | `0b489f63388b000f18242c5d401d8b7709c27dee` |
| [ming-particle-sliders](https://github.com/HyperGAN/ming-particle-sliders) | `93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5` |
| [music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders) | `b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe` |
| [yue2-particle-sliders](https://github.com/HyperGAN/yue2-particle-sliders) | `b33794d81b92c125891a011531f55554d65e2082` |
| [iris-particle-sliders](https://github.com/HyperGAN/iris-particle-sliders) | `6a226126623fb17c87fe22b86173fe2fcff4bf3e` |
| [qwen-image-particle-sliders](https://github.com/HyperGAN/qwen-image-particle-sliders) | `b418ff5b709dd687fdc7d9fbb5498c592ec21a58` |
| [moss-image-sliders](https://github.com/HyperGAN/moss-image-sliders) | `477278a2e87f77d8a56a38bd6b952741231f1b3e` |
| [Fibo-particle-slider](https://github.com/HyperGAN/Fibo-particle-slider) | `09758f187b869c14027014e85752c9e76dd2950a` |

No Sana derivative repository matched this verified visible inventory. Sana is
covered in the interface design and source-binding inventory, but its canonical
product target still needs resolution. Other model bindings in core do not
prove a separately migrated repository exists. Ming is a verified derivative
repository with no named packaged session adapter identified in this inventory;
its fresh task must inspect its actual entrypoints/formulation before choosing
a bridge. No tasks or new repositories were created here.

## Source inventory at the reviewed cdf/f797 lineage plus this local slice

The inventory below was made by inspecting the packaged response/builder source.
It is not a fresh qualification or audit of every derivative repository.

| Family | Existing shared facility | Migration status and next boundary |
|---|---|---|
| Iris | Native pixel-velocity adapter, real routed bank | Session factory/CLI implemented and CPU tested in this slice. Rendering remains disabled pending native FlowDPMSolver parity. |
| Qwen Image 2.1 | Native spatial VAE response adapter and layout helpers | Session factory/CLI implemented. Reverse product helper imports removed; products retain aliases to core helpers. Native dependencies stay in its product environment. |
| Krea2 | Native image adapter, PEFT leaves and edit/preservation rows | Session factory/CLI implemented. Preserve adapter GAN/particles=None; signed finite scales. Tiny fixture is response-only. |
| Fibo Edit Turbo | EDIT response adapter, verified staging, cached references/masks, real routed bank | Complete session factory/CLI and installed synthetic example implemented. Native Turbo training remains experimental; CPU mocks do not establish pretrained quality. |
| Anima | `AnimaResponseAdapter`, `AnimaRuntimeResponseAdapter`, staged runtime/bank binding | Adapter only in this inventory; no common session registration added. Product must supply factory/loading/cleanup, audit its actual update loop, and adopt a new explicit cohort. Keep real particle ownership and native trajectory rounding. |
| YuE2 | `YuE2ARResponseAdapter`, audio conditioning/critic and width gate | AR hidden-state response only; not a token sampler or waveform renderer. No session factory migrated. Preserve teacher/history policy and fail closed for unqualified wide continuous vectors. |
| MOSS | Native response adapter; lazy campaign provider/run builder and signed cached fields | Existing campaign path, no new session registration. Ongoing GPU experiments and source/result receipts stay on their original pins. Current campaign uses particles=None; do not invent a routed bank. One legacy reverse import remains explicitly inventoried below. |
| Sana | `ImageResponseAdapter('sana', ...)` with native timestep scaling | Response binding, no common product factory/loader/render entrypoint migrated here. Native sampler/decoder parity and product orchestration remain to audit. |
| SD/SDXL, SD3, Flux, Z-Image, Cascade | Named image conventions and native packing/scheduler-conditioned responses | Bindings only in this inventory. Preserve discrete input scaling, epsilon/v/flow distinctions, inverse time and non-RGB fields. Do not force a generic Euler renderer or claim complete lifecycle adoption. |
| Supra/final-form | Runtime response binding, cached campaign rows/run and parity gate | Existing staged/campaign interfaces, no new session registration. Keep native conditioning, mixed-precision trajectory rounding and explicit source/domain gates. |
| Music3 | Native Qwen3 AR host, last-hidden adapter and audio domain gate | Existing response/native facilities, no session factory migrated. AR response evaluation does not qualify audio generation. |
| Tiny Qwen/Bonsai | Native routed architecture/readout adapters and strict vector gates | Existing lower-level facilities, no common session registration added. Wide native vectors remain blocked unless the exact separately reviewed source profile is selected. |
| LTX 2.5 video | Native video velocity response adapter | Response binding only; no session factory, native video sampler/decoder or quality qualification migrated here. |
| Other/future families | Register/inject an adapter under the same dependency direction | Inventory native state/geometry, capabilities, lifecycle ownership and source compatibility before claiming support. |

## Enforced boundary and legacy exception

`tests/test_dependency_direction.py` scans installed core sources and rejects
new imports of known derivative packages and new `*_sliders`/`*_particle_slider`
packages. It also rejects product dependencies in core distribution metadata.
The new lifecycle/image/Fibo modules have no reverse product imports. The guard
has one exact grandfathered import:

```
pg09_moss_campaign.py: from moss_sliders.lora import Slider
```

This is an existing compatibility gap, not compliant new architecture and not a
claim that MOSS is migrated. Its source location, module and imported symbol
are fixed in the test allowlist; adding another reverse edge fails. Removing
this edge requires the MOSS product to inject a slider factory and own native
construction. Review/test that change as a separate source cohort, without
rebinding an active experiment. Do not hide the import behind dynamic lookup,
copy the MOSS implementation into core, or widen the allowlist for new work.

The current `image-session-v1` envelope is the first bounded implementation.
Non-image factories are not silently registered, and image reference/render
arguments must not be used to pretend that audio/video/text native contracts
are identical. Extend/version modality and capability contracts as needed,
then run each product's conformance and native qualification separately.

## Compatibility and migration order

1. Keep current cdf/f797 and research pins, environments, checkpoint schemas,
   results and receipts available. Neither a source import nor a dependency
   upgrade migrates a running experiment.
2. Audit each product's real entrypoints and declare its generator arm, native
   response/output modalities, references, strength/precision, sampler and
   qualification status. Extract repeated lifecycle logic into particle-sliders,
   with native callbacks/factories pointing from product into core.
3. Register only a complete factory. CPU-test exact resume/RNG, atomic invalid
   restore, EMA/export/load, supported strengths, failure cleanup and rendering
   gates using installed wheels outside source trees.
4. New source pins mean new cohorts unless an explicit migration is reviewed.
   Native model/GPU resource and quality qualification remain separate gates.
5. Use the clean main integration proposal in `image-lifecycle-integration.md`;
   do not bring the unrelated MOSS/audio/campaign stack into main through PR141,
   and do not retarget PR142. Review exact package diffs and derivative heads.

### Subsequent verified correction

Sana's archive-faithful prototype is in the shared repository, as confirmed by
task-14 audit 779333e9e8dd1af48bd9d892949cd0e7a8677fb7. It is not an eleventh
standalone derivative repository. Its online/provider/native/HNDL adoption is
pending. The ten repository default-head inventory above remains an observation
of defaults, not the selected unmerged integration heads.
