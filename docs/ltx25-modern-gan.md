# Opt-in LTX dummy video-field GAN preview

`ltx25-dummy-video-field-gan-hg-hndl-v2-preview` is an experimental **dummy CPU
fixture cohort** in an isolated environment. It is not a pretrained/native LTX
slider. Historical `ltx25_uni_embed`, the `ltx25_uni_velocity` ablation, their
implementations and checkpoint environments retain their original meanings.
The superseded v1 preview at `5e0608d72cf34ed715b82a173565a072071897ca` must use
its original environment; v2 does not relabel or restore those preview artifacts.

The historical working UNI teacher is post-connector video embeddings. This
preview separately exercises frozen-plus **video velocity** against an adapted
neutral-caption student through the real shared GAN lifecycle. Velocity UNI
previously failed to edit pretrained LTX. These finite CPU tests do not establish
that velocity is a useful concept teacher. The factory refuses all live execution
before model loading/hooks; no model weights, GPU execution or decoded media are
qualified or included.

## Bounded architecture and ownership

- The native fixture reuses the exact original tokenizer, encoder, connector,
  transformer, LoRA and stub classes. A product subclass changes initialization
  order only: capture a ledger **before** attaching LoRA, freeze every base root,
  set evaluation modes, then attach the original TE-last-four/video-connector
  adapters. The game arm is `adapter_gan`; there is no routed particle bank.
- The ledger validates exact host/leaf/parameter ownership and rejects unowned
  trainables, changed native registrations or training-mode base modules before
  forward execution. Cleanup restores every captured host by object identity,
  independent of the adapter list. It restores module registrations/modes,
  parameter gradient flags/prior gradients and the original network owner on
  failure or close. Partially attached factories and inference identity failures
  use the same rollback. Historical native source files are not edited.
- Teacher non-concept holds remain PRE-connector. Native padding/register masks
  and scaled timesteps are retained. Teacher/student share supplied latents and
  frozen neutral-caption audio conditioning. This does not establish invariant
  decoded audio through LTX's cross-modal transformer.
- Conditions are frozen valid-row means of the **neutral post-connector video
  features**, using the original `_masked_token_pool`, concatenated with the
  actual mean timestep divided by1000. No row index or positive-caption content
  enters the condition. This changes conditioning admission, not GAN mathematics.
- Fixture packing `[1,T*H*W,128]` is reversibly unpacked using its declared unit
  patches into `[1,128,T,H,W]`. Game and penalty see that entire field, including
  non-singleton time. This is not proof of real-model packing. There is no channel
  truncation, pooled response, temporal squeeze or wide-vector guard bypass.
- The verified HyperGAN develop builder executes the HNDL spatial critic per
  frame and averages frame scores. It has **no temporal-order sensitivity**.
  Internal frame batching belongs to the critic, not the shared penalty input.
  Graph, shape, temporal readout and builder identities bind provenance.

## Source/runtime admission and artifacts

The installed namespace is uniquely `particle_sliders_ltx25_dummy_preview`;
this distribution installs no `conceptmod` namespace. Build staging copies only
this binding and three original helper modules. Exactly two original absolute
imports in `ltx25_backend.py` become relative imports; replacement counts must
match exactly. Normalizing those import names reproduces the original helper
hashes. Computations and original repository helper bytes remain unchanged.

The generated package admission manifest checks all product Python source bytes
and the exact49-file core Python/JSON/HNDL closure from
`backends/ltx25-modern/core012-runtime.json` before native hooks. The admission
module's own digest is included in artifact runtime identity. Together with the
reproducible wheel hash, this identifies the complete built bundle; it is not an
external signature/security sandbox. Actual PG source and HG/HNDL builder gates
remain active. Changed artifact runtime identities are refused before hooks.

Backend, cohort, model identity, response kind and capability qualification all
explicitly identify a **dummy preview**. Generic `session.export_inference()` and
the shared default native export refuse; no default native `InferenceSpec` is
attached. The explicit `session.export_dummy_inference()` uses the shared artifact
container with dummy identities. Its separate inference registry advertises only
numerical `response`, not training or media rendering. Dummy-only inference can
run after training rows/teacher caches/critics/optimizers have been removed.

## Isolated package and builds

```bash
python3.12 -m venv /tmp/ltx25-dummy-preview
/tmp/ltx25-dummy-preview/bin/pip install ./backends/ltx25-modern
```

Distribution version0.2.0 pins core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`
from draft PR148, official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0 and `hypergan-hndl-v2`.
Do not combine the legacy HNDL0.8.2 extra or historical environments with this
preview. There is no live training CLI. Public fixture types/configuration are in
`particle_sliders_ltx25_dummy_preview.binding`; registry name is
`ltx25-dummy-video-gan-preview` for response and dummy-only inference.

Builds run setuptools in child processes with an isolated cwd; the parent never
changes cwd. Fixed build timestamps and normalized source archives make repeated
wheels, repeated sdists and sdist-rebuilt wheels byte-identical in the tested
setuptools84.2.1 build environment. Source archives
are self-contained and contain only safe relative paths. Every textual staging
replacement is count-checked before any artifact is emitted.

## Evidence and unresolved shared contracts

2026-10-31, Python3.12.15/Torch2.14.1+cpu:

- **31 installed-package checks** cover real HNDL modules; matched critic values,
  first/second derivatives; native output/parameter VJP equivalence; actual shared
  updates/exact resume/EMA; strengths0/0.5/1; frozen neutral conditions; all-base
  ownership; partial attachment/identity-failure cleanup; list-truncation cleanup;
  modes/gradient/owner restoration; exact runtime source rejection; dummy export
  gates and unique namespace isolation.
- **39 source checks**:37 unchanged historical UNI tests plus two packaging
  regressions, including concurrent builds with parent `os.chdir` forbidden,
  exact wheel/sdist/rebuilt-wheel bytes, source transformations and ambiguous
  replacement rejection. Installed tests use a separate venv with read-only
  qualified CPU dependency libraries; installation uses `--no-deps`. A fresh full
  dependency resolution/native environment remains unrun.
- Real-model packing/identity/gradients, pretrained/GPU/media quality and audio
  qualification are unrun. No `/dev/nvidia*` is available; occupancy is unknown,
  GPU0 is reserved. No model downloads/native training are part of this task.
- The working embed teacher is genuinely `[B,T,D]`. Shared response/domain
  admission currently refuses that geometry. It needs a reviewed token-response
  and mask contract: valid-token penalty semantics, masked critic reduction,
  explicit PG source/domain admission, exact shape/model identity and new
  checkpoint provenance. No pooling/recasting to pass admission is proposed.
- A routed bank and temporal-order critic remain unimplemented. They require
  reviewed native host/routing/bank/scale bindings and a real temporal architecture;
  this framewise adapter preview cannot qualify either.
- Root owns coordination of a shared pre-hook `admit(signature,runtime)` seam and
  identity-bound vector critic. Required admission must compare immutable source,
  graph, model, native axes/masks, conditioning and trainable ownership **before**
  native hook attachment, with cleanup ownership defined for failure/reentrancy.
  No speculative central API import or core-owner worktree edit is included here.

Existing broad PR140 remains untouched. This draft changes only an opt-in dummy
preview surface. No default migration, merge, package release, workflow change,
deployment, paid service or model-weight publication is included.
