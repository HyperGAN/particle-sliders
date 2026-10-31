# Opt-in LTX video-field GAN preview

`ltx25-video-field-gan-hg-hndl-v1-preview` is a **separate experimental
objective and environment**. The historical `ltx25_uni_embed` default,
`ltx25_uni_velocity` ablation, saved adapters and validated UNI cards retain
their original meaning and installation environment. Neither historical
checkpoint is accepted as a modern training/inference packet.

The working UNI teacher is post-connector video embeddings. The new preview
uses frozen plus-caption **video flow velocity**, with a neutral-caption
student, through the shared `ConditionalPG094` game and response lifecycle.
Velocity UNI previously failed to edit pretrained LTX; a different GAN loss
does **not** establish that velocity is an effective concept teacher. This
preview has **no pretrained, GPU, decoded-video or quality qualification**.
It refuses live execution before loading a model or attaching native hooks.
There is no new model download, native training campaign or weight publication.

## Native binding

- Reuses the existing LTX dummy/native implementation. TE last-four attention
  projections and video connectors own the only adapter leaves. The transformer,
  base TE/connector weights, audio connectors, VAEs and vocoder stay frozen.
  The arm is explicitly `adapter_gan`, without a routed particle bank.
- Teacher non-concept holds run on token-aligned TE features **before** padding
  and connectors. Existing attention masks, register insertion and native
  scaled timestep `725` are preserved. The student caption is neutral.
- Both branches use identical supplied video/audio latents. Teacher and student
  use frozen neutral-caption audio conditioning and masks. This does not claim
  invariant decoded audio from LTX's cross-modal frozen transformer.
- Unit-patch fixture packing `[1,T*H*W,128]` is reversibly unpacked to its native
  `[1,128,T,H,W]` video latent field. The shared game and penalty see that entire
  field. No channel truncation, pooled-vector response, temporal squeeze or
  wide-vector admission bypass occurs. Non-singleton temporal fixtures pass.
- The critic calls the real verified `hypergan.hndl_networks.build_network`.
  Its declared architecture is a **framewise spatial critic, averaging frame
  scores**. The HNDL graph executes Conv2d/SiLU/Conv2d, conditional FiLM and a
  scalar readout per frame. It has no temporal convolution or temporal-order
  sensitivity. Internal frame batching is only a critic implementation detail;
  the shared penalty differentiates the complete native 5D response.
- Product graph text, shape, temporal readout and verified builder identities
  appear in inspectable class source, `extra_repr()` and checkpoint provenance.
  Base fixture seed, frozen-weight digest and all three exact native helper
  source hashes bind native inference identity and reject changed helpers before
  hooks.
  HyperGAN/HNDL source verification runs before native hook attachment.

## Isolated installation

This distribution is opt-in; it does not change the umbrella's default core
or historical locks. Do not install it over a historical UNI environment.
It contains only the new binding and its three unchanged native helper modules.
It registers `ltx25-modern-video-gan` in both shared response and inference
backend registries.

```bash
python3.12 -m venv /tmp/ltx25-modern
/tmp/ltx25-modern/bin/pip install ./backends/ltx25-modern
```

Exact shared source is `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`
(published draft PR148; reviewed adoption/merge is still pending). Core0.12
uses official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0 and
`hypergan-hndl-v2`. Its qualification wheel SHA256 is
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.
The dependency is an immutable reachable VCS source, not an unreleased version
range. Do not combine the HNDL0.8.2 legacy extra with this environment.

The registry is usable through `particle_sliders.response.open_response` and
the explicit `config()` / `VideoRow` types in
`conceptmod.textsliders.ltx25_gan`. Callers supply two or more finite frozen
CPU fixture latents with exact declared geometry. There is intentionally no
live training CLI. Session train/resume/EMA/evaluation/export are the shared
APIs, and native-only artifacts use shared `open_inference(operation='response')`
with a new neutral caption and video/audio latents, without any training rows,
teacher cache, critic or optimizer construction. Native-only output is a
numerical video velocity field, not a generated video.

## Evidence and remaining admission gates

2026-10-31, Python3.12.15 / Torch2.14.1+cpu:

- **18 passed**, installed preview wheel outside the source checkout:
  `tests/test_ltx25_modern_gan.py`. Real HNDL module types, matched reference
  outputs, input/condition/parameter VJPs and input-gradient-penalty second
  derivatives; native adapter output/parameter VJP equivalence; actual shared
  updates; uninterrupted/split exact state and EMA; strength0/0.5/1; inference
  after training-factory removal; hook rollback; changed model/cohort/shape/
  profile/factory/builder refusal; native 3D/wide-vector refusal; wheel metadata
  and installed entry points.
- **1 passed**, source-archive rebuild regression in
  `tests/test_ltx25_modern_packaging.py`: safe relative archive paths, actual
  archive-to-wheel build and byte-identical selected native helpers.
- **37 passed**, original `tests/test_ltx25_slider.py` in the source checkout.
  This is historical regression evidence, distinct from installed preview tests.
- Fixture environment uses an isolated venv and read-only existing CPU dependency
  libraries. Wheel installation used `--no-deps` against that qualified core
  environment. A fresh full dependency resolver/native environment installation
  is **unrun**; the wheel metadata is not proof of that installation.
- Pretrained/GPU/decoded video and audio checks, native LTX unit-patch packing
  verification, staged model identity and actual-model gradient ownership are
  **unrun and blocked**. This executor has no `/dev/nvidia*`; occupancy is unknown,
  GPU0 is reserved, and model downloads/training are outside this task. The live
  guard is deliberate; remove it only in a separately reviewed qualified cohort.
- The shared core does not admit genuine `[B,T,D]` connector responses. Adopting
  the working embedding teacher as a GAN requires a reviewed native token-field
  contract with masks, penalty semantics, source identity and checkpoint domain
  admission. Recasting it as image axes or flattening/pooling to fit the guard is
  not an acceptable replacement. No shared API files were edited here.
- The framewise video critic is product-owned preview architecture; canonical
  shared naming and temporal critic qualification remain separate review work.

Review this focused draft independently from existing PR140, whose source-bound
video migration remains open at `0650d06c8d5fdeba314dc180041fec3d27fd86aa`.
No default objective migration, merge, package release or workflow change is
included. Installing a preview wheel is the only new release surface.
