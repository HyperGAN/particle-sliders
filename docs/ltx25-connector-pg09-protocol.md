# Proposed LTX default connector GAN response protocol

This port targets the default TE/video-connector adapter placement, not the
optional DiT velocity arm. It reuses native `LTX25Backend.encode_text`,
`encode_post_connector_video`, adapter leaves, tokenizer, and the exact native
preconnector hold functions. Only the shared `ConditionalPG09` trains it.
The core owner approved engineering of honest continuous sequence coordinates;
independent mask/hold review and model/GPU admission remain pending.

## Coordinates and masks

The caller declares the actual native `[1,tokens,features]` shape before any
response forward. The shared public validator must admit the exact ALZ profile
and explicit continuous penalty. Response order is contiguous token-major,
feature-minor `[1,tokens*features]`, retaining every coordinate without pooling,
truncation, feature projection or fabricated image axes.

For each immutable row, cache frozen positive and neutral output masks. Both
must be finite binary `[1,tokens]`. The fixed response mask is their intersection,
matching native same-length valid support. Multiply both target and generated
sequences by this frozen mask before vectorization. Inactive coordinates remain
present as zero, not removed or reindexed. Native masks, common mask, complete
shape, token IDs, hold flags, captions and source files are checkpoint-bound.
Changed student masks/tokenization, changed shapes, or empty intersections fail;
there is no native legacy mean-pooling/all-valid fallback.

This mask is part of the declared adversarial response coordinates, not an
optimized reconstruction objective. No per-row scalar error or MSE is optimized.
The frozen Gaussian-reference residual and its public critic penalty are still
owned solely by the shared game. Conditions retain complete neutral response
coordinates, common mask and explicit scale/time values, not row IDs.

## Frozen target, live student and hold

Frozen teacher: encode the positive and neutral captions at adapter strength0;
resolve concept IDs with the native helper; hold every nonconcept positive-token
feature to its matching frozen neutral feature **before padding/connectors**;
then run frozen video connectors. Concept IDs must exist. This is the same
native teacher construction as the old default embedding case.

Student: re-encode the neutral caption live through the adapted TE, then adapted
video connectors at strength+1. Do not cache adapted student embeddings. There
is no optimized hold loss, MSE, relative-L2, velocity, reconstruction or distillation
term. Hold defines frozen teacher conditioning only. The generator arm is
**adapter GAN**, `particles=None`; no disconnected bank is invented.

The adapter exposes the actual encoder/connectors through core's existing
`model` provenance walker so native Transformers/PEFT/Diffusers source files,
module configuration and architecture are bound. `backend.network` and the
optimizer/export still own only the actual adapter leaves. The existing repeatable
checkpoint helper controls recomputation scales. Native scale restoration now
also preserves exact PEFT dictionaries on encoder/connector roots.

## Software evidence

Immutable core: `3e94b13eb5d6cd1ceee3f018f019a68603f35696`.
Overlay: existing `/tmp/alz-pg09-domain-proposal`; no installed package changed.

**50 CPU tests pass**, no skips/failures: 19 explicit-profile tests (10 Tiny/Bonsai
and9 LTX connector) plus31 release/asset regressions. LTX cases prove native
preconnector hold inputs, full coordinate/order preservation, frozen masks,
no mask/shape fallback, both-preset GAN-only equations, positive continuous
penalties, adapter gradients/updates, frozen bases, safe checkpoint resume,
inference export/reload and exact zero.

The LTX tests include actual randomly initialized `Gemma4UnifiedForConditionalGeneration`
and `LTX2TextConnectors`, not just stand-ins, with two native text layers and
native video/audio connector branches. Both PG presets complete a full shared
GAN update with source/configuration binding and connected TE/video adapter
gradients. Those fixtures return native `[1,1024,32]`, vectorized to32768.
The lighter native backend fixture returns `[1,1026,8]`, vectorized to8208.
No pretrained component or staged cached tensor payload was loaded.

Six independent-process2+2 continuations pass: Tiny, Bonsai and LTX connector ×
both presets. Full parameters, optimizers, regularizer, bank where present,
RNG/coverage state, losses and seeded samples match exactly. Actual Tiny64
release↔ALZ transfers reject in separate processes with live state/global RNG
unchanged. Receipts:

- `evidence/pg09-text-video/connector-and-wide-tests.xml`
- `evidence/pg09-text-video/connector-release-regression.xml`
- `evidence/pg09-text-video/connector-fresh-process.json`

These establish CPU software behavior, not pretrained fit, convergence or semantic
quality. Tests forbid calls to MSE and legacy embedding/hold objectives.

## Remaining admission work

The native interface is shared with the separately qualified staged source
`c7096d475f70daee41fc161150cf8eeba1dfffc5`. Its text-only local loader should be
reused/pinned for a production text/connector stage; no old MSE worker or queue
will be executed. The current adapter source itself adds no component loader.
Exact staged source/runtime compatibility, full native response shape, fixed-mask
support and real adapter memory must be reviewed before full-model allocation.

The existing LTX model manifest records private R&D community terms. No agreement
or commercial/distribution applicability was newly accepted. Audio/video denoise
and actual decoder parity remain separate owner-admitted checks: text adapters
may affect audio conditioning through the shared TE even though this response is
video-only, matching the historical placement. No audio preservation or full AV
render quality is claimed.

Keep original held-out woman/man/fruit cases, native49frames544×960, seeds123/124/125,
the exact distilled schedule/CFG/STG/modality settings and Off/frozen-positive/
initial+1/strength controls from the staged recipe. Diagnostic MSE may be reported
without entering any optimized objective. Full clips and human review remain
required before a quality claim. GPU/process admission remains with the core owner;
no GPU request, model load, server launch, service or permission change occurred.
