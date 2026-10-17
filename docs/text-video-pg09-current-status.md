# Text/video integration status after the admitted Bonsai CPU canary

Follow-up: the admitted full comparison stopped before Torch/cache access at
zero updates because `/tmp` could not meet the disk reserve. Failure evidence is
frozen at `192696a`. The corrected NVMe-only proposal is
[bonsai-pg09-1200-proposal-v2.md](bonsai-pg09-1200-proposal-v2.md), with the same
1200-update protocol and separate final 20-update replay. It is pending a new
coordinator admission. The canary results below do not substitute for that run.

Code is frozen in `/mnt/nvme/sliders/text-video-adapters-pg09-20261017`, branch
`integration/text-video-pg09`. No shared core file was edited. The shared trainer
is immutable core `3e94b13eb5d6cd1ceee3f018f019a68603f35696`, module SHA256
`426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256`.
Explicit continuous arms use only the reviewed isolated ALZ overlay, patch SHA256
`c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571`.
Release/auto remains fail-closed at wide vector responses. No installed package
was patched, and no alternative penalty or training loop was introduced.

## Delivered code and evidence

| Commit | Deliverable |
|---|---|
| `56c91a8af7e92e8627c8d7d244e6eaa4d8c6c1d1` | Thin native TinyQwen and optional LTX video-velocity bridges; shared-core CPU/replay gates |
| `8c3194048cfe1b0f44313b97ddfdb4ca9695c84c` | Exact public repository/head inventory and initial qualification report |
| `fd3366fa20c6931efb2a51b30cb9f8ebff1f3334` | Metadata-only local asset preparation and original held-out/source pins |
| `1fd18c613084497fc7448255db7f4a006ef674f6` | Explicit public profile validation for native Tiny1024 and Bonsai5120; existing banks reused |
| `10124cfa56eddeb1d3254c0d43842990f8df1b6c` | Default LTX TE/video-connector GAN protocol, complete sequence coordinates and frozen teacher hold/masks |
| `44cf34c78067628442c1f29acff6ab33ca65ac72` | Bounded cached-real-readout Bonsai worker and synthetic full-flow tests |

The owner's common checkpoint fix is dependency `6e8ba6a8cf816cb060ac69c83d170f867a11a36f`
(canonical `331c3ff`); it was reused without another checkpoint wrapper.
The owner confirmed integration of the initial two commits as `e985db9` and
`b40d6e6`. Later commits are available for coordinator integration; their presence
in a live shared checkout is not asserted here.

## Repository inventory and coverage

This is the public organization API inventory captured on 2026-10-17, not an
assertion about unseen private repositories or later remote changes. Full API,
license and pinned README evidence: `evidence/pg09-text-video/repository-inventory.json`.

| HyperGAN repository | Exact default-branch head | Coverage |
|---|---|---|
| particle-sliders | `20fdf98d25455582e7e7191eef8ef51cf73173ae` | Embedded text/video owned here |
| anima-particle-sliders | `778ef38be9f1bec4e35cb02ade3a95e9c8dc8c14` | Image owner; existing bridge |
| supra-particle-sliders | `a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6` | Image owner; existing bridge |
| krea2-particle-sliders | `0b489f63388b000f18242c5d401d8b7709c27dee` | Image owner |
| ming-particle-sliders | `93cb5bfa9e6585da3dc0a89327fb56dc276e8cc5` | Image owner; scaffold limitations handed off |
| music3-particle-sliders | `b40d2aa6fed745eda3043d3d3b11e362ec2f8dfe` | Audio owner |
| yue2-particle-sliders | `b33794d81b92c125891a011531f55554d65e2082` | Audio owner |
| iris-particle-sliders | `6a226126623fb17c87fe22b86173fe2fcff4bf3e` | Image owner |
| qwen-image-particle-sliders | `b418ff5b709dd687fdc7d9fbb5498c592ec21a58` | Image owner |

No standalone TinyQwen, Bonsai, LTX or H3 public repository was found: all four
are actual embedded backends in the shared repository. No further public
text/video/other family was found. Initially unconfirmed image families were
reported; the parent subsequently assigned image coverage elsewhere. The existing
TinyQwen port `7039652` was inspected before adding a bridge to the canonical
trainer; its separate legacy trainer was neither copied nor invoked.

## Compatibility and remaining gates

| Backend | Response and status | Remaining action / status |
|---|---|---|
| TinyQwen | Actual native slider/bank and last-token1024 coordinates; CPU profile/replay gates pass | Pretrained loader/zero-scale/runtime fit and quality remain pending; no model/GPU admission |
| Bonsai | Native Torch head with its actual128x4 bank on frozen5120 GGUF readouts; admitted real-cache canary passes both presets | Held-out transfer fails after four updates; no convergence or generation claim, no further training admission |
| LTX default | Full native token-major/feature-minor sequence, fixed common-mask support, teacher-only preconnector hold, live neutral TE/video connectors; independent software QA passes | Reuse staged text-only loader pinned `c7096d475f70daee41fc161150cf8eeba1dfffc5`; pretrained shape/mask/runtime/memory/AV rendering and owner GPU admission remain pending |
| LTX optional velocity | Genuine5D native video coordinates, fixed audio context; CPU software gates pass | Optional arm does not cover default connector or full AV render; actual model/render admission remains pending |
| MiniMax H3 | Actual native jointly packed multimodal architecture inspected | Blocked on honest native AV geometry/critic protocol, staged source/asset/license and hardware-fit review; no invented vector/image response |

Tiny model pin: `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`,
Apache-2.0, already staged. Bonsai pin:
`prism-ml/Ternary-Bonsai-2-27B-gguf@b072e1d3b35a0a630cece372c2127528e0994386`,
Apache-2.0, existing PTQ1_0 file plus exact Prism runtime/pooling identity.
LTX pin: `Lightricks/LTX-2.5-Diffusers@da0b690960026eab794b8ceaf841233cf7064478`,
already staged; initial discovery HTTP401 was superseded by verified local
metadata. Its community terms are recorded for private R&D, without any new
agreement or commercial applicability accepted here. H3 pin:
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, custom
community license and large native multimodal hardware gate. Repository licenses
do not replace model licenses. Local asset manifests and fresh small metadata
hashes are in `assets-prepared.json`; large weights were statted against prior
attestations, never downloaded or rehashed for discovery.

## CPU and independent QA evidence

Final frozen-code regression: **52 passed, zero skips/failures**, comprising
21 explicit-overlay cases in6.62s and31 release/asset cases in3.86s. The two newest
cases use freshly generated synthetic cache tensors, not official cache data.
Receipts: `final-overlay-tests.xml` and `final-release-tests.xml`.

Six prior independent-process2+2 continuations cover Tiny/Bonsai/LTX connector
and both presets; parameters, optimizers, public regularizer, prior, sampling/RNG,
losses and seeded samples agree exactly. Two actual cross-profile processes reject
release/ALZ transfers transactionally with live state/global RNG unchanged.
Receipt: `connector-fresh-process.json`.

Independent QA replicated all9 LTX native tests (actual random Gemma4Unified and
LTX2 connectors, both presets) and added2 exact masked-coordinate Jacobian and
teacher/live-student role tests. No mask/hold source blocker was found. Receipts
are under `/ml2/codex-storage/future-workspaces/2026-10-17/task-3/qa_pg09/`:
`ltx-connector-10124cf-native-review.xml` and
`ltx-connector-independent-review.xml`. QA also completed static review of the
Bonsai worker with no source blocker before the core owner admitted execution.

## Admitted real cached Bonsai canary

Core-owner admission: `bonsai-cached-cpu-20261017-v1`, exactly the four sequential
commands in `bonsai-cpu-admission-proposal.json`, frozen worker44cf34c. No retries.
Worker SHA256: `9a3b987cba3d1b999ff8c9a4738be7d537630d01d1f53d1eefea46adacd79c36`;
critic SHA256: `b83322a2ef2192ac3693fe136d019259c7fd8bf26fba25bb254931ba25462a57`.
All four commands exited0 in4.551s,2.559s,2.160s,2.133s (120s budget each).
Both presets completed four updates, checkpointing at two, followed by an
independent-process replay of the last two. Twelve total updates include replay.

The15 official readouts bind exact identity
`358f97c219524fd96f9d510824f4bc7f874ad4e44927359e0911a3cb95bde5c7`.
Both presets changed G, D and the actual routed bank, retained frozen responses,
and produced positive continuous penalties. Full state, losses, held-out vectors
and safe inference-export reload matched exactly across process continuation.
Scale0 remained exact. No GGUF/model weights, inference server, CUDA context or
network download was used. Receipts and command logs are committed under
`evidence/pg09-text-video/bonsai-cached-cpu/`; tensor checkpoints remain under
`/tmp/pg09-bonsai-real-cache-20261017/` with hashes recorded in the evidence index.

**Held-out transfer did not pass.** At strength1 the three relative-edit MSEs
were194.50/127.54/120.13 for `bispf`, and2.54/2.06/2.11 for `bispf_optimized`,
against the initial no-edit value1. Canary relative RMS was12.39/1.09 respectively.
These are diagnostics only and never optimized. Four steps demonstrate real
cached-head execution and exact continuation, not useful transfer or convergence.
No hyperparameter sweep, extension, retry or quality claim follows from admission.

No service stop, permission change, legacy MSE restart, publication or merge
occurred. Core owner remains the sole GPU/process admission owner. Further actual
model or training execution is pending a new concrete admission; none is implicit.
