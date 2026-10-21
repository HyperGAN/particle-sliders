# Bounded follow-up to fd89e1c

Preserve fd89e1c and its receipts. Add an opt-in v2 inference packet and separate
registered inference factory. It receives only native adapter configuration and
local runtime options, builds a base/adapter binding without a game, and validates
source/model/architecture/protocol identity before loading tensors. Keep training
resume and old v1 loaders strict and separate; no automatic checkpoint rebinding.

Add fresh-process CPU tests through an installed fixture plugin, delete training
inputs, forbid training factories/game/critic/optimizer construction, and verify
raw/EMA labels, strengths, RNG, invalid-state atomicity and failure cleanup. Add
an inference CLI path that accepts no training configuration. No derivative
repositories are migrated by this follow-up.

Then create a separate branch from observed main 20fdf98, transplant an explicit
package dependency closure plus contracts/licenses/provenance/tests, and verify
its diff and installed wheel. Exclude campaigns/audio/text/video orchestration,
qualification receipts and unrelated historical commits. Preserve named existing
legacy APIs rather than redirecting their objectives. No publishing or merging.

## Consolidated derivative feedback boundary

Core 0.8 supplies native-only v2 loading, early inference-only session rejection,
backend-specific operation refusals, nested explicit CUDA RNG discovery,
allocation-free native request preflight, pure before-update admission,
observation contexts and opt-in factory cleanup transactions. The session gate
improves timing: PG already rejected updates from inference-loaded exports.
Factories still must adopt cleanup helpers and native request validation; no
blanket claim is made about unmodified derivative factories.

Sana is a shared-repository prototype, not a missing standalone repository.
Its adoption audit is in task-14/sana-audit at
779333e9e8dd1af48bd9d892949cd0e7a8677fb7. Strengths remain declared per backend;
observation contexts support online provider restoration. Archive-faithful
attention/native/HNDL and packaging qualification remain its task's work.

Music3 and YuE2 need a subsequent response-session modality/profile revision.
0.8 retains the image backend group/API and continuous PG profile. It does not
claim that dropping an AR/audio backend into image registration works. A clean
main candidate deliberately excludes historical pg09_audio_cli and unrelated
campaign modules. Music3 must retain its original pin for that module until a
reviewed explicit shared audio closure is transplanted and tested. Response and
waveform rendering must remain distinct; no waveform sampler is fabricated.

HyperGAN external reporting implementation is separately reviewed at local
66f696f2fc63000f01dd1d9f14d588af9d163361 (task-3). Existing on_step, detached
observation and manifest/artifact references are the extension seam; the parent
will assign the small bridge after this milestone. No dependency on internal
Execution or network-definition substitution is introduced here.
