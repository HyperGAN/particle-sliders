# Embedded video package audit

The video implementation is commit `397c69c2f1cdaf2b06098bc2649e04d6e4673b2e`,
stacked on family source `ed3e4a9a724aaaf9acc657be63f6d4556ebc45be`. It requires
replacement shared package candidate `4be0b5b29d2e2c741431f468d71ba8a7c884b724` and released
ParticleGAN **0.9.1**. Neither candidate is implicitly Ada-approved or merged.
All active family/core pins remain untouched. The earlier `42a2e371` candidate
is superseded because its optimizer-state restore could accept corrupt history.
Its original receipts remain preserved; the current dependency is locked in
`evidence/video-audit/current-dependency-lock.json`.

LTX's historical public trainer now dispatches into the installed
`particle_sliders.ltx25_cli`. The existing connector protocol, native host and
hold/mask helpers moved into the package; legacy imports are the same objects.
The shared `ConditionalPG09` owns every optimizer/loss/sampling/RNG update.
Complete neutral response and mask coordinates condition the critic. Training
is unipolar +1 with adapter leaves only, `particles=None`; no particle bank is
invented. Full checkpoints resume training; source/protocol-bound adapter exports
load inference response probes. Both old MSE/hold optimizer loops and native
`--load_ltx_lora` public inference route are closed.

Native adapter fixes retain FP32 leaves on BF16 hosts, preserve per-host devices
on fixture reload, preserve fractional PEFT alpha, and propagate attachment
failures instead of falling back after partial native mutation. Tensor frame
selection avoids ambiguous truth evaluation. Public production constructors,
trainers, native reload/export and AV rendering remain closed pending genuine
runtime qualification; explicit CPU component injection supports random native
class regressions. No placeholder MP4 is produced.

H3 has no honest shared-GAN native AV geometry/critic contract yet. Its public
trainer's MSE optimizer and fake MP4 fallback were removed. Native construction,
training, steps-zero reload, export, generation and live diagnostics reject before
loading or mutation. CPU architecture/diagnostic fixtures remain. This is an
honest closure, **not an H3 GAN runtime migration or qualification**.

## Exact installation and tests

The replacement producer wheel SHA256 is
`2e7621c370e790f65702acf76cfa2f428b393cd3f863245aa26d8a76ff096ada`.
It does not include video. The review composition combines that exact frozen
core Git head with the unchanged four video modules from `397c69c`. Result:
`particle_sliders_core-0.6.0-py3-none-any.whl`, SHA256
`f56eb2da52809ecbcd5ade6f6dca27c615f917d70e8f4db54d75aff6e905c209`.
Every shared Python source matches the producer wheel byte for byte; installed
core and video module fingerprints match the composed wheel. This is a local
review artifact, not a published package release or an Ada-approved merge.

The replacement rejects negative second moments, invalid optimizer clocks,
incomplete Adam fields and erased histories before live restore. Its history
schema distinguishes legitimately unused critic parameters from erased state.
Eight video CLI regressions check full receiver state, parameter identity,
ambient RNG, critic thermal cache and absent output after rejected resume.
No video implementation or shared loss mathematics changed during adoption.

The official PG 0.9.1 wheel SHA256 is
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
Its wide-vector automatic-domain limitation remains. LTX connector software tests
explicitly use the exact task14 `alz-continuous-v1`/`continuous` profile, copied
only into isolated test processes. No global/active environment is patched.
Release/auto rejects wide connector responses before backend construction.

- **139 CPU tests and 12 subtests pass**, no skips or failures, 16.22 seconds.
  All nine native adapter review cases re-pass within this installed-wheel suite.
- **8 independent restore regressions pass**, no skips, 4.29 seconds.
- Source-free installed-wheel tests run with isolated Python from an empty
  temporary directory. All video imports originate from site-packages, with no
  `conceptmod` import. Both presets match four uninterrupted updates to separate
  2+2 process continuation, including full state/optimizer/sampler/RNG. Inference
  probes match export reload exactly. Source tampering fails before output.
- A real synthetic checkpoint generated with the superseded core is rejected
  by the replacement before output, with its source file unchanged. No existing
  checkpoint is silently rebound to the new source lineage.
- CPU-only H3 closure tests run with `python -S` and copied source files, without
  Torch or the repository, and reject blocked routes before runtime imports.

Receipts, source hashes and exact composition are under
[`evidence/video-audit/replacement-4be0b5b`](../evidence/video-audit/replacement-4be0b5b/).
The original receipts at `evidence/video-audit/` describe the superseded wheel.
The replacement wheel remains at
`/ml2/codex-storage/future-workspaces/2026-10-17/task-22/video-candidate-replacement-4be0b5b/`.
Reproduction helpers are `scripts/build_video_candidate_wheel.py` and
`scripts/verify_ltx25_sourcefree.py`.

## Review and runtime gates

Task14 shared-package source must integrate first; then this narrow video delta
can be cherry-picked into a new candidate lineage and retested. Existing frozen
checkpoints cannot silently retarget relocated sources. Parent must route exact
heads through Ada before any reviewed integration merge; this session does not
merge independently.

LTX still needs the actual staged text-only loader, native full response
shape/mask/source validation, measured memory/runtime, license applicability,
AV denoise/decode parity, preservation and useful transfer. H3 additionally
needs genuine packed AV geometry and a reviewed critic protocol, verified source,
assets/variant identity and hardware fit. These actions are **pending**, not
executed or denied. Coordinator `01a147d7-f540-778e-ad79-4d8eb267403d` alone owns
runtime admission. No model/GPU job, service change, credential change, external
model publication or merge occurred.
