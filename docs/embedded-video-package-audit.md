# Embedded video package audit

The video implementation is commit `397c69c2f1cdaf2b06098bc2649e04d6e4673b2e`,
stacked on family source `ed3e4a9a724aaaf9acc657be63f6d4556ebc45be`. It requires
replacement shared package candidate `cdfb80a76b882b4b73861b12f555263bd64ffeb7` and released
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
`4abc83988fdeb6d7478de830e7ee8ec5d9bd99c20d52f89e4ce594d50dfd39a3`.
It does not include video. The review composition combines that exact frozen
core Git head with the unchanged four video modules from `397c69c`. Result:
`particle_sliders_core-0.6.0-py3-none-any.whl`, SHA256
`b1b12c081620198a9876394c5ac5b6ea1e33a5f67200ec1dedd223dbeb9d247c`.
Every shared Python source matches the producer wheel byte for byte; installed
core and video module fingerprints match the composed wheel. This is a local
review artifact, not a published package release or an Ada-approved merge.

The replacement rejects negative second moments, invalid optimizer clocks,
incomplete Adam fields and erased histories before live restore. Its history
schema distinguishes legitimately unused critic parameters from erased state.
Thirteen video CLI regressions check full receiver state, parameter identity,
ambient RNG, critic thermal cache and absent output after rejected resume.
The final delta binds original Adam G clocks to completed updates and validates
sparse D clock bounds. Its Iris import change does not affect the video path.
No video implementation or shared loss mathematics changed during adoption.

The official PG 0.9.1 wheel SHA256 is
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
Its wide-vector automatic-domain limitation remains. LTX connector software tests
explicitly use the exact task14 `alz-continuous-v1`/`continuous` profile, copied
only into isolated test processes. No global/active environment is patched.
Release/auto rejects wide connector responses before backend construction.

- **16 narrow consumer tests pass**, no skips or failures, 8.45 seconds:
  thirteen corrupt-history/clock cases, both presets' actual train/resume/export
  paths, and source/kind rejection.
- **13 independent restore regressions pass**, no skips, 5.56 seconds.
- The earlier broad **139 tests plus 12 subtests** remain preserved evidence
  against `4be0b5b`; unrelated suites were not rerun for the narrow final delta.
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
[`evidence/video-audit/final-cdfb80a`](../evidence/video-audit/final-cdfb80a/).
Earlier `42a2e371` and `4be0b5b` receipts remain unchanged in their original
locations. The final consumer wheel remains at
`/ml2/codex-storage/future-workspaces/2026-10-17/task-22/video-candidate-cdfb80a/`.
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

## Historical base and main integration

Historical main `20fdf98d25455582e7e7191eef8ef51cf73173ae` is the merge base
of the frozen family base `ed3e4a9a724aaaf9acc657be63f6d4556ebc45be` and CDF.
The family base is 22 commits ahead of that main; CDF is 101 commits ahead.
Neither branch contains the other. Exact ancestry is recorded in
`evidence/video-audit/final-cdfb80a/ancestry.json`.

PR #140 targets the historical family base. Merging that target PR alone does
not deliver video to main. Parent/Ada must first approve and reconcile the
main-to-core prerequisites, including the historical content outside PR #141's
narrow base comparison. Then apply only the reviewed video delta
`ed3e4a9..audit/embedded-video-shared-pg09` to the isolated CDF release stage.
A clean `git apply --check` against a detached CDF checkout verifies this delta
can be staged without merging the 22 historical family commits. Revalidate the
assembled candidate before the parent-controlled main merge. No merge has run.
The separately pending image `6aef` payload and the three launcher gates owned
by the Tiny/Bonsai audit are excluded from this video delta.
