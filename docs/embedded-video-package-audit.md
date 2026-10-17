# Embedded video package audit

The video implementation is commit `397c69c2f1cdaf2b06098bc2649e04d6e4673b2e`,
stacked on family source `ed3e4a9a724aaaf9acc657be63f6d4556ebc45be`. It requires
shared package candidate `42a2e371c537180bd189fd4a288b640f9f1c98fc` and released
ParticleGAN **0.9.1**. Neither candidate is implicitly Ada-approved or merged.
All active family/core pins remain untouched.

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

The first shared wheel SHA256 is
`7418efe0d89782b2fd266c6a8e536539243d06dd42ff5d393e409bf4b0f2c872`.
It does not include video. The reproducible review composition combines that
frozen core Git head with exactly four video modules from `397c69c`. Result:
`particle_sliders_core-0.6.0-py3-none-any.whl`, SHA256
`0941edb3a8467139ab49fb4ddb6a29ab59401c4cccb1504935bc10c5297c8641`.
The composition is a local review artifact, not a published package release.
The public shared PG source validator and loss math are unchanged by video.

The official PG 0.9.1 wheel SHA256 is
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
Its wide-vector automatic-domain limitation remains. LTX connector software tests
explicitly use the exact task14 `alz-continuous-v1`/`continuous` profile, copied
only into isolated test processes. No global/active environment is patched.
Release/auto rejects wide connector responses before backend construction.

- **131 CPU tests and 12 subtests pass**, no skips or failures, 11.33 seconds.
- **9 independent native review tests pass**, no skips, 2.19 seconds; both PEFT
  review defects are covered and resolved.
- Source-free installed-wheel tests run with isolated Python from an empty
  temporary directory. All video imports originate from site-packages, with no
  `conceptmod` import. Both presets match four uninterrupted updates to separate
  2+2 process continuation, including full state/optimizer/sampler/RNG. Inference
  probes match export reload exactly. Source tampering fails before output.
- CPU-only H3 closure tests run with `python -S` and copied source files, without
  Torch or the repository, and reject blocked routes before runtime imports.

Receipts, source hashes and exact composition are under
[`evidence/video-audit`](../evidence/video-audit/). The wheel remains at
`/ml2/codex-storage/future-workspaces/2026-10-17/task-22/video-candidate/`.
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
