# Anima Turbo training verification and benchmark

Measured on 20 September 2026 with the pinned Lumen Anima environment. These
results qualify the implementation and execution configuration; visual quality
still requires the development and held-out audits.

## Correctness

The captured ParticleGAN source is verified by SHA256 before evaluating its
actual gmix loss, parameter/input derivatives, particle VIC, and active/inactive
lazy gradient cap. Sixteen nonlinear cases passed with zero difference.
The same 16 cases also passed at the actual 65,536-coordinate velocity-field
dimension, including exact generator input derivatives and nonzero cap penalties.
All 52 CPU tests pass, including catalog-revision and retry-seed isolation,
unchanged teacher fields across an adapter-rendering handoff, and explicit
swapped-target, shuffled-timestep and incorrect-neutral probe checks. Target-reuse
tests also reject corrupt source files and mismatched paired teachers, and ensure
that changed prompts cannot inherit old velocity fields.

Streaming normalization was also compared directly with the pinned ParticleGAN
normalizer on 32 evenly spaced training shards from each variation, retaining
both trajectories and all ten full velocity fields. Maximum relative scale
difference was 4.84e-7, within FP32 rounding. The 1,600-update noise schedule
matched the pinned implementation exactly at all checked positions.

A separate diagnostic executes the pinned complete D/G/VIC update procedure
against Lumen's update on identical precomputed rows, noise and VIC draws.
All 16 high-precision fixtures passed, including active lazy-cap steps and
repeated rows: maximum parameter difference was 4.54e-10 and generator-gradient
difference 8.40e-16. This isolates the update equations from FP32 accumulation.
The corresponding FP32 strict parameter comparison failed: batch/grouping
roundoff, including near-zero attention key-bias gradients, is amplified by
Adam. Maximum parameter difference was 0.00180. That failed result is retained;
this diagnostic does not establish FP32 batch equivalence or approve a faster
microbatch. Production retains its separately verified deterministic
microbatch-one configuration.

Real-model verification covers exact checkpoint resume, optimizer and sampler
state, EMA, frozen teacher predictions, and exported Studio predictions at all
ten timesteps. Deterministic GPU execution and CPU placement of non-capturable
Adam step counters are required. No objective or hyperparameter was changed.

The singleton padding-mask correction has a separate runtime compatibility
certificate. All 240 single-image teacher comparisons were identical. This
certificate permits reuse of the original frozen target caches; larger
microbatches still have to pass their own numerical checks.

## Execution comparison

All accepted timings use the same fixtures and sampler streams, 20 warmup
updates, and 50 timed updates at effective batch eight. Eager execution uses the
pinned generator SDPA and plain-math critic attention for second derivatives.

| Host/configuration | Seconds/update | Peak VRAM | Result |
|---|---:|---:|---|
| Original RTX 3090, checkpointing on, microbatch 1 | 11.728 | 6.268 GiB | Passed |
| Original RTX 3090, checkpointing off, microbatch 1 | 8.535 | 10.355 GiB | Passed |
| Original RTX 3090, microbatch 2 | — | — | Rejected: prediction/gradient/update differences |
| Original RTX 3090, microbatch 4, checkpointing on | — | — | Rejected: numerical differences |
| Original RTX 3090, microbatch 4, checkpointing off | — | — | Rejected: out of memory |
| Replacement RTX 3090, checkpointing off, microbatch 1 | 6.094 | 10.355 GiB | Passed |
| Local RTX A6000, default kernels | 5.365 | 10.355 GiB | Rejected: cross-host state differences |
| Local RTX A6000, matched SM-82 kernel policy | 5.276 | 10.355 GiB | Passed: 70 exact updates |

Checkpointing on/off reproduced all 20,240 state tensors after 70 updates. The
replacement GPU also reproduced that complete 70-update state exactly. A separate
replay from the production step-60 checkpoint through step 64 reproduced the
entire saved state, update logs, and 768px EMA image exactly. Fresh embeddings,
both teachers at all ten timesteps on both shared trajectories, and the Off image
were identical across hosts. The selected configuration remains microbatch one,
checkpointing off. Compilation has not been accepted.

Moving to the user's A6000 required matching kernel-selection heuristics to the
3090's 82 multiprocessors. Setting only the main cuBLAS handle did not reproduce
backward updates. The accepted process-local policy configures every cuBLAS
handle, including the autograd worker's handle, and CUDA's reported SM count for
launch heuristics. Default autograd scheduling is retained. The physical A6000
still has 84 multiprocessors; this is not a GPU reservation or memory limit.
Other processes, including Music Studio, are unaffected. The complete 70-update
state (20,240 tensors) and every update record are identical to the reference.
Failed default, single-handle and single-thread attempts are retained separately.
The native policy sources, compiler, pinned CUDA packages and resulting shared
libraries are hashed by `scripts/anima_cuda_host.py`. The standalone launcher's
loaded ELF sections are identical to those used by the accepted benchmark.

## Preparation and overhead

The qualified dataset produced 1,224 paired trajectories in six caches, totaling
18,958,424,968 bytes. Timed target preparation took 3,756.55 seconds, excluding
earlier definition-review iterations. Every committed shard was checksum
verified after migration and is backed up locally.

The revised Theatrical v16 reference screen covers 24 training pairs and six
development pairs. Only its six training lighting clauses changed; the original
failed pilot, targets and catalog remain archived. All shared fields, neutral
clauses, other variations and unseen evaluation paraphrases are unchanged.
Minor reference differences remain documented, and some lighting clauses are
subtle; this screen does not qualify a trained slider.

During the new target build, `scripts/anima_reuse_neutrals.py` validated the
archived cache hashes and existing runtime compatibility certificate, then
compared all 660 already-generated neutral states/velocities exactly. It
published 339 remaining compatible neutral paths without overwriting the live
preparer. Every revised positive teacher prediction is still recomputed. The
validation/copy took 12.85 seconds. A Studio rerender of an archived Theatrical
checkpoint during preparation was also pixel-identical to its original image.

The revised build completed 384 training and 24 development trajectories in
1,341.06 seconds including runtime loading and the rendering handoff, storing
6,319,517,016 bytes. Rechecking its normalization against the pinned source on
32 evenly spaced training shards gave a maximum relative scale difference of
4.91e-7. The fresh pilot uses a new fixed baseline and normalization; its residual
values must not be compared directly with the archived pilot's values.

Both v16 pilot seeds completed 200 updates with finite gradients, improved best
held-out residual and exact resume/Studio parity, but failed the character-render
preservation gate. Their checkpoints, reviews and targets remain archived.
They are not qualified sliders.

A training-only diagnostic found that v16's `soft shadows` definition had negative
mean full-field edit cosine against every other theatrical definition
(-0.516 to -0.205). Different characters and states limit causal interpretation.
Version 17 replaces only that clause with `lower illumination`, after screening
all 24 training pairs and six development references. The 55 unaffected reference
images were pixel-identical; the five new positives retained principal appearance,
outfit, pose and scene, with documented minor changes. Final-test characters were
not used. Of the 408 target trajectories, 342 have identical complete paired rows
and were reused with exact tensor checks; the other 66 are recomputed. New
normalization and a fresh student pilot are required. The ParticleGAN formulation,
recipe and quality thresholds are unchanged.

The v17 build completed in 212.50 seconds including runtime loading, plus 16.71
seconds validating/copying the unchanged target shards. Its normalization again
matched the pinned source on 32 training shards (maximum relative difference
4.78e-7). Definition 05 now has positive mean full-field edit cosine against all
five other definitions (0.203 to 0.598); this resolves the measured direction
conflict, without establishing convergence or perceptual quality. The fresh
pilot's fixed development baseline is 2.47442.

The v17 pilot completed 200 updates with best development residual 2.44247
(raw residual at 200: 2.44393). Finite gradients and exact resume/Studio parity
passed, but the condition-concealed image review found three major clothing
regressions across six development comparisons, with no atmosphere wins. It
therefore failed pilot qualification and has not been promoted to a full run.

On the original host, the selected configuration took 7.45 seconds to load,
11.18 seconds to save, about 55 seconds per fixed development probe, and 0.99
seconds to release. Measured renders took 1.14–2.30 seconds at 512px and
2.32–2.97 seconds at 768px, excluding loading and queue handoffs. Those overhead
measurements belong to the original host; they should not be presented as new
host timings.

Observed event timings on the replacement host give a median 12.89 seconds
from the final rendered image to the next loaded trainer, and 2.46 seconds for
ordinary training save/release handoffs. These include runtime/cache/optimizer
restoration and exclude the special update-20 and 100-update probe/export
boundaries. Median 512px render times were 1.13 seconds Off and 1.72 seconds On.
The measurements are operational observations, separate from the controlled
20/50-update benchmark. They can be refreshed with
`scripts/anima_live_timing.py --since <host-start-epoch>`.

At the replacement's measured update speed, all 4,800 campaign updates alone
take about 8.13 hours. Fixed probes, 972 scheduled development images, 512 final
audit images, checkpoint writes, and GPU handoffs add time. Quality review may
prevent a checkpoint from qualifying regardless of elapsed training time.

The live Training page displays recent measured update speed separately from
these benchmark timings. Raw residual regressions remain visible alongside the
explicitly labeled best-so-far series.

## Fixed training-fit diagnostic

`scripts/anima_probe_training.py` evaluates immutable EMA exports without updating
the model. Its fixed training fixture covers all 24 rows at seed 1000, both paths,
and all ten positions: 480 full velocity fields. It uses the run's existing
training-only normalization, fixed projections and noise, and records checkpoint
and fixture hashes. These results do not enter development checkpoint selection.

For the original Candlelit run on the replacement GPU:

| Update | Training residual RMS | Development residual RMS |
|---|---:|---:|
| 100 | 0.9875 | 1.2335 |
| 400 | 1.0803 | 1.3798 |
| 800 | 1.3419 | 1.6630 |

Both residuals rise across these snapshots. Each training-fixture evaluation took
approximately 76 seconds, excluding model loading. Complete image reviews at
updates 400 and 800 each found three major preservation regressions among eight
cases. Neither checkpoint qualifies despite eight atmosphere wins each. The prescribed
training formulation, learning rates, noise schedule and selection thresholds
remain unchanged; the campaign continues through its fixed horizon.

At update 1,200 the raw Candlelit development residual was 1.55027, while its
best-so-far remained 1.23348 at update 100. All eight comparisons favored its
atmosphere, but four had major outfit, character, setting or medium regressions.
This checkpoint also fails selection. These ratings are condition-concealed
assistant image assessments, not a human-subject study.

A separate CPU diagnostic runs that same recipe and full 1,600-update horizon
through the production trainer on a tiny nonlinear frozen transformer. EMA
training residual fell from 1.0417 at initialization to 0.5632 at update 400,
0.1552 at 800, and 0.0870 at 1,600. Gradients remained finite and frozen weights
were unchanged. This sustained-learning check exercises the complete path but
does not establish Anima convergence or image quality. Reproduce it with
`scripts/anima_tiny_learning_diagnostic.py`; results are stored separately from
production and never enter checkpoint selection.

## Evidence locations

- `artifacts/anima/correctness/reference-reverification.json`
- `artifacts/anima/correctness/reference-full-field-reverification.json`
- `artifacts/anima/correctness/normalization-reverification.json`
- `artifacts/anima/correctness/normalization-reverification-theatrical-v16.json`
- `artifacts/anima/correctness/update-reverification-float32.json`
- `artifacts/anima/correctness/update-reverification-float64.json`
- `artifacts/anima/remote/correctness/gpu-resume.json`
- `artifacts/anima/remote/benchmarks/candlelit/report.json`
- `artifacts/anima/remote/benchmarks/candlelit/seventy-update-parity.json`
- `artifacts/anima/runpod-3090-candidate/hardware-benchmark.json`
- `artifacts/anima/runpod-3090-candidate/replay-verification.json`
- `artifacts/anima/runpod-3090-candidate/teacher-verification.json`
- `artifacts/anima/runpod-3090-candidate/target-verification.json`
- `artifacts/anima/runpod-3090-candidate/handoff-verification.json`
- `artifacts/anima/runpod-3090-candidate/operational-timing.json`
- `artifacts/anima/diagnostics/training-generalization/candlelit/report.json`
- `artifacts/anima/diagnostics/tiny-learning/report.json`
- `artifacts/anima/diagnostics/target-consistency.json`
- `artifacts/anima/definition-candidates/theatrical-consistent-lighting-v17/qualification.json`
- `artifacts/anima/remote/migrations/neutral-reuse.json`
- `artifacts/anima/migrations/archive-render-check.json`
- `artifacts/anima/browser/quality-live-report.json`

Large artifacts stay outside git. The replacement rental measured $0.234/hour
including storage when the original pod was retired; the RunPod account and
backup watchdog track the remaining credit.
