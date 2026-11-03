# MOSS: correcting the metric before repeating the LR search

On 2026-11-03, Mikkel preferred the actual MOSS step-960 slider samples to
step 128. The original evaluation ranked them in the opposite order. The error
was in the evaluation normalization, not evidence that training had regressed.
A fresh zero-update evaluation now confirms better teacher/student fit at 960.

Follow the common [evaluation and LR-search workflow](../packages/particle-sliders-core/EVALUATION-AND-LR-SEARCH.md).
The [machine-readable receipt](evidence/moss-teacher-student-20261103.json)
preserves exact values, source/input hashes and local artifact locations.
Those paths identify retained operator evidence; the large model weights,
raw responses and images are not distributed by this documentation commit.

## Experiment and measurement

This was an experimental MOSS-Image-SFT lighting-warmth slider: direct rank-2
linear LoRA with HyperGAN/HNDL residuals, official ParticleGAN 0.9.4, generator
LR `1e-5`, critic LR `0.001`, FP32 DiT and BF16 text encoder with IEEE
deterministic CUDA on GPU1. It has no routed particle bank and is not a
reproduction of the published historical Anima gmix formulation. Declared
wheel-dependency conformance was false; source/core qualification for another
cohort is not implied.

The student reads the neutral prompt at signed slider scale. At identical
latents/times the frozen teacher is
`neutral + sign * strength * (positive - negative)`.
The development grid contains two scenes (cup/book and portrait), two seeds
(171 and 172), two neutral-rollout positions and both signs: 16 response rows,
grouped into eight matched examples. These fixed neutral rollouts are shared
across checkpoints; they are not each student's own trajectories. Responses
have shape `[1, 128, 16, 16]` at the 256-pixel evaluation setting.

The old `mean_per_row_signed_velocity_nmse_v1` divides separately by each
row's teacher-edit energy, then averages. Tiny late teacher edits amplify
late errors out of proportion to their absolute size. The corrected
`matched_example_teacher_student_relative_l2_v2` pools SSE and teacher-edit
energy across the two positions **before division** for each scene/seed/sign,
then averages examples equally. This remains evaluation-only; the GAN objective
and checkpoint weights were unchanged.

## Actual native reevaluation

One evaluation ran from 22:27:48 to 22:31:21 UTC in 213.403 seconds:

| Saved checkpoint | Old per-row v1 error | Corrected pooled v2 error | Cup/book v2 | Portrait v2 |
|---|---:|---:|---:|---:|
| 128 | 0.9781120343 | 0.9783600822 | 0.9754370533 | 0.9812831111 |
| 960 | 4.2383090043 | 0.7300596835 | 0.6961327481 | 0.7639866189 |

Lower is better. OFF scores 1, the exact teacher 0, and a reversed teacher edit
4. At 960, relative error is 26.994% below OFF and 25.379% below checkpoint 128.
These are reductions in this response error, not percentages of image quality.
Both scenes improve, consistent with Mikkel's visual preference on the matched
rendered comparison. This small development grid does not establish general
perceptual quality or validate a universal slider scorer.

Checks passed:

- Saved 128 reproduced all 16 original legacy rows exactly (maximum difference
  zero), and its v2 score exactly matched the earlier CPU reconstruction.
- All 20 model payloads were verified; complete DiT/text-encoder checkpoint
  bytes were bound before use. The frozen DiT passed its final exact BF16
  roundtrip check while computing in FP32.
- All 336 saved adapter leaves matched before and after evaluation. OFF was
  exact under either adapter, across loads, and after detaching each adapter.
- Raw responses were retained. Independent CPU recomputation reproduced both
  scores exactly and verified identical teacher/neutral/latent/time tensors
  across the checkpoints.
- Zero optimizers were constructed and zero training updates were made.
  Owned processes were reaped and GPU1 leases released; other workloads were
  preserved. The single-use admission is consumed.

The selected training lineage still has 985 recorded steps and its latest full
checkpoint is 960. The requested 1,024 steps were not reached. Evaluation does
not add training steps. Historical v1 metrics and selection receipts remain
preserved; the former regression interpretation is superseded.

## What happens to the LR search

The corrected metric was first applied to the existing matched 128-step pilots,
with the critic fixed at `0.001`:

| Generator LR | Old v1 error | Corrected v2 error |
|---|---:|---:|
| `3e-6` | 0.9849987027 | 0.9933808684 |
| `1e-5` | 0.9781120343 | 0.9783600822 |
| `3e-5` | 1.1044827603 | 0.9357916754 |

The existing `3e-5` pilot is now the same-budget metric candidate. A full new LR
grid is not the first action. Byte-verify/export that saved pilot, compare it
with the `1e-5` pilot at the same budget, and inspect matched student/OFF/teacher
images before changing the rate. Keep 960 as the preferred saved reference,
clearly labelled as a different training budget. The lower 960 score does not
show which LR wins at 960 because `3e-5` has not been trained that far here.

Corrected rescoring of the existing `1e-5` lineage improved at every recorded
evaluation: 128 **0.978360**, 256 **0.946150**, 384 **0.907924**, 512
**0.869128**, 640 **0.828416**, 768 **0.788667**, 896 **0.748799**. The fresh
960 forward adds **0.730060**; it is not an interpolation of that history.

A future shared routed-bank MOSS implementation changes the architecture and
needs its own coherent formulation choice, export schema, native memory/update/
replay qualification and rate calibration. Neither `1e-5` nor `3e-5` is a
validated routed-bank default. Keep that effort separate from this metric fix.
