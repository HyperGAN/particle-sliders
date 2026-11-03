# Teacher/student evaluation and learning-rate search

Use this common workflow before selecting a learning rate or extending a slider
run. Validate the measurement first, compare candidates at equal budgets, then
check actual outputs. A lower GAN training loss is not a checkpoint ranking.

This is an experiment protocol, not a new core API or a change to the training
objective. Existing products must bind their evaluator explicitly; historical
scorers and checkpoints keep their original meaning. The
[MOSS case study](../../docs/moss-teacher-student-metric-20261103.md) records a
native validation of this protocol and the normalization error it corrected.

## 1. Freeze the teacher, student and comparison cohort

Write down the actual target formula and the surface being measured. For each
matched example, retain the same conditioning, latent or prompt state, noise
seed, native timestep, masks, coordinates and slider strength for teacher and
student. The frozen neutral response is the OFF reference. A fixed neutral
rollout grid measures response fit on that grid; it is not the student's own
generation trajectory.

For a unipolar product, the teacher may be the frozen positive-conditioned
response and the student the adapted neutral-conditioned response. For the
signed MOSS experiment, the teacher is instead:

```text
teacher = neutral + sign * strength * (positive - negative)
student = adapted neutral response at signed slider scale
```

Evaluate the target the product actually trains. A positive-prompt image alone
is not the signed MOSS teacher. State whether inference CFG adapts only the
conditional branch and whether evaluation measures velocity, a hidden field,
tokens, or decoded media. Do not substitute an unrelated pipeline.

Bind the base-model revision and loaded weights, adapter architecture/targets,
shared-core and ParticleGAN source pins, dependency versions, dtype/promotion,
device, deterministic settings, scheduler, prompts, seeds and evaluation grid.
Keep numerical cohorts separate. Direct LoRA and a routed particle bank are
different architectures, even when both use the same GAN engine.

## 2. Validate the selection metric before spending a search budget

For matched continuous response fields, use pooled teacher/student relative
**squared** L2 error as a starting protocol. Let `e` identify an example
(for MOSS: scene, seed and sign), and `t` range over its declared sampled
positions. `S`, `T` and `N` are student, frozen teacher and frozen neutral:

```math
R_e = \frac{\sum_t \lVert S_{e,t}-T_{e,t}\rVert_2^2}
           {\sum_t \lVert T_{e,t}-N_{e,t}\rVert_2^2},
\qquad R = \frac{1}{|E|}\sum_{e\in E} R_e.
```

Pool errors and teacher-edit energy across the matched positions **before**
division. Then weight examples equally; report per-scene and per-sign results
too. Record the coordinate system and any fixed weighting. If quadrature weights,
position coverage or coordinate normalization change, that is a new protocol.

Do not average a separate normalized error for every timestep by default. A
nearly zero teacher edit at a late timestep can make a small absolute error
dominate that average and reverse checkpoint rankings. Pooling does not delete
late-step errors: every declared position still contributes its absolute SSE.
It also does not reward arbitrary edit strength: overshoot and off-target
changes increase error.

Required analytical controls, within a declared numerical tolerance:

| Response | Expected pooled error |
|---|---:|
| Unchanged neutral / OFF | 1 |
| Exact teacher | 0 |
| Reversed teacher edit, `S = N - (T - N)` | 4 |

Also exercise weak late teacher edits, a large late student error, overshoot,
orthogonal corruption, coordinate partitioning, unit changes, incomplete grids
and duplicate labels. A state with zero teacher-edit energy is allowed when the
whole matched example has positive energy: its student error still counts. A
whole example with zero teacher-edit energy has no defined relative score.
Record it as unscoreable and investigate the target; do not silently drop it or
make it pass by adding an arbitrary epsilon. Predeclare any near-zero validity
threshold and retain absolute error diagnostics.

Check known no-op, useful and broken checkpoints against matched rendered
examples. When the metric contradicts a clear observed result, inspect the
formulation, normalization and inputs before retuning the learning rate. Do not
fit arbitrary weights merely to put one preferred example first.

This response metric measures teacher fit, not universal perceptual quality.
Token or audio products need surface-appropriate validation; do not apply a
waveform L2 score to independently generated songs and call it this protocol.
The [usable-slider selection work](../../docs/slider-selection-metric.md) and
its listening-validation limits remain separate.

## 3. Retain enough evidence to rescore without retraining

Save raw teacher, neutral and student responses when practical, with labels,
tensor shapes, hashes and the evaluator's source/version. Otherwise retain at
least these additive statistics for each state:

```text
element_count
student_teacher_sse = sum((S - T)^2)
teacher_edit_energy = sum((T - N)^2)
student_edit_energy = sum((S - N)^2)
student_teacher_edit_dot = sum((S - N) * (T - N))
```

Retain gain, cosine and orthogonal error as diagnostics, not substitutes for
teacher/student error. Validate finiteness, complete label grids and consistency
of the additive statistics. Preserve evaluator RNG state and each module's
original train/eval mode; evaluation must not update the model or optimizer.

A metric correction does **not** automatically require another LR sweep:

1. Version the correction and preserve the old outputs and source.
2. Rescore all reusable trials and checkpoints with sufficient saved evidence.
3. Label a reconstruction from rounded summary statistics as such, with its
   shape/precision assumptions; it is not a fresh model evaluation.
4. Reevaluate a saved control and the relevant finalist in the same numerical
   cohort. Check the control reproduces before trusting the new ranking.
5. If the corrected ranking changes, inspect the existing newly preferred
   candidate and matched media before launching more fitting.

Missing response data may require new zero-update forwards. That alone is not
a reason to repeat training.

## 4. Run a fair, bounded LR screen

Freeze a search manifest before the first candidate:

- Separate training prompts, development selection examples and a final heldout
  confirmation set. Record which examples informed metric design; they are not
  untouched validation data afterward.
- Use identical initial adapter/bank/critic weights, training seeds, sampled
  rows, reference-noise streams, batches and formulation/schedule horizon.
  Snapshot all RNG and optimizer state. Confirm the actual initial tensors,
  rather than assuming a shared seed guarantees them.
- Declare generator, critic and particle-bank rates separately when present.
  For a generator-rate screen, hold the other rates fixed. If changing a ratio,
  search a named rate pair and record the actual optimizer-group rates. Use the
  engine's supported configuration; do not infer them from a CLI label.
- Choose a small logarithmic bracket around a qualified starting point; record
  candidate values and a bounded compute budget. There is no universal MOSS,
  image, music or routed-bank learning rate.
- Choose common evaluation/checkpoint steps and a selection budget long enough
  to expose the behavior of interest. Keep the same total schedule horizon even
  for short pilots: a 128-step prefix of a 1,024-step schedule is not generally
  the same experiment as a schedule configured to end at 128.
- Predeclare feasibility rules, score direction, tie handling and failure
  treatment. Include OFF identity, finite results, usable edit direction and
  scene coverage. A failed or missing evaluation is unscoreable, not zero error.

Rank candidates at the **same completed step**. Do not select one candidate's
best early checkpoint against another's final checkpoint. Keep learning curves;
a short pilot can nominate a candidate but cannot establish its long-run rate.
Confirm finalists at another common budget and with additional training seeds
before claiming a generally better recipe. Paired intervals over a tiny prompt
set do not establish broad robustness.

## 5. Confirm useful outputs, then continue with exact state

Render a predeclared prompt/seed/strength ladder for finalists, OFF, and the
correct teacher reference. Hold native sampler settings, decoder, precision,
CFG and initial noise fixed. Check intended direction, useful intermediate
strengths, content preservation and artifacts. Use both signs only where the
product claims bipolar behavior. Keep velocity/hidden-response scores and
decoded-output judgments separately labelled.

Continue the selected **full** checkpoint: adapter/bank, critic, optimizer,
streams, RNG, schedule position and completed-step counter. Prove exact resume
with update/replay before a long continuation. Adapter-only exports suffice
for evaluation/inference, not an exact training resume. Keep periodic full
checkpoints and report actual updates, saved steps and replay calls separately.
If a stopped process reached 985 but its last checkpoint is 960, a restart from
that checkpoint begins at 960, not at 985.

Use current device ownership/resource admission and one executor for each
attempt. Stop only owned work. Preserve partial outputs and inspect failure and
cleanup receipts before retrying; do not overwrite an old run or reuse expired
admission. These operational checks protect a fair experiment as well as other
workloads.

## 6. Decide whether another search is justified

| Change or finding | Next step |
|---|---|
| Metric correction; sufficient retained responses/statistics | Rescore first, validate a saved control, check matched outputs; no automatic retraining |
| Missing evaluation evidence | Zero-update evaluation of existing checkpoints |
| Newly preferred existing LR candidate | Compare it at equal budgets and inspect matched media before changing the rate |
| Winner at bracket edge, uncertain/late rank reversal, unstable finalist | Run a targeted follow-up bracket or longer matched screen |
| New backbone, direct LoRA → routed bank, critic/normalization/noise/optimizer or precision cohort change | Qualify the new cohort, then calibrate its rates separately |
| New presentation-only output of unchanged evidence | No new LR search |

A final report names the exact cohort, metric/version/formula, grid and split,
rate pairs and actual group rates, common budgets, every candidate's outcome,
selection reason, matched media, checkpoint hashes, resume proof, actual steps
and cleanup. State whether the result is a candidate, a validated continuation,
or a general recipe claim. Keep a link to the superseded metric and explain why
the decision changed.
