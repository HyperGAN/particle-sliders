# Teacher/student evaluation and learning-rate search

Use this common workflow before selecting a learning rate or extending a slider
run. Validate the measurement first, compare candidates at equal budgets, then
confirm the selected raw response on held-out examples. A lower GAN training
loss is not a checkpoint ranking.

The 2026-11-04 cross-product LR campaign uses **one selection and confirmation
metric**: raw teacher/student equal-example pooled relative squared error,
defined below. Preservation drift, late/weak-field scores, rendered media,
listening, CLAP and subjective quality are optional diagnostics. They cannot
reject an LR candidate, block a recommendation, require a larger geometry or
trigger extra qualification work. Source/data/numerical identity, finite updates,
actual optimizer rates and exact continuation remain correctness requirements.

Disable model-weight EMA explicitly (`ema_decay=None`) for this campaign. Train,
rank, confirm, export and infer with raw current weights. Do not add an EMA or
checkpoint-averaging comparison. Keep the official optimizer update formula.
An official recipe may serialize the disabled value `ema_decay=0.0` while the
session records `None`; verify absent moving weights and raw export selection
without rewriting that saved metadata.

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
**squared** L2 error as the sole LR ranking criterion. Let `e` identify an example
(for MOSS: scene, seed and sign), and `t` range over its declared sampled
positions. `S`, `T` and `N` are student, frozen teacher and frozen neutral:

```math
R_e = \frac{\sum_t \lVert S_{e,t}-T_{e,t}\rVert_2^2}
           {\sum_t \lVert T_{e,t}-N_{e,t}\rVert_2^2},
\qquad R = \frac{1}{|E|}\sum_{e\in E} R_e.
```

Cast student, teacher and neutral tensors to FP64 **before subtraction**.
Pool errors and teacher-edit energy across all coordinates and matched positions
**before** division. Then weight examples equally. Per-scene and per-sign values
are optional diagnostics. Record the coordinate system and any fixed weighting. If quadrature weights,
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
Record it as unscoreable, retain its absolute error and exclude it explicitly
from the relative-score mean; never silently drop it or add an arbitrary
epsilon. Determine the valid/excluded example identities from the frozen teacher
and neutral tensors before ranking, and use that identical set for every LR
candidate. A student failure must not shrink the set. Record its identities and
counts. A cohort with no valid examples cannot rank an LR. The default threshold
is exactly zero; excluding positive-energy near-zero examples changes the
example set and metric cohort and must be separately declared before the search.
When a protocol requires a fixed positive-energy example set, a missing valid
target is a data failure rather than a preservation score.

Validate no-op, teacher and broken responses against the analytical controls
and retained native tensors. Inspect formulation, normalization and inputs when
those controls fail. Do not fit arbitrary weights to prefer an example.

This response metric measures teacher fit, not universal perceptual quality.
Token or audio products use their declared matched native response fields;
do not apply waveform L2 to independently generated songs and call it this
protocol. Listening and other media judgments are not LR gates in this campaign.
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
5. If the corrected ranking changes, inspect the newly preferred candidate's
   saved native responses and compare it at the same budget before more fitting.

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
- Choose a small logarithmic bracket around a source-qualified starting point; record
  candidate values and a bounded compute budget. There is no universal MOSS,
  image, music or routed-bank learning rate.
- Choose common evaluation/checkpoint steps and a selection budget long enough
  to expose the behavior of interest. Keep the same total schedule horizon even
  for short pilots: a 128-step prefix of a 1,024-step schedule is not generally
  the same experiment as a schedule configured to end at 128. Even when the
  effective rates are constant, preserve the checkpoint's serialized horizon
  and configuration identity on restore. A packet declared to end at 64 cannot
  silently become a 128-step packet; use a fresh common-horizon cohort or an
  explicitly source-qualified extension contract.
- Predeclare feasibility rules, score direction, tie handling and failure
  treatment. Require OFF identity, finite results and the complete declared
  native example/position grid. A failed or missing evaluation is unscoreable,
  not zero error. Do not add a separate scene, preservation, late/weak or media
  score as a veto. Every finite, source-correct candidate completes its declared
  budget; a large finite primary error is not an early-stop condition.

After native update mechanics have been qualified, report endpoint generator,
critic and bank displacement as diagnostics. A finite trajectory can return to
its starting values, so zero net displacement must not reject an LR candidate.
Retain the original native movement proof and the actual parameter-role, bank
shape, optimizer-rate, finite-update and native-call checks.

Use the common product keys `lr`, `d_lr_mult` and, where a real bank exists,
`prior_lr_mult`. Adapter-only products omit the bank multiplier. Changing G with
fixed multipliers changes D and bank too; label that as a coupled experiment.
For a G-only screen, compute multipliers that produce exactly common actual D
and bank rates, and verify the binary floating-point optimizer-group values
before updating. Nominal decimal labels are insufficient: `0.004 * 4.5` gives
`0.018000000000000002`. If no common value is representable for the chosen grid,
declare a new matched rate in a new plan; do not patch optimizer state.

A G screen recommends a tested combination with fixed D and bank rates; it does
not establish a joint optimum. To refine the critic, hold G and any bank rate
fixed and declare a separate D screen. Refine a real bank rate the same way,
holding the other absolute rates fixed. Retain the exact rates and scope of each
earlier recommendation. A later refinement needs a new unused confirmation set;
an earlier confirmation score must not guide the new choice.

Rank candidates at the **same completed step**. Do not select one candidate's
best early checkpoint against another's final checkpoint. Keep learning curves;
a short pilot can nominate a candidate but cannot establish its long-run rate.
Confirm finalists at a longer common budget and on held-out confirmation
examples using this same sole metric. Freeze the development nominee before
opening confirmation. Record a recipe-specific recommendation with its concept,
source/model/data/numerical cohort and measured budget. Additional seeds and
concepts are needed for a broader recipe claim; paired intervals over a tiny
prompt set do not establish broad robustness.

A predeclared halving plan may advance a fixed number of candidates at each
equal-budget stage, then continue the selected candidate to the final budget.
Retain each stage's ranking and actual budget; do not imply that eliminated
candidates completed the final budget.

Declare finite rate bounds and an extension budget before measuring a targeted
extension. A bracket-edge winner calls for a justified extension within that
budget. If the winner remains at the declared outer bound after the matched
comparisons, report it as the best tested rate at that bound, with the optimum
unbracketed. A prospectively frozen policy may then continue and confirm that
candidate; it cannot claim a global optimum or retune from confirmation.

A plan may retain already available equal-budget endpoints in later comparisons.
Declare that retention before observing the new candidates. Keep each endpoint's
actual source, initialization, teachers, numerical profile and completed budget;
prove the comparison is compatible and do not relabel its checkpoint as a new
producer. Complete every newly declared matched budget. Select one development
nominee before opening a new confirmation set.

## 5. Confirm the raw response, then continue with exact state

Use independent held-out examples and the same raw primary metric. Confirm the
native teacher/student surface at the artifact's declared geometry, strength,
precision and conditioning. A larger geometry or decoded-media ladder is not
required to recommend rates for the measured response cohort. Optional media
diagnostics retain their own sampler/decoder/CFG/noise identity and do not veto
the LR result.

Continue the selected **full** checkpoint: adapter/bank, critic, optimizer,
streams, RNG, schedule position and completed-step counter. Prove exact resume
with update/replay before a long continuation. Adapter-only exports suffice
for evaluation/inference, not an exact training resume. Keep periodic full
checkpoints and report actual updates, saved steps and replay calls separately.
If a stopped process reached 985 but its last checkpoint is 960, a restart from
that checkpoint begins at 960, not at 985.

Keep live progress and the last resumable step separate. When committing a new
checkpoint, atomically publish its full state and actual returned update metrics
before marking that step durably saved. Bind the saved step, source,
configuration and rates to the checkpoint and verify its readback. Journaling
must preserve caller RNG and make no additional native forwards. If the native
packet omits last-update metrics, keep them in a journal envelope around the
unchanged packet. Retain immutable attempt directories and any absolute journal
paths. Report committed calls, replayed calls and unknown in-flight work honestly.

If the target update is already committed but evaluation or export was
interrupted, restore that target and finish those observations with zero new
updates. Compare an existing RAW response golden only when one was actually
published and qualified; otherwise label the response as newly measured.

Use current device ownership/resource admission and one executor for each
attempt. Stop only owned work. Preserve partial outputs and inspect failure and
cleanup receipts before retrying; do not overwrite an old run or reuse expired
admission. These operational checks protect a fair experiment as well as other
workloads.

Record the owned GPU/allocator/RSS caps, global free-memory floors, timeout and
inherited physical/workload leases in each execution. An attempt may run beside
other processes when fresh actual capacity fits those declared controls. Unknown
foreign identities or future budgets do not by themselves veto that attempt.
Measure the owned PID's GPU usage and total device free memory throughout; stop
only the owned attempt when a hard cap or floor fails.

For host admission, current RSS is already charged to `MemAvailable`. Add known
remaining growth from source-bound phase forecasts once, including any later
model reload. Do not use a past RSS high-water mark as current residency, and do
not reserve growth for a completed, reaped phase. Recheck after source capture
and lock acquisition, before model loading. Forecasts and sampled admission are
not a proof that all later allocations will fit; keep the original hard caps,
live floors, exact owned cleanup and checkpoints.

Execution-only changes to device headers, resource bookkeeping or documentation
do not automatically require another model-proof matrix. Review and bind the
narrow change, retain the qualified native mathematics and prior evidence, and
verify actual initialization, teachers and placement for the new execution.
Preserve the real new execution identity rather than rewriting old packets or
claiming an unchanged hash. A changed numerical or scientific cohort still needs
fresh qualification.

An explicitly proved host-capacity interruption may use bounded continuation
after exact owned-child termination/reaping, fresh resource admission and
qualification of the committed prefix. Distinguish a source-proved fresh-start
interruption from a restored checkpoint; neither permits invented update counts.
Pure artifact audits may be rescheduled without repeating native training.
Source, numerical, GPU, owned-process RSS-cap and unknown-owned-attribution failures
remain terminal for that execution and require investigation. A retried final
confirmation uses the same frozen winner and held-out examples; its partial
observations cannot tune or replace the winner.

## 6. Decide whether another search is justified

| Change or finding | Next step |
|---|---|
| Metric correction; sufficient retained responses/statistics | Rescore first and validate a saved native control; no automatic retraining |
| Missing evaluation evidence | Zero-update evaluation of existing checkpoints |
| Newly preferred existing LR candidate | Compare it at equal budgets using the sole primary metric |
| Primary winner at bracket edge, uncertain primary ranking, primary rank reversal at a later update budget | Run a targeted follow-up bracket or longer matched screen |
| New backbone, direct LoRA → routed bank, critic/normalization/noise/optimizer or precision cohort change | Qualify the new cohort, then calibrate its rates separately |
| New presentation-only output of unchanged evidence | No new LR search |

A final report names the exact cohort, metric/version/formula, grid and split,
rate pairs and actual group rates, common budgets, every candidate's outcome,
selection reason, checkpoint hashes, resume proof, actual steps
and cleanup. State whether the result is a candidate, a validated continuation,
or a general recipe claim. Keep a link to the superseded metric and explain why
the decision changed.
