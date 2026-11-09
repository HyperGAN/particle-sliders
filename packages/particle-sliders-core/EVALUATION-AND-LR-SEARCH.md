# Evaluation and learning-rate selection

Select rates by **one scalar: the equal-case arithmetic mean of intended-effect
intensity ON minus exact matched OFF in finished outputs**. Higher wins. Teacher/
student relative error is diagnostic only, never a selector or tiebreak. Model
usage, supported inference and exact measured recipes belong in the
[product repositories](../../docs/products.md).

```text
case_gain = intended_effect(ON) - intended_effect(matched OFF)
rate_score = arithmetic_mean(case_gain over the declared cases)
retain every candidate attaining the highest rate_score
```

Keep negative gains, genuine native zeros and exact ties. A technical model,
render or parser failure is missing evidence, not zero. Preservation drift,
beauty, confidence and completion do not select, veto or break a tie. Do not
compare numerical scores across models, traits or numerical cohorts.

## Use existing RAWs and finished outputs

1. Reuse completed LR banks, source-bound RAW checkpoints, native proofs,
   conditioning, teachers and matched finished outputs. A metric or compatible
   software update does not require a repeated search, training or proof.
2. Freeze the intended effect, instrument, cases and inference settings before
   measuring. Check that the instrument measures that declared effect. Use the
   actual finished image, requested output frame, text, native discrete decision
   or rendered audio. Hidden responses alone are not finished-output evidence.
3. If measurements are missing, use the product's supported native inference/
   render path on its saved RAWs, producing only missing matched outputs. Reuse
   an existing matched OFF with its original provenance. Do not turn an
   inference export into a training-resume checkpoint or replay completed work.
4. Apply the frozen external instrument once and retain its first observations.
   Compute each ON-minus-OFF gain and the equal-case arithmetic mean. Use
   already measured candidates directly and preserve all tied maxima; no new
   top-two extension, worse-arm completion or fresh confirmation is required.

Compare rates within the same model, concept, native profile, completed update
budget and schedule horizon. Preserve separate fresh assessments and their
original limits; do not pool them into the selection cohort. Report scoped
best-tested rates, not a global/joint optimum or changed package default.
Record absolute generator, critic and participating-bank rates and exact option
literals; see [rate fields](LEARNING-RATE-CONTRACT.md).

## Existing SSE runner: historical diagnostic protocol

The current `particle-slider-search` implementation is a **fixed relative-error
runner**, not the finished-output selector above. It has no configurable output-
effect scorer/callback. Its `selection='raw'` field selects weight provenance,
not an objective. `--media-review` is diagnostic and does not change ranking.
Its SSE ranking, secondary tie rules and declared finalist/heldout checks remain
part of that unchanged historical runner; do not use them to select current
intended-effect rates or force a unique recipe from an effect tie. The
[runner API](PARAMETER-SEARCH.md) documents that implementation. It does not
implement automatic finished-output measurement or selection.

No new generic search flag or automatic product renderer is advertised here.
The current method uses supported product inference and an explicitly frozen
external first-observation instrument, followed by the arithmetic above. Shared
training execution/lifecycle APIs are not an effect selector: lifecycle
`evaluate()` measures native responses, not finished images, while `render()`
requires an actually supported product renderer.
Keep all completed SSE searches and saved results under their original labels;
add only genuinely missing finished-output measurements from existing RAWs.

### Retained relative-error diagnostic

For example $i$, student $S_i$, frozen target $T_i$ and frozen neutral baseline
$N_i$ in the same native response coordinates give:

$$
E_i = \frac{\sum_k(S_{i,k}-T_{i,k})^2}{\sum_k(T_{i,k}-N_{i,k})^2},
\qquad E=\operatorname{mean}_i E_i.
$$

Pool declared positions/coordinates within each example before dividing, then
weight examples equally. Convert original operands to CPU FP64 before
subtraction, not a rounded difference. The existing metric uses chunk-independent
accumulation. Lower finite error is diagnostically better; above one is valid.
Zero teacher-edit energy is undefined, not zero error. Preserve the original
target construction, operands, scoring grid and all outcomes. This formula and
its historical results remain unchanged; they do not rank current effect gains.

## Preserve scientific and checkpoint identity

Freeze model/source/data, target capture, numerical settings, splits, seeds,
initialization and horizon. Observation must restore training RNG, modes, buffers
and native state. Keep full checkpoints for exact continuation, RAW exports for
inference, actual update counts, all candidate results and their metric labels.
Resume only genuinely unfinished declared work from its exact latest FULL state
without replaying a committed prefix. A source/configuration change must be
recorded honestly without rewriting existing artifact identities or discarding
their scoped evidence.

A recommendation names product/core/ParticleGAN pins, model/data identity,
rates, budget, score and measurement limits. A software fixture establishes
orchestration, not native quality or a newly tuned rate. All training, selection,
exports and inference use RAW current weights. Fresh configs explicitly use
`ema_decay=None`; adapter, critic and particle-bank weight EMA are disabled.
ParticleGAN scalar optimizer statistics remain unchanged.
