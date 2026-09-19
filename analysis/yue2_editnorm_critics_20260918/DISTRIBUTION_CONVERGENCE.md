# Yue2 Particle-Bridge Distribution Convergence

## Objective

Do not use training cosine alone to decide that a slider has finished learning.
Cosine measures edit direction on the current training minibatch. It does not
establish that the predicted hidden-state distribution matches the teacher,
that edit magnitude is correct, or that every prompt family works.

Evaluate each checkpoint on a fixed bank of unseen continuation histories and
measure:

1. Paired prediction error.
2. Predicted-versus-teacher distribution distance.
3. Whether an independent evaluator can distinguish generated residuals from
   reference noise.
4. Performance at multiple slider strengths.
5. Stability across prompts and seeds.

## 1. Build a held-out probe bank

Create a separate `prepared-eval.pt` containing histories that are never
sampled during optimization.

Use:

- The same four prompt templates as training.
- At least 16 new continuation seeds per template.
- Preferably 32 seeds per template, giving 128 held-out examples.
- Seed values different from every training seed.
- The same history length and teacher extraction process as training.
- Both neutral and positive teacher hidden states.

Do not add these examples to `prepared.pt` or the training sampler.

For every probe row, retain:

```text
template identifier
continuation seed
token history
neutral hidden state
positive teacher hidden state
paired normalization scale
```

Keep another small set of prompts and seeds untouched for final listening.
Repeated checkpoint selection turns the probe bank into development validation,
not a final test.

## 2. Evaluate saved EMA checkpoints

Evaluate EMA exports because those are the likely production artifacts. Live
checkpoints can be measured separately for diagnosis, but should not be mixed
into the main ranking.

For the current experiment, evaluate:

```text
metal_gmix_t8_w48_l1:
  steps 100, 200, ..., 1600

metal_sn_mlp_w128_l1:
  steps 100, 200, ..., 1600
```

The milestone files already exist, so this can be done without retraining.

At every checkpoint, run each held-out row at slider strengths:

```text
0.25
0.50
1.00
```

For strength `s`, define the interpolated teacher target as:

```python
teacher_s = neutral + s * (positive - neutral)
residual = prediction_s - teacher_s
normalized_residual = residual / paired_edit_scale
```

This checks whether partial slider strengths behave sensibly instead of only
fitting the full-strength endpoint.

## 3. Record paired-error metrics

For every checkpoint and strength, calculate the following metrics.

### Normalized residual RMS

```python
sqrt(mean(normalized_residual ** 2))
```

This is the primary fidelity metric. Zero means the predicted hidden state
reached its paired teacher target.

### Row-level residual distribution

Calculate normalized residual RMS independently for each probe row, then
report:

```text
mean
median
p90
p95
maximum
```

The tail metrics catch seeds that fail despite a good global average.

### Edit gain

Let:

```python
predicted_edit = prediction_s - neutral
teacher_edit = positive - neutral

gain = dot(predicted_edit, teacher_edit) / dot(teacher_edit, teacher_edit)
```

The expected gain is approximately the requested strength:

```text
strength 0.25 -> gain 0.25
strength 0.50 -> gain 0.50
strength 1.00 -> gain 1.00
```

This exposes the failure where cosine is excellent but the slider is too weak
or too strong.

### Orthogonal error

Remove the component aligned with the teacher edit:

```python
orthogonal = predicted_edit - gain * teacher_edit
```

Report its norm relative to the teacher-edit norm. This measures unintended
hidden-state movement and leakage.

Cosine can remain in the report, but it is a secondary diagnostic.

## 4. Measure distribution matching

Use two complementary tests.

### Direct teacher-distribution comparison

Compare held-out predicted edit vectors with corresponding teacher edit vectors
using fixed random projections.

- Generate 256 fixed random unit vectors in 2048 dimensions.
- Save them once and reuse them for every checkpoint.
- Project predicted and teacher edits onto each vector.
- Calculate one-dimensional Wasserstein distance for every projection.
- Report mean, median, and p95 projected distance.

Reusing exactly the same projections is essential for meaningful checkpoint
comparisons.

### ParticleGAN game comparison

The training game compares:

```python
real = noise
fake = noise + normalized_residual
```

Generate these distributions using fixed random seeds and the same number of
samples at every checkpoint. Use a fixed evaluation noise standard deviation.

Do not use the checkpoint's current annealed noise value for model selection.
It changes with training step and would make checkpoint distances incomparable.
Use a fixed evaluation sigma such as:

```text
sigma = 1.0
```

Optionally report sensitivity at:

```text
sigma = 0.5, 1.0, 2.0
```

Calculate sliced-Wasserstein or energy distance between `real` and `fake`.
Lower is better; zero is ideal.

## 5. Run an independent two-sample classifier

Do not reuse the training discriminator. It has co-adapted with the adapter and
may be weak, overfit, or temporarily out of equilibrium.

For each checkpoint:

1. Generate many `real` and `fake` examples from held-out residuals.
2. Split by probe-row identity, not merely by noise sample.
3. Train a fresh small classifier on the training portion.
4. Evaluate it on entirely different probe rows.
5. Report ROC AUC with a bootstrap confidence interval.

Interpretation:

```text
AUC approximately 0.50: evaluator cannot distinguish the distributions
AUC 0.50-0.55: small or inconclusive detectable gap
AUC 0.55-0.65: meaningful remaining gap
AUC above 0.65: clearly not distributionally matched
```

AUC should not be the only selection metric because the estimate depends on
evaluator capacity and the number of probe rows.

## 6. Report subgroup failures

Produce metrics separately for:

```text
each prompt template
each requested strength
vocal configuration
tempo range
continuation seed
```

A global mean can hide one prompt family that consistently fails.

Flag a checkpoint when a subgroup has substantially worse residual RMS or
distribution distance than the rest, for example more than 1.5 times the
global median.

## 7. Decide when learning is finished

Evaluate every 100 steps. Call a run plateaued only when, across three
consecutive evaluations:

```text
normalized residual RMS improves by less than 1%
p95 residual improves by less than 1%
sliced-Wasserstein improves by less than 2%
edit gain remains close to the requested strength
no subgroup worsens materially
```

Then check that:

```text
independent evaluator AUC is approximately 0.55 or lower
the AUC confidence interval is compatible with 0.50
training remains finite and stable
audible held-out renders are not degrading
```

Do not require the training discriminator loss to equal exactly `log(2)`. That
is supporting evidence, not proof of convergence.

## 8. Select the production checkpoint

Do not automatically take the final checkpoint.

Rank checkpoints primarily by:

1. Held-out normalized residual RMS.
2. Held-out p95 residual.
3. Direct sliced-Wasserstein distance.
4. Correct gain at strengths 0.25, 0.5, and 1.0.
5. Independent evaluator AUC.
6. Matched listening evaluation.

Prefer the earliest checkpoint statistically indistinguishable from the best
later checkpoint. This reduces exposure to late adversarial instability without
sacrificing measured quality.

For the current gmix run, likely candidates are approximately steps 1000-1400,
but the probe must determine that rather than training cosine.

## 9. Integrate it into the repository

A clean implementation should add:

- A checkpoint evaluator that loads `prepared-eval.pt`.
- A persistent file containing fixed projection vectors and evaluation-noise
  seeds.
- An optional watcher that evaluates new `*_stepN.safetensors` files.
- `probe.jsonl`, with one record per checkpoint and strength.
- `probe-summary.json`, containing the current best checkpoint and stopping
  status.

Suggested output record:

```json
{
  "step": 1200,
  "weights_kind": "ema",
  "strength": 1.0,
  "examples": 128,
  "residual_rms": 0.0,
  "residual_p95": 0.0,
  "gain_mean": 0.0,
  "gain_p95_error": 0.0,
  "orthogonal_error": 0.0,
  "teacher_swd": 0.0,
  "game_swd_sigma_1": 0.0,
  "evaluator_auc": 0.0,
  "subgroups": {}
}
```

For future training runs, execute the fixed probe immediately after every
100-step checkpoint while the base model is already loaded. For the two current
runs, run a separate sequential checkpoint sweep after creating the unseen
probe bank. Avoid running another model instance concurrently if it would
contend with the active training GPU.
