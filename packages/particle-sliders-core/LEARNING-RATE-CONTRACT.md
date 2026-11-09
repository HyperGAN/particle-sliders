# Learning-rate fields

Products pass finite positive rate values through
`ConditionalPG09(..., recipe_overrides=...)`:

| Field | Meaning |
|---|---|
| `lr` | Generator/network base optimizer-group rate |
| `d_lr_mult` | Critic rate divided by `lr` |
| `prior_lr_mult` | Participating particle-bank rate divided by `lr` |

Omission retains the product's documented preset. Adapter-only products have no
bank and reject `prior_lr_mult`. A Boolean is not a numeric rate. The resolved
recipe and optimizer groups bind the checkpoint; changing rates is a new
experiment, not an exact continuation of the old optimizer history.

The common API standardizes meanings, not numerical defaults. Product CLI flags
and JSON options must resolve to the same recipe. An explicit alias can be kept
for compatibility, with conflicting values refused. Consult the
[product repository](../../docs/products.md) for its actual chosen rates and
upstream optimizer family. Normalized ParticleGAN step sizes are not directly
comparable with historical Adam/AdamW rates.

## Coupled and independent screens

Varying `lr` with fixed multipliers changes generator, critic and bank together.
Label it a **coupled rate screen** and record every absolute rate.

To vary only generator rate while holding critic rate `D` and bank rate `P`,
set `d_lr_mult=D/lr` and, when a bank exists, `prior_lr_mult=P/lr` for each
candidate. Independent screens vary actual optimizer roles; an absent bank or
optimizer is not an advertised parameter.

Use the [finished-output evaluation workflow](EVALUATION-AND-LR-SEARCH.md).
Select only the equal-case mean intended-effect intensity ON minus matched OFF;
retain negative gains, zeros and all tied maxima. Relative teacher/student
error is diagnostic only. Reuse completed searches, saved RAW checkpoints,
outputs and first measurements without additional fresh confirmation. Preserve
historical metrics/results under their original labels.

The [historical search runner](PARAMETER-SEARCH.md) still implements fixed SSE
ranking and its original finalist protocol; it has no output-effect scorer
callback. It is not the current finished-output selector. Use each product's
actual supported inference/configuration route and frozen output instrument.
Copy exact saved rate multipliers rather than recomputing them from rounded
displayed rates when reusing a measured recipe or continuing its original state.
