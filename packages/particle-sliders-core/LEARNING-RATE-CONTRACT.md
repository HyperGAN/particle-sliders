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

Use the [evaluation protocol](EVALUATION-AND-LR-SEARCH.md) and
[search API](PARAMETER-SEARCH.md). Relative teacher/student error is the accepted
default proxy; output-effect measurement is optional. Keep each declared search's
metric and completed results, compare equal completed budgets, and reuse saved
raw checkpoints rather than rerun completed training.
