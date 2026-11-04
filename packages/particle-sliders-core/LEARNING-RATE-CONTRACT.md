# Model-specific learning-rate configuration and optimization

The common configuration uses three optional, finite positive numeric fields:

| Field | Meaning |
| --- | --- |
| `lr` | Generator/network base optimizer-group rate. |
| `d_lr_mult` | Critic rate divided by `lr`. |
| `prior_lr_mult` | Participating particle-bank rate divided by `lr`. |

Omission preserves the product's documented preset/default. Do not give every
model the same numerical rate merely to standardize its interface. Adapter-only
products have no bank and reject a supplied `prior_lr_mult`. A Boolean is not a
numeric rate. All three fields, their resolved recipe and optimizer groups bind
the training checkpoint; a changed rate starts a new experiment rather than
silently resuming an old optimizer history.

Products pass these fields through `ConditionalPG09(..., recipe_overrides=...)`.
`build_fibo_game` also accepts that pass-through. The game, optimizer and schedule
remain owned by upstream ParticleGAN. API, common session configuration and any
product CLI must select the same defaults. Product CLIs use `--lr`,
`--d-lr-mult`, and, for a real bank, `--prior-lr-mult`; the common dispatcher
receives them in the product's JSON `options`. Existing documented aliases such
as Qwen Image's `g_lr` remain explicit aliases, with conflicting values refused.
Adoption is recorded per product; this contract does not activate missing native
model operations or make old source-bound packets portable.

For official PG0.9.4 `bispf_optimized`, the nominal defaults are G `0.012`,
D `0.018` (`1.5x`), and bank `0.03` (`2.5x`). Both rate floors are one, so the
effective rates are constant despite the cosine field. These are normalized
DualNorm step sizes, not directly comparable with historical Adam/AdamW rates.
Explicit upstream `bispf` has different defaults. Record the actual preset,
optimizer family and resolved group rates in every trial.

## Search and selection

Follow [the evaluation and search protocol](EVALUATION-AND-LR-SEARCH.md).
Prepare a new source/model/data cohort and verify its metric controls before
searching. Freeze initial tensors, data splits, seeds, batches, reference noise,
numerical settings and total schedule horizon across candidates. Keep training,
development selection and final confirmation examples separate.

A base-rate screen with fixed multipliers changes G, D and the bank together:
label it a **coupled rate screen** and record all three absolute rates. For a
G-only screen at fixed critic rate `D` and bank rate `P`, each candidate uses
`d_lr_mult=D/lr` and `prior_lr_mult=P/lr`. Coordinate screens can then vary D or
P while holding the other absolute rates fixed. There is no universal selected
rate for image, audio, text, routed-bank and direct-LoRA architectures.

Compare candidates at equal completed updates, using held-out teacher/student
fit and the product's actual target law. Retain additive error/energy statistics,
per-example results, update ratios, finite-state checks, peak memory and wall
time. GAN training loss alone does not rank checkpoints. Report failures and
unscoreable evaluations as such, never as zero error. Extend finalists to a
common longer budget and additional seeds; inspect matched native outputs,
OFF identity and strength controls. Prove full-state continuation and portable
inference before calling a candidate a qualified continuation rate.

Every recommendation names exact product/core/PG/model sources, optimizer and
precision, concept/data/evaluator identity, G/D/bank rates, comparison budget,
all trial outcomes, held-out confirmation, checkpoint hashes and resume proof.
Use `provisional` until the declared confirmation criteria pass. A CPU fixture,
an inherited default, an old MSE result, or a different native source cohort is
not a newly tuned current-model rate.

GPU jobs use shared physical-device admission, a distinct workload lock, a
measured process memory cap and headroom for existing workloads. Recheck actual
capacity before launch; terminate only the owned attempt on a resource failure.
Keep native model caches and earlier experiment outputs immutable.
