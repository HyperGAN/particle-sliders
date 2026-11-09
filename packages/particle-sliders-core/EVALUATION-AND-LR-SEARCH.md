# Evaluation and learning-rate search

Choose one declared ranking metric and keep it fixed across candidates. Current
shared search uses raw teacher/student relative error as a proxy for the learned
effect. Model usage and actual tuned recipes belong in the
[product repositories](../../docs/products.md).

## Default proxy

For example $i$, compare student $S_i$, frozen target $T_i$ and frozen neutral
baseline $N_i$ in the same native response coordinates:

$$
E_i = \frac{\sum_k(S_{i,k}-T_{i,k})^2}{\sum_k(T_{i,k}-N_{i,k})^2},
\qquad E=\operatorname{mean}_i E_i.
$$

Pool all declared positions/coordinates within each example before dividing;
then weight examples equally. Convert original operands to CPU FP64 before
subtraction, not a previously rounded difference. The search implementation uses
chunk-independent accumulation. Lower finite scores rank first. A value above
one is valid and worse than the neutral baseline, not a reason to stop training.
Zero teacher-edit energy makes that example undefined; it is not zero error.
The frozen scoring grid must not change to hide student failures.

The target is the backend's actual training target, including guided/signed
construction. Critic loss, preservation drift and media beauty are separate
diagnostics. They are not additional ranking gates or tiebreakers.

## Optional output-effect objective

A fixed instrument can measure intended effect ON minus OFF in matched finished
outputs, averaged equally across cases. This may improve selection when response
error is an imperfect proxy for what users perceive. It is optional: existing
proxy searches and saved results remain usable without a new render,
confirmation pass or training run.

Declare an output-effect scorer, matched cases and ranking direction as a
separate explicit protocol; do not mix it with the proxy or relabel historical
scores. The current [search runner](PARAMETER-SEARCH.md) implements its fixed
relative-error protocol, including its declared finalist checks. Those checks
are part of that runner, not a universal requirement for using an existing
completed search.

## Compare actual experiments fairly

Freeze model/source/data identity, target capture, numerical settings, splits,
seeds, initialization and schedule horizon. Compare candidates at equal actual
completed updates within that protocol. Record absolute generator, critic and
participating-bank rates, not just a nominal `lr`; see
[rate fields](LEARNING-RATE-CONTRACT.md).

Keep development selection and any declared heldout assessment separate.
Observation must restore training RNG, modes, buffers and native state. Report
all outcomes and undefined/nonfinite scores instead of quietly dropping rows.
Do not call an inherited default or CPU fixture result a newly tuned native rate.

## Reuse evidence

Reuse completed searches. Render or score saved raw checkpoints for a missing
measurement. Resume unfinished declared work from its exact full state without
replaying completed updates. Changed sources, recipes or model/data identities
start a new cohort; they do not justify rewriting checkpoint signatures.

Retain full checkpoints for continuation, raw exports for inference, original
response operands, actual step counts, all candidate results and the declared
metric. A recommendation should name its product/core/ParticleGAN source pins,
model/data identity, rates, budget, score and measurement limits. A software
fixture establishes orchestration behavior, not model quality.

All model weights used for selection are raw. Fresh configs use
`ema_decay=None`; adapter, critic and particle-bank EMA are disabled.
