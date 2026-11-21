# Native distillation comparisons

Qwen3.5 already evaluates the frozen base, a routed particle teacher, and its
ordinary LoRA student on GSM8K. The historical release reports that all four
teacher responses hit the generation cap. The release includes the matched
four-case student subset and a separate sixteen-case baseline and distilled
student comparison. Those longer summaries mix different case sets with the
teacher probe, and the base and distilled answers also hit the cap. The
generation records include decoded text, token counts, and truncation flags.
They lack emitted token IDs and paired teacher and student distributions on
the same prefixes. They can support finished-answer audits on completed
matched cases. Token overlap and on-policy effects remain unmeasured.
[Historical Qwen3.5 release evidence](https://github.com/HyperGAN/qwen35-particle-sliders/blob/aea38ab88505558c2165193ecdba6dc4cc13beef/evidence/final-experiment.json)

The current [response lifecycle](../../packages/particle-sliders-core/src/particle_sliders/response.py)
provides numerical responses and preserves exact checkpoint identity. The
[autoregressive adapter](../../packages/particle-sliders-core/src/particle_sliders/audio.py)
binds positive and neutral prompts to the same native continuation. Glimmer
uses that boundary for final hidden states. Its response session has no token
sampler or next-token distillation objective. Use a model-specific runner
for the comparisons below.

## Record the comparisons

1. Freeze the model and tokenizer revisions, RAW teacher checkpoint, student
   initialization, held-out prompts, seeds, decoding settings, and budgets
   before fitting. Retain EOS and truncation as separate outcomes. Record
   generated IDs directly from the sampler. The tokenizer may normalize
   decoded text, so tokenizing that text again cannot recover the exact trace.
2. For each training response, record the exact prompt and rollout-prefix IDs,
   source arm, candidate and checkpoint identity, step, vocabulary, and valid
   token mask. Compare the teacher and student on identical rollout histories.
   For distribution losses and KL diagnostics, retain full native logits or a
   verified exact reduction covering the full valid vocabulary, including stop
   tokens. A selected vocabulary changes the loss being measured.
3. Keep matched fresh-student, frozen-initial-student, and teacher-rollout arms
   separate. Predeclare the initial token-overlap statistic and its strata.
   For each arm log distinct generated, evaluated, and supervised token counts,
   model calls, wall time, and actual completed updates. A frozen-only run can
   measure fit on that support, while a refreshed-vs-frozen comparison requires
   both arms.
4. Reload RAW student weights through the product's ordinary adapter loader.
   Verify OFF and exactly one ON application. Keep training checkpoints and
   evaluation firsts. Rank by the equal-case mean finished-answer score ON
   minus its matched OFF. Retain ties and report token overlap, loss, and
   response SSE as diagnostics. Qualified catastrophic failures stop the
   affected candidate before another update, resume, or selection. Any
   uncertain or missing assessment is unmeasured.

## Product experiments

| Paper PR | Comparison in the product | Required evidence |
| --- | --- | --- |
| [#210](https://github.com/HyperGAN/particle-sliders/pull/210), [#217](https://github.com/HyperGAN/particle-sliders/pull/217) | Hold Qwen rollout histories fixed. Compare a direction derived from verified correct within-budget continuations, an unsteered teacher, the same reasoning-mode prompt without reference solutions, and worked-solution conditioning. Compare against reversed or shuffled directions at the same scale. | Disjoint calibration and evaluation cohorts, verification of answer correctness and stop reason, and an explicit record of which reference each teacher saw. |
| [#218](https://github.com/HyperGAN/particle-sliders/pull/218), [#226](https://github.com/HyperGAN/particle-sliders/pull/226) | Validate the paired-history boundary first. Then compare perturbing the student prompt while holding the frozen teacher reference and history fixed, perturbing the teacher reference alone, and untouched controls. Compare action-sequence selection only on a task with genuine alternate successful trajectories. | Verified transformations and stable selected reference hashes. Glimmer's matched history checks provide the admission invariant. A shared suffix alone says nothing about which reference improves a student. |
| [#223](https://github.com/HyperGAN/particle-sliders/pull/223), [#227](https://github.com/HyperGAN/particle-sliders/pull/227) | Compare dense supervision with divergence-masked positions and stochastic horizons in separate arms. Include a fixed short horizon matched by supervised positions. | Same native rollout budget and exact accounting of scored positions, teacher queries, and updates. A teacher call over a prefix can count as generation cost even when most positions are masked. |
| [#246](https://github.com/HyperGAN/particle-sliders/pull/246), [#249](https://github.com/HyperGAN/particle-sliders/pull/249) | Compare refreshed and frozen student rollouts by initial overlap. In a separate comparison hold the rollout protocol fixed while varying tokenwise versus sequence-level objectives and divergence direction. | A native runner that implements the objectives and retains the required probabilities. Record prefix depth, position-level signal and final effect. The original #246 reversal remains specific to its toy. |
| [#241](https://github.com/HyperGAN/particle-sliders/pull/241) | Compare matched full-distribution supervision and sampled-label supervision at the same number of native carriers. | Exact aligned vocabularies and independent carrier manifests. Report spectral errors as diagnostics. The activation covariance used by routed ridge is a different object from the token likelihood Fisher. |
| [#213](https://github.com/HyperGAN/particle-sliders/pull/213), [#240](https://github.com/HyperGAN/particle-sliders/pull/240), [#242](https://github.com/HyperGAN/particle-sliders/pull/242), [#256](https://github.com/HyperGAN/particle-sliders/pull/256) | Compare observed full native edits, projected direction or calibration-mean constant, and the existing fitted ordinary LoRA. For weight-to-activation translation, measure at several fixed strengths on the same frozen host and downstream response. | Exact native layer and schedule, separate calibration and held-out prompts, and matched OFF/ON outputs through the same native loader. The current [ridge fit](../../packages/particle-sliders-core/src/particle_sliders/distillation.py) and [ordinary loader](../../packages/particle-sliders-core/src/particle_sliders/native_lora.py) support the fitted arm. A fit residual or constant share is a diagnostic. |
| [#230](https://github.com/HyperGAN/particle-sliders/pull/230) | Compare explicit adapter application against a separately merged export and its clean reload. Only the active, unmerged form can be used by the [named PEFT scope](../../packages/particle-sliders-core/src/particle_sliders/peft_scaling.py). | Base revision, factor identity, independent OFF and ON, and a fresh host before any reapplication. This covers deployment integrity. Hook-guided training and trajectory-policy gap require separate native experiments. |
