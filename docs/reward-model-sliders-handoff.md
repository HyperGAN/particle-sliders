# Handoff: reward-derived sliders, starting with Content Enjoyment

Date: 2026-09-07. This document defines the experiment. Implementation and current
execution evidence are tracked in the [research package](../conceptmod/textsliders/reward_sliders/README.md)
and [investigation record](../analysis/reward_sliders_20260907/README.md).
The specification itself does not establish a successful direction or LoRA.

## Goal and interpretation

Test whether higher-scoring music exposes a reusable direction in the Music 3
composer, then express that improvement through the **same unidirectional LoRA
slider formulation** as the current studio controls. Build the reward selection
and teacher construction as reusable additions; Audiobox Content Enjoyment (CE)
is the first reward implementation.

The proposed sequence is **scored matched takes → activation direction → causal
audio test → standard LoRA student → composition and transfer tests**. Runtime
scoring and extra candidate songs are unnecessary if the learned slider works.
The activation hook is an experimental teacher, not a replacement weight format.

Interpret “same formulation” as the existing frozen-base, Off→plus, additive
low-rank adapter and host-energy contract. Preserve the contrastive teacher/student
residual formulation when constructing training examples. The reward supplies the
positive target instead of a hand-authored positive caption. Extending the current
trainer to generation spans is real implementation work, described below; this
is not an existing command-line option or an unchanged training-data recipe.

**Expectation:** the method can be generic across reward definitions. One trained
CE slider may improve average enjoyment across styles, but that is a hypothesis.
It is not yet a universal quality direction, and it will remain specific to the
base model and calibration domain. Use the research name `reward-ce-v1`; reserve
“general quality” for demonstrated transfer with acceptable preservation.

## Existing evidence and code

| Reference | What to reuse / limitation |
|---|---|
| [Combination and energy study](../analysis/studio_combo_energy_20260907/README.md) | 30 scored clips, three pairs, two seeds, **one lyric/caption fixture**. Useful for plumbing; insufficient for generic-quality training or independent validation. |
| [Metric selection](../analysis/studio_combo_energy_20260907/metric-selection.md) | Mean CE selected settings automatically. No mandatory listening gate. |
| [Metric agreement](../analysis/studio_mergers_20260907/metric-agreement.md) | CE agrees with 13/18 dependent pairwise preferences; a fixed method ordering gets 12/18. Weak evidence for subjective universality. |
| [Creation probe](../analysis/studio_creation_probe_20260907/README.md) | Warm 10-second scorer forward: median 0.567 seconds on four CPU threads. This excludes generation, loading and preprocessing. |
| [Feature/scorer implementation](../slider_selection/features.py), [measurement wrapper](../analysis/studio_combo_energy_20260907/measure.py) | Reuse pinned models and auditable preprocessing. Keep CE separate from PQ and style diagnostics. |
| [Current palette recipe](../analysis/uni16_20260906/README.md) | Fresh rank-8/alpha-8 LM sliders; original GAN warmup then bounded continuation. The successful 660-step setting is not an established optimum for a new reward task. |
| [Teacher rows](../conceptmod/textsliders/gan_v2/data.py), [objective](../conceptmod/textsliders/gan_v2/engine.py), [export](../conceptmod/textsliders/gan_v2/train.py) | Reuse shared-history residual construction, applicable objective machinery, and ordinary LoRA export. Current main contrast is on **lyric spans**. |
| [Resolver](../../app/sliders.py), [generator](../../app/generator.py), [registry](../../app/sliders.json) | Preserve resolved energy, exact-base restoration, and full-delta merging. |

The existing WAVs do **not** include the intermediate residual activations and
complete semantic trajectories required here. Collect them during new generation,
or rerun a frozen configuration with capture. Do not assume arbitrary saved WAVs
can be converted back into the original composer states.

## Keep the slider formulation and account for energy

For projection `j`, retain:

```text
W_j = W_base,j + sum_i m_i * (alpha_i / rank_i) * B_i,j @ A_i,j
```

The candidate should use rank 8, alpha 8, `kind: language_model`,
`target_replace: [Qwen3Attention]`, `prefix: lora_te`, `delimiter: '-'`,
`train_method: full`, `unit_scale: 1.0`, and the current LoRANetwork keys. The
current palette covers 144 q/k/v/o attention projections across 36 layers.
Export an Off→plus control with the existing 0–1 fader convention. Verify actual
checkpoint keys, alpha buffers and module count against the live host before use.

At rank=alpha and unit gains, fader values `p_i` yield
`m_i = E * p_i / sum(p)`. A lone nonzero fader receives the entire LM energy;
changing its fader alone does not sweep its strength. Sweep energy for solo tests.

Run two explicitly different composition comparisons:

1. **Isolate the added adapter:** hold existing effective style multipliers fixed.
   For example, female/pop at 1.68/1.12 plus quality at 0.20 requires total E=3.0
   and shares 0.56/0.373333…/0.066667…. Compare with 1.68/1.12 at E=2.8.
   A research renderer may specify equivalent direct multipliers. Respect the
   host limit; never silently clamp or create a new independent energy budget.
2. **Test the actual studio contract:** hold total E fixed and add a quality
   share. Style strengths then decrease. Include a style-only control at those
   same reduced multipliers, so reduced style energy cannot masquerade as a
   benefit of the quality adapter. Log every resolved multiplier.

The direct activation teacher has its own dimensionless coefficient; it does
not participate in the studio's LoRA energy resolver. Fit and calibrate the LoRA
student independently. Adding a constant vector is not algebraically equivalent
to an attention projection's `B @ A` update; do not rename vector tensors as LoRA
weights or assume the teacher's strength transfers numerically.

## Reusable reward and capture interfaces

Keep research code outside the production generator initially, for example in
`conceptmod/textsliders/reward_sliders/`, with a run directory under
`analysis/reward_sliders_20260907/`. The implementation now uses these paths;
execution status and evidence are kept in the investigation record.

- `RewardSpec`: identifier, orientation, scorer/revision hashes, preprocessing,
  window rule, aggregation, valid range and invalid-result policy. Return a
  scalar reward plus separate diagnostics; never silently blend CE and PQ.
- `Observation`: exact sheet/caption hash, prompt-family split, seed, base and
  style-weight hashes, resolved multipliers, sampler settings, GPU, full audio
  hash, window scores, activation/trajectory references and timing.
- `CaptureSpec`: model/source identity, layer and hook location, residual width,
  CFG branch policy, generation-position mask, frame/time mapping, dtype and
  normalization statistics. Persist semantic codes/frame inputs needed for
  teacher forcing; a pooled feature alone cannot reconstruct a trajectory.
- `RewardTeacher`: fitted direction, training-only normalization, dev-selected
  layer/strength, and shared-history baseline/steered target construction.
- Standard LoRA student/export plus a reward provenance sidecar. Loading a
  different reward requires new fitting and validation, not just changing a label.

First spec: higher Audiobox **CE** is better; retain model revision
`9b1dd8e5df9af7216e836a98974fe3b82c56ded6` and record all actual installed source
and weight hashes. Short-screen primary score is mean CE over two disjoint
10-second windows in the first 20 seconds. Match the previous study's RMS 0.1
float-copy normalization, then the existing scorer's resampling. Preserve raw
audio and raw-level diagnostics. This normalization is an experimental protocol,
not a claim that it removes every loudness preference.

For final full songs, freeze a separate primary score: duration-weighted CE over
all nonoverlapping 10-second windows, including the actual shorter tail. Normalize
one full-song copy, not each window independently. Also report beginning, middle,
ending and raw-audio scores; do not select only the most flattering window.
Invalid, missing, silent, truncated and failed generations stay in the manifest;
report valid-pair CE and failure rates separately. Never hide failed arms by
rerolling their seed, and do not promote an arm with unresolved output failures.

## Stage 1: matched data and a small direction probe

Use existing clips only for capture/scoring smoke checks. Proposed fresh pilot:
16 distinct lyric/caption families × four seeds = 64 baseline clips, with the
split fixed before scoring: eight training, four development, four test families.
This is a feasibility pilot, not enough independent families for a generic claim.

Balance fixtures across electronic/acoustic and dense/sparse arrangements, slower
and faster time-feels, female/male/unspecified leads, instrumental cases, and
no-slider/solo/mixed settings. Include the three familiar combinations, while
using new sheets. Keep every variant of a family in the same split. Reserve new
combinations and arrangement families for the later transfer study.

Within each exact sheet/caption/style/energy cell, rank seeds by CE. Compare
high- and low-scoring takes **within that cell**, never a high-scoring pop take
with a low-scoring folk take. Pool activations over comparable audio-frame windows
and give each family equal weight. Seed differences produce different histories:
do not pretend their token positions are semantically aligned training targets.

Start with the mean of within-cell high-minus-low pooled residual differences,
normalized to unit L2 length per candidate layer. Fit normalization on training
families only. Score a held-out take by projection onto the direction and report
within-cell rank agreement. This tests association; only new steered audio tests
whether the direction causes improvement. Avoid training a large reward network
or searching many representations in this first pilot.

Capture two predeclared intermediate layers near one-third and two-thirds depth
(zero-based 11 and 23 if the inspected host has 36 layers). Define insertion at
the decoder block's residual output, before subsequent blocks, with:

```text
h_l,t <- h_l,t + s * median_training_residual_L2(l) * unit_direction_l
```

Initial dev grid: `s = 0.01, 0.03`, one layer at a time, plus off. These are
proposed perturbation sizes, not calibrated slider units or known safe strengths.
Use four dev families × two seeds × five arms, at most 40 short renders including
off. Exact matching baseline captures can be reused with verified provenance.
Select the highest mean-CE layer/strength; off is eligible. Exact ties prefer off,
then smaller strength, then lower layer index. Do not use test scores to tune.

## Stage 2: causal activation test

Freeze the chosen direction and run the four untouched test families × four seeds
with off, positive, reversed, and one predeclared norm-matched random direction:
at most 64 short renders. Use the same initial seed per comparison and preserve
all RNG states; divergent sampled histories are an expected treatment effect.
The random direction is a control, not another candidate to optimize.

Read the installed pipeline before implementing hooks:
`/home/mikkel/anaconda3/envs/minimax-music3/lib/python3.11/site-packages/diffusers/modular_pipelines/minimax_music3/encoders.py`.
Record its hash. Current semantic generation uses conditional/unconditional rows,
CFG 1.5, and a residual-code decoder before the acoustic stages.

- Initially steer both CFG rows at generation positions, with identical injection
  convention. A shared LoRA also acts on both rows. Log both branches; nonlinear
  downstream computation means this does not guarantee CFG cancels the shift.
  Conditional-only steering is a separately declared ablation, not a hidden change.
- Leave prompt/lyric positions untouched. Specify the audio-start boundary:
  the last prefill position predicts the first frame, so either steer just that
  position or explicitly exclude it. Use one policy consistently in capture,
  generation and teacher forcing; a `sequence_length == 1` test alone is inadequate.
- Steer at an intermediate block. A constant shift immediately before a linear
  LM head is just a constant vocabulary-logit bias; it does not establish the
  proposed context-dependent intervention through later blocks.
- Keep downstream semantic sampling, residual-code conditioning and stored frame
  hidden states consistent with the intervention. Account for acoustic chunk
  overlap and context when mapping a scored audio window to semantic frames;
  window scores are not exact token-level reward labels.
- Scope hooks and configuration to one job and pipeline/device. Remove hooks on
  success and exception. Zero must bypass all arithmetic and random draws, with
  baseline output parity on the same device/configuration. Check zero after a
  nonzero job too, plus concurrent GPU isolation and cancellation cleanup.

A direction that predicts CE but fails to improve fresh audio is a failed causal
probe. Reversed steering need not lower CE symmetrically, but positive steering
should beat off and the random control to support a reward-specific explanation.
Do not move a failed direction into the catalog.

## Stage 3: learn the same LoRA slider from the successful teacher

Proceed only if the activation experiment supports improvement. Freeze its
layer, normalization and coefficient. Train a fresh zero-output rank-8/alpha-8
adapter on the training families, with the original base and style adapters frozen.

For each saved history `x`, compute the baseline and positively steered teacher
on **that same history**, using identical branch and position masks. Construct:

```text
real = H_steered_teacher(x) - H_baseline(x)
fake = H_base_plus_style_plus_reward_LoRA(x) - H_baseline(x)
```

Use aligned generation spans for these residuals. This preserves the existing
contrastive slider formulation and permits reuse of RpGAN/b_cap and feature
matching machinery. Add an explicit reward-row preparation and student-forward
path; do not pass generation features through the old lyric-span masks unchanged.
The current `prepare_rows` builds its main positive-minus-neutral target on lyric
positions. A teacher that only steers generation would produce zero there.

Start from the documented bounded baseline optimizer/objective settings where
applicable; freeze the new recipe and its span geometry in the manifest. Recompute
the critic's scale calibration on the new features. Keep end behavior and prompt/
lyric preservation diagnostics explicit. Any extra KL, supervised distillation
loss or different critic is a named follow-up arm, not an undocumented repair.
Use bounded checkpoints selected on dev audio CE; 600/660 is a reference budget,
not an automatic winning checkpoint or reason to continue a broken run.

An ordinary merged LoRA also affects prompt positions and may approximate the
activation teacher imperfectly. Check prompt drift and then **free-running audio**;
low hidden-state or critic loss does not establish preserved enjoyment. Export
the normal live rank-8 adapter, not a rank-32 EMA or dense teacher disguised as
rank 8. Save full training state separately from the inference weights.

Calibrate the LoRA's effective strength on dev data with fixed style multipliers.
Then compare off, frozen activation teacher and frozen LoRA on held-out audio,
including the energy-matched controls above. Pilot test families already inspected
for the teacher cannot serve as a fresh final validation of the whole method.

## Stage 4: transfer, selection and the quality claim

Predeclare a new transfer set after pilot design decisions: at least 12 fresh
lyric/caption families, two seeds each, full intended song durations, and frozen
off/LoRA treatments. That is a minimum 48-song comparison, with extra composition
controls budgeted separately. Include unseen style combinations and no-style
baselines; apply **one fixed adapter and calibrated strength rule** throughout.
This minimum is a starting study size, not a promise of adequate statistical power.

CE selects candidates automatically. Report paired mean changes, seed win rates,
and bootstrap intervals clustered by whole lyric/caption family; windows and
multiple seeds of one sheet are not independent samples. Provisional evidence
for transfer requires positive mean full-song CE with a 95% interval above zero.
An interval overlapping zero means inconclusive; preserve off as the default.
Report family/genre/voice results even when the overall average improves.

Keep PQ, PC, CU, fixed-description style/voice similarities, lyric diagnostics,
duration, silence, clipping and output diversity visible. Freeze any numerical
preservation tolerances before the final test, based on baseline variability.
Do not silently optimize a CE/PQ mixture or require new human ratings. Listening
may help diagnose a result but is optional for numerical selection.

Distinguish the conclusions:

- **Generic mechanism:** the scorer/capture/teacher pipeline can fit a new reward
  without changing the slider format. This does not imply shared weights across
  rewards, or across base model revisions.
- **CE improvement:** frozen LoRA raises held-out CE. If it also changes desired
  style, voice, words or dynamics materially, report that tradeoff and retain the
  CE-specific name; a metric gain alone is insufficient for a quality claim.
- **Broad quality candidate:** the gain transfers across intended styles, mixes,
  voices and later song sections, with preserved intent and no output collapse.
  Even this is an average improvement in the tested domain, not a guarantee for
  every song or a universal definition of musical quality.

If only a few styles improve, report a scoped reward slider rather than quietly
introducing per-genre directions. If direct steering works but LoRA transfer
fails, report that separately; the requested same-formulation addition remains
unproven. A pure production-quality target may ultimately need the acoustic host,
which would be a separate experiment from this LM enjoyment test.

## Runtime, operation and deliverables

Measure capture-free activation overhead and merged-LoRA generation time on matched
fixtures, including job setup/merge time and peak memory. The activation path adds
a residual vector operation at each selected generation step; measure Python-hook
overhead rather than claiming zero cost. A merged LoRA keeps the normal inference
graph, with adapter preparation cost when the mix changes. A proposed runtime
target is no more than 2% median end-to-end overhead after warmup; report variability
and treat this as an engineering target, not an already measured result.

Use `/home/mikkel/anaconda3/envs/minimax-music3/bin/python` and existing caches;
never install the repository's old requirements file. Follow [AGENTS.md](../../AGENTS.md):
no real artist/band/songwriter/producer/album names in any emitted song package,
prompt, notes or sidecar. Validate authored fixtures before capture and training.
If a checkpoint's prompt provenance contains prohibited names, strip the names
and retrain; do not ship those weights.

The user previously authorized both cards for this investigation. Before executing
a future stage, inspect current ownership and studio queue state. Use physical
GPU 1 for training/rendering while the studio occupies GPU 0; use both only when
available under that authorization. Preserve studio settings, Keep mode and jobs.
Do not restart the studio just to write or inspect this handoff. The budgets above
are proposed staged ceilings, not jobs launched by this document.

Required deliverables:

1. Frozen reward spec, clean prompt families/splits, source/weight hashes and exact
   render budget before collection; reusable CPU scoring and per-job capture code.
2. Capture/zero-parity audit; matched observations, raw audio, trajectories,
   direction tensors, fit diagnostics and the full dev/test arm manifest.
3. Causal result with automatic CE selection and explicit failures. Stop before
   expensive LoRA training if no direction beats off meaningfully.
4. If supported, a normal LoRA checkpoint/sidecar, full training state, numerical
   merge/off-restoration audit, strength/composition calibration and transfer report.
5. A concise decision record: failed, inconclusive, scoped CE improvement, or broad
   quality candidate. No production registry change until an artifact and its
   evidence exist; this handoff itself makes no deployment change.

Research basis: [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
computes paired activation differences and adds them during text generation. Its
application to CE and Music 3 here is an unvalidated adaptation. [Audiobox
Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) supplies the
separate enjoyment and production-quality axes; it does not establish that one
universal composer direction controls them all.
