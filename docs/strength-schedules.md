# Scheduled slider strength

Use scheduled strength when qualifying a slider for a multi-step generator.
Keep the RAW adapter fixed and vary its contribution during generation. Each
product uses the schedule that wins on that product, slider and native sampler.
Keep constant strength as the comparison and as the fallback for profiles with
no qualified schedule winner.

The product comparisons here test fixed schedules. Their gains are computed
from the declared curve and native sampler, then evaluated on finished
outputs. The cited token-steering papers use activations to decide where or
how strongly to intervene during generation.

The [product map](products.md) identifies the generation stage for each public
product. Use the generation stage and scheduler from its native source cohort.

## Search alpha on a saved adapter

### Jev command

```sh
particle-slider alpha-tune path/to/slider-definition.json
# The standalone alpha-tune command is installed with the same package.
alpha-tune path/to/slider-definition.json
```

Run this in the native product's inference environment. The definition pins a
saved RAW adapter and the case images. The command opens one registered native
renderer, searches the declared strengths, and saves
`slider-definition.alpha-tune/tuning.json` beside the definition. The receipt
contains the checkpoint hash, nominal alpha, timing profile, sampler identity
and resolved gains for replay.

The command uses Jev through `NANOGPT_API_KEY`. It scores the concept in each
image independently and measures an identical OFF image once. A separate
Jev check screens visible rendering defects and severe replacement of the
original subject or setting. A severe-defect probability of 0.8 or higher
rejects a setting. Probabilities from 0.5 up to 0.8 trigger a focused Jev
clarification and stay unmeasured if the clarification is uncertain.
An ambiguous anatomy flag can be resolved with questions about visible
detached, fused, or extra parts.
Within the settings that pass, the comparison
uses mean concept gain. A definition may include known passing and failing
image pairs to check this rubric. Reserved cases can validate the chosen
setting. A failed or uncertain run leaves its evidence and produces no tuning
file.

A definition uses schema `particle-slider-alpha-tune-v1` and includes these
fields.

| Field | Value |
| --- | --- |
| `selection`, `ema_decay` | `"raw"` and `null` |
| `checkpoint` | `{"path", "bytes", "sha256"}` for the saved adapter |
| `renderer`, `renderer_options` | Installed renderer entry point and native options |
| `concept` | What Jev should score in a finished image |
| `profiles` | A map from `constant`, `early`, or `late` to the declared native alpha grid |
| `refinement_rounds` | Number of local refinement rounds, one by default |
| `cases` | Rows with `id`, `split` of `search` or `reserved`, native `request`, descriptive `preserve` text, and pinned `off` image |
| `judge_fixtures` | Optional rows with `id`, pinned `off` and `on`, `preserve`, and `expected` verdict |

Relative file paths resolve beside the definition. Resuming the same definition
reuses committed renders and Jev receipts. If the definition changes, use a
new filename for the new run.

Products register a context-managed factory under
`particle_sliders.alpha_renderers`. The factory receives the checkpoint
`FileRecord` and native options. The renderer supplies `sampler` metadata,
`interval_widths`, and `render(request, effective_strengths, output)`. It
writes the requested finished PNG and returns its `FileRecord`. The product
owns adapter loading, native strength limits, and accelerator resources.

### Callback API

Run `particle_sliders.alpha_search.search_alpha` after exporting a RAW adapter.
Bind the checkpoint, a fixed concept rubric, and matched cases with the same
prompts, seeds and OFF outputs. Declare a coarse grid of valid native alphas and
a refinement budget. The search measures the grid and then refines near the
best measured point. An effect can peak and then disappear as alpha rises,
so binary search for a monotone threshold answers a different question.

The scoring callback returns one `AlphaObservation` per case. A product can
render its ON output at the requested alpha and ask a fixed Jev rubric to score
the blinded OFF/ON pair. Pass the qualified integrity verdict separately.
Missing or uncertain judgments stay unmeasured. A candidate with a qualified
catastrophic failure stays in the trial record and is ineligible. Mean
intended-effect gain is the sole ranking score. Exact ties remain available.

```python
from particle_sliders.alpha_search import AlphaObservation, search_alpha
from particle_sliders.candidate_failure import FileRecord
from particle_sliders.strength_schedule import interval_strengths

checkpoint = FileRecord.from_path(raw_path)

def evaluate(alpha):
    gains = interval_strengths(alpha, native_interval_widths, profile="early")
    observations = {}
    for case in cases:
        # Product-owned callback. Reuse exact saved renders and judge receipts.
        off, on, verdict = score_finished_pair(checkpoint, case, gains)
        observations[case.id] = AlphaObservation(off=off, on=on, integrity=verdict)
    return observations

result = search_alpha(
    checkpoint=checkpoint,
    case_ids=[case.id for case in cases],
    alphas=(0.5, 1.0, 1.5, 2.0),
    evaluate=evaluate,
    refinement_rounds=1,
)
```

Save the chosen alpha, resolved per-step gains and exact native sampler for
replay. To calibrate timing as well as alpha, use the same cases for a
constant-strength comparison and for the desired timed profile. Respect the
product's allowed range for each profile. Keep any follow-up panel separate
from the cases used to choose the setting.

## Product entry points

Pass gains to the entry point for the saved RAW session. Most `step_gains`
arguments multiply the nominal strength. VoxCPM2's native EOS helper takes
effective strengths directly. The linked sources define gain count, limits
and native ownership requirements.

| Inventory product | Entry point and coordinate | Source |
| --- | --- | --- |
| Anima | `python -m release_tools.infer --step-gains` for native Euler steps | [Render guide](https://github.com/HyperGAN/anima-particle-sliders/blob/main/docs/scheduled-strength.md) |
| Arete | `arete_particle_sliders.scheduled_text.generate` or `arete_native_scheduling.generate` with `step_gains` for prefill and cached positions | [Text helper](https://github.com/HyperGAN/arete-particle-sliders/blob/main/src/arete_particle_sliders/scheduled_text.py), [strict RAW helper](https://github.com/HyperGAN/arete-particle-sliders/blob/main/native-candidate/src/arete_native_scheduling/__init__.py) |
| Flux | `ps_flux_native.scheduled_render.render` with `step_gains` for native flow steps | [Renderer](https://github.com/HyperGAN/flux-particle-sliders/blob/main/native-candidate/src/ps_flux_native/scheduled_render.py) |
| GLiDE | Scalar strength on the native two-choice readout, with one scoring forward | [Native scorer](https://github.com/HyperGAN/glide-particle-sliders/blob/main/src/glide_particle_sliders/native.py) |
| Glimmer | `glimmer_scheduling.generate` with `step_gains` for cached or full-prefix token positions | [Strict RAW helper](https://github.com/HyperGAN/glimmer-particle-sliders/blob/main/native-candidate/src/glimmer_scheduling/__init__.py) |
| Iris | `iris-slider-infer --step-gains` for native solver intervals | [CLI](https://github.com/HyperGAN/iris-particle-sliders/blob/main/src/iris_sliders/infer.py) |
| Klein | `klein_sliders.scheduled_render.render` with `step_gains` for native Euler steps | [Renderer](https://github.com/HyperGAN/klein-particle-sliders/blob/main/src/klein_sliders/scheduled_render.py) |
| Krea2 | `krea2.scheduled_render.render` with `step_gains` for native denoising steps | [Renderer](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/krea2/scheduled_render.py) |
| LTX 2.5 | `ps_ltx25_native.scheduled_render.render` with `step_gains` for native coupled audiovisual denoising | [Renderer](https://github.com/HyperGAN/ltx25-particle-sliders/blob/main/backends/ltx25-native/src/ps_ltx25_native/scheduled_render.py) |
| Ming | `ming.scheduled_render.render` with `step_gains` for shared denoising steps across both ordered image frames | [Renderer](https://github.com/HyperGAN/ming-particle-sliders/blob/main/ming/scheduled_render.py) |
| MiniMax H3 | `minimax_h3_scheduling.render_latents` with `step_gains` for coupled video and audio latent steps | [Native helper](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/main/native/src/minimax_h3_scheduling/__init__.py) |
| MiniMax H3 Image | `minimax_h3_image_sliders.scheduled_native.denoise_with_gains` with `step_gains` for the six native PDD heads | [Native helper](https://github.com/HyperGAN/minimax-h3-image-particle-sliders/blob/main/src/minimax_h3_image_sliders/scheduled_native.py) |
| Music3 | `music3.scheduled_semantic.generate` with `step_gains` for prefill and semantic frames before the original acoustic decoder | [Semantic sampler](https://github.com/HyperGAN/music3-particle-sliders/blob/main/music3/scheduled_semantic.py) |
| P-H3 Image | `ph3_native_scoped.scheduling.rollout` with `step_gains` for packed image and audio latent denoising | [Native helper](https://github.com/HyperGAN/p-h3-image-particle-sliders/blob/main/backends/ph3-native-scoped/src/ph3_native_scoped/scheduling/__init__.py) |
| Qwen Image | `qwen_image_sliders.scheduled_inference.render` with `step_gains` for native flow steps | [Renderer](https://github.com/HyperGAN/qwen-image-particle-sliders/blob/main/src/qwen_image_sliders/scheduled_inference.py) |
| Qwen3.5 | `qwen35-slider evaluate-math --step-gains` for prefill and cached token positions | [Evaluation CLI](https://github.com/HyperGAN/qwen35-particle-sliders/blob/main/src/qwen35_sliders/cli.py) |
| Ripple | `ripple_native_public.scheduled_text.generate` with `step_gains` for autoregressive text positions | [Native helper](https://github.com/HyperGAN/ripple-particle-sliders/blob/main/backends/ripple-native/src/ripple_native_public/scheduled_text.py) |
| SD | Retired Stable Diffusion scaffold with no native renderer | [Repository status](https://github.com/HyperGAN/sd-particle-sliders/blob/main/README.md) |
| Sophon | `sophon_particle_sliders.scheduled_text.render` with `step_gains` for tagged text, or `sophon_speech.scheduled_audio.render` with `frame_gains` for speech | [Tagged text](https://github.com/HyperGAN/sophon-particle-sliders/blob/main/src/sophon_particle_sliders/scheduled_text.py), [speech](https://github.com/HyperGAN/sophon-particle-sliders/blob/main/src/sophon_speech/scheduled_audio.py) |
| Supra | `supra.scheduled_inference.render` with `step_gains` for native denoising steps | [Renderer](https://github.com/HyperGAN/supra-particle-sliders/blob/main/supra/scheduled_inference.py) |
| Tiny Aya | `evidence_dial.portable.Portable.generate` with `step_gains` for prefill and cached grounding tokens | [Grounding API](https://github.com/HyperGAN/tiny-aya-particle-sliders/blob/main/src/evidence_dial/portable.py) |
| VoxCPM2 | `voxcpm2_native.frame_strength.eos_patch_schedule` for effective strengths through native EOS, or `frame_gains` for fixed-length waveforms | [Patch schedule](https://github.com/HyperGAN/voxcpm2-particle-sliders/blob/main/backends/voxcpm2-native/src/voxcpm2_native/frame_strength.py) |
| YuE2 | `yue2_sliders.scheduled_semantic.generate` with `step_gains` for native semantic tokens | [Semantic sampler](https://github.com/HyperGAN/yue2-particle-sliders/blob/main/yue2_sliders/scheduled_semantic.py) |
| Z-Image | `embedded_zimage_shared.scheduled_render.render` with `step_gains` for native denoising steps | [Renderer](https://github.com/HyperGAN/zimage-particle-sliders/blob/main/src/embedded_zimage_shared/scheduled_render.py) |

## Choose the coordinate

| Native operation | Where to apply strength | Starting comparison |
| --- | --- | --- |
| Image, video or audio diffusion and flow | Use the native denoising trajectory. Scope the multiplier to the adapter's denoiser calls. | Compare constant, stronger early and stronger late using the same native noise intervals. |
| Autoregressive text, codec tokens or music planning | Use the current generation position or a causal feature from the current activation. Apply at the adapter residual. | Compare constant, a position-based control and a causal gate. Shuffle the applied gains for an allocation control with the same budget. |
| Fixed multi-pass inference | Use declared pass or refinement positions when the adapter actually participates on those passes. | Compare constant and a redistribution across equal native passes. |
| A single adapted forward, hidden-state probe or one-shot renderer | Use the product's scalar strength. A sweep provides the comparison for any future multi-step path. | Compare matched OFF and scalar ON. |

In a hybrid product, identify which stage owns the adapter. Scheduling a
denoiser cannot steer an adapter that only runs in a text or codec generator.
For autoregressive generation, use inputs available at the current position
and honor KV-cache and full-prefix refresh semantics. A static sequence
envelope is a position control. A learned token gate requires a calibrated
activation signal that predicts where the edit helps on held-out examples.

## Reproduce the early and late denoising comparison

Let `s` be the user's signed strength. Use the sampler's actual positive
interval widths `w_i = abs(sigma[i+1] - sigma[i])`, or the equivalent native
integration coordinate. Index the intervals in execution order. For each
interval, let `f_i` be the fraction of that interval lying in the first half
of the total traversed distance. A step that straddles halfway receives a
fractional gain.

The midpoint split and gain bounds of `0.5` and `1.5` are experiment settings.
Their average over the two equal-distance halves is `1`, which preserves the
constant profile's total dose. A gain between the bounds comes from the
fraction of an interval on each side of the midpoint. Record the bounds and
split with the sampler's sigma grid so the schedule can be regenerated.

For a measured fixed-direction edit, pass the resolved coefficients and
native endpoint coordinates to `match_binary32_interval_dose` or
`match_binary64_interval_dose` for the coefficient precision used by the
native hook. Both return the nearest representable constant with ties
rounded to even, plus both exact signed coefficient-space doses. For
two equal native passes, use endpoints `(0, 1, 2)`. Check equality of
the returned doses when exact matching is required. Record rounding in
the native tensor edit when comparing finished outputs.

When a constant control ends at a different native position, pass its own
endpoints as `constant_coordinates`. For example, a token gate that fires
once over five cached predictions has dose 1. A constant on four cached
predictions then needs strength `0.25`. A constant on three needs the
nearest representable value of `1/3` and has a nonzero dose residual.
After generation, use the control's observed endpoints and applied
coefficients to verify its dose. Compare only the eligible intervals.

To verify two measured schedules, call `interval_dose` on each schedule
with its native endpoint coordinates and the measured `coefficient_format`.
It returns the exact signed dose as a `Fraction`. The `binary32` mode
requires coefficients already resolved at that precision. For a sampler
with multiple clocks, compare the two schedules separately on each clock
using the recorded endpoints from matching runs.

| Profile | Effective strength per interval |
| --- | --- |
| Constant | `s` |
| Stronger early | `s * (0.5 + f_i)` |
| Stronger late | `s * (1.5 - f_i)` |

All three have the same nominal dose `sum(w_i * abs(strength_i))`. The
realized effects can differ because the adapter residual and latent state
change throughout generation. Use actual intervals rather than the number of
denoiser calls. Keep solver sub-evaluations in their native interval, retain
the product's original conditional/unconditional routing, and validate all
effective strengths against the product's supported range before sampling.
For a maximum supported absolute strength of 1, for example, choose a
comparison strength at or below `1 / 1.5` for these profiles.

In core source, compute the three candidate profiles with
`interval_strengths` and pass each result to the native sampler:

```python
from particle_sliders.strength_schedule import interval_strengths

# Use the actual integration intervals in the native execution order.
deltas = (0.4, 0.2, 0.4)
for profile in ("constant", "early", "late"):
    gains = interval_strengths(1.0, deltas, profile=profile)
    print(profile, gains)
```

The [native inference contract](../packages/particle-sliders-core/NATIVE-INFERENCE.md)
shows where to apply these gains.

Anima Turbo Candlelit RAW256 provides an initial native result. With the same
ten-step sampler and six matched prompt–seed pairs, mean finished-image
Candlelit gain was **+0.00609** stronger early, **+0.00379** constant and
**+0.00186** stronger late. Early beat constant on four cases. The constant
replay matched its archived PNG byte for byte. All 20 new images passed a
separate catastrophic-output check. This supports the early profile for this
tested configuration. Other sliders and samplers need their own comparison.
The measured panel used cached prompt embeddings. Use the original
conditioning and numerical profile when replaying the exact benchmark images.
[Per-case evidence](evidence/anima-candlelit-strength-schedule-20261121.json)

## Qualify and use a schedule

1. Start from the saved eligible RAW adapter and the product's native sampler.
   Hold prompts, seeds or starting noise, references, preprocessing, scheduler,
   guidance and output geometry fixed. Use the native OFF path for each case.
   Replay a known constant render when introducing a new integration.
2. Render each candidate on the same cases. For continued deployment, compare
   the scheduled winner with the best constant from a small fixed-strength
   sweep and verify on additional cases. Record the checkpoint, scheduler,
   ordered intervals or positions, resolved gains, and resulting outputs.
3. Use the mean of finished-output intended-effect intensity ON minus matched
   OFF, giving each case equal weight. Retain ties, negative gains and the
   separate qualified catastrophic-output verdicts. A qualified catastrophic
   failure rules out the affected configuration. Teacher error and residual
   norms can help explain results.
4. Use the qualified schedule for that profile's inference, including examples
   and replay. Record it alongside the scalar strength and RAW identity. Reuse
   saved weights and completed learning-rate evidence.

When native observers have already scored the finished outputs, use the same
reducer as alpha search. Supply one `(case_id, off_score, on_score)` tuple for
each case in the registered roster. Record the categorical integrity verdicts
separately and apply the same reducer again with the matched constant's score
in the baseline position to measure the timing comparison.

```python
from particle_sliders.alpha_search import matched_effect_mean

gain = matched_effect_mean(case_ids=registered_case_ids, observations=matched_scores)
```

## Activation-dependent steering

### CFP and CLABS

*Right Token, Right Time* keeps the edit strength `alpha` fixed and calibrates
where the edit fires. Fit `v` with `contrastive_sae_direction` from target and
neutral corpora. The inputs are token-averaged SAE codes for each example and
the decoder matrix from the selected residual layer. The paper ranks features
by the signed difference of corpus means and normalizes the sum of the top
`k` decoder rows. `activation_projection` takes the dot product of the
current, pre-edit activation with `sign(alpha) * v`.
[Equations 1–6](https://arxiv.org/html/2611.18274v1#S3)

- CFP fires when alignment is at or below a threshold measured on calibration
  trajectories with steering active.
- CLABS standardizes alignment with calibration mean and standard deviation.
  Its error is `setpoint - standardized_alignment`. It updates the running
  error with the paper's exponential weighting before computing the clipped
  probability `bias + proportional_gain * error + integral_gain * running_error`.
  It samples one Bernoulli gate. Calibrate the bias with steering active.
- The matched random control fires independently at the same target rate.
  Log the realized fraction for every arm. For a sampler with unequal spans or
  variable edit norms, also measure the interval-weighted or residual dose.

```python
from particle_sliders.adaptive_steering import (
    ActivationGate, GateUsage, activation_projection, calibrate_gate_budget,
    steer_residual,
)


def run_cfp(case, threshold):
    gate = ActivationGate("cfp", threshold)

    def edit_current_residual(h):
        score = activation_projection(h, direction, strength=alpha)
        decision = gate.step(score)
        return steer_residual(
            h, direction, strength=alpha, gate=decision.fired,
            preserve_norm=preserve_norm, projection_bounds=projection_bounds,
        )

    # The native generator calls the hook once for each eligible new position.
    output = native_generate(case, edit_current_residual)
    return output, gate.usage


def active_rollout(threshold):
    usage = [run_cfp(case, threshold)[1] for case in calibration_cases]
    return GateUsage(sum(u.fired for u in usage), sum(u.eligible for u in usage))


calibrated = calibrate_gate_budget(
    active_rollout, target_rate=budget, bounds=threshold_bracket,
    tolerance=declared_tolerance, max_iterations=calibration_run_limit,
)
held_out_output, usage = run_cfp(held_out_case, calibrated.parameter)
```

For CLABS, instantiate `ActivationGate("clabs", bias, seed=gate_seed,
clabs=parameters)` with `CLABSParameters` from the same calibration protocol
and bisect the bias instead of the threshold. Reuse calibration cases and
seeds across candidate parameters. Use fresh cases and seeds for evaluation.
For an exact count tolerance, pass both the target and tolerance as
`fractions.Fraction` values. A 20% target with one allowed firing of
tolerance over 300 eligible positions uses `Fraction(1, 5)` and
`Fraction(1, 300)`. Measured counts from 59 through 61 qualify, including
both endpoints. Float arguments retain binary64 comparisons.
To resume after an interrupted lower endpoint, supply `lower_usage_bound`
with the largest possible firing count and exact eligible count for that
endpoint. The entire range must fall below the accepted rate interval.
Every setting returned by calibration has a measured firing count.
If the observations cannot match the declared tolerance at the measured
resolution, leave that budget unqualified. The paper reports `k=10`,
`epsilon=1e-6`, setpoint `0.75`, and gains `(1.0, 0.2, 0.2)` for its tested
setups. Its shared edit rescales to the pre-edit norm and then clamps the
projection onto `v` to `[0, 40]`. It selects `alpha` on the calibration
split with a fluency constraint. Record the configuration and the data behind
any settings used on a new product. [Paper calibration protocol](https://arxiv.org/html/2611.18274v1#A2)

Each generation request owns its gate. Context-only prefill, padding, unadapted CFG
branches, repeated prefix positions, and repeated solver evaluations do not
advance the controller. Full-prefix decoding reapplies recorded decisions at
old positions and decides only the new position. A non-token product must
define its native event, how it observes pre-intervention alignment, and how
it maps the gate to a causal intervention. Multiplying a saved LoRA residual
by the gate uses the paper's allocation equation with the product's edit in
place of the paper's fixed direction. Report that choice with the evidence.

### PSR with paired activations

*Steer Like the LLM* learns `lambda = relu(h @ w + b)` and adds
`alpha * lambda * z` to the residual. `PromptSteering` fits the affine probe
and direction together. To train on positive examples, generate a response
from the attribute prompt and replay that same response under the neutral and
attribute prompts. Run the neutral input with the PSR hooks active, and use
frozen activations from the attribute input as targets at the same aligned
positions. Supply the annotated attribute score as training `alpha`. Select
inference `alpha` on a development roster disjoint from the fit data, then
freeze it before evaluating on further held-out cases.
[Equations 7–8 and training pipeline](https://arxiv.org/html/2605.03907v1#S3)

```python
from particle_sliders.prompt_steering import (
    PromptSteering, psr_location_penalty, psr_mse_loss,
)

# Construct one module per intervened layer. Train only the PSR parameters.
layer_psr = PromptSteering(hidden_size, device=device)
steered, coefficients = layer_psr(pre_edit_residual, attribute_scores, mask=edit_mask)
# Continue the native forward from `steered` and collect aligned layer outputs.
loss = psr_mse_loss(edited_layers, attribute_teacher_layers, aligned_position_mask)
loss += regularization_weight * psr_location_penalty(coefficients)
```

S-PSR intervenes at one layer and evaluates the loss at that layer and all
later layers. A-PSR intervenes at all layers and optimizes them jointly, so
later probes see the activations produced by earlier interventions. Sum the
location penalty over intervened layers, using the declared training weight.
For an image, video, or audio adaptation, qualify paired targets at the same
native trajectory coordinate and use the same initial noise, references,
and native sampler. Inference requires fitted probe weights and a selected
strength. The historical fixtures verified synthetic coefficients and a
caller-supplied LoRA envelope. Product-level gains from activation-dependent
steering require native held-out results.
[PR #209](https://github.com/HyperGAN/particle-sliders/pull/209) ·
[PR #212](https://github.com/HyperGAN/particle-sliders/pull/212)

## Run a token-position control

For native LoRA products whose forward receives the full causal prefix,
`LoRANetwork.set_seq_gain([...], mode="prefix")` sets gains by absolute
position. Include the prompt positions in the envelope. The last gain repeats
on longer prefixes, so end with zero when the edit should stop. Restore the
request's envelope after generation and use a request-owned adapter scope.
Incremental KV decoding needs the gain for the current absolute position
injected at the generation step. The length of a one-token tensor cannot
identify that position.

[PR #212](https://github.com/HyperGAN/particle-sliders/pull/212) exercises
a caller-supplied prefix envelope on native LoRA. The controlled fixture keeps
the sum of per-token residual norms equal between the uniform and targeted
edits. Ungated positions match OFF, including after a full-prefix refresh.
This is the position control for a product experiment. A learned activation
gate still needs a finished-output comparison on the product.
