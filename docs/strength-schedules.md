# Scheduled slider strength

Use scheduled strength when qualifying a slider for a multi-step generator.
Keep the RAW adapter fixed and vary its contribution during generation. Each
product uses the schedule that wins on that product, slider and native sampler.
Keep constant strength as the comparison and as the fallback for profiles with
no qualified schedule winner.

The [product map](products.md) identifies the generation stage for each public
product. Use the generation stage and scheduler from its native source cohort.

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

The per-token research is complementary. The fixed-budget gate in
[Right Token, Right Time](https://arxiv.org/abs/2611.18274)
and the learned token coefficients in
[Prompt Steering Replacement](https://github.com/HyperGAN/particle-sliders/pull/209)
motivate a causal comparison. Before using an activation gate on a product,
check held-out signal quality and compare finished outputs with both constant
steering and a budget-preserving shuffled control. Use only deployment-available
information.

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
