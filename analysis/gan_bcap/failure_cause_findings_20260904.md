# What the saved states tell us about the late smoke failure

The user confirms 900 is audibly broken and qualifies 750 as very feminine and
appealing but possibly beginning to degrade. These are listening observations,
not numeric ratings or proof that 750 is optimal.

The leading hypothesis is a destabilizing generator update followed by a
feedback loop involving raw feature matching and changed hidden states. The
probe below establishes the unprotected path and the resulting drift; it does
not prove which gradient or optimizer update initiated the failure.

## Crossing saved critics with fixed adapter outputs

The read-only probe crosses saved critics at 600/750/900 with hidden spans
from each adapter on the same four training prompts. It uses production span
extraction and critic code, with no optimizer updates. All 12 slider-off
hidden comparisons are exact. Three fake-equals-real controls yield exactly
zero feature-matching loss and zero input gradient.

| Critic | Adapter | Raw FM loss | FM input-gradient norm | Adversarial input-gradient norm | Mean calibrated score-gradient norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 750 | 750 | 0.417 | 0.217 | 0.104 | 0.603 |
| 750 | 900 | 62.745 | 2.717 | 0.174 | 0.179 |
| 900 | 750 | 1.775 | 0.392 | 0.570 | 0.587 |
| 900 | 900 | 78.924 | 3.472 | 0.180 | 0.186 |

With the critic fixed at 750, the 900 adapter has about 150 times the FM error
of the 750 adapter. Thus the large final loss is not merely the later critic
changing its grading scale: the generator has moved far away even under the
old critic. Conversely, the 900 critic assigns a much smaller FM error to 750
than to 900. The critic's teacher-feature RMS is 2.626 at 750 and 1.760 at 900;
these endpoints do not support a simple story that all critic features grew
without bound.

The cap penalty on fake inputs is zero in all four table entries. At 900, the
FM input-gradient norm is about 19 times the adversarial input-gradient norm.
These gradients stop at the detached hidden spans. They do not attribute full
LoRA parameter gradients and omit the end-margin loss.

## A concrete gap in what the cap constrains

`SpanTransformerD.forward` scores `head(out_norm(pooled))`, while `features`
returns unnormalized `pooled` values. `b_cap` penalizes input gradients of the
normalized scalar score, not the full feature Jacobian or the generator update.

Doubling both pooled feature sets multiplies raw FM loss by exactly four, while
normalized critic scores barely change: maximum absolute change is 0.00353
in every crossed combination. Thus the normalization hides much of the feature
magnitude change from the scalar score. Simply normalizing FM is not a validated
remedy; it would also remove magnitude information the successful recipe uses.
The original [feature-matching formulation](https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf),
section 3.1, defines matching in the learned feature coordinates; it does not
guarantee that this implementation's gradient cap controls that separate loss.

Input calibration and optimizer settings are identical in all three saved
states. Generator LR stays at 0.0005, critic LR at 0.00075, and beta1 is zero.
There was no scheduled LR jump or input-scale recalibration. The generator
clips individual gradient values at 1 rather than clipping the global norm.
Logged norms are measured before clipping and are not actual update sizes.
Raw FM, end-margin gradients and Adam's adaptive update remain candidate
contributors to an overshoot.

## Timing and the qualified 750 preference

Smaller bursts in FM, gradients and end-margin drift occurred around updates
640 and 700, then subsided before 750. In updates 721–750, mean FM was 0.413,
mean gradient norm 18.63 and hidden alignment about 0.92. From 826–836, mean
FM was 33.52 and the gradient norm 124.73. A much larger collapse follows.
These logs do not disprove audible degradation at 750 or establish that it
has the exact same cause as the later sharp failure.

Identifying the initiating cause requires replay near the onset with per-loss
parameter gradients and actual update norms. Controlled follow-ups should
change one variable at a time: reduce generator step size while preserving the
objective, and separately control FM feature scale or gradient magnitude.
Use 600 as an earlier comparison given the qualification on 750. Neither
intervention was trained or declared a fix in this probe.

Limits: prefix-only bf16 forwards can differ numerically from training geometry;
post-state crossed critics are not a replay of pre-update losses; four training
prompts do not establish generalization. Original weights and full GAN states
were preserved throughout.

Artifacts: [probe results](critic_failure_probe_20260904.json),
[implementation](critic_failure_probe.py), cached detached spans in
`critic_failure_probe_20260904.pt`, and the
[original failure timeline](continuation_findings_20260904.md).
