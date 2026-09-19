# Attempts to stabilize the smoke GAN after the late failure

The user asked to keep testing and attempt a fix. The listening reference is
600, with 750 described as very feminine and appealing but possibly starting to
degrade. The original 900 is audibly broken. These observations do not establish
an optimal update count or an overall numerical quality metric.

## Controlled branches

All branches start from the same complete 600-update state, preserving the
LoRA, critic, optimizer moments and RNG state. The source training files, prompt
rows, batch, rank, adversarial and feature-matching weights, end regularizer,
critic cap, and critic learning rate remain fixed. Interventions are explicit
research hooks rather than changes to the default recipe.

| Branch | Generator LR | Critic LR | FM gradient intervention |
| --- | ---: | ---: | --- |
| Historical continuation / new replay control | 0.0005 | 0.00075 | None |
| Smaller generator updates | 0.000125 | 0.00075 | None |
| Bounded FM input gradient | 0.0005 | 0.00075 | Per-row calibrated norm capped at 1 |

The initial target was 900 total updates. The bounded-FM branch reached that
target with its cap inactive on all 1200 row gradients across the 300 new
updates. Thus its update rule was unchanged throughout that segment. This
supplies the planned unchanged-update comparison; a redundant third replay
was not launched. Both branches were extended with a planned target of 1200,
then stopped at logged 1004 after saving their full 1000 states when user
feedback redirected the work toward lyric preservation. Neither reached 1200.
The FM cap activated on two rows at 929; that later segment has no paired
uncapped counterfactual.
A lower learning rate also slows learning, so surviving 900 is not by itself
evidence that it fixes the failure at a comparable degree of adaptation.

## What the FM intervention does

For the existing batch-mean feature-matching surrogate, take its gradient with
respect to one row's span of hidden-state changes. Multiply its norm by the
fixed teacher input scale to express it in the same coordinates as `b_cap`.
Apply the detached multiplier `min(1, 1 / norm)` to that loss's backward signal.
The usual uniform batch weight is then applied by the trainer.

The loss's numerical value and the raw feature vectors are unchanged. Small
gradients pass through exactly; large gradients retain their direction and
are shortened. This preserves magnitude information in feature matching,
unlike normalizing every feature vector. The cap does not bound the LM's full
Jacobian, the end-margin gradient, or the final optimizer step, so it is a
specific intervention to test, not a stability guarantee.

## Verification and measurements

The focused CPU checks record 33 passes. They cover the original trainer,
unchanged-hook reproduction with identical weights and logs, bounded gradient
norm and preserved direction/loss value, correct generator-only LR changes,
refusal to resume an intervention through the ordinary trainer, and exact
resumption of the bounded-gradient branch compared with uninterrupted training.

The real branches' first forward-pass losses exactly match the historical
601st update. Gradient norms differ slightly at floating-point precision, so
GPU bitwise trajectory equality is not claimed. The first parameter update
norm is 0.3110 in the smaller-LR branch and 1.2442 in the FM-cap branch, which
has an inactive cap at that point. This verifies the intended fourfold change
in generator update size while the starting forward computation is shared.

Every new update records post-value-clipping gradient norm, actual parameter
update norm and relative norm, and each row's raw/limited FM input-gradient
norm. Every 25 updates also measures the aggregate raw and limited FM gradient
with respect to LoRA parameters. All full states are archived at the 50-update
save interval. These intervention states include this harness's fingerprint
and configuration; ordinary continuation rejects them rather than silently
dropping the intervention.

Training diagnostics screen for recurrence of the large late failure. Their
minima do not select musical quality. Promising checkpoints will be evaluated
with the same cached next-token policy histories and compared by matched audio
at slider +1, prompt row 0, generation seeds 7/23 and a 20-second duration cap.
Existing 600/750/900 references are reused byte for byte, with no seed retries.

## First results at 900

| Run | Last-30 hidden-shift alignment | Norm ratio | FM loss | Active cap rows |
| --- | ---: | ---: | ---: | ---: |
| Historical broken 900 | 0.0384 | 6.7978 | 80.5083 | Not measured |
| Quarter generator LR, 900 | 0.9199 | 0.9495 | 0.1323 | No cap |
| Original LR with inactive FM cap, 900 | 0.9274 | 0.9516 | 0.4348 | 0 / 1200 |

Both new runs avoided the historical large training excursion. This **does not
show that the cap fixed it**: the cap did nothing in this segment, and the
original update rule also survived. The earlier collapse is not a deterministic
900-update deadline. Numerical trajectories differ despite the shared complete
600 state and matching initial forward losses; this investigation has not
isolated every source of GPU floating-point variation.

At both 800 and 900, all 432 inference tensors exactly match the corresponding
full training state. The optimizer rates, harness fingerprint, and seven core
source fingerprints also match their declared configurations. These checks
exclude a changed trainer or a mismatched exported adapter in these branches.

The [900 audio comparison](../../eval/listen/gan-bcap-stability-20260904/index.html)
contains all three historical references and both new candidates with generation
seeds 7/23. Verification covers 40 WAV files, 14 distinct audio hashes, all 24
reused reference files, matching off/positive references, and finite samples.
No retries or listening-based sample filtering were used.

The fixed-history semantic-policy evaluation also avoids the original 900
excursion. On positive-caption histories, continuation KL is 0.010484 for the
smaller-LR 900 and 0.011343 for the inactive-cap 900, versus 0.010007 at 600,
0.010298 at historical 750, and 9.203376 at broken 900. The repeated 600
reference exactly reproduces prior per-position metrics and history hashes;
all 24 slider-off policy checks pass bitwise. This metric excludes depth/flow
prediction paths and does not rank feminine character or musical quality.

The effective adapter weights also changed: relative to 600, the smaller-LR
900 has norm ratio 1.0505 and cosine 0.8416; the original-rate 900 has ratio
1.2186 and cosine 0.7784. Neither is simply an unchanged 600 adapter, although
weight-space change is not a measure of perceptual progress.

## Listening changed the next experiment

The user then reported, "theres a bit of gibberish idk at the 900s". The exact
candidate, generation seed and timestamp were not specified, so this is a
concern about the comparison set rather than an equal failure rating for each
candidate. It is direct evidence that avoiding the old numerical collapse did
not settle the lyric-quality question.

The longer branches were deliberately stopped after saving their complete
1000 states (1004 updates were logged). Neither reached 1200. At saved 1000,
last-30 alignment / FM were 0.9285 / 0.1245 for the smaller LR and 0.8992 /
0.7757 for the cap branch. The cap acted on two rows at update 929, with factors
0.9683 and 0.8931, and no other rows through 1000. This has no paired uncapped
counterfactual after 900 and does not prove the cap prevented anything.

GPU time shifted to [lyric-preservation trials](lyric_preservation_findings_20260904.md).
No numerical or musical fix has been accepted.

Artifacts: [experiment plan](stability_plan_20260904.json),
[research harness](stability_experiment.py), and the
[preceding failure investigation](failure_cause_findings_20260904.md).
