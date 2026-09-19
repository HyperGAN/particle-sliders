# Lyric-preservation attempts after the 900 listening feedback

The lyric-hold plus parameter-step-bound trial completed 900 updates without
the large numerical excursion seen in the failed trials. The user describes
its 750-to-900 transition as clearly audible and appealing. This is positive
listening feedback, without an explicit checkpoint preference, specified seed,
or verdict that the lyrics are clean. On an additional fixed prompt, ASR
recovers the first two supplied lines at 750 and none at 900; listening is
needed to assess that discrepancy. This is a promising research candidate,
with lyric generalization still unresolved.

Subsequent user feedback prefers 600 in a particular listened example, with
explicit uncertainty about sample variation. The page and seed were not
specified. This qualifies any reading of the earlier appealing-transition
feedback as a preference for the later checkpoints. The
[metrics comparison](metric_interpretation_600_20260905.md) shows higher ASR
word coverage at 600 in two of the three matched prompt/seed examples and a
tie in the third, while fixed-history semantic KL stays almost unchanged.
This is limited evidence about lyric coverage, not a general quality ranking.

The user subsequently clarified that this bounded 900 is not broken, sounds
cool, and could subjectively be better than 600. Retain that listening judgment
alongside the ASR concern; do not describe the current bounded 900 as an audible
failure. The next authorized test is the step limit alone, with lyric hold off,
as recorded in [the new plan](stepcap_only_plan_20260905.json).

This round followed the user's report of some gibberish in the preceding
900-update comparisons despite stable training and conditional semantic KL.
The exact candidate, generation seed and timestamp were not specified.

## Intervention and calibration

The repaired smoke card sets `lyrichold_weight=0`. These branches resume the
same original complete 600 state; the initial branches change only that weight.
The final trial also bounds the actual parameter update as described below. Generator LR
remains 0.0005, critic LR 0.00075, adversarial and batch-mean FM weights 1/1,
critic `b_cap` unchanged, and no FM gradient clamp. Training and rendering use
physical GPU 1. The production trainer and default launcher remain fixed.

The existing hold penalizes squared differences from neutral-teacher hidden
states at the prompt lyric positions. It averages over positions but **sums
over hidden dimensions**. It is an indirect lyric constraint, not a guarantee
of intelligibility, nor an audio-start target or a full generated-sequence KL.

The first calibration mistakenly treated weights 0.1 and 1.0 as mild and
strong settings. Their shared initial unweighted hold was about 1098.596;
their first actual parameter-update norms were 10.156 and 24.328, versus
1.244 in the unchanged replay. The first GAN terms were identical, isolating
the changed hold term. Those oversized trials were stopped after their full
650 states were saved; 658 and 657 updates were logged. They are not claimed
as completed 900 trials or judged by listening.

The calibrated trials restart from the untouched original 600 state:

| Run | Hold weight | Initial added hold loss | Planned total updates |
| --- | ---: | ---: | ---: |
| `smoke-lyrichold-w0p0001-s7-20260904` | 0.0001 | about 0.110 | 900 |
| `smoke-lyrichold-w0p001-s7-20260904` | 0.001 | about 1.099 | 900 |

The original initial total loss is about 4.22. Loss magnitude alone does not
determine gradient influence; actual parameter updates are also recorded.
These are controlled research branches, not promoted replacement weights.

The calibrated first updates are 1.24433 and 1.25045, close to 1.24422 in
the unchanged replay. Their first total losses are 4.32838 and 5.31711.
This removes the oversized initial update seen in the first calibration.

## Verification and comparison plan

Forty focused CPU checks pass, including the underlying trainer and lyric
hold, unchanged-driver reproduction with identical weights/logs, changed
hold influence, correct settings in the full state, exact resumption, and
rejection of a mismatched hold strength. The research driver and inherited
stability hooks are fingerprinted in every state.

The initial listening plan proposed comparing 750 and 900 within both
calibrated branches. Their failures superseded that plan: final listening
compares the bounded branch's 750 and 900 against earlier references, on the
same row, generation seeds 7/23, slider +1 and a 20-second cap. There are no seed retries.
An additional fixed arrangement and lyric outside the four training rows was
selected before these results, with generation seed 7. The development policy
fixtures are unchanged; their KL minima do not select musical quality.

CPU ASR transcripts use the existing local Whisper backend and bag-word lyric
recall only as an aid to inspect the words. ASR can hallucinate words or miss
gibberish, bag recall ignores order and extras, and the short duration can omit
valid later lyrics. These readings do not overrule the user's listening.

All 14 distinct clips on the first 900 page were transcribed with the same CPU
float32 backend. One new 900 clip receives full bag-word recall despite the
recognizer adding an off-sheet tail. Some older references also have repeats
or extra transcribed words; these are not exclusive to 900. The broken
historical 900 clips have zero recall. The transcript text is machine output,
not a verified annotation of what the singer actually says.

## Calibrated holds still failed

Both uncapped calibrated trials developed large training excursions. The
0.001 branch first crossed hidden magnitude ratio 2 with alignment below 0.5
at 664; its largest actual update was 22.169 at 665. The 0.0001 branch crossed
the same descriptive thresholds at 720; its largest update was 24.428 at 722.
These thresholds describe the recorded failures; they are not quality gates.

The 0.001 branch was stopped at logged 709 with its full 700 state preserved.
The 0.0001 branch was stopped at logged 784 with its full 750 state preserved.
Neither is a completed 900 trial. Their checkpoints and logs remain available;
no listening verdict is claimed for them.

In the 0.001 failure, actual updates grew from approximately 2 to 9.553 at 661,
16.238 at 662, 15.995 at 663, and 21.822 at 664. The maximum calibrated FM
input-gradient norm across rows was below 1 in each of those four updates.
Thus an FM-only cap at 1 would not directly constrain those particular steps.
End-margin drift also rose sharply. The telemetry does not assign the entire
LoRA gradient to a particular loss, and does not prove that this is the exact
initiating mechanism in the historical unregularized 900 failure.

## Direct parameter-step bound

The next trial is `smoke-lyrichold-w0p001-stepcap2-s7-20260904`. It again starts
from the original full 600 state, keeps the 0.001 hold and original G/D rates,
and shortens the joint LoRA parameter displacement when its L2 norm exceeds 2.
Inactive steps pass through exactly. Adam moment updates are retained. The
bound applies to actual trainable LoRA parameters, up to float rounding; it
does not bound effective delta-W, policy divergence, or musical change.

Forty-two focused checks pass, including global bound and preserved direction,
inactive identity, actual recorded update norms, exact full-state continuation,
and rejection of a changed bound on resume. The bound's implementation and
settings are included in the full-state signature.

Matched audio is complete for this branch at 750 and 900 against six
already-rendered references, followed by the additional fixed prompt. The
earlier plan for nonexistent 900 uncapped-hold checkpoints was explicitly
superseded in the listening plan. No audio is fabricated or selected by ASR.

## Completed bounded trial

The bounded trial reached 900. Its last-30 training alignment is 0.9271,
magnitude ratio 1.0335 and FM loss 0.8614. The limiter acted on 20 of the 300
new updates. The largest proposed step was 4.60764; the largest applied step
was 2.000000056, consistent with the bound and float32 rounding. Both the
750 and 900 exports exactly match all 432 tensors in their full states; all
seven core sources and three research-source fingerprints match. Independent
step telemetry agrees exactly on every applied update norm.

This is a promising stabilization result for one trajectory, not proof of a
general fix. The uncapped GPU trajectory was not bitwise identical before the
cap activated, and no musical winner has been selected by these numbers.

The [main listening comparison](../../eval/listen/gan-bcap-lyric-hold-20260904/index.html)
contains the new 750/900 plus the earlier 300, 600, 750 and three 900 references.
The original 300 is from a separate earlier run. Verification covers 64 WAVs,
20 distinct audio hashes, 48 byte-identical reused files, matching controls,
finite samples and all 20 players. The supplied lyric lines are visible on
the page. No seed retries or audio filtering were used.

CPU ASR on the new seed-23 900 stays much closer to the supplied lines than
the preceding unregularized 900 transcripts, but the new seed-7 750/900
transcripts capture only part of the lyric sheet. Coverage is timing-sensitive
under the 20-second cap and ASR may miss or invent words. These are mixed
diagnostics, not evidence that gibberish has been eliminated.

The fixed-history semantic-policy evaluation remains near the earlier 600:
positive-history continuation KL is 0.010060 at bounded 750 and 0.010024 at
bounded 900, versus 0.010007 at 600 and 9.203376 for historical broken 900.
The repeated 600 per-position metrics and trajectory hashes match exactly;
all 24 slider-off checks pass bitwise. These small KL differences do not
establish lyric clarity or voice quality.

On the additional arrangement, the recognizer recovered no supplied lyric
words from the new bounded 900, while it recovered some from older adapters.
The WAV is a finite, nonsilent 20-second render, so this is not a missing file.
ASR can misrecognize singing or a clip may have a long intro; this is an
unresolved concern requiring listening, not a verified transcript of failure.
It weakens the case for a lyric-quality fix. The completed
[follow-up comparison](../../eval/listen/gan-bcap-lyric-heldout750-20260904/index.html)
adds bounded 750 to the exact same prompt/seed comparison, keeping all 16
existing WAVs byte-identical. At bounded 750 the recognizer recovers the first
two supplied lines (bag-word recall 0.60), versus no supplied words at bounded
900 (0.00). This does not establish the cause or override human listening.
Verification covers 20 finite WAVs, seven distinct audio hashes, seven players
and matching controls. The prompt is outside the four training rows but is
part of the existing development fixtures, not a fresh unseen final test.

This round covers one training seed, two generation seeds on the main prompt,
and one generation seed on the additional prompt. The user's positive main
comparison feedback and the additional-prompt ASR concern are both retained.
GPU training and rendering are complete; no bounded continuation beyond 900
was run, and no default recipe or model alias was promoted.

The bound first activated at 653, shortening a proposed 2.36025 update to
approximately 2. Through 662 it acted on nine updates; the largest proposal
was 2.62957 and the largest measured applied norm was 2.00000005 (float32
rounding). At that point the last-10 alignment was 0.9322, magnitude ratio
0.9620, and end-margin drift 0.1125. This is an interim training result.
Its first forward GAN terms match the uncapped 0.001 run, but the initial
gradient norm differs slightly (15.11669 versus 15.11631), so a bitwise paired
GPU trajectory before the intervention is not claimed.

Artifacts: [plan](stability_plan_20260904.json),
[driver](lyric_preservation_experiment.py),
[parameter-step implementation](parameter_step_limit.py),
[completed verification](step_limit_verification_final_20260904.json),
[additional-prompt ASR](asr_lyric_hold_heldout750_20260904.json),
[first numerical-stability attempts](stability_findings_20260904.md).
