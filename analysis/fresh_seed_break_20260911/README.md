# Fresh-seed Lo-fi: step 2000 → 5000, looking for the break

The 1,000 → 2,000 run did not fall apart, so the user asked to keep going and
see **where, or whether, this run breaks**. This experiment continues the exact
full step-2000 state to **step 5000**: the same single prompt, the same constant
learning rate, the same critic, generator, optimizer, sampler and RNG states,
and one fresh base-model continuation sampled per update. Nothing here stops on
a metric. Degradation is the object of study, not a failure of the run.

Sampling seeds continue the parent rule unbroken, `1000001 + update - 601`,
running 1001401 through 1004400. The 1,400 earlier continuations stay in the
full history and in the duplicate check, so no sampled continuation is ever
reused or silently substituted.

## Queueing

This run does not start on its own clock. `start_when_ready.py` waits for the parent run
(`analysis/fresh_seed_stress_20260910`) to report `complete` at step 2000, then:

1. records the parent state and weight hashes into `protocol.json` and
   `setup-audit.json`, and checks the first seed continues the parent rule;
2. starts training on **GPU 1**;
3. waits for the parent's own step-2000 listening measurement, which is this
   run's reference, then starts the renderer/scorer on **GPU 0**.

The recorded hashes are *observed* at queue time, not predeclared — the parent's
step-2000 state does not exist until it gets there. The real verification is in
`train.py`, which restores that state and asserts the network matches the
exported safetensors, the critic, both optimizers, the sampler and all RNG
states match the blob, and `completed == 2000`, before a single update runs.
`lofi-fresh-break/preflight.json` records every one of those checks.

If the parent run fails before 2000, the queue stops and writes
`queue-failure.json` rather than starting from an unintended state.

## Listening

GPU 1 trains; GPU 0 renders at **every 200 updates from 2200 through 5000** —
15 listening checkpoints. Each uses the same two held-out arrangements and
generation seeds 101/303, 20 seconds at strength 1, with **step 2000 as the
reference**. Reference audio is copied with hash checks; no replacement seed is
chosen after scoring. Every recording is kept, including short, quiet or
degraded results.

Meta Audiobox Content Enjoyment and Production Quality, plus ASR lyric
agreement, are diagnostics only. The page charts mean CE against step so a
collapse is visible at a glance, and marks any checkpoint carrying a duration,
near-silence or clipping flag. Listening remains the assessment.

Audio diagnostics flag clips shorter than 19 seconds, RMS below 0.0001, or at
least 1% of samples near clipping. Audible repetition, lost structure, drifting
pitch or other degradation has to be judged from the retained recordings.

## Stopping

Training stops at step 5000 or on an execution/numerical failure — non-finite
weights raise before the checkpoint is written, and every preceding checkpoint
and clip stays on disk for inspection. There is no metric threshold, no early
stopping, and no automatic checkpoint selection or Hub publication.

Recovery checkpoints are saved every 20 updates; full states at each listening
milestone are retained separately. At ~15.3 s/update the 3,000 updates take
roughly 13 hours.

## Services

- `music-fresh-break-queue-20260911` — the waiter described above
- `music-fresh-break-train-20260911` — training, GPU 1
- `music-fresh-break-finish-20260911` — rendering, scoring and publishing, GPU 0

All three are transient `systemd-run` units, like the parent run's: they do not
survive a reboot. Training resumes exactly from `lofi-fresh-break/state.pt` if
restarted, and the evaluation picks up at the first unrendered milestone.

Live page, on the listener bound to `0.0.0.0:8888`:
<http://100.90.104.57:8888/fresh-seed-break-lofi-20260911/>

## Limits

Fresh histories change the ending penalty. Prompt-state style targets remain
fixed. One prompt and one training trajectory: no generalization claim, no
promotion claim. The monitoring cases are carried forward from the parent runs
and do not independently confirm any selected checkpoint.
