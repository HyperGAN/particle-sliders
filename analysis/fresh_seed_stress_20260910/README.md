# Fresh-seed Lo-fi beyond step 1000

The user listened to step 1000, liked its direction, and requested continued
training to observe whether it degrades or breaks. This run continues the
exact full step-1000 state to **step 2000**. Listening is the main assessment.
Meta Audiobox Content Enjoyment and Production Quality remain diagnostics,
alongside ASR lyric agreement and basic audio measurements. A lower metric
score does not stop training or discard a sample.

The prompt and lyrics stay identical. Sampling seeds continue from 1000401
through 1001400, one new base-model continuation per update. The previous
400 fresh continuations remain in the full history and duplicate checking.
The original constant learning rate, critic, generator, optimizer states,
sampler, and RNG states carry forward. The original frozen runtime is
unchanged. Style teachers still use fixed prompt states; history resampling
continues to affect the ending penalty.

GPU 1 trains; GPU 0 renders the listening checkpoints at **1200, 1400, 1600,
1800, and 2000**. Each uses the same two held-out arrangements and generation
seeds 101/303, for 20 seconds at strength 1. Step 1000 is the reference.
Every recording is kept, including short or quiet results. Reference audio
is copied with hash checks; no replacement seed is chosen after scoring.

Recovery checkpoints are saved every 20 updates, and full states at each
listening milestone are retained separately. Training stops at step 2000
or on an execution/numerical failure; failures and the preceding valid
checkpoints remain available for inspection. The original recipe's existing
optimization limits remain unchanged.

Audio diagnostics flag clips shorter than 19 seconds, RMS below 0.0001, or
at least 1% of samples near clipping. These are inspection aids, not a test
of musical quality. Audible repetition, lost structure, or other degradation
should be judged from the retained recordings. The same monitoring cases
are reused; there is no claim of independent generalization evidence.

Live page, on the existing listener bound to `0.0.0.0:8888`:
<http://100.90.104.57:8888/fresh-seed-stress-lofi-20260910/>

Durable services: `music-fresh-stress-train-20260910` and
`music-fresh-stress-finish-20260910`. The second service publishes samples,
scores and progress automatically at every milestone. `protocol.json`
records the experiment, `setup-audit.json` checks the source and seed
boundary, and `lofi-fresh-stress/preflight.json` records exact GPU restoration.
`summary.json`, `result.md`, per-step scores and evaluation audits are
updated as checkpoints are evaluated. This run makes no automatic
checkpoint selection or publication to the model Hub.
