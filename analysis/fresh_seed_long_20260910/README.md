# Longer fresh-seed Lo-fi continuation

The user requested longer training of the fresh-seed arm. This run extends
its complete step-660 state to step 1,000 on GPU 1: 340 more updates, with
the same single prompt and new continuation seeds 1,000,061–1,000,400.
The fixed-seed arm remains a reference from the preceding experiment.

The original constant learning rate, recipe, critic, generator, both optimizer
states, and RNG states are preserved. `lofi-fresh-long/preflight.json` records
exact restoration checks. The original frozen runtime and source checkpoint
are hash-checked. The new seed schedule continues the previous 60 updates
without overlap, and duplicate sampled continuations stop training rather
than silently selecting a replacement seed.

Exports and resumable state are saved every 20 updates. Full states at 800
and 1,000 are retained separately. GPU 0 renders those two checkpoints while
GPU 1 trains. Each checkpoint uses the same two held-out arrangements and
generation seeds 101/303 as the previous step-660 evaluation, for 20 seconds
at strength 1. The original Off and caption-reference recordings are copied
with hash checks. Every candidate is kept, including low-scoring recordings.

The declared comparison is each longer checkpoint versus **fresh step 660**
on raw mean Content Enjoyment across all four cases. Per-case CE deltas,
production quality, and ASR-based lyric agreement are also retained. These
reused monitoring cases are not independent confirmation of a selected
checkpoint. This is one continued trajectory on one training prompt.

Changing the history still affects the ending penalty; the current style
teachers read prompt states and therefore remain unchanged. This run does
not introduce a new objective or promote/publish a checkpoint.

Live listener, bound to `0.0.0.0:8888`:
<http://100.90.104.57:8888/fresh-seed-long-lofi-20260910/>

The durable user services are `music-fresh-long-train-20260910` and
`music-fresh-long-finish-20260910`. The latter renders and scores declared
checkpoints automatically and publishes previews and metrics to the live
page. Failures are written to `lofi-fresh-long/failure.json` or
`evaluation-failure.json`; logs and partial outputs are retained.

`protocol.json` records the fixed protocol, `summary.json` is updated when
each checkpoint is measured, and `scores-800.json` / `scores-1000.json`
retain the detailed measurement records. `result.md` is generated with the
available results. Training completion is recorded separately from rendering
and scoring completion.
