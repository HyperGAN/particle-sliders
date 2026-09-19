# Automatic GAN selection and catalog campaign

The user explicitly authorized continuing the metric search, automatically
choosing a suggestion, training all sliders with it, and preparing a strong
presentation without requiring further user decisions. This supersedes the
earlier proposal to require blind listening before choosing the next branch.
Human feedback remains evidence: the previous lyric-hold plus step-limit 900
is described as cool and not broken, with the subjective preference open.

## Current pilot

The step-limit-only run resumes the original complete 600 state, with the
original repaired b_cap recipe, original generator/critic learning rates,
lyric hold 0, and an actual generator parameter-update norm bound of 2. It
completed 900. The bound acted on 35/300 added updates; the largest proposed
update was 2.9325 and largest applied update 2.000000075 (float32 rounding).
The saved 750/900 exports exactly match their 432 full-state tensors, and
the source fingerprints and base recipe settings match.

Two prompts and generation seeds 7/23 are fixed for the pilot. The additional
prompt is an existing training-heldout development fixture. Every first
render is kept. The two listening pages each contain 48 WAV files and 16
distinct audio clips, with matching controls. Reused reference WAVs are
verified by content hash. The original-rate 900 reference is the earlier
replay whose FM cap was inactive through 900, so its update rule was unchanged.
GPU trajectories are not claimed bitwise paired.

Fixed-history semantic KL is 0.010007 at original 600, 0.010683 at step-limit-only
750, and 0.011135 at step-limit-only 900. Repeated 600 records match exactly;
all 24 slider-off policy checks match exactly. This diagnostic is recorded
separately from the automatic audio score.

## Automatic rule

The actual coefficients and complete per-clip components are recorded in
`state.json` and each `pilot-*.json` report. The score is a **heuristic
suggestion**, not a trained or validated overall music-quality metric.

The initial v1 instrument was audited against existing controlled audio
attacks. Gain +3 dB earned +0.067 and high-frequency hiss earned +0.016, while
a quarter-second repeat scored -0.871. The known historical broken 900 scored
-0.811 and -2.250 on its two main-prompt seeds. These exposed specific blind
spots rather than certifying the metric. V1 scores and the old search history
are preserved under `v1_archive`; training and original audio remain unchanged.

The same controlled attacks under v2 scored -0.000075 for gain +3 dB,
-0.000080 for gain -3 dB, -0.160806 for high-frequency hiss, and -0.829262
for a quarter-second repeat (`attack-audit-v2.json`). This validates these
specific corrections, not general subjective quality.

V2 corrects those instrument weaknesses before the catalog decision. Perceptual
models measure separate float32 copies normalized to stereo RMS 0.1; the user
audio is untouched. Raw audio still supplies level checks and ASR transcripts.

- 40% concept margin from fixed CLAP sound-description pairs, scaled by 0.05.
- 20% predicted content enjoyment and 20% predicted production quality from
  the existing audio model, each scaled by 0.5.
- 20% lyric preservation: the mean of transcript phrase accuracy and supplied
  word recall, scaled by 0.25. This can be affected by ASR errors and timing.
- A penalty with coefficient 0.2 for high-frequency power above both the
  slider-off and positive-caption reference: tolerate 0.002 excess above
  14 kHz, then scale the excess by 0.02. This addresses a band the perceptual
  model's 16-kHz input cannot adequately judge, while allowing reference brightness.

Each component is the change against that clip's exact slider-off control,
clipped to [-3, 3]. Audio windows are weighted by their share of covered time,
so a nearly duplicate tail window does not double-count the ending. Prompt/seed
examples have equal weight. A near-silent render relative to its own control
is ineligible; no candidate is declared broken solely by its ASR transcript.
Judge descriptions were fixed before the first scoring pass. V2's level and
noise correction was determined from controlled attacks, not by tuning on
candidate listening preferences. Candidate scores are recomputed under v2;
v1 and v2 numbers are not mixed.

The pilot extends the step-limit-only state in blocks of 150 updates. It
stops after three evaluated extensions without improving the best mean score
by more than 0.05. The initial 2400-update budget automatically extends if
improvement continues. This finds an observed plateau of this metric and this
sample set; it does not establish a global optimum. Storage below 15 GiB
interrupts the search while retaining the measured checkpoints.

The two highest-scoring eligible measured pilot candidates proceed to an
automatic confirmation on two additional arrangements and lyric sheets with
generation seeds 101/303. These examples were not rendered during the search.
The original four search examples and the four confirmation examples have
equal weight in choosing the suggested duration and intervention settings.
Their use to confirm/select makes this development validation, not an untouched
final test. The original 600 and original-update 900,
and lyric-hold plus step-limit 750/900, remain eligible comparisons. A higher
score for a reference may therefore select the original recipe or the lyric
hold, rather than automatically favoring the newest experiment.

## Catalog and presentation

Scope is the 28 LM controls in the current studio catalog. Their training
prompt files were resolved from the existing uni-lyric adversarial catalog;
the project's name validator flagged none. No prompt names are introduced.

Each control uses a fresh original-recipe warm-up through 600 (or the suggested
duration when earlier). If the selected duration is later, it resumes its own
complete warm-up state with the selected bound and hold. This mirrors the
pilot's intervention timing instead of applying a new loss from step one.
The complete state retains the critic, optimizer moments and RNG state.

The suggested endpoint, earlier available checkpoints and 300 are compared
automatically on the original example and a fixed new lyric sheet, with seeds
7/23. The highest-scoring eligible checkpoint is exported to the gallery.
A 60-second render with seed 101 provides a third seed and a longer example.
That render is retained as evidence, not presented as proof of natural ending
quality. Existing studio assignments and published model catalogs are not
automatically replaced by this research gallery.

The service is `music-gan-autonomous-20260905.service` in the user's systemd
manager. It waits for physical GPU 1, uses CPU for audio scoring, records
progress to `state.json`, and writes a live gallery under
`eval/listen/gan-autonomous-20260905`. Interrupted training stages resume from
their saved state into fresh directories. Failed artifacts remain available;
the runner records failed controls and continues the remaining catalog.

The implementation checks currently record 50 passing tests, covering the
parameter bound, inactive identity, exact optimizer-state continuation,
configuration mismatch rejection, fixture equality, overlap time weighting,
plateau reconstruction and interrupted-stage recovery. These verify the
workflow, not the validity of the subjective quality heuristic.

## Live evidence page

`eval/listen/gan-autonomous-20260905/pilot.html` presents all completed pilot
checkpoints, matched samples, the training curve and weighted score components.
The separate CPU-only `music-gan-pilot-page-20260905.service` refreshes this
presentation from saved reports every 20 seconds; it cannot change training
or checkpoint selection. All 44 audio URLs in the first seven-checkpoint
version were checked over HTTP, and the rendered page was visually inspected.

User lingering is enabled so the campaign can continue after logout. Persistent
user-unit copies and default-target links are installed for the controller and
evidence page, allowing saved-state recovery after a user-manager restart.

The historical failure control was also recomputed under v2: seed 7 scored
-0.780860 and seed 23 scored -2.300708. This is the original failed 900,
not the later hold-plus-limit 900 the user describes as cool.

The Female catalog comparison also retains the confirmed pilot winner, so a
fresh warm-up cannot discard the already measured best candidate. Its scoring
fixtures and coefficients are identical to the fresh candidates.

After confirmation, the evidence page contains nine pilot checkpoints and all
eight additional candidate samples. Eighty-eight HTTP resources, including
the confirmed adapter and metadata, were checked successfully. The final page
was visually inspected. The first fresh catalog run completed its initial
updates with finite training metrics; the rest of the batch remains automatic.
