# A measured route to better studio blends

September 7, 2026. Recommendation: confirm the listener's preferred energy 2.8
on another fixture before changing the merger. If gain alone does not hold up,
investigate bounded layer-specific blending against combined-caption examples.

**Execution update:** all 25 requested renders and all 25 CPU measurements
completed on September 7. [Listen in the studio](http://localhost:7860/studio-mix-20260907/)
or inspect the [results](results.md) and [file/delivery audit](audit.json).
The frozen `screen.json` retains its original planned status as input provenance;
`renders.json` is the authoritative execution record.

**User listening result:** the user preferred **energy 2.8 in all three pairs**.
The exact A/B/C mappings, clip hashes and original feedback are preserved in
[`listening-feedback.json`](listening-feedback.json).

| Pair | Best to worst energy | User qualification |
|---|---|---|
| Female + Pop | 2.8, 2.5, 2.0 | 2.5 was close to 2.8 |
| Country + Indie Rock | 2.8, 2.0, 2.5 | No closeness stated |
| House + Acoustic Folk | 2.8, 2.0, 2.5 | No closeness stated |

This makes a simple 40% gain increase the leading candidate for confirmation.
The automated diagnostic tradeoffs below do not override the user's preference.
It does not establish that all intermediate increases help: energy 2.5 placed
last in two pairs. Next compare 2.0 against 2.8 on reserved row 3 / seed 303 for
these three pairs (six renders), with ratio, captions and lyrics held fixed
within each pair. Preserve solos and production defaults until confirmation;
these rankings concern two-adapter blends on one fixture and seed. A successful
gain-only confirmation would set the baseline that a more elaborate merger
needs to beat, without establishing weight-norm normalization as the cause.

All renders used the studio merge implementation on physical GPU 1. Every
20-second excerpt exactly matches its full recording. Full durations span
29.83–50.05 seconds; 12 reached the normal 50-second ending cap. There were no
render failures, empty outputs or measurement failures. Total measured render
time, including initial model loading, was 41.46 minutes.

The automatic diagnostics give no uniform reason to raise energy: Female + Pop
trades a stronger vocal margin for a weaker Pop margin, while Country + Indie
Rock strengthens both style margins at 2.8. House + Acoustic Folk also improves
both margins at 2.8 relative to 2, but its middle setting shifts the balance
toward acoustic character. Quality predictions and early transcript accuracy
do not improve together across the screen. These are model measurements, not
listening preferences. Sparse early transcripts in two solos were additionally
checked in later sections (`later-transcripts.json`); some later words are
recognizable, so early ASR absence should not be equated with a silent song.

The next decision is confirmation of energy 2.8 against 2.0 on row 3 / seed 303.
Production energy and mixing
behavior have not been changed. In particular, the pair at energy 2 already
applies multiplier 1 to each adapter, the upper end of all six sidecars'
recommended ranges. Geometric norm retention is not a calibrated measure of
under-strength musical control.

The studio ran on GPU 0 during the study. New active and queued studio jobs
arrived before cleanup, so GPU 1 was released without interrupting them.
`music-studio-mix-restore-20260907.service` waits for an empty, inactive queue
with Keep disabled, then restores the original two-GPU service and current
Keep settings. `restore-status.json` records whether that restoration is still
waiting or has completed.

## What was actually checked

The studio uses sixteen independently trained rank-8, alpha-8 LM attention
adapters, each covering the same 144 projections. It sums full updates `B @ A`,
then merges into pristine model weights. It does not average the factors and
therefore does not introduce the cross terms that factor-wise averaging would.
The current energy knob fixes the sum of effective coefficients, not a weight
norm, activation norm or perceptual quantity.

`geometry.py` read all sixteen deployed files on CPU and computed their full
parameter-update Gram matrices using low-rank products, with float64 arithmetic.
Checkpoint, sidecar, registry and script hashes are recorded in `geometry.json`.
No model was loaded and the studio was not restarted.

| Measurement | Result |
|---|---:|
| Largest / smallest adapter update norm at multiplier 1 | 3.067 |
| Pairwise update cosine, minimum / median / maximum | 0.0046 / 0.0197 / 0.0612 |
| Equal pair mix norm / mean solo norm at the same energy | 0.711–0.795; median 0.720 |
| Equal sixteen-control mix norm / mean solo norm | 0.296 |

The largest norm belongs to Reggaeton, the smallest to Male. These are parameter
sizes, not rankings of perceived strength or quality. Do not turn their inverse
norms into production gains. The existing Reggaeton preview failure also does
not establish that its larger update caused the failure.

There is an especially useful counterexample to treating this as semantic
geometry: Female and Male have an update cosine of only 0.0174 even though their
single-lead instructions differ on the same attribute. Small weight overlap
cannot establish musical compatibility. Global statistics can also hide layer
structure: Country / Indie Rock reaches cosine 0.373 in the final output
projection despite its global cosine of 0.061.

For normalized fader shares `p` and update Gram matrix `G`, the current merged
weight norm is `E * sqrt(p.T @ G @ p)`. The weighted mean solo norm is
`E * sum(p_i * sqrt(G_ii))`. Their ratio produces the retention values above.
These numbers motivate a gain experiment; they do not diagnose audible failure.

## First experiment: 25 matched renders

`screen.json` freezes the jobs, resolved studio multipliers, captions, lyrics,
seed, source hashes and checkpoint hashes. It is **planned, not rendered**.

Use physical GPU 1 and `app.generator.generate` with the studio's default merge
implementation. The user has authorized restarting the studio on a single card
if needed. Before doing so, allow current jobs to finish and preserve queued
work and keep-loop settings; those are held in memory by the server. Do not
restart a busy two-card studio just to obtain these CPU measurements.

One reserved training-evaluation fixture, row 2, seed 101, is shared across:

- Female + Pop: a vocal attribute with a production style.
- Country + Indie Rock: compatible guitar-band styles, with the highest global
  overlap among these exports.
- House + Acoustic Folk: contrasting arrangements that can form a coherent
  fingerpicked song over a club pulse.

The 25 jobs comprise one shared Off, six solos at energy 1 and 2 (12), three
equal mixes at energies 2, 2.5 and 2.8 (9), and three combined-caption references
with adapters off (3). Solo energy 1 matches each component's contribution in
the baseline equal mix at energy 2. Solo energy 2 matches the entire studio
budget. This distinguishes composition from the simple act of halving a solo.

All mix arms keep the same 50/50 proportions. Energies 2.5 and 2.8 are modest
gain-only alternatives available through the existing knob, not new merger
methods or declarations of safe musical strength. Caption references describe
the desired combination at the same BPM, lyric sheet and section structure.
They are examples of the target behavior, not ground-truth recordings.

Request 20 seconds consistently. The studio adds its normal ending allowance;
retain actual full outputs and durations, and compare matched 20-second listening
excerpts. Keep the first failures and silent/early-ending outputs. Randomize
presentation and conceal arm labels for the initial preference pass.

Assess each concept separately, intelligible/correct words, natural vocal
delivery, coherent groove and overall preference. A stronger concept score
alone is insufficient. A clip that only resembles one member is not a successful
combination. Fixed-history hidden errors and predictive KL are diagnostic, given
their earlier disagreement with listening in this repository.

Only advance a gain if it improves combined character without sacrificing words
or preference. If gains help, repeat the selected arm and baseline with the
reserved row 3 / seed 303, then check unequal shares and a three-control mix.
One fixture and seed cannot justify a studio default. If neither gain helps,
retain the current default and move to the diagnostic below.

## Next diagnostic: measure the response to real inputs

Capture inputs to the adapted projections on shared, teacher-forced audio
histories with adapters off. Include caption/lyric prefill, audio start, and
early/middle/later audio positions separately. Reuse the existing history
preparation helpers, with fixture and history hashes. Calibration data must be
separate from final listening confirmation.

For each projection and sampled input `x`, compute the actual local residual
`r_i = (alpha_i / rank_i) * B_i @ A_i @ x`. Accumulate the small matrix
`G_ij = mean(dot(r_i, r_j))`, normalizing each projection by its base output
mean-square magnitude before aggregating. Also retain per-projection results
and absolute residual levels: aggregation must not hide one damaging layer.
This requires only the base forward plus cheap low-rank contractions for the
sixteen adapters; it does not require sixteen separate full-model passes.

This is local linear response on fixed inputs. It misses upstream changes in
the composed model. Separately run selected singles and mixtures on the SAME
histories and inspect full-model behavior and late-position drift. Do not compare
hidden states from unrelated free-running histories as if their difference were
caused only by the merge. Do not automatically amplify a tiny mixed residual:
it can reflect conflicting controls or a poorly conditioned measurement.

If the gain study succeeds, a possible future automatic correction is
`g = sum(p_i * sqrt(G_ii)) / sqrt(p.T @ G @ p)`, with a small validated upper
bound, a near-zero-denominator fallback to 1, the existing energy ceiling, and
exact single-slider endpoints. This preserves the user's proportions. It is
only gain adjustment; any advanced merger must beat an equally well tuned
gain-only baseline. The output norm does not replace listening as the objective.

## If gain is insufficient: bounded layer-specific mixing

Keep adapter factors frozen and fit a small number of nonnegative mixing gates
per LM layer or projection. Start at the existing mixture and constrain changes
near it. Compare against the strongest accepted scalar-gain baseline. The same
coefficients can be merged into model weights before generation, retaining the
studio's normal inference cost.

Use explicit combined-caption teachers with matching lyrics and audio histories,
plus solo/end-point preservation examples. A teacher with only one constituent
caption cannot specify the desired combination. For vocal changes in particular,
semantic-token KL alone has previously missed audible attributes here; use
multiple diagnostics and accept models by held-out listening. Fit a small
representative pair screen first, without creating permanent special-case
adapters for every pair. Arbitrary three-way mixtures remain a separate test.

Any eventual studio mode must preserve zero, existing solo behavior, continuous
fader response and the user's relative intent. Store the algorithm version,
checkpoint identities and applied coefficients with the song so a future mixer
does not silently alter recall. Production adoption follows confirmation, not a
better training loss.

## How the published mergers fit

- [ZipLoRA](https://arxiv.org/abs/2311.13600) learns column-level gates while
  keeping adapter weights frozen and preserving constituent behavior. Its
  image results support trying learned gates, but not assuming they transfer
  to musical blends. The layer-gate experiment above is a simpler adaptation,
  not an implementation or reproduction of ZipLoRA.
- [TIES](https://arxiv.org/abs/2306.01708) trims small updates and resolves
  disagreeing signs. It is a reasonable offline comparator if gain and gates
  fail. Sign disagreement is not proof of unwanted musical interference;
  pruning can remove wanted character and threshold decisions can make fader
  changes discontinuous.
- [KnOTS](https://arxiv.org/abs/2410.19735) aligns LoRA updates using a joint SVD
  so other mergers can operate in a shared representation. A full-rank change
  of basis followed by the same linear sum leaves our current sum unchanged.
  Benefits require a changed merge operation or truncation, with their own
  tests. Basis alignment alone is not a missing correctness fix in the studio.

The method descriptions above come from those papers. Their prioritization and
the proposed music experiments are deductions from the local implementation,
training history and measurements, not published Music 3 results.

## Reproduction and validation

Run from `/ml2/music` with the installed environment:

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python sliders-conceptmod/analysis/studio_mix_20260907/geometry.py
/home/mikkel/anaconda3/envs/minimax-music3/bin/python sliders-conceptmod/analysis/studio_mix_20260907/build_screen.py
```

The low-rank Gram contraction was checked against explicit dense products using
different ranks. Analytical identical, orthogonal and opposing-update mixtures
were checked as well. Screen jobs resolve through the live studio registry and
pass its slider and project name validation. No weights, live studio code or
service settings were changed by this study.
