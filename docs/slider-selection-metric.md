# One objective for selecting slider losses

Proposal and implementation, 2026-09-04. Scope: LM sliders first. The
[calibration workflow](../slider_selection/README.md) is implemented; its
first listening pilot is complete. The initial automatic mixture failed its
development checks; see [the results](slider-quality-pilot-results.md).
An improved predictor and independent validation are still required.
Existing scoring contracts and thresholds are unchanged. The original transcript
audit used cached measurements; subsequent work adds frozen audio-model
measurements, listening collection and new development renders.

## Recommended target: usable-slider rate

Optimize the probability that a listener would keep the slider's result:
the intended change is clearly audible, the requested words remain
intelligible, the music remains coherent and appealing, and attributes
outside the requested change remain acceptable.

One observation is a fixed prompt/lyric/seed **ladder**, including zero and
the declared operating settings. For bipolar sliders both poles must work.
The success label is the conjunction of those requirements over that ladder.
At zero, require the unchanged base behavior; audible movement is assessed
only at nonzero settings where the declared control calls for it.
Increasing effect strength beyond the useful range earns no extra credit;
a no-op fails the audible-change requirement. A technically intact but
unappealing result does not earn a success label.

The automatic implementation estimates this joint success probability from
rendered-audio measurements and listening labels:

```text
p_i = calibrated P(listener would keep ladder i | its measured features)

usable_score = 100 * LCB90(mean over prompt rows(mean over seeds(p_i)))
```

`LCB90` is a one-sided 90% lower confidence bound. Equal prompt weighting
prevents an easy prompt with more seeds dominating. Use a paired, hierarchical
bootstrap of prompt rows and seeds for candidate comparisons, with fixed
evaluation coverage. Model fitting/calibration uncertainty must also be
estimated; a bootstrap of render rows alone only measures render variability.
Do not call an interval with very few independent rows a robustness guarantee.
Even a calibrated score of 80 is evidence about the benchmark distribution,
not a guarantee for every song or a proof that a judge cannot be exploited.

This is one scalar for an outer experiment optimizer: train a candidate,
render it, measure it, compute the score. It need not be differentiable and
does not prescribe a new training loss. The underlying measurements remain
visible for diagnosing why a loss won or failed.

If the optimizer also needs feasibility in the same number, use disjoint
ranges: `-1 - V/(1+V)` for a failed validated rejection rule, and
`usable_score` for feasible candidates. `V >= 0` is normalized violation
severity; a stronger effect cannot compensate for a failure. Missing required
measurements produce **UNSCOREABLE**, never a passing numeric value. Freeze
the rejection rules before loss search. The historical frozen verdict remains
separately reported: this proposal does not silently remove its vetoes.

## Measurements to combine

| Requirement | Existing evidence to retain | Missing or improved measurement |
| --- | --- | --- |
| Audible concept control | DSP and embedding changes as diagnostics | A concept-specific audio readout, checked on independent prompt pairs and listeners; compare movement with the same prompt's slider-off distribution |
| Intelligible requested lyrics | Current ASR recall | Time-local, order-sensitive lyric alignment, insertions and omissions, permitted phrase repeats, and an independent audio intelligibility judgment |
| Musical quality | Level, silence and ending checks; artifact diagnostics | Vocal naturalness and musical coherence judged from audio, including windows around transitions |
| Preservation | Existing song-distance diagnostic | Explicit per-axis nuisance attributes: tempo may change for a tempo slider; perceived vocal presentation may change for its intended axis |
| Reliability and control | Seed and scale metadata | Multiple heldout prompts/seeds; useful intermediate settings; both poles when claimed; long-form behavior |

Fit a small regularized success predictor first, with very few predeclared
features. Keep separate component labels to find blind spots, and calibrate
the **joint** keep/reject probability. Multiplying marginal probabilities
would incorrectly assume the criteria are independent. An arbitrary weighted
sum or geometric mean has no automatic interpretation as a success rate.

Concept detectors should use fixed descriptions of sound, identical nuisance
context on opposite poles, and frozen model versions. Music-text similarity
can be one candidate input; it needs local validation for each concept. For
energy, evaluate performance intensity separately from a gain-only control.
Do not infer perceived vocal presentation from pitch alone. Do not normalize
effect by the magnitude of one caption-swap render or reuse the failed LM
embedding monotonicity/null tests as proof of efficacy.

Quality judgments must consume audio. An ASR transcript or an explanation
generated from it cannot establish that the actual vocal sounds intelligible.
A frozen external audio judge is a candidate feature extractor, not ground
truth; validate its errors against the same listening examples. A judge trained
only on clean speech or instrumental music needs additional singing validation.

For long renders, measure the whole file in windows. The current
`WhisperBackend.measure` calls the processor once and pools a single encoder
sequence; it does not implement a whole-song window loop. Full-file DSP ending
checks do not imply full-file lyric or coherence coverage.

## What the present evidence rules out

The [current LM contract](../LM-SCORING.md) calls its scalar `E` uncertified:
its magnitude depends on one caption swap, its direction tests do not establish
efficacy, and it was not validated as a ranking among acceptable sliders.
The [GAN comparison](lm-gan-bcap.md) also records a later run with better
hidden-state metrics but worse reported intelligibility. These are development
counterexamples, not an untouched validation set for a new metric.

I recomputed two cheap transcript diagnostics on the cached unit-scale clips:
sheet-vocabulary precision (fraction of transcribed words belonging to the
sheet) and ordinary word error rate against one copy of the sheet.

| Run | Seed | Existing recall | Vocabulary precision | Ordinary WER |
| --- | ---: | ---: | ---: | ---: |
| Repaired smoke | 7 | 1.000 | 0.696 | 0.769 |
| Later paired mined | 7 | 1.000 | 0.800 | 1.308 |
| Repaired smoke | 23 | 0.923 | 0.833 | 0.923 |
| Later paired mined | 23 | 1.000 | 0.484 | 1.385 |

The listening report prefers smoke's intelligibility at the run level; it does
not provide a new formal label for each seed. Precision exposes the extra-word
problem at seed 23 but ranks the later run higher at seed 7. WER ranks smoke
better at both seeds, but is not a solution: the locked ears-PASS tempo run has
WER **1.846**, while the locked ears-FAIL energy-v16 plus clip has **1.769**.
Valid repeated phrases can inflate ordinary WER. Neither statistic should
become a gate without proper alignment and fresh listening validation.

The 14 locked labels are folder verdicts, including borderline and unit-scale
coverage issues. They are not a preference ranking among good candidates.
No response JSONL files were found under `eval/listen/abtest/`; that directory
contains a prepared session, not observed listening responses in this checkout.
There is insufficient evidence here to fit and validate a trustworthy automatic
selection function today. Rearranging the current metrics cannot supply the
missing observations of concept control and vocal quality.

Reproduce the transcript audit with the standard library:

```bash
python analysis/slider_selection/audit_cached.py
```

The [CSV](../analysis/slider_selection/cached_lyrics.csv) includes source score
hashes, unit-scale measurements, and the original folder labels. It emits no
transcripts or song text and does not overwrite existing scores.

## Smallest useful experiment

1. Freeze a compact benchmark of neutral prompts and lyrics outside training,
   at least three generation seeds, and the same scale ladder for all candidates.
   Include distinct lyric sheets and arrangements. Use GPU 1 when GPU 0 hosts
   the studio. Match training budgets when comparing losses, and include several
   training seeds for conclusions about a recipe rather than one checkpoint.
2. Backtest candidate features on existing acceptable, weak and broken runs,
   including smoke and the later regression. Add no-op, reversed direction,
   gain-only, hiss, repeated-word, and new-song controls. Use legitimate
   caption-only references to check that a target is realizable, without treating
   a single reference waveform as the unique correct song.
3. Collect blind labels for concept movement, lyric intelligibility, musical
   quality, nuisance changes, and overall keep/reject. Include paired preference
   judgments among acceptable candidates to check that the scalar chooses the
   more appealing results. A pilot of several dozen judgments can expose holes;
   its sample size is not a validation guarantee.
4. Fit and compare simple combinations using grouped splits. Hold out whole
   prompt/lyric families and checkpoint/recipe families; related seeds, scales,
   shared references and nearly identical checkpoints must not leak across folds.
   Report false acceptance of broken audio, probability calibration, and how
   often the selected winner agrees with blind preference. Overall correlation
   alone is insufficient for an optimizer that selects the top candidate.
5. Freeze the successful scorer and run the loss search on a development suite.
   Confirm finalists on a separate locked suite and 60–90-second renders.
   Do not tune thresholds on those results. Check finalists against fresh
   listening labels, including a few low-score controls, to detect score gaming.

If this experiment fails, add or replace a measurement that sees the missed
audible property. Do not keep adjusting weights until the old examples sort
correctly. If it succeeds, the repeated loss experiments get the one automatic
number requested, with a measured connection to listening quality.

## External evidence and its limits

[A large human preference study](https://arxiv.org/html/2506.19085v1) found
music-trained CLAP variants useful for text alignment and as embeddings for
distributional quality evaluation. Its clips were ten-second instrumentals,
and its main comparisons were between generation models; this does not validate
detecting singing garble or ranking nearby slider checkpoints. FAD is a
distribution comparison and does not by itself measure whether an individual
slider obeys the requested edit.

[MusicEval](https://arxiv.org/abs/2501.10811) demonstrates learning music
assessment from expert ratings, with 2,748 clips and 13,740 ratings. It supports
trying a calibrated learned combination. Whether such a judge generalizes to
these slider artifacts remains a local experimental question.
