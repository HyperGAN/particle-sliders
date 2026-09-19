# Slider quality pilot: first listening results

The first automatic composite is **not ready to optimize**. The new listening
answers give a useful target, but neither the full fitted mixture nor the
smaller alternatives reliably select accepted examples across recipe families.
The first model and all original responses are preserved. `optimizer_score`
remains null.

## What the listener selected

All 23 trials were completed; both controls and both repeat checks passed.
The 17 actual ladders received six direct keep votes and eleven rejections.
Under the predeclared joint target, four passed every requirement, eleven
failed at least one, and two remain uncertain because preservation was unsure.
Those two were not converted to rejections or invented positive labels.

The frozen pilot groups both GAN checkpoints into one recipe family. That
conservative grouping is retained in the folds and bootstrap. An alias check
also keeps newer explicit smoke/paired names in that same family for leakage
detection; relabeling a recipe cannot make known weights become fresh evidence.

| Requirement | Yes | No | Unsure |
|---|---:|---:|---:|
| Intended change is audible | 11 | 5 | 1 |
| Lyrics are understandable | 9 | 8 | 0 |
| General audio quality is acceptable | 16 | 1 | 0 |
| Acceptable preservation | 10 | 4 | 3 |
| Would keep | 6 | 11 | 0 |

The smoke checkpoint won both direct matched preferences, at seeds 7 and 23.
Both paired-checkpoint examples were rejected for lyrics while their general
quality and direction received yes votes. This isolates the immediate missing
measurement: a polished, clearly changed output can still have bad singing.

## Comparison of automatic mixtures

The table uses 15 definite joint labels, holding out each recipe family in
turn. Normalization and fitting use the other families only. Lyric domains
overlap across these folds, so this is development evidence, not independent
validation. There are only four positives; these estimates are very uncertain.

| Model | Brier error, lower is better | Ranking AUC | Accepted positives at p >= 0.5 | False accepts |
|---|---:|---:|---:|---:|
| Training-fold prevalence constant | 0.229 | — | 0 / 4 | 0 / 11 |
| All eleven features | 0.419 | 0.523 | 1 / 4 | 5 / 11 |
| Lyric text diagnostics | 0.403 | 0.375 | 1 / 4 | 5 / 11 |
| Audio features without lyric text | 0.216 | 0.682 | 0 / 4 | 1 / 11 |
| Old recall, song distance and level | 0.475 | 0.432 | 1 / 4 | 6 / 11 |
| Concept change and enjoyment | 0.244 | 0.364 | 0 / 4 | 0 / 11 |

The audio-only variant is a modest ranking lead, but its current operating
point misses every accepted result. Calling that a successful selector would
hide the failure behind a mostly-reject decision rule. The full model also
learned implausible signs, rewarding some nuisance changes and penalizing
recall, consistent with confounding in this small mixed-axis sample.

## Follow-up measurements

An independent development probe decoded all pilot audio in full and in
contiguous ten-second chunks, without the lyric sheet, prompts, retries or
human labels. It measured raw decoder log probabilities using pinned
`whisper-large-v3-turbo` weights. Decoder confidence is a candidate diagnostic:
confident errors, musical intros and transcription failures remain possible.
Whisper itself uses [average token log probability as a decoding diagnostic](https://github.com/openai/whisper/blob/main/whisper/transcribe.py).

Against the lyric component ratings in this same pilot, fixed readout AUCs
were 0.292 for word recall, 0.715 for phrase alignment, 0.771 for chunked phrase
alignment and 0.806 for whole-clip decoder confidence. These are **in-pilot
associations**, not cross-validated accuracy. In particular, the accepted
distortion example still defeats the transcriber, and an unwanted repeated
phrase can receive high decoder confidence.

Eight subsequent two-component probability probes combined concept presence
or change with whole/chunk confidence or chunked phrase alignment, with ridge
penalties 1 and 10. Their recipe-heldout AUCs ranged from 0.159 to 0.318 and
none recalled an accepted example at p >= 0.5. The promising lyric association
has therefore not translated into a working general selector. These probes
were selected after reading the pilot and cannot certify themselves.

The old frozen gates have a separate problem: they veto two of the four
fully accepted examples (smoke seed 7 and the distortion example) and pass
six of the eleven rejected examples. Their thresholds and historical records
remain intact for comparison. The new validator now evaluates the gate-adjusted
probability and requires usable-example recall, so a raw probability model
cannot certify a selector whose gates discard the good audio. This protocol
revision precedes any new validation labels.

## What is ready for the next experiment

GPU 1 rendered all 18 new matched cases: two checkpoints, three new lyric and
arrangement fixtures, and three seeds each. Each case includes 0, 0.5 and 1;
base and reference audio are byte-identical across checkpoints for a matched
prompt and seed. Short and silent first draws are retained, with no retries.
The automatic feature manifest uses unit endpoints; intermediate clips remain
available and the old gate report also checks them.

These cases broaden the development benchmark. They do not provide new human
labels, and their known checkpoint families make them unsuitable for terminal
validation. The useful next judge needs audible concept and vocal clarity
readouts that survive these failures, followed by a fresh listening comparison.
More loss optimization against the present mixture would not answer that gap.

All 18 cases have complete features and score previews. Both candidates are
unscoreable because the model is unvalidated and the recipes overlap with
calibration; smoke also contains features outside the calibration range. The
old gates pass 1/9 smoke cases and 2/9 paired cases. These are proxy outcomes,
not listening verdicts, and cannot establish a new winner given the measured
gate errors above.

The target remains the conservative expected keep rate over matched prompt
and seed ladders. The learned approximation to that target has failed its
first generalization checks; no winning weights are being presented as a
validated objective.

## Reproduce and inspect

```bash
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
$PY analysis/slider_selection/ablate_features.py
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 $PY analysis/slider_selection/acoustic_probe.py
$PY analysis/slider_selection/pilot_report.py
$PY -m pytest tests/test_slider_quality.py tests/test_lm_score.py -q
```

- Original answers and first model: `eval/slider-quality/pilot/listening/`.
- Mixture folds and predictions: `eval/slider-quality/pilot/feature-ablation.json`.
- Component ratings, probes and gate disagreements: `eval/slider-quality/pilot/survey-analysis.json`.
- Raw confidence/crop measurements: `eval/slider-quality/pilot/acoustic-probe.json`.
- New render manifest: `eval/listen/quality-heldout/benchmark.json`.
- New unit feature manifest: `eval/slider-quality/heldout/`.
- Workflow and CLI: [slider_selection/README.md](../slider_selection/README.md).

Verification: 34 tests passed, including response persistence, playback
auditing, immutable mappings, gate-adjusted validation and recipe alias leakage.
The browser was also exercised with actual media playback and resume. GPU 1
completed the benchmark and feature jobs; the listening server remains available.
