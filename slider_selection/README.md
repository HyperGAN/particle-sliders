# Slider quality calibration

**Pilot completed:** the first combined metric failed the development checks.
See [the listening results](../docs/slider-quality-pilot-results.md). All responses
and the first model are preserved; the optimizer score remains unavailable.

The implemented workflow measures actual LM renders, collects blind listening
judgments, fits a regularized probability model, and produces one conservative
selection score. The first model is **exploratory**. A separate disjoint
validation suite must pass before `optimizer_score` is numeric. Missing
measurements, insufficient prompt/seed coverage and calibration overlap fail
closed. Existing frozen LM gates remain vetoes.

The target and its limits are in
[the proposal](../docs/slider-selection-metric.md). The current implementation
evaluates exact unit endpoints, including both poles of bipolar sliders.
Intermediate settings and long-form endings require their own evaluations;
short endpoints do not certify those behaviors.

## What runs

- [features.py](features.py): frozen CLAP concept margins, Audiobox enjoyment
  and production quality, full-band stereo spectral measurements, phrase-aware
  lyric diagnostics, and existing song-distance/level measurements. Every
  audio file is covered in windows; cache keys include content, model revisions
  and package versions. The feature reducer is hashed into the measurement ID.
- [dataset.py](dataset.py): immutable source manifests and anonymous study
  assets. Historical folder labels are diagnostic metadata, never training
  labels. Exact audio hashes prevent silently changing the clips after freezing.
- [model.py](model.py): ridge logistic fitting, recipe-bootstrap model draws,
  prompt/seed-bootstrap scoring, and disjoint validation. Shared lyrics,
  prompts, recipe families or audio cannot masquerade as fresh validation.
- [server.py](server.py) and [listen.html](listen.html): playback checks,
  keep/reject and component ratings, matched preference comparisons, two
  rejection controls and two repeat checks. Answers persist on the server and
  in the browser, with download and resume. Private mappings are outside the
  served directory. Synthetic test responses exist only in temporary tests.

The probability model has eleven input features. Weights are learned from
the new listening answers, rather than assigned from historical PASS labels.
The bootstrap includes coefficient variability and render variability; it
does not prove robustness to systematic judge errors. Validation criteria are
development acceptance criteria, not a universal guarantee. After the pilot
exposed false vetoes, validation protocol `gated-selector-v2` additionally checks
gate-adjusted probabilities and requires recall of at least half the usable
examples. A raw classifier cannot certify a selector with broken gates.

## Environment

Use the existing `minimax-music3` environment. Its numerical stack is unchanged.
The optional quality package is installed separately under the task cache:

```bash
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
$PY -m pip install --no-deps --target /ml2/music/.cache/slider-quality/python audiobox-aesthetics==0.0.4
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/ml2/music/.cache/huggingface
```

The CLI adds that dependency directory itself. Model weights have already been
downloaded into the Hugging Face cache. For subsequent offline runs set
`HF_HUB_OFFLINE=1`. Do not install the repository requirements file.

Pinned sources:

- [CLAP](https://huggingface.co/laion/clap-htsat-unfused), revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.
- [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics),
  model revision `9b1dd8e5df9af7216e836a98974fe3b82c56ded6`.

These are candidate readouts, not validated slider judges. In the local probes,
gain changes move CLAP slightly; a repeated fragment increases its vocal margin;
and high-frequency hiss above the quality model's input bandwidth barely moves
its quality score. Full-band HF power and flatness expose that last change.

## First study

From the repository root:

```bash
$PY scripts/slider_quality.py prepare --out eval/slider-quality/pilot
$PY scripts/slider_quality.py measure \
  --manifest eval/slider-quality/pilot/manifest.json \
  --out eval/slider-quality/pilot/features.json
$PY scripts/slider_quality.py session \
  --manifest eval/slider-quality/pilot/manifest.json \
  --out eval/slider-quality/pilot/listening
$PY scripts/slider_quality.py serve \
  --session eval/slider-quality/pilot/listening \
  --features eval/slider-quality/pilot/features.json \
  --bind 0.0.0.0 --port 8902
```

The first manifest and 23-trial session are already built. Creation refuses
to overwrite them; resume with `serve`. The current pilot was frozen with
17 ladder trials, two no-op/silence controls, two repeats and two matched
preferences. Later sessions can request more comparisons with
`session --max-preferences 10`; existing sessions are unchanged.

When every trial is answered, the server attempts fitting automatically and
writes `model.json`, `score-preview.json`, and `calibration-status.json` beside
the response file. Controls and repeats must pass; at least eight definite
ladder labels, including three usable and three rejected, are required even
for an exploratory fit. “Unsure” is not a fabricated middle-quality label.
If any requirement is “No,” the joint keep label is zero; all requirements
must be “Yes” to make it one. Controls and repeats do not inflate the training
sample count. `Previous trial` permits correcting an answer.

Manual fitting, including downloaded response recovery:

```bash
$PY scripts/slider_quality.py fit \
  --features eval/slider-quality/pilot/features.json \
  --session eval/slider-quality/pilot/listening \
  --responses eval/slider-quality/pilot/listening/responses.jsonl \
  --out eval/slider-quality/pilot/listening/model.json
```

## New renders and loss comparisons

`scripts/render_quality_benchmark.py` freezes three new arrangement/lyric
fixtures and seeds 101, 202 and 303, renders the smoke and paired checkpoints at
0, 0.5 and 1, and writes explicit prompt/candidate/recipe metadata. It loads the
pipeline once and copies byte-identical base/reference files across checkpoints.
It retains short and silent first draws and never retries seeds. Checkpoint
weights are strictly loaded and remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/render_quality_benchmark.py
CUDA_VISIBLE_DEVICES=1 $PY scripts/lm_score.py score \
  eval/listen/quality-heldout/*-s* --cache_dir eval/lm_score_cache --device cuda:0 \
  --out eval/slider-quality/heldout-lm-score.tsv
$PY scripts/slider_quality.py prepare --out eval/slider-quality/heldout \
  --folders eval/listen/quality-heldout/*-s*
$PY scripts/slider_quality.py measure \
  --manifest eval/slider-quality/heldout/manifest.json \
  --out eval/slider-quality/heldout/features.json
```

Those renders are development evidence outside slider training, **not terminal
metric validation**: the checkpoint families already occur in the first study.
Do not weaken the overlap check to certify them. Future validation needs new
recipe families as well as new prompt/lyric families, both classes of listening
labels, enough matched preferences, and the frozen model's measurement protocol.

```bash
$PY scripts/slider_quality.py validate --model PATH_TO_EXPLORATORY_MODEL \
  --features PATH_TO_DISJOINT_FEATURES --session PATH_TO_DISJOINT_SESSION \
  --out PATH_TO_VALIDATION_RESULT
$PY scripts/slider_quality.py score --model PATH_TO_VALIDATED_MODEL \
  --features PATH_TO_CANDIDATE_FEATURES --out PATH_TO_SCORE_BOARD
```

`score` emits `optimizer_score=null` while the model or evaluation is
unqualified. `preview_lower90` is an exploratory estimate only. For qualified
candidates, failed frozen gates occupy a negative score range and feasible
candidates get `100 * lower90(mean usable probability)`. Every candidate must
cover the same prompt/seed slots; missing or duplicated observations cannot
improve the score. Prompt rows receive equal weight.

Run the checks without GPU:

```bash
$PY -m pytest tests/test_slider_quality.py tests/test_lm_score.py -q
```

The browser was also tested with real media playback, server saving, resume,
and a narrow viewport. The high-frequency cancellation test verifies that
stereo content cannot disappear from the spectral diagnostic through downmixing.
