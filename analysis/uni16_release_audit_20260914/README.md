# Per-slider checkpoint release shortlist

Live page: <http://100.90.104.57:8888/uni16-release-audit-20260914/>

The current selection policy favors later checkpoints when both quality
measures are close. Description similarity remains visible as a diagnostic
and has no influence on selection. Lyrics are informational only, following
the user's report that they are consistently fine.

## Current decision rule

`selection-policy.json` and `selection.py` define `quality-later-v2`.
For each slider, exclude candidates with integrity failures, technical flags,
or near-identical waveform flags. Find the highest clean mean enjoyment (CE)
and highest clean mean production quality (PQ), separately. Shortlist
candidates within **0.2 original score points on both dimensions**, then prefer
the latest checkpoint from the new run. No combined quality score is used.

The bounds always refer to the best clean values across all measured
candidates. Several small checkpoint-to-checkpoint differences cannot chain
into a large overall regression. If no candidate meets both bounds, the
report calls for listening to resolve the tradeoff. The legacy released 660
comes from a different run: it is recommended only when no new-run candidate
meets both bounds. It is not treated as 60 updates after the new step 600.

The tolerance is a provisional preference rule, not an estimate of audible
indifference, statistical equivalence or confidence. Every slider shows the
picks at 0.1, 0.2 and 0.3 to expose threshold sensitivity. Quality anchors,
shortfalls, per-arrangement means, worst individual CE/PQ values and paired
differences remain visible. After the shortlisted group, other clean rows
are ordered latest first for inspection; that order is not a quality ranking.
All four matched cases receive equal weight: two arrangements times two seeds.

The CLAP diagnostic is a change in the target-versus-comparison text margin
relative to adapter-off audio. Its wording is not calibrated to the user's
style preferences. It is neither a quality score nor a direct measurement
of audible slider strength. Listening is still needed to decide whether a
checkpoint produces the intended style.

## Measurements and provenance

Two low-priority CPU workers reuse installed CLAP and Audiobox Aesthetics
models and content-addressed caches. Existing Whisper transcripts are read
when available; no fresh transcription pass is performed. Missing lyric
measurements remain null. Workers score complete four-case sets at new
steps 600, 1000, 2000, 3000 and 3400 when present, plus released 660.
Every candidate shares adapter-off and positive-caption controls, rows 2/3,
seeds 1709/2903, strength 1 and a 20-second request. Workers follow future
milestone recordings through the approved training budgets. This audit
launches no training or GPU rendering.

`protocol.json` freezes the measurement sources and **historical** decision
rule. The original worker JSON files retain that rule's composite scores for
reproducibility. The current publisher applies the separate selection policy
to their unchanged raw measurements. In public `summary.json`,
`measurement_protocol` contains the frozen history and `selection_policy`
is authoritative for current suggestions. The selection source and policy
hashes accompany every report. The CSV follows the current default policy.

Previously, the blend used 50% scaled style gain, 25% scaled enjoyment and
25% scaled production, plus a 25% excess high-frequency penalty. Before
component clipping, 0.05 style-margin gain contributed as much as one full
CE point or one full PQ point. Those exchange rates were not validated
against listener preferences, and clipping could further change their
relative influence across cases. That blend is no longer used for selection.
The prior report, page and documentation are preserved in
`evidence/before-quality-selection-v2/`.

## Technical screening and limits

Technical flags include relative silence, severe shortening, clipping,
excess high-frequency energy, stereo cancellation, channel loss, unusually
long interior gaps, exact repeated half-second blocks, identical output
across seeds and output identical to adapter off. Musical rests, loops and
brightness can be intentional. Flags retain the original clips for inspection.
Training loss, update count and weight norm are not evidence of good audio.
Every export is checked for finite tensors; new exports are compared
tensor-for-tensor with their pinned full state. Released 660 has an
export-only audit.

`waveforms.py` separately compares prompt/seed cases within a checkpoint
using absolute normalized correlation after mono box filtering and
interpolation to 4 kHz. A warning requires correlation above 0.995 while both
corresponding controls stay below 0.98. It tolerates gain and polarity
changes, not timing shifts. Its method and source hash are saved. A flagged
candidate is excluded from current quality anchors and recommendations.

Recommendations are provisional. Two seeds per prompt cannot clear the
existing four-seed diversity gate. Exact or near-duplicate detection does
not establish healthy musical variety. Pitch, subtle timbral artifacts,
full-song structure, other strengths and slider combinations remain
unvalidated. The CSV lists exact paths and hashes; it does not publish
or replace studio weights.

The model roles are documented by the original projects:
[Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
predicts enjoyment and production among its aesthetic dimensions;
[CLAP](https://github.com/LAION-AI/CLAP) supplies audio/text representations.
Their scores remain proxies.

The first two step-600 candidates received fresh ASR before the scope was
narrowed to cached transcripts. That protocol and those measurements remain
under `evidence/`; changing the lyric scope did not change the other metrics.

## Operation

```bash
systemctl --user status music-uni16-release-audit@0.service music-uni16-release-audit@1.service
journalctl --user -u music-uni16-release-audit-page.service -n 15 --no-pager
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m pytest -q test_audit.py test_selection.py
```

The publisher refreshes `summary.json`, `index.html` and `recommendations.csv`
every five seconds. The listening server binds 0.0.0.0:8888. Original audio
stays in the training campaign folder. Selection tests verify fixed quality
bounds, independent dimensions, later preference, style/lyric invariance,
legacy-run handling, invalid measurements, technical flags, matched cases,
paired evidence and automatic selection of future checkpoints.
