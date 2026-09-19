# Bounded baseline versus FM gradient limiting, continued

> **Current mode: fast sampling.** The user requested only one or two samples so
> training can continue. The [fast protocol](fast-protocol.json) supersedes the
> broad evaluation below: one fixed prompt, seeds 7 and 23, two clips per recipe
> per checkpoint. Existing controls and 750 clips are reused. CPU scoring runs
> alongside subsequent training; no full diversity audit or large render survey
> runs during the sweep. Both recipes still target 2100, with sparse severe
> audio checks and numerical failure stops. The previous broad artifacts remain
> preserved. The older convergence survey stays paused.

The user liked both 660-update candidates and requested continuing both to see
whether one keeps improving for longer. Both resume their complete saved games:
generator, critic, optimizer moments, sampler, RNG and training history.

The only recipe difference remains the weighted feature-matching parameter
gradient cap of 10. Both retain raw critic features, original training prompts,
constant learning rate and the final parameter-step limit of 2. The original
signed schedule fields remain 600/660 because their schedule is constant; the
resource endpoint advances without changing the resume signature.

The [frozen protocol](protocol.json) declares checkpoints at 750, 900, 1050,
1200, 1350, 1500, 1650, 1800, 1950 and 2100. Each stage trains both surviving branches before
comparing them on eight reserved lyric sheets and four fixed seeds. The first
two prompt scores are published early as a descriptive screen. All eight
prompts contribute to the completed comparison. The first two prompts were
already examined in the 660 screen; the other six expand its coverage.

Original 600 and both 660 checkpoints remain audio anchors. Identical controls
and existing anchor audio are reused only with matching fixture and checkpoint
hashes on GPU 0; every reuse has a provenance record. The old study's prompt 0
was rendered on GPU 1 and is rendered again here on GPU 0. There are no seed
retries or removal of inconvenient clips.

The fixed proxy and the separate lyric, target-description, tail-score and
two-view diversity diagnostics are unchanged. Five paired comparisons per
stage are planned: FM minus baseline; each arm minus its own 660; and each arm
minus its previous checkpoint. Intervals average seeds within prompts and
allocate family alpha .05 across 100 interval tails. A gain of .05 is the
predeclared proxy tolerance. These approximate intervals concern this metric
and prompt sampling assumptions, not independently validated musical quality.

An arm stops on nonfinite training, at least two consensus diversity alarms
among eight prompts, at least four near-silent clips among 32, or widespread
severe lyric mismatch: at least eight clips across at least three prompts with
word precision and phrase accuracy both below .35 while both references reach
.65 on both measures. This is an operational ASR alarm, not a certainty about
audible gibberish. Low sheet coverage alone never stops it: slower phrasing
can explain reduced coverage. The [break-check amendment](break-check-amendment.json)
was made before any continuation render or score. All failures remain reported. The 2100 ceiling is a resource stop,
not convergence; best-over-checkpoints selection is exploratory and needs
fresh confirmation. There is only one shared training ancestry and seed.

The useful comparison is each recipe's best usable checkpoint and the range
over which it stays useful. More survivable updates help only when the target
effect and audio quality remain competitive. No catalog weights or slider
defaults are promoted by this experiment.

The controller runs as
`music-gan-paired-continuation-20260905.service` on physical GPU 0. It pauses
the earlier convergence audio job and resumes that job when this campaign
finishes. The earlier samples, scores and its eight-to-sixteen-prompt expansion
remain intact. Unrelated GPU processes are not stopped; new jobs wait for
enough free memory. Training and rendering are sequential on GPU 0, with one
CPU scorer working alongside rendering. Restart resumes saved full states
into fresh run directories and retains partial audio.

[Live listening and measurements](../../../eval/listen/gan-paired-continuation-20260905/index.html)
· [Controller](../paired_continuation.py) · [Status](status.json)

The user extended the minimum requested continuation beyond 2000 before any new
audio scores were collected. The [amendment](extension-amendment.json) preserves
the original 1800 protocol and raises the ceiling to 2100, with the interval
comparison budget adjusted before evaluation. This changes no training recipe.
