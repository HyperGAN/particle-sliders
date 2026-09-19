# Repaired smoke: training length and metadata

**User listening update:** the 300-update smoke checkpoint is preferred.
The user described its audio as "fire" and wants to explore further steps;
300 is not established as the optimum. This supersedes the initial
recommendation to keep 120 based on last-token error. The historical baseline
is `repaired-tx-smoke`, trained for 120 updates.
The historical records do not establish an optimal training length. They do
show why a single GAN loss cannot provide a stopping rule for these runs.

## What the existing logs establish

Every original smoke update uses all four training rows. These are generator
updates, not epochs over a large song dataset. Its final 20-update windows are:

| Updates | Mean direction cosine | Mean normalized teacher error | Mean change / teacher change |
| --- | ---: | ---: | ---: |
| 41–60 | 0.829 | 0.574 | 0.932 |
| 61–80 | 0.890 | 0.459 | 0.933 |
| 81–100 | 0.895 | 0.445 | 0.926 |
| 101–120 | 0.908 | 0.418 | 0.946 |

The teacher error continues to decline through the final window. There is
no observed post-120 plateau in that run because training ended there.
The existing launcher saves only final weights unless `SAVE_EVERY` is set;
the original run has no intermediate weights to reload or listen to.

The later runs are not evidence for a particular step count: they also change
conditioning, feature matching, mining, adversarial weight and/or LR schedule.
Their loaded-weight audit is retained in the existing repair report.

| Checkpoint | Updates | Heldout teacher error | Listening evidence |
| --- | ---: | ---: | --- |
| Original smoke | 120 | 0.517 | User preferred; no gibberish in listened clips |
| Repaired with neutral lyric hold | 120 | 1.365 | Conflicting hidden-state targets; no inferred audio verdict |
| Unconditional with mining and LR decay | 300 | 0.784 | No inferred audio verdict |
| Paired, conditioned, mined, lower adversarial weight | 180 | 0.497 | User reported gibberish despite better hidden fit |

## How to read the metadata

- `pperc` is normalized last-token L2 teacher error: unchanged neutral is 1,
  exact teacher is 0. It is a useful fixed-target progress diagnostic.
- `cos_pos` measures the direction of the last-token change. Read it with
  `mag_ratio`; a correctly directed change can still be too small or too large.
- `g_adv`, `fm` and `d_loss` depend on the changing critic. Comparing their
  minima across runs or treating their convergence as audio quality is not
  justified. Even feature matching uses a learned, moving feature space.
- `d_pen = 0` means sampled critic gradients are within the cap. It is not
  convergence or proof of a dead generator gradient. The original smoke has
  many zero-cap steps while making substantial teacher-directed progress.
- `grad_norm` is the total LoRA gradient before elementwise clipping, not
  the actual Adam update size. `gadv_norm` measures only the weighted
  adversarial term on the first batch row. Their ratio is not a decomposition
  of the complete update into objectives.
- `edrift_p` is end-margin drift on cached teacher-forced continuations.
  It is an ending diagnostic, not a transcription or long-form audio verdict.
- Missing historical metadata stays unknown. Historical default zero gradient
  fields do not establish dead gradients.

This separation also agrees with the evaluation literature's treatment of
distribution matching, mode loss and overfitting as distinct requirements;
it does not supply a music-quality score for this project. See
[An empirical study on evaluation metrics of generative adversarial networks](https://arxiv.org/abs/1806.07755).

The log labeled step N is written after update N but its forward metrics were
computed before that update. The curve therefore places those metrics at
N−1 completed updates; a saved `stepN` file contains N completed updates.
Saved checkpoints need their own reload evaluation.

## Controlled test

One continuous seed-7 run on physical GPU 1 retains the smoke card, with
constant generator LR 0.0005, cap coefficient/threshold 1, no conditioning,
batch feature matching, weights 1/1, no mining, no neutral lyric hold, and the
existing end regularizer. It trains to 300 updates and saves every 30 updates.
The only experimental variable within this trajectory is completed updates.
It is a fresh repeat, not an exact resume of the original GAN optimizer and
critic: the original checkpoint contains LoRA weights only.

The command is:

```bash
STEPS=300 SEED=7 SAVE_EVERY=30 bash scripts/train_lm_gan_bcap.sh 1 smoke-steps-s7-20260904
```

Source snapshots and hashes are in the run's `provenance/` and
`experiment.json`. The launcher keeps its 120-update, seed-7, final-only
defaults and refuses to overwrite an existing run.

The audio comparison fixes row 0, scales 0/+1, generation seeds 7/23,
and 20-second duration, with no seed retries: original smoke and the new
60/120/180/300-update weights. Each set includes positive-caption references.
Selection among these candidates requires listening to voice direction,
lyric order/additions, song continuity and artifacts, separately from hidden
teacher fit. Word recall alone misses additional nonsensical words.

## Choosing the next budget

Use checkpoints to find a region where heldout teacher fit and the desired
audible effect improve, then choose the earliest checkpoint whose audio is
competitive without regressions. Do not automatically select the lowest
adversarial loss, highest cosine or final update. A plateau or rising heldout
error can narrow the listening shortlist; neither certifies audio quality.

After a candidate interval is identified, repeat it with independent training
seeds and more prompt/lyric contexts. A training seed changes optimization;
a rendering seed changes sampling. Two rendering seeds cannot substitute for
multiple training runs. Repeatedly used heldouts become development validation;
reserve fresh prompts for confirmation. The user's subsequent listening
comparison favors 300 over the earlier recommendation of 120.

## Completed checkpoint sweep

The 300-update run completed successfully. All numeric training records are
finite. Reload evaluation covered ten distinct saved weights plus the original
smoke, using four training and four heldout rows. All **88/88** full-sequence
zero-scale checks were bitwise identical to the pristine base model.

| Updates | Training last-token error | Heldout last-token error | Heldout direction cosine | Heldout lyric-span error |
| --- | ---: | ---: | ---: | ---: |
| 30 | 0.758 | 0.784 | 0.660 | 0.817 |
| 60 | 0.492 | 0.636 | 0.774 | 0.765 |
| 90 | 0.410 | 0.527 | 0.853 | 0.739 |
| 120 | 0.413 | **0.516** | **0.859** | 0.728 |
| 150 | 0.487 | 0.552 | 0.836 | 0.718 |
| 180 | 0.462 | 0.574 | 0.820 | 0.715 |
| 210 | 0.499 | 0.588 | 0.810 | 0.725 |
| 240 | 0.489 | 0.548 | 0.838 | 0.719 |
| 270 | 0.452 | 0.522 | 0.855 | 0.745 |
| 300 | 0.496 | 0.543 | 0.841 | **0.711** |

The independently reloaded original smoke scores 0.517 heldout last-token
error, cosine 0.860 and lyric-span error 0.726, closely matching the repeat's
120-update point. The runs are not bitwise identical; small training-log
differences begin before the first intermediate save. No exact-reproduction
claim follows from sharing a seed and training card.

Within this trajectory, 120 has the lowest sampled heldout last-token error.
The difference between 120 and 270 is small (about 0.006), so this is evidence
for using the earlier useful region, not a precise optimum at step 120.
300 updates costs 2.5 times the update budget and finishes with about 5%
higher last-token error. Training error also worsens, so describing the later
behavior as classical overfitting would be premature; the game oscillates.

The lyric-span diagnostic has a different preference: 300 improves it slightly
over 120. That directly illustrates why even the fixed-teacher measurements
do not collapse to one quality score. Neither improvement establishes that
the lyrics in generated audio are intelligible.

The initial recommendation to concentrate around 90–120 was based on the
last-token diagnostic and is withdrawn following the user's listening report.
The next refinement explores beyond 300 with matched audio samples, retaining
the preferred 300-update weights as the reference. KL over token predictions
is being investigated as a separate diagnostic, not assumed to be a quality
metric merely because it might favor the preferred checkpoint.

Artifacts:

- [Training curves and checkpoint measurements](step_sweep_20260904.png)
- [Full reload audit and per-row measurements](lm_steps_20260904_evaluation.json)
- [Historical metadata and window summaries](step_sweep_20260904.json)
- [Checkpoint table](step_sweep_20260904.tsv)
- [Matched listening comparisons](../../eval/listen/gan-bcap-steps-20260904/README.md)

The plot and machine-readable comparison are reproducible with:

```bash
python analysis/gan_bcap/step_report.py \
  --run-dir models/gan-bcap-repair/smoke-steps-s7-20260904 \
  --evaluation analysis/gan_bcap/lm_steps_20260904_evaluation.json \
  --output-prefix analysis/gan_bcap/step_sweep_20260904
```

## Automated checks on the listening set

The existing scorer completed all ten folders. Its exit status is 1 because vetoes fired; this is not a crashed scoring run. Every candidate fails the seed-7 song-preservation proxy and passes at seed 23. The 180-update seed-7 clip additionally fails the lyric gate.

| Checkpoint | Seed 7 word recall | Seed 23 word recall | Seed 7 additional issue |
| --- | ---: | ---: | --- |
| Original smoke | 1.0000 | 0.9231 | None beyond song-distance veto |
| 60 updates | 1.0000 | 1.0000 | None beyond song-distance veto |
| 120 updates (repeat) | 0.9231 | 0.9231 | None beyond song-distance veto |
| 180 updates | 0.6154 | 1.0000 | Lyric recall veto |
| 300 updates | 1.0000 | 1.0000 | None beyond song-distance veto |

The repeat at 120 updates has extra words in its ASR transcript despite matching the original's hidden-state scores closely. The 60-update seed-7 transcript contains the expected two lines, while the 180-update seed-7 transcript omits the chorus. These are transcription diagnostics; ASR can mishear, and repeated sung refrains are not inherently gibberish. The subsequent user feedback prefers the 300-update checkpoint; ASR is not a replacement for that listening judgment.

Nine of ten candidate clips have short-clip ending warnings. Absence of that warning in the remaining 20-second clip does not establish a natural long-form ending.

The original smoke's base, +1 and positive-reference WAVs reproduce byte for byte at both seeds. Every checkpoint shares byte-identical base and reference audio for each seed. All 40 WAV files are present (14 distinct renders, with shared references copied).

[Open the side-by-side audio players](../../eval/listen/gan-bcap-steps-20260904/index.html). Full transcripts and measurements are in [the audio summary](step_audio_20260904.json) and each listening folder's `lm_scores.json`.
