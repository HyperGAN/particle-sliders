# LM GAN with the Gaussian reference cap

**Current listening update:** the user prefers the 300-update constant-LR
smoke checkpoint from the [step sweep](../analysis/gan_bcap/step_findings_20260904.md).
This is `smoke-steps-s7-20260904_last.safetensors`, not the older mined
`gender-gan-bcap` 300-update ablation below. The original smoke remains the
historical reference. More training lengths are being sampled; last-token
hidden error is not the checkpoint-selection rule.

The earlier listening baseline was **`repaired-tx-smoke`**. The user reports no
gibberish in the smoke clips they heard and gibberish in the later
**`gender-paired-mined-bcap`** clips. That intelligibility regression
overrides the later checkpoint's slight hidden-metric advantage. The
launcher is restored to the actual smoke recipe: 120 updates, constant
LR, no row conditioning, batch-mean feature matching, adversarial/FM
weights 1/1, and no row miners.

Both checkpoints use **`b_cap`**, with coefficient and threshold 1,
fixed teacher-RMS inputs, a span transformer, and `mean_last` readout.
The user's shorthand "bcap" identifies the later checkpoint here, not
the penalty alone. The experiments also change training length, LR
schedule, conditioning, feature objective, adversarial weight, and mining,
so they do not isolate which change caused the gibberish.

The smoke checkpoint's reloaded heldout direction cosine is 0.860 and
normalized teacher error is 0.517. The later paired mined checkpoint
scores 0.871 and 0.497, with magnitude ratio 0.957. Those numbers and all
72 exact zero-scale identity checks remain valid; they did not predict
the reported intelligibility regression. The smoke audio also fails the
seed-7 song-preservation proxy despite full word recall. Selecting it as
the listening baseline does not establish robust audio quality or a
production release.

The main problem was the game formulation and its implementation, rather
than evidence that a transformer discriminator cannot work. The repaired
run still uses a two-layer span transformer. The diagnosis and exact
upstream provenance are in [the reference audit](lm-gan-bcap-reference.md).

The implementation corrects several interacting failures:

- Critic inputs use a fixed scale calibrated from positive-minus-neutral
  teacher deltas. Dividing each individual fake by its own norm was
  singular at the initially zero LoRA delta. Fixed scaling preserves
  magnitude and has a finite derivative at zero.
- The cap covers real and fake samples, including zero fake deltas, in
  the calibrated coordinates. Its coefficient and one-sided form match
  the Gaussian reference: `0.5*c*(mean(relu(n_real-k)^2) +
  mean(relu(n_fake-k)^2))`, with `c=k=1`. A cap permits small gradients;
  it does not force gradients to remain at one.
- The discriminator no longer changes its definition between a batch D
  step and a single-row G forward through a batch-standard-deviation
  channel. Padded tokens are excluded, and the `mean_last` readout gives
  the audio-start token its own feature channel alongside the lyric span.
- Batch-mean feature matching preserves its exact weighted-mean gradient
  under sequential LM forwards. Paired matching additionally constrains
  each row to its own teacher features. The previous per-row loss against
  a global teacher centroid penalized legitimate teacher diversity.
- Gradient accounting measures the actual feedback into LoRA, checks
  finite gradients, and records critic-input norms. Historical default
  zero diagnostic fields were not measurements of dead gradients.
- The launcher uses Adam first-moment coefficient zero for the GAN and
  constant LR for the listening baseline. The optional delayed cosine
  schedule retains full LR for 60% of training, then decays to a 5% floor;
  it was used in later ablations. Cached end-regularizer paths and optional
  row mining are covered by the integration checks.

For a deterministic caption slider, unconditional distribution matching
admits a permutation of teachers across prompt rows. `RowConditionalD`
adds an optional projection on a learned row/polarity embedding. The row
ID stays fixed for each real/fake pair, including the cap calculation;
it is not concatenated to the differentiated hidden input. This identifies
the condition during training. The critic and row IDs are training-only,
so inference still uses the neutral caption and LoRA alone.

Critic conditioning alone did not eliminate the live errors. The paired
feature objective constrains the correct row correspondence explicitly.
For fake and teacher learned features `f_i` and `r_i`, let `e_i=f_i-r_i`:

```text
paired FM = mean_i ||e_i||²
          = ||mean_i e_i||² + mean_i ||e_i - mean_j e_j||²
          = batch-mean FM + variance of row errors.
```

Every term vanishes when each fake matches **its own teacher**. The old
incorrect objective `mean_i ||f_i - mean_j r_j||²` instead pushes all
fakes toward one centroid and suppresses teacher diversity. Paired FM is
learned-feature supervision for one teacher per condition; it uses paired
targets in addition to adversarial ranking. It is not hidden-state MSE
in the generator's original coordinates.

A three-seed CPU fixture using the production conditioned span critic and
directly learnable fake rows supports paired FM on that fixture. With adversarial weight
0.1 and FM weight 1, paired FM reaches mean whole-span normalized RMSE
**0.085** and last-token cosine **0.991**, compared with **0.455** and
**0.839** for batch-mean FM. All gradients remain finite. Increasing paired
FM weight to 10 is worse on two seeds. Direct fake rows are less constrained
than a shared LoRA. This justified a live comparison, but its favorable
fixture metrics do not supersede the user's intelligibility feedback.
See [the probe and limitations](../analysis/gan_bcap/paired_fm_findings.md).

The positive caption changes hidden states throughout the identical lyric
span. Simultaneously asking a discriminator to match those positive span
states and a hold loss to preserve neutral span states creates opposing
targets. A matched 120-update comparison makes the conflict visible:

| Saved checkpoint | Training last-shift cosine | Heldout last-shift cosine | Heldout teacher error | Heldout lyric-shift cosine |
| --- | ---: | ---: | ---: | ---: |
| **Listening baseline: repaired smoke, neutral lyric hold disabled, 120 updates** | **0.913** | **0.860** | **0.517** | **0.700** |
| Same repaired GAN with neutral lyric hold | 0.359 | 0.270 | 1.365 | 0.275 |
| Initial baseline, 32 updates | 0.072 | 0.081 | 0.996 | 0.132 |
| Existing checkpoint, 400 updates | 0.170 | 0.142 | 1.051 | 0.104 |
| Unconditional batch-mean FM, 300 updates, 64 miners | 0.751 | 0.694 | 0.784 | 0.663 |
| Row-conditioned batch-mean FM, 180 updates, no miners | 0.774 | 0.776 | 0.631 | 0.612 |
| Row-conditioned batch-mean FM, 180 updates, 64 miners | 0.827 | 0.787 | 0.619 | 0.673 |
| Row-conditioned paired FM, 180 updates, no miners | 0.801 | 0.805 | 0.652 | 0.692 |
| Later row-conditioned paired FM, 180 updates, 64 miners | 0.893 | 0.871 | 0.497 | 0.717 |

The 300-update unconditional run regresses from the 120-update smoke.
Critic row conditioning with batch-mean FM still falls short. The
conditional mined run improves over its unmined counterpart on these rows,
but that does not establish a general mining advantage. None of these
three follow-ups uses the new paired-feature objective.

In the matched paired-FM comparison, 64 miners improve heldout cosine from
0.805 to 0.871 and reduce teacher error from 0.652 to 0.497. The mined
checkpoint's four heldout cosines range from 0.842 to 0.895 and errors
from 0.451 to 0.544, so the improvement is not carried by a single row.
Its heldout magnitude ratio is 0.957, versus 1.077 without miners.
Lyric-span cosine is 0.872 on training rows and 0.717 on heldouts; normalized
span error is 0.503 and 0.713 respectively. This favored the mined model
under the hidden-state audit, but the later user listening report rejects
it as an improvement in intelligibility. It does not establish a general
mining benefit. The live paired-FM runs also lower adversarial weight from 1
to 0.1; the separate three-seed fixture isolates the FM objective itself.

The 120-update smoke run's mean last-token magnitude ratio is 0.887 on training
rows and 0.790 on heldout rows. It makes progress on the teacher's lyric
span as well: normalized span error is 0.584 on training rows and 0.726
on heldout rows, versus approximately 1 for the baseline. Neutral lyric
hold reduces hidden drift but removes most teacher-directed span movement
and damages the last-token change. Accordingly, the repaired launcher
sets `--lyrichold_weight 0`; it retains the end regularizer. A hidden-state
hold is not a transcription metric, so neither a small nor a large hidden
drift alone establishes whether the words are sung correctly.

For checkpoint verification, the evaluator loads the bf16 base model once,
encodes all pristine neutral and positive teachers before attaching LoRA,
and strictly reloads each final `.safetensors` file using its sidecar
topology. It evaluates all four YAML rows plus four newly written
arrangement/lyric combinations. Neutral and positive lyric token IDs must
match exactly. All 72 checkpoint/row combinations passed **bitwise exact
zero-scale identity** for the full hidden sequence. Per-row results,
heldout prompts, checkpoint hashes, and source hashes are retained in
[the initial checkpoint audit](../analysis/gan_bcap/lm_checkpoint_comparison.json),
[unconditional audit](../analysis/gan_bcap/lm_unconditional300_evaluation.json),
[conditional audit](../analysis/gan_bcap/lm_conditional_evaluation.json),
and [final paired audit](../analysis/gan_bcap/lm_paired_evaluation.json).
Historical runs have unequal budgets and other settings; the 120-update
hold comparison and each pair of mined/unmined 180-update conditional
trials are matched ablations.

The 120-update smoke checkpoint was rendered and scored at +1:

| Render seed | Scorer verdict | Lyric recall at +1 | Reported issue |
| --- | --- | ---: | --- |
| 7 | FAIL | 1.0000 | `U4_same_song`; `U5_ending` warning |
| 23 | PASS | 0.9231 | `U5_ending` warning |

The seed-7 failure occurs despite perfect word recall: recognizing the
expected words does not guarantee preserving the song. One passing seed
does not establish robust behavior. These scores describe the earlier smoke
checkpoint. Exact smoke records are in
[seed 7 scores](../analysis/gan_bcap/smoke_audio_scores.tsv) and
[seed 23 scores](../analysis/gan_bcap/smoke_s23_audio_scores.tsv).

The **later paired mined checkpoint** was separately rendered at both
seeds using row 0, 20 seconds, neutral-caption LoRA at +1, and prompt-only
references. There were no seed retries. Seed 7 also includes a +2 stress
sample; the prompt YAML's recommended range remains 0 to 1.

| Render seed | Scorer verdict at +1 | Whisper word recall | Song distance / limit | Reported issue |
| --- | --- | ---: | --- | --- |
| 7 | FAIL | 1.0000 | 0.09713 / 0.07894 | `U4_same_song`; `U5_ending` warning |
| 23 | PASS | 1.0000 | 0.06070 / 0.07720 | `U5_ending` warning |

Both slider-off WAVs are byte-identical to the corresponding earlier smoke
baselines. Both +1 renders have positive reference projections in Whisper
and DSP features and RMS ratios near one (0.947 and 0.974). The transcripts
contain additional words despite recall 1.0: recall measures expected word
membership, not exact lyric order or absence of additions. The +2 stress
sample has recall 0.8462.

The frozen scorer was not modified. Its PASS covers its level, lyric-word,
direction-sign, and song-distance proxies; it has no dedicated singer
direction classifier. A short-clip `U5_ending` warning means a hot tail at
the duration cap, and cannot validate a natural long-form ending. These
two seeds do not satisfy the separate catalog promotion requirement of
at least three seeds and a 60–90-second ending check. Subsequent user
listening reports gibberish in the paired mined clips despite these proxy
scores, while reporting none in the smoke clips they heard. The paired
checkpoint is retained as an intelligibility regression, and broader
audio validation remains necessary before any promotion.

Listen to [seed 7](../eval/listen/gan-bcap-repair/gender-paired-mined-bcap-s7/LISTEN.md)
and [seed 23](../eval/listen/gan-bcap-repair/gender-paired-mined-bcap-s23/LISTEN.md).
Full measurements and transcripts are retained in the folders' `lm_scores.json`
files and in [seed 7 scores](../analysis/gan_bcap/paired_s7_audio_scores.tsv)
and [seed 23 scores](../analysis/gan_bcap/paired_s23_audio_scores.tsv).

The Gaussian result also reproduces independently using the unchanged
upstream loop. At 7,000 updates with seed 1234 and actual movable latent
particles, `b_cap` captures all 100 modes with **98.771%** of samples
within three true standard deviations of a center, compared with
93.461% for matched R1/R2. The core width ratio is 0.8672, consistent with
the reference's 0.866. All 20,000 latent particles move. See
[the reproduction](../analysis/gan_bcap/gaussian_findings.md) and
[training curves](../analysis/gan_bcap/gaussian_comparison.png).

The LM's optional `--parts` remains **adaptive prompt-row mining**, not
the Gaussian model's movable latent prior: deterministic caption inputs
feed the LM, and the learned table selects training-row weights. The
repaired miner reserves 10% uniform mass and penalizes missing row
coverage using `KL(uniform || mean query weights)`. This allows individual
queries to specialize while preserving restoring gradients against
collapsed row coverage. The table is training-only; final inference loads
only LoRA. The 120-update smoke comparison above has mining disabled.
The completed paired trials compare 0 and 64 miners. Their hidden-state
audits favor the mined model within that pair, but the user's listening
comparison favors the earlier smoke model. Mining is therefore disabled
in the restored listening-baseline defaults; its implementation and
artifacts remain available for controlled comparisons.

From `/ml2/music/sliders-conceptmod`, reproduce the smoke recipe on
an assigned free GPU (GPU 1 in this example):

```bash
bash scripts/train_lm_gan_bcap.sh 1 repaired-tx-smoke-repeat
```

The script uses the existing `minimax-music3` environment and refuses to
overwrite an existing run. The example uses a fresh second argument to
preserve the completed checkpoint. Its generator objective uses
`--adv_weight 1 --fm_weight 1 --pole_weight 0`
with rank/alpha 8, LR 0.0005, seed 7, batch 4, calibrated cap coefficient
and threshold 1, and the span transformer's `mean_last` readout. Defaults
are `MINERS=0`, `STEPS=120`, `CONDITION=none`, `FM_MODE=batch`,
`ADV_WEIGHT=1`, `FM_WEIGHT=1`, and **`SCHEDULE=constant`**. Adam beta1
remains zero, neutral lyric hold remains disabled, and the end regularizer
remains enabled. These settings restore the smoke card rather than
substituting the later delayed-LR recipe.
The LoRA is applied to the neutral caption at +1;
feeding the positive caption as well would apply the concept twice.

To reproduce the later paired mined ablation, set every changed option
explicitly; this is an experimental comparison, not the listening baseline:

```bash
SCHEDULE=delayed_cosine CONDITION=row FM_MODE=paired ADV_WEIGHT=0.1 \
FM_WEIGHT=1 MINERS=64 STEPS=180 \
  bash scripts/train_lm_gan_bcap.sh 1 gender-paired-mined-repeat
```

Reload a final checkpoint on an assigned free GPU, then refresh the
portable report on CPU:

```bash
CUDA_VISIBLE_DEVICES=0 /home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  analysis/gan_bcap/lm_evaluate.py \
  --weights models/gan-bcap-repair/gender-paired-bcap/gender-paired-bcap_last.safetensors \
            models/gan-bcap-repair/gender-paired-mined-bcap/gender-paired-mined-bcap_last.safetensors \
  --output analysis/gan_bcap/lm_paired_evaluation.json

/home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  analysis/gan_bcap/lm_report.py
```

`lm_report.py` discovers the initial audit and `lm_*evaluation*.json`
files by default; `--evaluations file1.json file2.json ...` accepts an
arbitrary explicit audit set. It merges audits, reads saved training JSONL files,
and emits [a standalone PNG](../analysis/gan_bcap/lm_report.png) and
[a compact JSON summary](../analysis/gan_bcap/lm_report.json). It includes
critic condition, FM objective and weights, miners, hold, schedule, and
sidecar provenance. Historical unrecorded settings remain null. The
current sidecar hash is checked against the one captured during inference.
It plots cap and gradient trajectories without interpreting unmeasured legacy
zeros as dead gradients. If a training log is still being written, its
completed records can be plotted, but that is not a final checkpoint
evaluation.

The focused checks cover cap double backward and zero inputs, calibrated
units, padding invariance, batch-independent critic outputs, exact
sequential feature-mean gradients, paired targets and row conditions,
row coverage and saturated recovery,
schedule endpoints, and cached/legacy regularizer execution:

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m pytest -q \
  tests/test_lm_adv.py tests/test_lm_adv_tx.py tests/test_lm_adv_condition.py \
  tests/test_lm_gan.py \
  tests/test_lm_particles.py tests/test_lm_gan_trainer_integration.py \
  tests/test_lm_lyric_hold.py
```

The broader LM suite recorded 465 passing tests and one failure in the
unchanged pre-existing high-dimensional oscillation fixture,
`test_window_mean_is_the_stable_read_when_the_reply_is_not_a_mirror`.
The full result is saved in
`models/gan-bcap-repair/final_lm_tests.log`; the suite is not reported as
fully passing.

All existing measurements and artifacts are retained. User listening
selects `repaired-tx-smoke` as the baseline and identifies the later paired
mined checkpoint as an intelligibility regression in the clips heard.
The smoke's seed-7 song-preservation failure remains an unresolved limit;
no claim of robust audio quality follows from restoring its recipe.
