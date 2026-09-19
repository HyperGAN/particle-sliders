# Versioned GAN architecture repairs

This implements the changes identified in the
[architecture review](../architecture_review_20260905.md), while preserving the
seven original trainer files and all baseline checkpoints. The new implementation
lives in [`conceptmod/textsliders/gan_v2`](../../../conceptmod/textsliders/gan_v2).
It is an executable research recipe. The [completed matched screen](results.md)
does **not** establish that the combined architectural changes beat 600.

The bounded baseline and FM gradient-limit arm improve the fixed proxy on both
reserved prompts. Their overall scores are 0.7957 and 0.7804, versus 0.5300 for
600. The combined live recipe scores 0.4911. Normalization and decay each improve
the first prompt and regress on the second; the broader combined arm changes
several things together, so this screen cannot identify which component causes
its result. No candidate triggers the two-view gross-collapse alarm. All candidates
remain `evaluate_more` under the declared coverage requirement.

The launcher now defaults to the **FM gradient-limit arm with the original four
prompts**. This is a conservative engineering choice that closes the known FM
gradient-budget gap while staying close to the bounded baseline. It is not a
statistically selected musical optimum, and its lyric flags remain visible.
The full combined recipe and broader data remain explicit experimental options;
no slider-catalog defaults or deployed weights were changed.

## What changed

| Review finding | Implemented change | Verification |
| --- | --- | --- |
| Raw feature scale can increase FM without increasing the capped score | FM uses fixed RMS normalization, without a trainable gain; the weighted full-minibatch FM parameter gradient has its own limit | Learned-norm gain cannot rescale FM; explicit gradient norm postcondition |
| Parameter step 2 depends on LoRA factorization | Retain that safeguard and additionally bound the Frobenius norm of the effective `BA` change, computed without materializing dense weights | Dense-matrix parity, factor-gauge invariance, post-update bound |
| Critic projection discards 3968 directions | Full-width, paired trust tubes around the positive teacher on both prompt spans and sampled continuation positions | Detects perturbations in the old critic's nullspace; no imitation gradient inside the tube |
| Batch FM permits wrong row assignments | Configurable paired FM component and a critic conditioned on the frozen neutral hidden context | Permuted-row and centroid counterexamples now get correction gradients |
| Set critic discards explicit position | Lyric-relative sinusoidal positions, with the existing separate audio-start readout | Order changes observable; padding and absolute caption offset remain excluded |
| End-margin loss misses semantic changes | Teacher-to-student semantic-plus-EOS KL hinge on shared histories, plus the full-width continuation guard | Semantic-logit permutations with identical EOS margins trigger the guard |
| Four examples repeated indefinitely | 32 prompts crossing eight new lyric sheets and four arrangements; multiple history seeds supported | Eight evaluation lyric sheets are disjoint; the earlier convergence holdouts stay reserved |
| Constant LR and no averaging | Absolute, fixed-horizon delayed cosine; rank-compressed EMA of effective adapter weights | The final scheduled update receives the floor; resume preserves horizon and EMA |
| `collapse=0` and incomplete gradient attribution | Unavailable collapse is null; optional full-batch gradients for every active term, pairwise cosine, pre/post-clip norms and actual updates | Diagnostics leave updates unchanged in a deterministic tiny-model test |
| Mean audio score hides failures/diversity | Separate clip flags, per-prompt rates, absolute description margins, score tails, and two within-prompt diversity views | Collapsed samples with an outlier are detected; missing coverage/calibration cannot pass |
| EMA may have a different LoRA rank | Renderer safely detaches one adapter topology before attaching another, sharing baseline/reference audio | Exact detach/reattach identity tested at different ranks |

The full-width guards use **soft penalties outside trust tubes**, not hard teacher imitation:
the default radius is one teacher-delta RMS per token. They cover large escapes
the critic projection misses while leaving smaller deviations unconstrained.
They do not enforce a hard hidden-distance or KL ceiling, make the critic injective, or prove that hidden-distance preservation
corresponds to preferred sound.

The FM limit is on the accumulated, weighted **parameter gradient**. It is
distinct from the scalar critic-input b_cap penalty. The effective-step limit
uses a factorization-invariant norm; the optimizer and interpolation path can
still depend on the factorization. Both limitations are recorded explicitly.

EMA averages effective products `BA`, not the two factors separately. A thin
QR/SVD keeps its rank budget at 32 and records the last compression error.
Periodic observations approximate the intervening trajectory; this is not an
uncompressed EMA of every optimizer update. Live and EMA adapters are evaluated
separately, with their actual ranks and alphas in their metadata.
In this short 600→660 continuation, the nominal source-600 mass is still
`0.995**60 = 0.7403` before compression. Thus the EMA is heavily influenced by
the starting adapter; its step label does not mean an average centered on 660.
The final observed relative compression error is `1.424e-5` for that compression
operation, not a bound on all accumulated trajectory error.

## Controlled experiment

[`v2_study.py`](../v2_study.py) declares these arms before collecting new audio:

- Bounded baseline: original critic/features/objective, constant LR, parameter step 2.
- Normalized FM only, otherwise that same baseline.
- Capped FM gradient only, otherwise that same baseline.
- Delayed cosine only, otherwise that same baseline.
- Combined repaired recipe, also using the broader training set.

Each starts from the same complete original 600 state and runs to 660. The
first four preserve the critic and both optimizer states. The combined recipe
restores the generator and its optimizer, and initializes the changed critic
explicitly. Its data/architecture changes are combined, so that arm alone does
not attribute an improvement to one component.

The fixed schedule origin is 600 and horizon is 660. The resource endpoint
does not redefine the schedule. **660 is a screening budget, not a recommended
training duration.** Four matched seeds on two reserved prompts provide the
first audio screen for the original 600, five live candidates and the combined
candidate's effective EMA. They do not establish convergence.

The study writes a live [matched comparison page](../../../eval/listen/gan-v2-20260905/compare.html)
with slider-off and positive-caption controls, plus [run details](../../../eval/listen/gan-v2-20260905/index.html),
the declared manifest, status, complete logs and audio-screen results here.
The slider catalog stays paused. The earlier convergence audio job is paused
to release its GPU and is resumed after this screening completes. The user
reassigned the work to **GPU 0** during rendering. Prompt 1 remains a complete,
matched GPU 1 group; prompt 2 is restarted entirely on GPU 0. Its five partial
GPU 1 WAVs are retained in an archive. This resource-driven restart is recorded
in [the reassignment manifest](gpu-reassignment.json); no audio score determined
which samples were restarted.
The [cross-GPU repeat check](gpu-repeat-check.json) found all five repeated
clips byte-identical between these two RTX A6000s. This is an observed result
for those clips, not a general GPU determinism guarantee.

## Validation

- All five declared arms completed **60 real-model updates**, from 600 through
  660, on GPU 1. The [training validation](training-validation.json) checks all
  300 updates for finite gradients and the declared update limits. The separate
  FM cap activated on 21/60 updates; the combined effective-weight cap activated
  on 51/60. Both scheduled arms reached the 0.05 LR fraction at 660.
- The combined run's last minibatch has mean fixed-history policy KL 0.01175
  and maximum 0.17869 nats. The KL hinge is a soft penalty: this is neither a
  hard 0.05 ceiling nor a free-running audio-quality result.
- The [source check](source-validation.json) verifies that all 15 signed source
  files still match the declared study and every run's manifest.
- [All 56 matched candidate clips](render-validation.json) are finite and
  nonempty, their seven checkpoint hashes match the declaration, and all
  candidates share byte-identical controls within each prompt/seed.
- The initial combined recipe completed three real-model updates on GPU 1.
  Gradients were finite and the effective-weight step bound activated when
  needed. These initial smoke weights precede the final schedule-boundary fix
  and are retained only as diagnostic artifacts.
- A separate real-model decay run saved at 604 and resumed through 606. It
  reached the declared LR floor at update 606; the complete state includes
  optimizer moments, the sampler, EMA, RNG states and all source fingerprints.
- Deterministic CPU tests verify interrupted versus uninterrupted updates and
  EMA exactly. This is not a general promise of bitwise GPU reproducibility.
- The initial [saved-failure guard check](saved_failure_guard.json) gives loss
  0.000664 at 600, 0.000024 at 750 and 2.040 at the original failed 900, on
  previously cached prompt-only spans. It detects a known large escape without
  proving its cause or musical quality.
- The fixed clip diagnostics flag lyric regression in **both archived failed-900
  samples** checked in [the known-failure audio check](known-failure-audio-check.json).
  One has positive description-similarity gain despite that failure. These are
  retrospective examples from the original failed branch, not the later bounded
  900, and do not independently calibrate the judge.
- The first two-prompt audio screen found no consensus gross diversity collapse
  among 600/1050/1200/1350 in the tested views. Individual technical flags remain
  visible. Two prompts do not estimate general failure rates reliably.

The [test log](tests.txt) records **124 passes** after the final regression run. The tests
exercise meaningful counterexamples from the review, complete trainer updates,
gradient accounting, resume, and averaging algebra.

The final [audio audit](../quality_audit_v2.py) extends the initial frozen screen
with near-duplicate group occupancy, so two repeatedly generated modes can be
flagged even if their separation makes total spread look healthy. It also
requires a candidate to meet calibrated endpoint thresholds; a judge-validation
boolean alone cannot pass it. These audit refinements live outside the frozen
training sources and do not change the preference score or candidate weights.
Their separate regression tests are recorded in `quality-audit-tests.txt`.
All **three** of those tests pass. The [report publisher](../v2_report.py)
automatically runs the final audit after the initial screen finishes, reusing
its frozen embeddings, and publishes all candidates with their controls.
The page also exposes ASR word match, phrase match and lyric-sheet coverage
separately. The first prompt includes a combined-repair clip with 100% word
match but only 58% sheet coverage: the combined lyric score can flag slower
coverage without proving wrong lyrics. The frozen scoring rule and thresholds
were not changed after seeing this example.

## Running and resuming

Use the existing conda environment; no training dependency installation is needed.
The launcher defaults to physical GPU 0 under the latest user assignment:

```bash
bash scripts/train_lm_gan_v2.sh a-fresh-run-name
```

This reproduces the small FM gradient-limit intervention on the original data.
Use `ARM=repaired` to run the combined experimental recipe on the broader
training set. An explicit second positional argument overrides the prompt file.

`MUSIC_GAN_GPU`, `ARM`, `SOURCE_STATE`, `SCHEDULE_ORIGIN`, `SCHEDULE_HORIZON`, and `UNTIL` can be
set explicitly. To resume, set `RESUME_STATE` and keep every signed recipe and
schedule setting identical; set `UNTIL` beyond the saved update. Use a fresh
run name for the new attempt. `USE_EMA`, `EMA_EVERY`, `DIAGNOSTICS_EVERY`, and
`SAVE_EVERY` are exposed, with defaults matching the declared study arm. The
standalone Python CLI also supports multiple history seeds and a signed JSON
file of recipe overrides. This migration was validated with the supplied
rank-8, alpha-8 legacy source. The rank argument does not automatically convert
an existing adapter or its optimizer state to a different topology.

Checkpoints include source copies under `provenance/`. Old legacy states cannot
silently become new experiments: migration is explicit. V2 states reject changed
source, prompts, history-cache content, sampler geometry, model file identity,
optimizer recipe, EMA settings or schedule. Resource stops are recorded as
`budget_complete` or `paused`, never as convergence.

## Limits that remain scientific questions

No loss here is a validated overall musical-quality metric. CLAP description
margin is not a calibrated voice classifier, ASR can misread singing, and the
two diversity representations do not enumerate known musical modes. The new
quality gate therefore stays unvalidated without independent judge calibration,
even when technical screens pass. It can reject clear failures without declaring
every surviving sample good.

The trainer still uses teacher-forced histories rather than differentiating
through complete generated audio. Free-running audio is evaluated separately.
Rank 8 remains the live-adapter baseline because the review did not establish
it as the active capacity bottleneck. A larger-rank training experiment needs
a compatible source and separately checked migration. Automatic catalog
promotion is outside this study; the catalog remains paused.
