# Numerical convergence checks

The Training page (`/#training`) has a **Has the slider settled?** panel.
`GET /api/training/convergence` serves the same numerical evidence, bound to
the current model and full run identity. A replacement run cannot inherit a
previous run's assessment. These measurements supplement the existing raw
residual, edit alignment/gain, orthogonal error and SWD plots.

## Fixed prediction movement

For each of Candlelit and Moonlit, evaluate saved updates 1200, 1300, 1400,
1500 and 1600 on the same 12 development prompt rows: seed 29001, both teacher
trajectories, all ten denoising positions, 512px and Energy 1. This gives 240
full velocity fields per checkpoint. Compare live adapter weights and EMA
weights separately. EMA is the version used to render images; its smoothing
can conceal movement in the live training weights.

For checkpoint k define the normalized edit
`e_k = (v_k(neutral) - v_frozen(neutral)) / training_scale[t]`.
The reported change is `RMS(e_b - e_a)`, relative to both `RMS(e_a)` and the
normalized teacher edit. It measures changes on identical inputs, including
direction changes that a flat magnitude or loss curve can miss. It is not a
percentage of changed pixels or a perceptual difference score. A zero prior
edit makes that denominator undefined, not zero. Full-field RMS, per-field
p95, direction cosine, and definition/character/framing/timestep/bare-prompt
breakdowns are also retained in the report.

Each prediction pass repeats its first ten fields and requires exact equality.
Cached FP32 predictions carry checkpoint and diagnostic identity hashes.
No additional sample gallery is generated.

## Frozen-opponent response

At updates 1400 and 1600, restore the saved live adapter, critic and optimizer
states into disposable copies. For each of two sampler seeds (811 and 947),
run 16 D-only updates with the adapter fixed, and independently 16 G-only
updates with the critic fixed. Each trial starts from the original checkpoint.
Updates use training rows only, independent D/G streams, effective batch eight,
microbatch one, original learning rates and Adam states, original particle VIC,
and the exact gradient cap every fourth update. The full campaign noise
schedule is retained; all these late updates are on its sigma=1 hold.

Measure the tested side's objective before and after on the same 240 development
fields with common evaluation noise (seed 3803, two draws per field). The D
objective includes the cap at its time-averaged weight. The G objective includes
VIC averaged over four fixed 64-particle subsets (seed 3804). Adversarial and
regularizer components remain available separately in each raw trial file.
**Positive improvement means a lower objective.** Intervals use 2000 bootstrap
resamples of whole prompt rows, never individual pixels or denoising positions.
They quantify variation within these fixed development rows, conditional on
these evaluation draws; they are not population-wide confidence intervals.
Several rows share a character, so independent-character generalization is
not established by this interval. Trials are displayed individually without
claiming a multiple-testing-adjusted significance level.

A successful local response demonstrates remaining improvement for that side
against this particular frozen opponent. Failure to find improvement in a
short search does not prove optimality. This is a bounded local response test,
not an exact duality gap or proof of a Nash equilibrium. Critic and slider
objectives are reported separately because their regularizers differ. A lower
adversarial objective also does not establish better images or preservation.

## Execution and provenance

Use the pinned environment and verified GPU launcher. The regular coordinator
must first finish any active update and release its GPU lease. Only one owner
may use the Anima lease at once. Restore the coordinator when the diagnostic
exits, including on failure. Both current full training runs must be complete.

```bash
.venv-anima/bin/python scripts/anima_cuda_host.py \
  --gpu 1 --sm-count 82 --kernel-policy \
  --script scripts/anima_convergence.py -- --gpu 1
```

Results are stored outside git under
`artifacts/anima/remote/diagnostics/convergence/{candlelit,moonlit}/`.
`identity.json` binds the code, full run identity, checkpoint hashes, normalization,
cache fingerprints, development indices, GPU policy and trial settings.
`report.json` powers the dashboard. `predictions-*.pt`, `objective-*.json` and
`response-*.json` preserve the raw measurements. Completed units can be reused
only with identical provenance; archive a prior assessment before changing
fixtures or diagnostic code. Production run files and checkpoint hashes are
verified unchanged at completion. Final-test data is never opened.

The diagnostic serves up to four queued Studio images between measurement
units. It does not enqueue more training, change the formulation, promote a
checkpoint, or qualify Theatrical's unsuccessful pilot.

## Verification

CPU tests compare D-only and G-only parameter and optimizer updates with the
production phases, on both active and inactive cap steps. They also check that
the frozen opponent and base stay unchanged, that rotation is detected despite
unchanged edit magnitude, that duplicating correlated timesteps cannot narrow
the bootstrap interval, and that fixed objective evaluation is repeatable and
does not update weights or optimizers. API tests reject stale run identities.
Desktop/mobile browser checks exercise the deployed panel.

## Results, 2026-09-20

Candlelit completed all ten prediction passes and eight response trials. Every
repeated prediction matched exactly, all response gradients were finite, frozen
opponents stayed unchanged, and original checkpoint/run files passed the final
hash check. Recomputing all existing probe metrics from these new EMA predictions
matched to within 1.2e-7 (residual, cosine, gain, orthogonal error and both SWD
measures), saved in `probe-parity.json`. Both variations have completed all ten prediction passes and eight response
trials each. Total GPU diagnostic wall time was about 36 minutes; the ordinary
Lumen worker was restored successfully (supervisor exit code 0).

Candlelit's normalized live prediction change over the last four 100-update
intervals was 50.7%, 62.1%, 56.8% and 54.0% of the previous edit. EMA change was
38.7%, 37.2%, 37.6% and 37.7%; it was not diminishing toward zero. Across
1200–1600 the EMA change was 59.4% of its starting edit. In comparison, the
existing raw development residual fell from 1.5503 to 1.4876 (4.0%). Prediction
movement and residual level answer different questions.

At Candlelit update 1600, D-only trials lowered the objective from 0.001590 to
0.001472 and 0.001402. Improvements were 0.000118 [0.000011, 0.000244] and
0.000188 [0.000124, 0.000262]. G-only trials changed their objective from 8.5897
to 8.9389 and 8.4592: improvement -0.3493 [-0.5665, -0.1640] and 0.1304
[-0.0065, 0.2853]. Thus both critic trials found remaining local improvement;
slider-only improvement did not reproduce reliably across seeds. The assessment
does not support declaring the game converged or assuming that a few more
slider updates will improve held-out images.

A separate exploratory `character-sensitivity.json` reports response changes
for each of the four development characters. It resamples character-level means
(256 possible draws) to check sensitivity to grouping; the primary prompt-row
intervals remain unchanged. This post-hoc check does not expand the dataset or
establish population-wide certainty. Reproduction helpers for this assessment
are saved beside the browser evidence under
`artifacts/anima/convergence-browser/{probe-parity,character-sensitivity}.py`.

Moonlit's live movement over the same four 100-update intervals was 57.6%,
60.4%, 55.4% and 64.6%; EMA movement was 40.4%, 36.5%, 36.3% and 37.6%.
Across 1200–1600 its EMA change was 64.9% of the starting edit. Raw development
residual fell from 1.6670 to 1.5850 (4.9%). All five Moonlit probe evaluations
also reproduce within 1.2e-7. For both variations the largest normalized
100-update prediction changes occur at late denoising positions; the complete
breakdowns are retained with each movement measurement.

At Moonlit update 1600, D-only objectives fell from 0.001476 to 0.001235 and
0.001417: improvements 0.000241 [0.000010, 0.000592] and 0.000059
[0.000005, 0.000142]. G-only objectives fell from 8.9691 to 8.8944 and 8.9299,
but both improvement intervals crossed zero: 0.0747 [-0.0893, 0.2490] and
0.0392 [-0.0794, 0.1706]. Thus neither final G-only trial establishes a clear
held-out gain within this search budget.

Both completed assessments found residual local improvement for the critic.
Their predictions continue moving, while short G-only responses do not produce
reliable held-out objective improvement. These results do not establish a
converged game, nor do they demonstrate that extending training will improve
images. Both original runs remain completed at 1600 updates. Theatrical remains
unqualified and was excluded from this assessment; development selection and
untouched-character/mixture audits are separate outstanding campaign work.

Final verification: 18 relevant CPU/API tests passed; all 20 GPU prediction
passes passed the exact repeat check on their first ten fields; all 16 response trials kept the opponent frozen and
had finite gradients. All ten saved probe evaluations reproduce within 1.2e-7.
The original model/run/checkpoint files remain unchanged. Desktop and mobile
checks of the deployed completed reports pass with no JavaScript/HTTP errors or
horizontal overflow.
