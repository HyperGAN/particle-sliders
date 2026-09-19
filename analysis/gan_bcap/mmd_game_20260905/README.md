# MMD leaderboard game

Each round is a challenge to improve a real audio score. The champion survives
unsuccessful challenges; their evidence survives too. There are two records:
MMD's personal best and the overall leader, including existing GAN competitors.
The champion's running score cannot fall because the game keeps the best entry.
Individual candidates can regress. That bookkeeping property is separate from
the prototype's monotonically nonincreasing training objective.

Start with the [fresh-context instructions](/ml2/music/MMD-GAME.md).
The [leaderboard](LEADERBOARD.md) is generated from [state.json](state.json).
[protocol.json](protocol.json) fixes the judge; [opponents.json](opponents.json)
inventories 29 historical checkpoints without pretending their old scores are
comparable. No actual leaderboard entries exist at initialization.

## Rules and score

The initial board is the unidirectional gender concept, positive label Female,
negative label Off, scale +1, with exact slider-off and positive-caption controls.
It uses rows 0 and 1 of `analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml`,
seeds 7 and 23, 20 seconds, and physical GPU 1. Each checkpoint has four scored
clips. Controls are reused byte for byte after their first measurement. The
renderer still writes the complete normal listening package for each seed.

The fixed `render-heuristic-v2` rule combines candidate-minus-off improvements:

| Component | Weight | One scaled unit |
| --- | ---: | ---: |
| Concept text/audio margin | 0.4 | 0.05 |
| Enjoyment proxy | 0.2 | 0.5 |
| Production proxy | 0.2 | 0.5 |
| Mean of phrase accuracy and supplied-word recall | 0.2 | 0.25 |
| Excess high-frequency artifact penalty | 0.2 | 0.02 power fraction |

Each improvement is clipped to [-3, 3]. The artifact component is a penalty only:
power at or above 14 kHz exceeding both references by more than 0.002, scaled by
0.02 and capped at 3. Concept and aesthetics models measure separate float32
copies normalized to RMS 0.1; raw audio supplies transcripts, levels and artifact
fractions. Windows are weighted by covered time, then clips equally. A clip below
2% of its off-control RMS makes that checkpoint ineligible. All individual clips,
worst score, transcripts, component values and hashes remain in each entry.

This existing heuristic is the leaderboard judge, not a validated human-quality
measure. Do not change coefficients, normalization, descriptions, seeds, prompts,
duration or failure eligibility to help a candidate win. `load_game` rejects
changed protocol, prompt or frozen evaluation sources. If a genuine judge defect
requires a change, preserve this board and create a new version that re-evaluates
all compared entries. Never splice scores across protocols.

Repeated use makes these four examples development fixtures. A new leader needs
separate confirmation with additional prompts, seeds and longer audio. Predeclare
that confirmation set before rendering. Keep its results under the round's
`confirmation/` directory, outside this board's aggregation. Compare candidate
and strongest reference on exactly the same added fixtures. Record lyric errors,
per-example quality regressions and cross-seed diversity. A pair of different WAV
hashes is not a useful diversity guarantee; embedding spread alone can reward
noise. Report diversity as unmeasured until the relevant comparison exists.

## Commands

Run from `/ml2/music/sliders-conceptmod` with the existing interpreter:

```bash
MMD_PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
MMD_GAME=analysis/gan_bcap/mmd_game_20260905/game.py
"$MMD_PY" "$MMD_GAME" status
```

`init` is idempotent and has already run. It never resets an existing board.
The script's `evaluate` command explicitly selects physical GPU 1. Check current
GPU usage before a model job; wait for your previous process to exit fully even
if its completion marker already exists. Do not kill unrelated processes. The
existing renderer requires substantial free GPU memory (about 28 GiB in this
environment). Scoring uses the installed CPU quality/ASR setup and shared caches;
no dependency installation is needed.

Bootstrap in this order to establish comparable controls and strong opponents:

```bash
"$MMD_PY" "$MMD_GAME" evaluate --id gan_original_600 --family reference \
  --weights models/gan-bcap-repair/smoke-steps600-s7-20260904/smoke-steps600-s7-20260904_last.safetensors
"$MMD_PY" "$MMD_GAME" evaluate --id mmd_seed_720 --family mmd \
  --weights models/conditional-mmd-cfg-research-20260905/conditional-mmd-cfg-research-20260905_step720.safetensors
"$MMD_PY" "$MMD_GAME" evaluate --id gan_fm_cap_1950 --family reference \
  --weights models/gan-v2/fm_capped-to1950-20260905/fm_capped-to1950-20260905_step1950.safetensors
"$MMD_PY" "$MMD_GAME" evaluate --id gan_bounded_1200 --family reference \
  --weights models/gan-v2/baseline-to1200-20260905/baseline-to1200-20260905_step1200.safetensors
```

The first `begin` requires a scored MMD seed and at least one scored reference.
Benchmark the two strong additional opponents before interpreting a first win as
beating the prior GAN work. Add remaining pending opponents progressively, with
their inventory IDs. Historical results can guide order, not substitute for a
measurement. Beating the current leader is different from beating all 29 opponents.

Start one experiment, using the actual hypothesis and change:

```bash
"$MMD_PY" "$MMD_GAME" begin \
  --hypothesis 'The late plateau is partly caused by an oversized Adam proposal' \
  --change 'Resume the MMD champion with adaptive proposal scale and unchanged objective' \
  --budget 150
```

This creates `rounds/round-0001/round.json` (or the next number), records the current
champions and selected parent, and refuses a second active round. `--parent` can
select another scored entry for a justified branch; the default is the MMD champion.
The budget is a recorded starting plan, not an automatic trainer or proof of
convergence. Read the handoff's research-state loader warning before training.

The agent now implements, validates and trains the proposed experiment in a fresh
research directory. Keep a recipe, source snapshot, full state and diagnostics
with the round. Use `CUDA_VISIBLE_DEVICES=1` for research training commands.
Once the candidate actually exists, run (substitute the real ID, path and round):

```bash
"$MMD_PY" "$MMD_GAME" evaluate --id round0001_adaptive --family mmd \
  --round round-0001 --weights models/YOUR_NEW_RUN/YOUR_CHECKPOINT.safetensors
```

`evaluate` renders each row using the frozen renderer, scores with the frozen
judge, and registers the complete result. It writes logs in `evaluations/ID/`,
audio in `eval/listen/mmd-game-20260905/ID/`, and archived score reports in
`entries/ID/`. Existing renders are reusable only under their identical locked
specification. Preserve the checkpoint's topology and polarity metadata sidecar.
Do not modify checkpoints in place after rendering.

Finish after writing a real experiment note:

```bash
"$MMD_PY" "$MMD_GAME" finish --round round-0001 \
  --notes-file analysis/gan_bcap/mmd_game_20260905/rounds/round-0001/notes.md \
  --next-experiment 'State the next specific experiment justified by this result'
```

The note must explain the hypothesis, actual recipe and compute, initialization
and optimizer state, loss behavior, the score versus both previous leaders,
individual regressions, diversity/confirmation evidence or gaps, and the next
test. Use `--failed` if training or evaluation failed before any candidate could
be scored, with the cause and retained logs in the note. A scored losing candidate
is a normal completed round, not a reason to discard it. Do not end a round merely
because an arbitrary number of update attempts elapsed if the intended candidate
has not been rendered/scored; resolve the failure or record it explicitly.

## Persistence and recovery

- `state.json`: authoritative champions, entries, controls and active/completed rounds.
- `rounds/ID/round.json`: hypothesis, parent, starting scores, outcome and next experiment.
- `entries/ID/`: immutable registered entry plus exact score reports and registration inputs.
- `evaluations/ID/`: render/scoring logs and reports, including partial work.
- `LEADERBOARD.md`: regenerated readable view, with pending-opponent coverage.

Resume an active round rather than calling `begin` again. Re-run an interrupted
`evaluate` with the same inputs to reuse completed audio; registration can resume
after an interrupted state write. A registered entry ID cannot be replaced.
If input files need changing, preserve the old work and use a new ID. The game
serializes state mutations with a file lock. Run one model job at a time; this
lock is not a GPU job scheduler.

For already-produced reports, `register` accepts the same ID/family/weights/round
arguments plus `--scores REPORT...`. It checks all four fixtures, checkpoint and
audio hashes, exact controls, renderer/checkpoint specification, game GPU
provenance, source/rule identity and score recomputation. It does not independently
re-run perceptual models; reports should come from `evaluate` and retained logs.
Synthetic unit fixtures verify bookkeeping, not end-to-end GPU generation.

Validation at creation: 11 game tests pass, including changed score/rule/renderer,
missing/duplicate fixtures, different controls, checkpoint mutation, interrupted
registration recovery and preservation of wins and failures. Run:

```bash
"$MMD_PY" -m pytest -q tests/test_mmd_game.py
"$MMD_PY" analysis/gan_bcap/mmd_game_20260905/averaging_probe.py
```

## Why keep the averaging canary?

[Plot](averaging-probe.png) · [numbers](averaging-probe.json) · [code](averaging_probe.py).
For one deterministic prediction and two equally probable targets at -1 and +1,
MSE and a broad paired RBF loss minimize at their mean. Our five-kernel paired loss
instead minimizes near either individual mode. Neither represents both choices.
Two correctly placed generated particles achieve zero empirical distributional
MMD; two particles at zero give 0.97106485 in this canary.

Full squared MMD uses `E k(fake,fake') + E k(real,real') - 2 E k(fake,real)`.
Its distribution comparison is described in the primary
[MMD paper](https://jmlr.org/papers/v13/gretton12a.html). The missing conceptual step
for the slider prototype is not simply adding the word MMD: it is defining the
right conditional distributions and sampling them. Current paired deterministic
hidden matching and the Gaussian experiment's generated point distribution are
different setups. The plot is an analytic counterexample, not a measurement of
music diversity or a claim that the current checkpoint audibly averages voices.

## First challenge: adaptive continuation

Round 0001 established the four required comparable baselines and tested a
research-state continuation with restored Adam moments, adaptive proposal
fractions and a negative-gradient fallback. All 48 trial proposals across three
attempts failed the unchanged objective, activating the declared stall stop.
The candidate retained the seed's exact weights and reproduced its four audio
files byte for byte. It tied the MMD seed at 0.7491774; FM-cap 1950 remains ahead
at 0.7966217. Read the [round notes](rounds/round-0001/notes.md) and
[listening comparison](rounds/round-0001/index.html).

The research-state loader now exists in [adaptive.py](adaptive.py), with the
full trainer in [train_adaptive.py](train_adaptive.py). It verifies starting
outputs and loss, preserves optimizer/RNG state, and archives fixed histories
and source code. The old `live.py` remains unchanged. Do not spend the next round
repeating the same failed fraction grid: test whether bf16 rounding obstructs
descent, using repeated evaluations and directional finite differences before
a separately documented float32 continuation. Keep the audio judge fixed.

The additional-prompt confirmation trigger did not fire because this candidate
set no new record. MMD diversity and human listening verdicts remain unmeasured.
The remaining 26 inventoried opponents still require this board's measurements.
Use `game.py status` for the current authority on later rounds and champions.

[Later user feedback](rounds/round-0001/listening-feedback.md) reports that the
individual score columns align reasonably well with listening judgments, with
a tentative slight preference for adaptive over MMD. The corresponding audio
files and displayed metrics are identical; that preference is preserved without
attributing it to a model change. The feedback concerns individual components
and does not validate the composite weights or overall ranking.


## Round 0002: precision diagnosis and audio regression

Float32 arithmetic restored reliable local descent: all 150 updates were accepted,
and the new fixed objective fell 7.37%. The scored candidate nevertheless lost:
**0.5309238** versus retained MMD/adaptive **0.7491774**. Read the
[round notes](rounds/round-0002/notes.md), [listen](rounds/round-0002/index.html),
and inspect the [saved-state/precision audit](rounds/round-0002/state-and-precision-audit.json).
Lower fixed-history hidden loss did not produce better audio. The float32 teacher
calibration differs from bf16; raw losses across them do not rank checkpoints.

The additional inventoried baseline GAN 660 became the development leader at
**0.799029**, narrowly above FM-cap 1950's **0.796622**. A separately predeclared
30-second comparison on rows 2/3 and seeds 101/303 favored baseline 660 on all
four scores: mean **0.214274** versus **-0.292926**, with mean supplied-word recall
**0.9173** versus **0.7143**. One baseline clip had worse lyric recall. Full-mix
cross-seed diagnostics are retained in the
[confirmation report](rounds/round-0002/confirmation-baseline/summary.json);
two samples per condition do not establish full mode coverage or human preference.
25 historical opponents still await this board. No triggered confirmation remains
unfinished for round 0002; the losing float32 candidate did not trigger its own.

The next experiment is an equal mixture of the original frozen neutral histories
and one additional frozen parent-generated +1 history per training prompt. The
prepared round-0003 trainer starts from retained MMD 720, keeps prompts/rank/kernel
fixed, and checks that original float32 data match round 0002 bit for bit. Sampling
finishes before optimization. This is still a frozen-history surrogate, not full
trajectory-distribution MMD. Its larger batch costs more compute, which must be
reported. `resume_fixed.py` also provides a float32-state continuation entry point
that restores prepared data, Adam moments and RNG and checks starting outputs and
loss; it has compiled but has not yet been exercised in a live continuation.

The user has authorized continuing new experiments toward the top game score,
with thorough confirmation, and permits GANs or other methods. Keep method labels
honest and keep the audio judge unchanged. The ongoing search handoff is
[SEARCH-STATUS.json](SEARCH-STATUS.json). Do not stop after merely closing round 0002.


## Round 0003 and the authorized wider search

The active-history coverage candidate reduced fixed loss14.07% across150 accepted updates, yet scored0.4947456 and lost all4 development clips to MMD/adaptive. [Round3 notes](rounds/round-0003/notes.md) retain the recipe, source/state audit, separate history-group fits, per-clip metrics and listening gallery. Added-history fitting improved substantially, but concept score and production regressed; the candidate's extra-prompt/diversity confirmation did not trigger.

The user explicitly authorized continued new experiments, including GANs and other methods. The game now accepts distinct `gan`, `distillation`, `calibration` and `hybrid` labels alongside `mmd` and historical `reference` entries. MMD's record still requires family `mmd`. Round construction via `begin --family calibration --budget 0` is allowed for a declared checkpoint transformation with no optimizer updates. Round outcomes include all research candidates while excluding historical references from the new-experiment result. The locked score, prompts, settings and four frozen evaluation sources are unchanged.22 relevant tests passed, and the bookkeeping source is archived with round4.

Round4 tests one predeclared1.25 strength calibration of baseline660, with exact native-LoRA parity checks in float32/bf16, original trained matrices and full parent state retained, and renderer +1. A new overall development lead triggers fresh rows4/5, seeds515/727,30-second confirmation against baseline660 and FM-cap1950. No broad parameter sweep is queued. See live SEARCH-STATUS.json rather than treating this paragraph as completion evidence.


## Round 0004 complete: stopped for user review

The single1.25 strength calibration scored **0.5892729** versus baseline660 **0.7990291**, winning two development clips and losing two. It changed144 alpha buffers8→10, retained all288 trained tensors bitwise, and used zero optimizer updates.22 launch checks and the final state/source audit passed. The locked judge and protocol are unchanged. No new overall record means the predeclared fresh confirmation did not trigger; this candidate's generalization and diversity remain unmeasured.

[Review all compared audio](rounds/round-0004/index.html) · [Measured findings and per-clip regressions](rounds/round-0004/notes.md) · [State/source audit](rounds/round-0004/state-and-source-audit.json).

The user asked to stop after this experiment for review. Round4 is closed; no experiment process, follow-up or opponent benchmark remains queued. Resume only on a new user request. The current overall champion is baseline660 at0.7990291; MMD/adaptive remains0.7491774. The new float32, coverage and strength candidates all lost.25 historical opponents remain unmeasured, so this is the current measured board, not a claim to beat every inventoried checkpoint.
