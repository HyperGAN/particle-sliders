# Handoff: build a repeatable reward-slider game and keep improving it

Date: 2026-09-08. Workspace: `/ml2/music`.

## Execute this assignment

Build a reusable experiment game so the user can repeatedly change training,
submit a LoRA checkpoint, and compare its audio against the same controls. Then
use that game to try different approaches until a candidate demonstrates a
credible improvement. The user trusts the automated Content Enjoyment (CE)
scores, prioritizes consistent wins, and authorized research on both GPUs.
Natural full-song completion is secondary to this fixed-duration quality task.

After an unsuccessful experiment, record the result, choose a materially
different next experiment, and continue. A recipe's update limit or a rejected
checkpoint ends that attempt. The search remains active. Stop the research loop
when the success criteria below are met, the user stops it, or an actual resource
or external dependency prevents further work. Report those states accurately.
Do not promise that further training must eventually succeed.

The immediate deliverable is **the repeatable game**, followed by new attempts
using it. The previous scripts each ran a finite recipe and stopped when it
failed; they do not provide this persistent research loop.

## Prepared artifacts and implementation status

The [game specification](../analysis/reward_slider_game_20260908/README.md) is
ready. It includes (the three JSON files are not in git):

- `benchmark.json`: 16
  fixed development cases, eight families, two seeds, and fixed 4/8/16-case stages.
- `reference-scorecards.json`:
  Off and the original LoRA at effective strengths 1 and 0.5 on those same cases.
- `audit.json`: the 48 cached
  reference WAVs were hash-verified; no new audio was generated for this setup.
- [build_benchmark.py](../analysis/reward_slider_game_20260908/build_benchmark.py):
  a reproducible builder that refuses to overwrite a changed game specification.

**The evaluator and controller CLI described below are not implemented yet.**
Creating this handoff did not start another GPU campaign. Implement the CLI in a
new package, such as `conceptmod/textsliders/reward_game/`, and reuse the existing
model, rendering, scoring, capture and export code through explicit interfaces.

The benchmark's cases are already exposed development material. Their purpose
is repeated comparison. They must never be described as untouched validation.
Keep training families separate from these eight benchmark families.

## What the evidence says

There is no confirmed broadly improving incumbent. Keep Off eligible and retain
the original LoRA as a comparison. Half strength was a development selection,
not a demonstrated universal improvement or a new production default.

| Experiment | Result | Interpretation |
|---|---|---|
| Original LoRA, step 600, strength 1 | 12/16 wins; +0.3998 mean CE on its pilot comparison | Promising pilot; limited transfer evidence |
| Original LoRA, 24 additional 20-second excerpts | 16/24 wins; +0.0145 CE; family bootstrap interval crossed zero | No demonstrated average transfer gain |
| Half-strength calibration on the new 16-case development set | 9/16 wins; +0.1545 CE | Average score improved, consistency remained weak |
| Half-strength follow-up stopped at the user's request | 13/27 wins; -0.0903 CE; five unstarted bundles retained | Incomplete evaluation; useful negative diagnostic |
| Broader activation teacher | -0.2006 CE vs Off and -0.2491 vs random on its independent check | Failed; new GAN students were not trained from it |
| Preference continuation of the existing LoRA | Two learning rates, 30 and 60 updates; four new checkpoints; 20 new clips | Every checkpoint failed the small development gate |

The preference comparison used the same four cases for every row:

| Adapter | Wins vs Off | Mean CE gain | Worst CE delta |
|---|---:|---:|---:|
| Original LoRA at half strength | 3/4 | +0.1824 | -0.3674 |
| Gentle preference training, 30 updates | 2/4 | +0.0317 | -0.4858 |
| Higher-rate preference training, 30 updates | 3/4 | -0.1905 | -1.6114 |
| Gentle preference training, 60 updates | 3/4 | +0.0424 | -0.6260 |
| Higher-rate preference training, 60 updates | 2/4 | -0.6304 | -2.2929 |

That trial really trained and exported LoRAs. All 48 training histories passed
exact replay checks. New sidecars now record the failed evaluations. Do not
mistake `checkpoint_ready`, successful unit tests, a decreasing training loss,
or a zero process exit code for improved music.

Sources: [pilot and original study](../analysis/reward_sliders_20260907/README.md),
excerpt diagnostic (`analysis/reward_quality_followup_20260908/results.json`),
[stopped strength study](../analysis/reward_search_20260908/README.md),
stopped-study statistics (`analysis/reward_search_20260908/audit/stopped-final-diagnostic.json`),
preference results (`analysis/reward_preference_20260908/results.json`).
The three JSON sources are not in git.
Different cohorts' win counts cannot be compared as if they were the same test.

## Define the game once

Use the prepared `benchmark.json` as development game v1. Before implementing
evaluation, verify its source manifest, checkpoint identities, style components,
reward specification and audio references. The reference-scorecard benchmark
digest uses canonical JSON; `audit.json` records file-byte SHA-256 hashes.

Preserve the existing reward: frozen Audiobox CE, higher is better, one RMS 0.1
float copy of the first 20 seconds, mean CE over disjoint 0–10 and 10–20 second
windows. The renderer's 20.4-second cap provides its existing rounding allowance.
Keep original FLOAT WAVs unchanged. Short, silent, missing and nonfinite outputs
remain invalid. Do not change loudness normalization, reward weights, windows,
seed, prompt, style energy or sampler to make a candidate look better.

Each checkpoint has one fixed effective multiplier for a scorecard. Comparing
strengths means registering separate candidates. Never assemble a candidate's
scorecard from whichever strength won each individual case. Record adapter
content hashes, actual rank/alpha, source hashes, generation settings and GPU.

Keep each candidate/control comparison on the case's declared physical GPU.
Use separate processes for the GPUs. Off references can be reused only when all
relevant inputs and renderer identities match. If the generation implementation
changes, establish compatible controls or create a new game version; retain the
old leaderboard separately.

The game v1 reference scores on all 16 cases are:

| Reference | Wins vs Off | Equal-family mean CE gain | Worst delta |
|---|---:|---:|---:|
| Off | 0/16, all ties | 0 | 0 |
| Original LoRA, strength 1 | 7/16 | -0.0043 | -0.7455 |
| Original LoRA, strength 0.5 | 9/16 | +0.1545 | -0.5483 |

These are the reproducible starting scorecards. The historical 12/16 is from a
different set of prompts and is not this game's starting score.

### Small stages and a stable scorecard

The following are new operational thresholds, chosen for this game. They are
not claims of statistical significance and were not gates for the old studies.

| Stage | Required wins vs Off | Minimum mean CE gain | Worst allowed loss |
|---|---:|---:|---:|
| First 4 cases | 3/4 | +0.02 | -0.50 |
| First 8 cases | 6/8 | +0.05 | -0.50 |
| All 16 cases | 13/16 | +0.10 | -0.50 |

The full stage also requires at least 10/16 wins and +0.05 mean CE against the
original LoRA at strength 1. Every scheduled comparison must be valid to advance.
An exact tie is not a win; separately report meaningful wins of at least +0.02 CE.
Aggregate mean CE gains with equal family weight. The two audio windows are not
two independent musical examples.

Rank completed 16-case scorecards by: gate pass, wins vs Off, worst delta, mean
gain vs Off, then mean gain vs the original LoRA. Also show the half-strength
comparison, meaningful wins, individual losses, per-family and voice-group
results, invalid outputs, new clips, cache hits and elapsed time. Keep partial
4/8-case rankings separate. Show `2/4, early rejected`, not an extrapolated
`8/16` or an invented final score.

Default evaluation stops at the first failed stage. Four new clips per weak
candidate is the normal cost, with existing controls reused. A deliberately
requested full development run may bypass early stopping, but must be labeled
and logged. Thresholds and case order never change within a game version.

## Implement a reusable command interface

The commands below are the implementation contract. Make them work and document
them before launching a new search. Use the existing `minimax-music3` environment
and run from `/ml2/music/sliders-conceptmod`.

```bash
# Verify the frozen game and initialize its ledger without generating audio.
python -m conceptmod.textsliders.reward_game init \
  --spec analysis/reward_slider_game_20260908/benchmark.json

# Test any compatible checkpoint. Auto means the fixed 4 -> 8 -> 16 ladder.
python -m conceptmod.textsliders.reward_game evaluate \
  --checkpoint /absolute/path/candidate.safetensors \
  --multiplier 1.0 --stage auto --tag attempt-001

# Explicit, recorded development diagnostics and comparisons.
python -m conceptmod.textsliders.reward_game evaluate \
  --checkpoint /absolute/path/candidate.safetensors \
  --multiplier 1.0 --stage 16 --tag attempt-001-full
python -m conceptmod.textsliders.reward_game leaderboard
python -m conceptmod.textsliders.reward_game compare --left RUN_ID --right RUN_ID
python -m conceptmod.textsliders.reward_game inspect --run RUN_ID

# The bounded-attempt controller resumes the overall search ledger.
python -m conceptmod.textsliders.reward_game search --resume
```

Return both machine-readable JSON and a short Markdown scorecard. A scientifically
rejected but valid candidate is a normal result with `advance: false`. Distinguish
it from a malformed checkpoint, corrupted cache, interrupted job or hardware
error. The controller must understand that distinction.

Cache renders by the full generation identity: checkpoint bytes and structural
sidecar fields, multiplier, case content, base/style hashes and resolved energy,
sampler/duration, renderer version and physical GPU. Cache scoring separately by
raw-audio hash and reward-specification hash. Paths and human tags are not enough
to identify a candidate. Repeating the same successful evaluation must reuse all
outputs and reproduce the same scorecard without additional renders.

Provide a resume mechanism that retains completed observations and runs only
unattempted work. Keep failed/partial audio and its original attempt record.
Engineering repairs get an explicit version or amendment; do not reroll seeds or
erase a poor outcome. Reject concurrent claims of the same case atomically.

Each attempt directory should contain its hypothesis, parent checkpoint, changed
variables, immutable recipe/source signature, training state, adapter/sidecar,
generation jobs, audio observations, scorecard, decision and next action. Maintain
an append-only ledger plus a readable leaderboard and current `status.json`.
Allow manually supplied checkpoints as well as registered training methods.

### Acceptance checks for the game itself

1. Rebuild the prepared reference scorecards from their observations and obtain
   the exact saved values. Verify all 48 reference audio hashes.
2. Test cache hits, content-hash invalidation, multiplier changes, partial resume,
   double-claim prevention, failure accounting, and separation of 4/8/16 stages.
3. Reproduce one declared Off case on its physical GPU and compare decoded PCM
   with the saved reference. Preserve the rerender as an explicit engineering
   audit, not a new independent quality observation.
4. Check a known failed checkpoint through the same candidate interface. Use
   matching prior observations where available; otherwise a four-case screen
   suffices. Confirm that rejection advances the research loop to another attempt.
5. Verify exact Off restoration, checkpoint topology, and that a local adapter
   update leaves base weights frozen. Finish appropriate existing/new tests.

Do not repeatedly reload and render every baseline to test the evaluator.

## Run an ongoing research loop

For each attempt: state the failure mechanism being tested, change a small number
of relevant variables, register the recipe and local update/render budget, train
or construct the adapter, then evaluate it through the same game interface.
Use checkpoint hashes to attach scores to the actual evaluated weights.

Keep training, development selection and fresh validation separate. Repeatedly
examining game cases is expected development use. Do not optimize a training
loss on their audio, tokens, exact sheets or labels. Use training-side examples
with similar musical properties to investigate their failures. If development
cases are deliberately moved into training, retire that game version and build
a new development bank before claiming progress.

Use at most two training arms concurrently. Start with short checkpoints, such
as 10/30/60 new updates, and save the actual intermediate weights as well as full
optimizer/RNG states. Pick update spacing from measured update cost and observed
behavior. Keep the model resident across small tests when practical. Learning
rate or strength grids should be small, justified experiments.

After two comparable failures within one method, review the evidence and move
to a different mechanism. Do not spend an entire search repeating the failed
semantic-only preference loss with larger rates or ever longer training.
Per-attempt budgets protect resources; there is no user-specified total attempt
limit. Report cumulative attempts, GPU time and new clips honestly.

The software controller can execute registered methods. The supervising agent
owns hypothesis selection and implementation. If the registered backlog empties,
read the failure ledger, implement the next justified method and replenish it.
Do not report the task complete merely because the current Python loop exited.
Keep the user informed when changing approach. Existing authorization covers
local experiments; recurring permission questions are unnecessary.

## Concrete experiments to try next

These are hypotheses to test, not established explanations of the failures.
Begin with the cheap audit, then prioritize a materially different learning
signal. Use the same game to compare every exported candidate.

### 1. Audit the training forward against deployed merging

Training currently uses attached low-rank forwards in
`/ml2/music/app/lora_runtime.py`, while audio evaluation uses
`/ml2/music/app/generator.py`'s ordinary weight merger. The latter
adds pristine base and every scaled style/reward delta in CPU FP32, then casts
once to the host dtype. Separate low-rank arithmetic in BF16 need not produce
identical outputs.

On existing training histories, measure hidden error, guided semantic KL and
top-candidate changes between attached and actually merged forms of the same
adapter, including styles. Compare the discrepancy with the adapter's own
effect; do not assume it is the cause. This diagnostic needs no new audio.

If the discrepancy is material, test a training forward whose numerical output
matches the real merged projection while retaining gradients to rank-8 factors.
Include pristine base plus all style deltas before the final cast; adding a
reward delta to an already rounded style merge would still differ. Verify live
merged output and gradients on a small model and the actual host before training.

### 2. Learn from the current adapter's actual regressions

The last preference dataset used high/low **Off** takes from 24 training families.
It did not contain matched good/bad outcomes from the current adapter. Test
preferences collected from the current policy against Off on training families.

Start with four balanced training families and two seeds. Reuse matching Off
captures; render eight new adapter takes with capture. Preserve instrumentation,
voice, styles and exact effective multipliers. Include training-side instrumental
dance and female guitar arrangements, without borrowing the game sheets.
Choose the higher-CE take within each matched prompt/seed pair, whether it came
from Off or the adapter. Keep successes as well as regressions in the data and
balance family/voice exposure. Freeze the pair labels before updating weights.
An eight-take collection is a training pilot; expand only if it helps.

### 3. Test a richer policy objective or a simpler positive-target objective

The implemented preference loss is a semantic-policy surrogate. It omits the
seven residual-code probabilities and acoustic likelihoods, and drops sampling
top-k during teacher forcing. Its failure does not test all reward learning.

In a separate version, extend exact replay to retain all eight frame codebooks;
the existing replay already reconstructs them, then discards the residual codes.
Teacher-force the residual decoder with its actual codebook ordering and CFG.
Keep its weights frozen but allow gradients through its input from the LM. Check
warmup alignment, both branches, real gradient flow and agreement with captured
codes. Do not detach the very path being optimized. A semantic-plus-residual
objective still is not a full waveform likelihood; describe it accurately.

Compare that approach with reward-weighted imitation of chosen training takes,
with a reference/prompt preservation term. This tests whether pushing rejected
sequence probability downward is harmful. Keep family exposure balanced and
compare alternatives individually. Use small fixed subsets of generation positions
if needed for memory, retaining every selected position's full causal history.

### 4. Try direct CE search over a small adapter parameterization

If surrogate losses keep improving their own metrics without helping audio,
test direct score feedback. For example, learn bounded gains on a few depth or
projection groups of an existing LoRA, with each gain folded into its `lora_up`
weights. That preserves the ordinary rank-8 export. Verify zero and unchanged
gain settings reproduce their intended controls.

Use paired small perturbations on a rotating **training-side** mini-bank, with
fixed seeds and a predeclared objective that penalizes negative CE deltas.
Start with two proposals and four training cases, not an exhaustive layer grid.
Submit only the selected fixed adapter to the development game. Report this as
direct adapter-parameter search, not as fresh GAN training. If useful, expand the
trainable parameterization or distill its result with a separately tested method.

If these approaches fail, use the error patterns to propose another intervention:
different layer support, stronger preservation on failure categories, revised
credit across audio sections, or a different LoRA host as an explicitly separate
experiment. Preserve the ability to compare exported adapters through the game.
Research new methods from primary sources when needed; retain a concrete recipe
and a falsifiable reason for each attempt.

## Confirm improvement before ending the search

Passing development means the candidate merits confirmation. It does not end
the task. Freeze the checkpoint, multiplier, recipe and confirmation protocol
before producing confirmation audio.

Use at least eight fresh families with two seeds each, balanced across voices,
instrumental cases, acoustic/electronic arrangements and style combinations.
Render Off, the original step-600 LoRA at strength 1, and the candidate: normally
48 clips, only after the candidate has passed all 16 development cases. A later
confirmed incumbent can be an additional explicitly budgeted comparison. Do not
launch this larger test for a 9/16 development candidate.

Require the same practical full-stage targets on the fresh batch: at least 13/16
wins and +0.10 equal-family CE against Off, at least 10/16 wins and +0.05 against
the original LoRA, every result valid, and no loss worse than -0.50 CE versus Off.
Report family-level uncertainty; the interval for mean gain must support a
positive effect against both controls. Check voice/style/lyric preservation and
diversity separately using the existing diagnostics, including the instrumental
failures. Natural song completion remains a separate recorded issue.

Repeated attempts to pass a held-out gate are themselves a selection process.
Record every confirmation attempt, never reuse a scored confirmation batch as
untouched evidence, and predeclare how to handle repeated testing. Use an
appropriate multiple-attempt adjustment or require an additional independently
frozen replication of a selected winner before declaring success. If the result
is promising but uncertain, label it that way and collect the predeclared extra
evidence. Do not weaken gates after inspecting a failure.

Before claiming the slider works in the studio, also run a small composition
check under the actual fixed-total-energy contract, with the style-only control
at the same reduced style multipliers. The main game holds style strengths fixed
to isolate the added adapter. Those are different comparisons, and the original
LoRA failed the earlier fixed-energy composition study.

Completion requires the ordinary audited LoRA, exact recipe and provenance,
passed fresh evidence, an honest account of remaining limitations, a scorecard,
and a local listening page including regressions. Preserve all unsuccessful
attempts. Production registry changes or external publication are separate from
this research handoff and require the user's deployment instruction.

## Assets, reuse and operating rules

All paths below are relative to `/ml2/music/sliders-conceptmod` unless absolute.

| Asset | Location / use |
|---|---|
| Original ordinary LoRA | `analysis/reward_sliders_20260907/student/reward-ce-v1_step600.safetensors` |
| Original weight SHA-256 | `34b6aca72508d70aa35cecc7facabc937e0c8b61922e6c360dc4048ee79de63b` |
| 96 scored training captures, 24 families | `analysis/reward_search_20260908/training-observations.json` |
| 24 frozen high/low preferences | `analysis/reward_preference_20260908/preferences.json` |
| 48 verified semantic histories | `analysis/reward_preference_20260908/tokens/` |
| Cached initial-half policy references | `analysis/reward_preference_20260908/reference.pt` |
| Failed continuation weights/full states | `analysis/reward_preference_20260908/students/{gentle,faster}/` |
| GPU-selectable ordinary renderer | `conceptmod/textsliders/reward_search/renderer.py` |
| Exact code recovery and preference trainer | `conceptmod/textsliders/reward_preference/` |
| Scoring, observations, integrity helpers | `conceptmod/textsliders/reward_sliders/` |
| Existing tests | `tests/test_reward_sliders.py`, `tests/test_reward_search.py`, `tests/test_reward_preference.py` |

The original LoRA is rank 8, alpha 8, `kind: language_model`, all 144 q/k/v/o
projections across 36 layers, 432 tensors, `target_replace: [Qwen3Attention]`,
`prefix: lora_te`, `delimiter: '-'`, `train_method: full`, `unit_scale: 1.0`.
Export a genuine additive LoRA, with exact Off restoration and the ordinary
host-energy contract. An activation vector or a runtime best-of-N selector is
not a substitute for this artifact. Other ranks/hosts are separate declared
experiments with their own format and compatibility checks.

The preference warm start scales the original `lora_up` tensors by 0.5 and trains
at multiplier 1. Changing objective explicitly creates a new optimizer; restoring
an old GAN optimizer under a preference label would not be an unchanged resume.
Reference caches are tied to the initial policy, tokens, numerical forward and
source signature. Verify compatibility before reuse; regenerate incompatible
targets explicitly. Existing capture files avoid repeating expensive audio
collection. Exact token recovery needs only the composer and residual decoder.

Use `/home/mikkel/anaconda3/envs/minimax-music3/bin/python`. Never install
`legacy/requirements.txt`. Read [MUSIC3.md](../MUSIC3.md) and
`/ml2/music/AGENTS.md`. Never put real artist, band, songwriter,
producer or album names in authored prompts, lyrics, captions, titles, listening
notes or sidecars. Validate sound-only fixtures and their provenance. If existing
prompt provenance contains prohibited names, strip it and retrain those weights.

At the last verified check, the preference campaign had finished and the studio
was restored to GPU 0 with one worker; GPU 1 was free. Recheck live ownership.
Train/render on GPU 1 while the studio owns GPU 0. Borrow GPU 0 only after the
studio queue drains, keep playback available, and restore its previous settings
when releasing it. Do not cancel user songs or interfere with unrelated GPU jobs.

Use a new run directory and new service names. Preserve all frozen earlier
sources, manifests, audio and checkpoints. The workspace contains extensive
unrelated changes. Do not clean or reset it. Keep experiments isolated from
production generator and registry edits.

Use persistent user services for long jobs. Preserve resumable state and cleanup
on errors/cancellation. Transient service definitions can disappear when stopped;
recreate missing ones instead of repeatedly trying to start a nonexistent unit.
Do not reset a failed transient unit before retrieving its definition/logs.
The studio has a persistent unit at
`/home/mikkel/.config/systemd/user/music-studio.service`; use atomic restarts for
GPU-assignment changes. Resource helpers in the old packages contain date-bound
service names: parameterize them for the new game rather than controlling old runs.

The old temporary studio GPU override was
`/home/mikkel/.config/systemd/user/music-studio.service.d/20-reward-search.conf`.
Verify ownership before changing an override. Never remove another run's lease.
Log studio snapshots, GPU assignment, restoration, load time, update time and
render time. Report actual optimizer steps separately from loading/preparation.

## First actions for the next agent

1. Verify the prepared benchmark and existing artifacts; inspect current GPU and
   studio ownership. Do not restart a completed old campaign.
2. Implement and validate the reusable evaluate/cache/scorecard/leaderboard CLI.
3. Implement the attempt ledger, method registration and resume loop, including
   a test that a weak candidate triggers another attempt.
4. Run the training-versus-merged numerical audit on saved training histories.
5. Choose and register the first new learning experiment, then run short
   checkpoint checks through the game. Continue with new approaches until a
   verified improvement or an explicit stopping condition is reached.
