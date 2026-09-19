# Reward-slider development game

Run in `/ml2/music/sliders-conceptmod` with
`/home/mikkel/anaconda3/envs/minimax-music3/bin/python`. No installation is needed.
The frozen benchmark has 16 exposed development cases. Its 4 → 8 → 16 ladder,
physical GPU assignments, ordinary merger, 20.4-second render cap and normalized
Audiobox CE reward remain fixed. Off stays eligible. A development pass requires
fresh confirmation, independent replication or predeclared multiplicity handling,
preservation diagnostics and the fixed-total-energy studio composition check.

```bash
python -m conceptmod.textsliders.reward_game init \
  --spec analysis/reward_slider_game_20260908/benchmark.json
python -m conceptmod.textsliders.reward_game audit-off
python -m conceptmod.textsliders.reward_game evaluate \
  --checkpoint /absolute/path/candidate.safetensors \
  --multiplier 1.0 --stage auto --tag attempt-001
python -m conceptmod.textsliders.reward_game evaluate \
  --checkpoint /absolute/path/candidate.safetensors \
  --multiplier 1.0 --stage 16 --tag explicit-full-diagnostic
python -m conceptmod.textsliders.reward_game leaderboard
python -m conceptmod.textsliders.reward_game inspect --run RUN_ID
python -m conceptmod.textsliders.reward_game compare --left RUN_ID --right RUN_ID
python -m conceptmod.textsliders.reward_game register --recipe /absolute/path/recipe.json
python -m conceptmod.textsliders.reward_game search --resume
```

Place `--home /absolute/path/game-ledger` **before** the command to use another
ledger. The default is `analysis/reward_game_v1_20260908`. `--checkpoint off` can
inspect the cached Off reference through the same interface. Strengths have
separate content identities. A tag or checkpoint path does not change identity.
Only ordinary rank-8/alpha-8 LM q/k/v/o adapters are currently supported; other
topologies require a declared compatible implementation.

Commands emit JSON to stdout. Evaluations also emit a short Markdown scorecard
to stderr and save both files. Scientifically rejected candidates return exit 0
with `advance: false`; malformed inputs and engineering failures return exit 2.
Repeated completed evaluations return their original scorecard exactly. A new
ledger event records zero new clips and the invocation's cache reuse; the saved
scorecard retains its original cost. Explicit stages are logged as development
diagnostics. Comparisons only use shared valid cases and do not extrapolate.

Initialization hashes the model, style and reward assets, checks source/package
identities, verifies all 48 reference WAVs, and reproduces the saved scorecards
exactly. It generates no audio. Later checks use those verified file sizes and
nanosecond timestamps and hash any changed file. Renderer source changes fail
closed. The separate Off rerender audit compares decoded PCM rather than the WAV
container timestamp. It is an engineering audit, not another quality observation.

Renders are indexed by checkpoint bytes and structural metadata, fixed strength,
full case, exact style/base identities, sampler, duration, rendering sources and
physical GPU. Scoring has its own raw-audio/reward-specification cache. Original
FLOAT WAVs are preserved. Claims use nonblocking OS locks plus permanent attempt
records. Concurrent work cannot claim a case twice. Completed observations remain
available after interruption; only unattempted cases run on resume. A failed or
interrupted attempted case cannot silently rerender. Engineering repairs require
an explicit recorded amendment or a new version; retain the original attempt.

The worker calls the existing `SearchRenderer.generate` interface. Separate GPU
processes execute each stage. One runs at a time by default to bound host RAM;
set `REWARD_GAME_MAX_GPU_WORKERS=2` when host resources support two model loads.
The parent owns resource leases and waits for
workers to exit before restoring the studio. GPU 0 is borrowed only after its
queue drains with continuous generation disabled. Playback stays available.
Another run's override is never removed. GPU 1 is used while the studio owns 0.
Use unique persistent user services for long campaigns; do not stop an unrelated
job or reuse an old campaign's service name.

Registered attempt recipes require `hypothesis`, `failure_mechanism`, `method`,
`parent_checkpoint`, `changed_variables`, `budget`, and `sources` (path → SHA-256).
`method: checkpoint` evaluates the declared `checkpoint` at `multiplier`.
`method: command` also takes an argument-vector `argv`, optional `cwd`/`env`, an
elapsed budget `budget.max_seconds`, and the expected adapter/training-state
paths. Its training program owns optimizer/RNG persistence and actual step logs.
The command must be resumable; an exit code never establishes audio improvement.
The supervising agent registers the next method after reviewing the failures.

`search --resume` records a rejection and immediately tries the next registered
attempt. It pauses on an engineering failure, a development pass requiring fresh
confirmation, a local `STOP` marker, or an empty backlog. An empty backlog reports
`method_selection_required`, **not completion**: the supervising agent must
replenish it with a justified method. There is no total attempt ceiling. At most
two training arms may run concurrently. No deployment or publication is implied.

Artifacts include an append-only `ledger.jsonl`, `status.json`, reference audit,
separate stage leaderboards, immutable attempt recipes/decisions, source hashes,
training logs/state pointers, generation jobs/logs, raw observations, and JSON /
Markdown scorecards. The no-artist-name rule covers every authored song field,
prompt, listening note and sidecar. Development sheets and scores cannot be used
as training targets without retiring this benchmark version.

Validation:

```bash
python -m pytest -q tests/test_reward_game.py tests/test_reward_sliders.py \
  tests/test_reward_search.py tests/test_reward_preference.py
```

A command may declare `training_gpu: 1` for a controller-owned GPU lease. Commands
that dispatch their own leased workers omit it. Cancellation or a command timeout
terminates the command's process group. Continuations declare `initial_updates`
alongside `expected_updates`; the difference is checked against the local optimizer
budget. Direct parameter searches may declare `allow_no_candidate: true` and a
`selection_result` path. A verified `no_candidate_selected` training status records
an Off selection, archives all training observations, and advances the search
without inventing a development scorecard.

Matched-policy experiments use `collect_matched`, `prepare_matched`, and
`train_matched_imitation`. The collector freezes a parent and eight matching Off
controls before audio, retains every new take, audits capture parity, and freezes
all pair labels before updating. Recovery merges each chosen take’s actual source
policy and requires 501 exact feedback frames for 500 eight-code training targets.
The separate trainer loads the exported parent at unit strength and starts a new
optimizer and reference cache. `matched_sequence` manages the bounded collection
dependency and leased preparation/training processes.

Run `python -m conceptmod.textsliders.reward_game.costs` to rebuild observed
campaign cost accounting. The listening page includes matched training pairs in
a separate section from development scorecards.


## Combined evidence overview

The derived overview combines completed development results only when the child
game has the same frozen cases, controls, reward, sampler and gates, and a passed
Off compatibility audit. It rebuilds the saved scorecards before ranking. Early
and incomplete screens retain their actual denominators; fresh batches remain
separate cohorts. A development or first fresh pass never marks research complete.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m conceptmod.textsliders.reward_game.overview \
  --home analysis/reward_game_v1_20260908
```

This writes `eval/listen/reward-game-current/index.html` and `overview.json`.
Add `--watch` to refresh derived data every thirty seconds. It creates no audio,
updates no model, and changes no scientific decision. The listening server serves
[the current overview](http://192.168.1.90:8888/reward-game-current/).
The overview's persistent service definition and verification are recorded in
`analysis/reward_game_v1_20260908/audit/overview-service-v1.txt` and
`overview-verification-v1.json`.
