Fresh evidence is separate from the exposed development game. No confirmation
or composition audio has been generated as of preparation of this document.
The reusable helpers require a strict completed 16-case development pass.

Run from `/ml2/music/sliders-conceptmod` with the existing environment. Replace
`PASSED_EVALUATION` only with the actual passing evaluation ID.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m conceptmod.textsliders.reward_game.confirmation \
  --home analysis/reward_game_v1_20260908 --name fresh-001 freeze \
  --evaluation PASSED_EVALUATION \
  --fixtures analysis/reward_game_v1_20260908/recipes/confirmation-fixtures-draft-v1.json
```

The freeze operation binds checkpoint bytes, effective multiplier, exact sheets,
seeds, physical GPU 1, Off/original/candidate arms, renderer sources, reward
preprocessing, practical gates and whole-family uncertainty rules. It records
48 scheduled clips. The `run` action must execute in a persistent user service
with `CUDA_VISIBLE_DEVICES=1`, `HF_HOME=/ml2/music/.cache/huggingface`,
`HF_HUB_OFFLINE=1`, and
`PYTHONPATH=/ml2/music:/ml2/music/sliders-conceptmod`.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -u -m conceptmod.textsliders.reward_game.confirmation \
  --home analysis/reward_game_v1_20260908 --name fresh-001 run
```

The same `run` command resumes only unattempted work. Failed or interrupted audio
is retained without rerolls. `score` rebuilds JSON, Markdown and raw-audio players
without rendering. A passing first fresh batch is still incomplete research.
Every confirmation attempt stays in the ledger; exposed fixtures cannot be used
again as untouched evidence.

After a fresh CE batch passes, run independent diagnostics on CPU in a persistent
service. Off observations freeze baseline-only tolerance values before any
candidate diagnostics are measured.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -u -m conceptmod.textsliders.reward_game.confirmation_intent \
  --home analysis/reward_game_v1_20260908 --name fresh-001
```

A selected winner needs an independently frozen replication using new families
and seeds while keeping the candidate bytes and multiplier fixed:

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m conceptmod.textsliders.reward_game.confirmation \
  --home analysis/reward_game_v1_20260908 --name replication-001 freeze \
  --evaluation PASSED_EVALUATION --replication-of fresh-001 \
  --fixtures analysis/reward_game_v1_20260908/recipes/replication-fixtures-draft-v1.json
```

Run and diagnose this batch using the same commands with its new name. Both
batches must independently pass the practical gates and positive family-level
intervals, with every scheduled comparison valid.

The separate composition check freezes four designated fresh families, two seeds
each, with two vocal and two instrumental arrangements. It compares one unit of
reduced LM style energy alone against those same reduced style multipliers plus
one unit of the fixed LM reward adapter, giving two total units in the combined
arm. The instrumental style assignments are explicit in its immutable protocol.

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python -m conceptmod.textsliders.reward_game.composition \
  --home analysis/reward_game_v1_20260908 --name composition-001 freeze --confirmation fresh-001
```

Use its `run` action in a persistent GPU-1 service. It schedules 16 clips and
requires 6/8 wins, +0.05 mean CE, no loss below -0.50, and all comparisons valid.
Its `score` action rebuilds the evidence and listening page without rendering.

Only after both fresh batches, their preservation reviews and the composition
check pass may `reward_game.complete_confirmation` package the ordinary adapter
and exact evidence. It does not modify the production registry. Its supported
claim is confined to the frozen first 20 seconds; natural full-song completion
remains a separate unestablished property.

These helpers currently support the declared LM adapter format. An acoustic
transformer adapter needs a separately declared format and per-host energy
protocol; do not apply the LM fixed-energy recipe across different hosts.


Acoustic host amendment: use `reward_game.acoustic_confirmation`, `acoustic_confirmation_intent`, `acoustic_composition` and `acoustic_complete_confirmation` with `--home analysis/reward_game_v1_20260908/acoustic-v1`. The parent `recipes/confirmation-template-acoustic-v1.json` retains all practical and uncertainty gates and independent replication. Its composition protocol follows separate host allocation: two units of LM style energy remain unchanged while the candidate alone receives one unit of transformer energy. No transformer style competes for that allocation. The acoustic fresh renderer freezes these caps in its manifest, dispatches the original LM and acoustic candidate through their separate validators, and uses ordinary generation. No fresh audio has been rendered by these helpers.
