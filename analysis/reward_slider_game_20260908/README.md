# Repeatable reward-slider game specification

The [handoff](../../docs/reward-slider-game-handoff.md) defines the next work:
implement a reusable evaluator and an experiment controller that keep trying
different methods after weak candidates, until a better LoRA is confirmed.

The following artifacts are ready now:

- [Frozen development benchmark](benchmark.json): 16 cases, eight families,
  two seeds, and fixed 4/8/16-case stages.
- [Reference scorecards](reference-scorecards.json): Off and the original LoRA
  at strengths 1 and 0.5, on exactly those cases.
- [Reference audit](audit.json): all 48 existing observation/audio references
  verified; no new clips rendered.
- [Reproducible builder](build_benchmark.py): rerunning verifies the sources and
  refuses to overwrite a different game specification.

All these cases have already been exposed during development. They are useful
for repeated attempts and cannot serve as independent final validation. Fixed
stage gates screen out weak candidates; only a promising candidate receives a
separate fresh test. Counts from different stages or historical cohorts are not
directly comparable.

The evaluator/controller commands in the handoff are an **implementation
contract**, not runnable commands yet. No new training campaign was started by
creating this handoff or benchmark specification.

Rebuild and verify the prepared artifacts from `/ml2/music`:

```bash
/home/mikkel/anaconda3/envs/minimax-music3/bin/python \
  sliders-conceptmod/analysis/reward_slider_game_20260908/build_benchmark.py
```
