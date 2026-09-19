# Reward CE search — stopped at user request

The user stopped the remaining half-strength evaluation to prioritize training for more consistent wins. The final stage retained 27 completed matched bundles; five unstarted bundles remain unrendered. No generated result was discarded or rerolled. This is an incomplete evaluation, not a completed confirmatory result.

The original LoRA at half strength scored 9/16 wins on development prompts. The expanded activation teacher failed its independent check, so no student was trained from it. Studio rendering was restored to GPU 0 with one worker after the active bundles drained.

See the [user-directed stop record](audit/user-stopped-evaluation.json), [archived progress report](audit/report-before-user-stop.md), and [new preference continuation](../reward_preference_20260908/README.md). The original frozen sources, observations and checkpoints remain intact.
