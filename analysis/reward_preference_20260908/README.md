# Reward preference continuation

Current stage: **complete**.

The user stopped the larger half-strength evaluation to prioritize training for more consistent wins. This experiment continues the original rank-8 LoRA weights, initialized at their effective half strength. It uses a new preference optimizer; the unsuccessful activation teacher is not used.

Twenty-four same-prompt high/low CE pairs reuse 48 existing training histories and their scores. Semantic codes must reproduce 501 saved feedback frames exactly. No new training audio is generated. The differentiable objective favors the higher-scoring semantic sequence relative to the initial LoRA, with reference-policy KL and prompt preservation. It is a semantic-policy surrogate; it does not differentiate the audio scorer or claim the full audio likelihood.

Two learning rates train for 30 and 60 updates, with four comparisons per checkpoint. Step 120 is allowed only after a 3/4-win screen with mean CE gain at least .02 and no loss worse than -.30. Selection ranks wins, then the worst regression, then mean CE; the initial half-strength LoRA remains eligible. Only a promising new candidate receives four fresh Off/candidate pairs (eight clips). There are at most 36 new audio clips across this entire bounded experiment.

[Protocol](manifest.json) · [Training preferences](preferences.json) · [Validation](audit/tests.json)

Method reference: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). This music-specific adaptation uses mean semantic log probabilities and omits sampling top-k and residual/acoustic likelihoods.

| Checkpoint | Wins vs Off | Mean CE gain | Worst delta | Pass |
|---|---:|---:|---:|---|
| incumbent | 3/4 | +0.182 | -0.367 | False |
| gentle-30 | 2/4 | +0.032 | -0.486 | False |
| faster-30 | 3/4 | -0.191 | -1.611 | False |
| gentle-60 | 3/4 | +0.042 | -0.626 | False |
| faster-60 | 2/4 | -0.630 | -2.293 | False |

These are reused development comparisons. They cannot establish shipping readiness.

Outcome: **no_promising_continuation**.

[Recorded results](results.json)

Studio playback stays available during the run. GPU 0 returns to one studio worker afterward. Original weights and production slider registration remain unchanged.
