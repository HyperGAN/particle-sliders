# Metrics alongside the preference for 600

The user prefers 600 in a particular listened example and explicitly allows for sample variation. Exact page and seed are unspecified. The available lyric proxy is consistent with that preference, but the stable training metrics and fixed-history KL do not establish musical quality.

| Checkpoint | Semantic KL, positive history | Main prompt, seed 7 word recall | Main prompt, seed 23 word recall | Additional prompt, seed 7 word recall |
| --- | ---: | ---: | ---: | ---: |
| Original 600 | 0.010007 | 92.3% | 92.3% | 80.0% |
| Lyric hold + step limit 750 | 0.010060 | 61.5% | 92.3% | 60.0% |
| Lyric hold + step limit 900 | 0.010024 | 61.5% | 92.3% | 0.0% |

Word recall is an ASR diagnostic, not the percentage of lyrics sung correctly. Historical broken 900 had positive-history continuation KL 9.203376, compared with approximately 0.010 for these candidates: the diagnostic detects that large failure but has not ranked the appealing, numerically stable candidates.

The additional-prompt recognizer recovers portions of three supplied lines at 600, the first two at new 750, and none at new 900. Main-prompt seed 23 ties by word coverage; seed 7 favors 600. This leaves both prompt dependence and sample variation unresolved.

- KL is agreement with a positive-caption teacher on fixed histories; it does not evaluate the model following its own generated audio history or depth/flow predictions.
- ASR supplied-word recall ignores extras, order and repeats; recognition errors and the 20-second cap affect the result.
- Only two generation seeds on one training prompt and one seed on one additional development prompt are available; no reliable checkpoint preference rate or sampling-variance estimate is established.
- The new 750 and 900 change both the lyric hold and parameter-step bound relative to the 600 parent. This comparison cannot isolate training duration or either intervention.

600 remains the comparison reference. A useful next comparison would hold these three checkpoints fixed and use several predetermined seeds across multiple arrangements and lyrics, with blind listening recorded separately for voice character and lyric intelligibility. No new training or rendering was started for this metrics review.

Sources: [policy](policy_lyric_hold_20260904.json), [main ASR](asr_lyric_hold_20260904.json), [additional-prompt ASR](asr_lyric_hold_heldout750_20260904.json), [user feedback](listening_preferences.json).
