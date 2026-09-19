# Pilot stopping result

The fixed v2 heuristic search stopped at 1350 after three additional checkpoints without a gain above the previous best by more than 0.05. This is an observed stopping point, not a global musical optimum.

| Updates | Mean heuristic |
| --- | ---: |
| 600 | 0.950077 |
| 750 | 0.691833 |
| 900 | 0.608087 |
| 1050 | 0.847509 |
| 1200 | 0.851237 |
| 1350 | 0.614496 |

The 600 point is the original repaired-smoke state. Later points use the step limit alone, applied after 600.

Additional-sample confirmation compares original 600 with lyric hold 0.001 plus step limit 2 at 750. Each candidate gets two additional arrangements and generation seeds 101/303. Catalog selection uses the combined eight-sample mean.

The full catalog starts automatically after confirmation; no further user decision is required.

## Confirmed result

Original 600 scored 0.893195 on the four additional samples, versus 0.758464
for lyric hold plus step limit 750. Combined eight-sample scores were
0.921636 and 0.828828 respectively. The automatic catalog suggestion is
600 original repaired-smoke updates, lyric hold 0, with no step limit.

The 28-control batch started on physical GPU 1. The first fresh control is
Female; its candidate comparison also retains the confirmed pilot winner.
Earlier 300/450 checkpoints remain eligible for each control. Matched examples,
selected adapters and longer renders will populate the live gallery as jobs
finish. No completed catalog controls are claimed at batch startup.
