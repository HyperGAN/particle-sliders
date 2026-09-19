# Settings selected by Content Enjoyment

The user clarified that this experiment should use the metric to choose settings.
Selection now maximizes mean Content Enjoyment (CE) across the two shared seeds for each pair.
Production Quality (PQ) and the two style margins are visible diagnostics. They do not silently change the objective.
No listening ranking is required to make these selections.

| Pair | Energy | Balance in pair order | Mean CE | CE gain vs reference | PQ gain vs reference |
|---|---:|---:|---:|---:|---:|
| female + pop | 2.8 | 60/40 | 8.142 | +0.451 | +0.216 |
| country + indie-rock | 2.8 | 60/40 | 7.844 | +0.175 | +0.298 |
| house + acoustic-folk | 3.2 | 50/50 | 7.889 | +0.127 | -0.058 |

Reference: energy 2.8 at 50/50. CE and PQ gains are score points, not percentages or measured human preference.

Female + pop and country + indie rock improve CE and PQ over the reference on both seeds.
The female-voice CLAP margin drops on one selected female + pop sample, so quality improvement is not uniform style improvement.
House + acoustic folk at E3.2 has the highest mean CE, only about 0.024 ahead of E2.4. Its mean PQ falls by 0.058 versus the reference,
and its CE improvement reverses on seed 909. E2.4 is the stronger alternative for quality and consistency, but E3.2 is the winner of the stated CE objective.

These are the best tested settings on this search sample, selected after the sweep. They have not been validated on fresh seeds or songs.
No automatic production routing or global slider default has been installed. The presets are saved in `selected-presets.json`.

[Metric-selected results](http://192.168.1.90:7860/studio-combo-energy-20260907/selected.html).
