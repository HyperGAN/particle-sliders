# Combination and energy diagnostics

All 30 studio renders and CPU measurements completed. These results describe the first 20 seconds.
Aesthetics and CLAP use RMS-normalized copies; listening previews retain the generated levels.
Following the user clarification, mean Content Enjoyment selects the best tested settings.
[Selected presets and tradeoffs](metric-selection.md). Listening is optional, not a prerequisite for selection.

[Blind listening page](http://192.168.1.90:7860/studio-combo-energy-20260907/).

Each table averages two fresh seeds on the same existing lyric sheet and caption. Differences across
seeds are retained in `results.csv` and `paired-changes.json`; two samples are not a confidence interval.
Both cards rendered production recuts. Energy splits between faders: 50/50 gives 1.2/1.2 at E2.4,
1.4/1.4 at E2.8 and 1.6/1.6 at E3.2. At E2.8 the unequal balances give 1.12/1.68 or 1.68/1.12.

CE = content enjoyment; PQ = production quality; PC = production complexity. Higher complexity is
not necessarily desirable. Each CLAP margin measures one requested concept separately. ASR is a
diagnostic: an instrumental intro can score poorly against the full lyric sheet.

## Metric observations

- At equal balance, E2.4 scores above E2.8 on PQ in all six matched comparisons and on CE in five.
  These are six comparisons from only two seeds on one song, and some gaps are small.
- The highest energy-triplet CE score is at E3.2 for seed 707 and E2.4 for seed 909, for every pair.
  The female + pop seed-707 CE lead over E2.8 is only about 0.005.
- At E2.8, 60/40 improves CE and PQ over 50/50 on both seeds for female + pop and country + indie rock.
  The country seed-707 PQ increase is only about 0.002. For female + pop, the feminine-voice CLAP margin
  decreases on seed 909 despite the quality-score gains; this is not uniform improvement on all goals.
- House + acoustic folk is sensitive to the seed: 60/40 improves CE/PQ on 707 and reduces both on 909.
  Both unequal balances also reduce the acoustic-folk margin on seed 909.

The numerical selection is reported separately from these diagnostics. Combining E2.4 with 60/40
would be an untested interaction; the screen has not rendered that combination.

## female + pop

| Setting | CE | PQ | PC | First concept | Second concept | Lyric phrase |
|---|---:|---:|---:|---:|---:|---:|
| E2.4, 50/50 | 7.939 | 8.466 | 6.404 | 0.089 | -0.029 | 0.500 |
| E2.8, 50/50 | 7.690 | 8.122 | 5.651 | 0.056 | -0.100 | 0.864 |
| E3.2, 50/50 | 7.927 | 8.275 | 6.207 | 0.028 | -0.072 | 0.482 |
| E2.8, 40/60 | 7.551 | 8.125 | 5.801 | 0.106 | -0.221 | 0.482 |
| E2.8, 60/40 | 8.142 | 8.338 | 6.309 | 0.034 | -0.013 | 0.904 |

Changes from E2.8 at 50/50, shown separately for each seed:

| Setting | CE Δ seed 707 | CE Δ seed 909 | PQ Δ seed 707 | PQ Δ seed 909 |
|---|---:|---:|---:|---:|
| e24 | -0.032 | +0.528 | +0.058 | +0.628 |
| e32 | +0.005 | +0.469 | -0.138 | +0.443 |
| balance40 | -0.263 | -0.016 | -0.013 | +0.017 |
| balance60 | +0.074 | +0.829 | +0.035 | +0.396 |

## country + indie-rock

| Setting | CE | PQ | PC | First concept | Second concept | Lyric phrase |
|---|---:|---:|---:|---:|---:|---:|
| E2.4, 50/50 | 7.745 | 8.360 | 6.060 | 0.182 | 0.191 | 0.800 |
| E2.8, 50/50 | 7.669 | 8.155 | 4.861 | 0.154 | 0.074 | 0.818 |
| E3.2, 50/50 | 7.746 | 8.213 | 5.081 | 0.205 | 0.216 | 0.500 |
| E2.8, 40/60 | 7.602 | 8.048 | 4.673 | 0.193 | 0.134 | 0.182 |
| E2.8, 60/40 | 7.844 | 8.453 | 6.377 | 0.168 | 0.180 | 0.895 |

Changes from E2.8 at 50/50, shown separately for each seed:

| Setting | CE Δ seed 707 | CE Δ seed 909 | PQ Δ seed 707 | PQ Δ seed 909 |
|---|---:|---:|---:|---:|
| e24 | +0.078 | +0.076 | +0.016 | +0.396 |
| e32 | +0.211 | -0.058 | +0.041 | +0.076 |
| balance40 | -0.345 | +0.212 | -0.670 | +0.457 |
| balance60 | +0.098 | +0.252 | +0.002 | +0.594 |

## house + acoustic-folk

| Setting | CE | PQ | PC | First concept | Second concept | Lyric phrase |
|---|---:|---:|---:|---:|---:|---:|
| E2.4, 50/50 | 7.865 | 8.384 | 4.874 | -0.223 | 0.120 | 0.893 |
| E2.8, 50/50 | 7.762 | 8.005 | 4.492 | -0.273 | 0.113 | 0.844 |
| E3.2, 50/50 | 7.889 | 7.947 | 5.386 | -0.259 | 0.039 | 0.500 |
| E2.8, 40/60 | 7.252 | 7.758 | 5.178 | -0.294 | -0.069 | 0.795 |
| E2.8, 60/40 | 7.215 | 7.784 | 5.561 | -0.283 | 0.040 | 1.000 |

Changes from E2.8 at 50/50, shown separately for each seed:

| Setting | CE Δ seed 707 | CE Δ seed 909 | PQ Δ seed 707 | PQ Δ seed 909 |
|---|---:|---:|---:|---:|
| e24 | +0.031 | +0.175 | +0.309 | +0.448 |
| e32 | +0.516 | -0.262 | +0.241 | -0.357 |
| balance40 | -0.939 | -0.083 | -0.656 | +0.163 |
| balance60 | +0.121 | -1.216 | +0.244 | -0.686 |

## Scope

This is a local cross-shaped sweep. Energy comparisons use equal balance; balance comparisons use E2.8.
It does not establish the best balance at other energies, a universal optimum, a solo energy default,
or a result for other songs or three-slider combinations. The repeated center recording in the energy
and balance listening sections is one observation, not two independent renders.

The selected presets maximize mean CE on these search samples; fresh-seed generalization remains untested.
The production generator, weights and default energy are unchanged.
