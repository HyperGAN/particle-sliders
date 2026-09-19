# Studio mix listening results

Render status: **complete** (25/25). Measurement status: **complete**.

[Listen in the studio](http://localhost:7860/studio-mix-20260907/) · [Protocol](README.md) · [Exact jobs](screen.json)

The table compares the same 50/50 mix at three energies, with the same caption, lyric sheet and seed.
Concept columns are CLAP margins against fixed descriptions. Compare changes within each column; different concepts do not share an audible strength scale.
Phrase match is a Whisper transcript diagnostic for the first 20 seconds. Low values can reflect late vocals or ASR failure as well as wrong words. PQ and CE are model predictions, not listener preference.
All excerpts use their original amplitude. The measurement models use separate RMS-normalized copies; the source recordings are preserved.

Each of these six checkpoints lists a recommended multiplier range of 0–1. An equal pair at studio energy 2 already applies multiplier 1 to each adapter; energy 2.8 applies 1.4 to each. A solo at energy 2 applies multiplier 2. The reduced norm of a blend relative to a solo at energy 2 therefore does not by itself establish that the blend is too weak.

## female + pop

| Energy | female margin | pop margin | Phrase match | PQ | CE |
|---:|---:|---:|---:|---:|---:|
| 2.000 | 0.027 | -0.040 | 0.943 | 8.449 | 7.719 |
| 2.500 | 0.088 | -0.122 | 0.953 | 7.886 | 7.434 |
| 2.800 | 0.066 | -0.068 | 0.000 | 8.464 | 7.976 |

## country + indie-rock

| Energy | country margin | indie-rock margin | Phrase match | PQ | CE |
|---:|---:|---:|---:|---:|---:|
| 2.000 | 0.106 | 0.183 | 0.000 | 8.215 | 7.774 |
| 2.500 | 0.095 | 0.173 | 0.952 | 8.351 | 7.850 |
| 2.800 | 0.245 | 0.352 | 0.850 | 8.217 | 7.586 |

## house + acoustic-folk

| Energy | house margin | acoustic-folk margin | Phrase match | PQ | CE |
|---:|---:|---:|---:|---:|---:|
| 2.000 | -0.107 | -0.027 | 0.837 | 8.323 | 7.654 |
| 2.500 | -0.146 | 0.032 | 0.750 | 7.882 | 7.295 |
| 2.800 | -0.055 | 0.092 | 0.808 | 8.309 | 7.459 |

## User listening result

The user preferred energy **2.8 in all three pairs**. These preferences take precedence over the automatic diagnostics above.

| Pair | Best to worst energy | Qualification |
|---|---|---|
| female + pop | 2.8, 2.5, 2 | Top two were close |
| country + indie-rock | 2.8, 2, 2.5 | No closeness stated |
| house + acoustic-folk | 2.8, 2, 2.5 | No closeness stated |

Next confirmation: energies 2.0 and 2.8 on reserved row 3 / seed 303 for all three pairs (six renders). No studio default is changed by these one-fixture rankings.

## Scope

One fixture, one seed, three authored pairs. The study does not establish a new studio default or test all sixteen adapters in combination.
No confirmation runs on row 3 / seed 303 are included. Those remain reserved until a listening preference or clear failure hypothesis selects the next comparison.

Raw render records: `renders.json`. Model measurements and transcripts: `measurements.json`. All individual results: `results.csv`.
