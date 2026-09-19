# Merger study diagnostics

[Final audit](audit.json): passed all 24 recordings, 24 measurements and 48 audio URLs. [Listen in the studio](http://192.168.1.90:7860/studio-mergers-20260907/).

Human listening preferences are recorded for all six comparisons and take precedence over the proxy scores below. No production default has changed.

Recordings: 24/24 complete. Six are verified reuse; eighteen are new. Full audio and exact first-20-second excerpts are available in the gallery.

## User listening results

[Exact feedback and decoded key](listening-feedback.json). Rankings are best to worst; the close and uncertain judgments are retained.

| Song | Pair | Ranking | Qualification |
|---|---|---|---|
| familiar | female + pop | KnOTS-TIES > Ordinary E2.8 > TIES | None stated |
| familiar | country + indie-rock | Ordinary E2.8 > KnOTS-TIES > TIES | None stated |
| familiar | house + acoustic-folk | KnOTS-TIES > Ordinary E2.8 > TIES | None stated |
| confirmation | female + pop | TIES > KnOTS-TIES > Ordinary E2.8 | The user said the ranking was hard to tell; preserve this uncertainty. |
| confirmation | country + indie-rock | Ordinary E2.8 > TIES > KnOTS-TIES | C and B were close; both were ranked well above A. |
| confirmation | house + acoustic-folk | Ordinary E2.8 > KnOTS-TIES > TIES | None stated |

Ordinary E2.8 ranks first in three groups, KnOTS-TIES in two, and TIES in one. The TIES win is the explicitly uncertain female + pop confirmation group. TIES ranks last in four groups. Counts summarize order only; they do not measure how large the preference was.

KnOTS-TIES beats ordinary mixing for female + pop on both songs, although the confirmation ranking was hard to distinguish. Ordinary wins country + indie rock twice; on confirmation it is close to TIES, with KnOTS-TIES well behind. House + acoustic folk splits between KnOTS-TIES and ordinary, with TIES last on both songs.

Recommendation: retain ordinary full-delta addition as the studio default. KnOTS-TIES merits a focused follow-up for female + pop and a possible optional mode; it does not justify a universal replacement or automatic pair-based routing. The current TIES setting is a lower priority given its four last places and uncertain lone win.

The user did not rank the energy-2 references. These results select among merger methods at matched parameter strength; they do not independently confirm energy 2.8 over energy 2 on the second song.

## Geometry

Each of the 144 projection norms matches ordinary energy 2.8. The cosine below measures update direction; it is not musical similarity or quality.

| Pair | Method | Cosine to ordinary | Maximum relative norm error |
|---|---|---:|---:|
| female+pop | ties | 0.8766 | 5.3e-08 |
| female+pop | knots_ties | 0.9439 | 5.7e-08 |
| country+indie-rock | ties | 0.8718 | 5e-08 |
| country+indie-rock | knots_ties | 0.9182 | 5.4e-08 |
| house+acoustic-folk | ties | 0.8728 | 5e-08 |
| house+acoustic-folk | knots_ties | 0.9269 | 5.4e-08 |

## Audio diagnostics

The CSV includes CLAP concept margins, aesthetics, lyric recognition and signal checks. These proxies can disagree with listeners. ASR recall against the complete lyric sheet is also duration-dependent; an instrumental opening is not automatically a vocal failure. No overall score or automated winner is computed.

Interpret listening group by group, then compare the same pair on the second song. A single 30% pruning setting does not test every possible TIES/KnOTS variant.

Observed proxy tradeoffs: ordinary E2.8 has higher country and indie-rock CLAP margins than either merger on both songs. For female + pop, KnOTS-TIES has higher PQ/CE predictions and pop margins than ordinary E2.8 on both songs, with lower female margins. House + acoustic folk changes are mixed across songs and metrics. These patterns motivate listening comparisons; they do not establish a preferred merger or a universal replacement.

Concept columns are CLAP margins against fixed descriptions; compare within each concept column. PQ and CE are aesthetics-model predictions. Phrase match is a first-20-second ASR diagnostic, not a human intelligibility rating.

### familiar: female + pop

| Method | female | pop | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | 0.066 | -0.068 | 8.464 | 7.976 | 0.000 |
| TIES | 0.026 | -0.000 | 8.439 | 7.705 | 0.885 |
| KnOTS-TIES | 0.049 | 0.030 | 8.531 | 8.046 | 0.897 |
| Ordinary E2 reference | 0.027 | -0.040 | 8.449 | 7.719 | 0.943 |

### familiar: country + indie-rock

| Method | country | indie-rock | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | 0.245 | 0.352 | 8.217 | 7.586 | 0.850 |
| TIES | 0.074 | 0.090 | 8.443 | 7.745 | 1.000 |
| KnOTS-TIES | 0.100 | 0.171 | 8.184 | 7.825 | 0.846 |
| Ordinary E2 reference | 0.106 | 0.183 | 8.215 | 7.774 | 0.000 |

### familiar: house + acoustic-folk

| Method | house | acoustic-folk | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | -0.055 | 0.092 | 8.309 | 7.459 | 0.808 |
| TIES | -0.158 | 0.154 | 8.198 | 7.395 | 0.667 |
| KnOTS-TIES | -0.060 | 0.067 | 8.335 | 7.988 | 0.923 |
| Ordinary E2 reference | -0.107 | -0.027 | 8.323 | 7.654 | 0.837 |

### confirmation: female + pop

| Method | female | pop | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | 0.111 | -0.093 | 8.384 | 7.788 | 1.000 |
| TIES | 0.055 | -0.089 | 8.328 | 8.030 | 1.000 |
| KnOTS-TIES | 0.076 | -0.056 | 8.500 | 8.135 | 0.941 |
| Ordinary E2 reference | 0.062 | -0.051 | 8.510 | 7.980 | 0.923 |

### confirmation: country + indie-rock

| Method | country | indie-rock | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | 0.214 | 0.357 | 8.333 | 7.644 | 1.000 |
| TIES | -0.059 | -0.104 | 8.339 | 7.636 | 0.969 |
| KnOTS-TIES | 0.077 | 0.079 | 8.188 | 7.583 | 0.833 |
| Ordinary E2 reference | 0.170 | 0.200 | 8.424 | 7.984 | 0.969 |

### confirmation: house + acoustic-folk

| Method | house | acoustic-folk | PQ | CE | Phrase match |
|---|---:|---:|---:|---:|---:|
| Ordinary E2.8 | -0.147 | 0.057 | 8.447 | 7.718 | 0.870 |
| TIES | -0.099 | 0.099 | 8.426 | 7.856 | 0.000 |
| KnOTS-TIES | -0.117 | 0.248 | 8.380 | 6.970 | 1.000 |
| Ordinary E2 reference | -0.102 | -0.049 | 8.246 | 6.988 | 0.000 |
