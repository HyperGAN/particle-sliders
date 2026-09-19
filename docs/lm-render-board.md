# LM render board — every scored listen ladder (2026-09-02)

Discovery board for `scripts/lm_score.py` after its one and only calibration
pass. Contract and frozen gate definitions: `LM-SCORING.md`. Machine-readable
source: `eval/lm_score_board.tsv` plus `lm_scores.json` inside every folder.

**94 folders scored** out of 106 attempted; 12 were skipped for missing REF
clips (`uni-v2/live-lm-uni-v2`, `v16/breath-lm-v16`, all four `v17` breath /
distortion / energy / tempo, `v19/rhyme` + `v19/tempo`, `v22/rhyme` +
`v22/tempo`, and both `v23` folders, which contain slider clips but no REFs).

## How to read a verdict on this page

**PASS is a no-harm certificate, not an efficacy certificate.** The calibration
pass demoted every channel that was supposed to certify that the concept moved
(`LM-SCORING.md`, "What the calibration pass decided"): U6's null rejects all
six ears-PASS folders, and rank monotonicity anti-correlates with ears. What
remains vetoes silence (U1), lyric garble (U2), a flagrantly reversed plus pole
(U3), song replacement (U4), and a hot ending at ≥45 s (U5). So:

- **PASS** = "this render still sings the line and is still the same song".
  43 of 94 folders. It does **not** say the slider does anything.
- **FAIL** = one of those five things is broken, with the gate named.
- **UNSCOREABLE** = the ladder has no rung at the product setting `|s| ≤ 1`.
  One folder: `gender-lm-20s`, rendered ±2 only.

**Every verdict here is single-seed (seed 7) and therefore provisional.**
`"seeds": 1` is written into every `lm_scores.json`. Promotion needs ≥3 seeds
and a 60–90 s ending check, where U5 becomes a veto. Nothing on this page
promotes anything.

**E is an uncertified ranking heuristic**, not an effect size: signed
`dsp_proj` at the unit rung times the lyric factor, min over poles. It is a
fraction of the same-seed caption swap — the kind of magnitude the contract
forbids as a *gate* — is not comparable across seeds, and includes `ln rms`
among its 8 components, so on energy-like axes a slider that only turns up the
gain scores well. It orders gate-passers deterministically and nothing more.
`U5_ending` warns on 89 of 94 folders because every ladder is 20 s; that warn
is not a defect signal.

## Recipe families

| family | what it is | n |
|---|---|---|
| `uni-lyric` | **the current shipped catalog** — uni sliders, one LoRA per pole (`grit` and `grit-smooth` are two different checkpoints) | 28 |
| `uni-v2` | the previous uni wave, the one `docs/lm-uni-v2-garble.md` listened through | 14 |
| `uni-v3` | prefix-hold uni variants (`*-lm-uni-prefix`) | 8 |
| `v16`–`v22` | bipolar waves, one LoRA spanning both poles | 28 |
| `v9-ritual` | the `lm-v9` catalog wave, bipolar | 14 |
| `v3-era` | the two original 20 s bipolar ladders (`energy-v3-lm-20s`, `gender-lm-20s`) | 2 |

## 1. What actually works: best PASS-ing candidate per axis, by E

The catalog answer, subject to every caveat above — in particular that E does
not certify the concept moved and that each row is one seed. Where a pole
variant wins, the winning **pole** is named: `Smooth` winning the `grit` axis
means the smooth-direction checkpoint is the one that survives the gates, not
that "grit works".

| axis | best PASS by E | family | pole | E | rec@unit+ | same_song ÷ anchor | dsp+ | passers / scored |
|---|---|---|---|---|---|---|---|---|
| **breath** | `uni-v2/breath-lm-uni-v2` | uni-v2 | Breathy | +0.28 | 0.92 | 0.87 | +0.31 | 2 / 7 |
| **distortion** | `uni-lyric/distortion-clean-lm-uni-lyric` | uni-lyric | Clean | +0.21 | 0.92 | 0.58 | +0.21 | 2 / 7 |
| **energy** | `uni-lyric/energy-lm-uni-lyric` | uni-lyric | Loud | +1.30 | 1.00 | 0.75 | +1.30 | 10 / 12 |
| **gender** | `uni-v2/gender-lm-uni-v2` | uni-v2 | Female | +1.11 | 0.92 | 0.77 | +1.20 | 4 / 11 |
| **grit** | `uni-lyric/grit-smooth-lm-uni-lyric` | uni-lyric | Smooth | +0.77 | 0.92 | 0.61 | +0.77 | 2 / 5 |
| **hurt** | `uni-v3/hurt-lm-uni-prefix` | uni-v3 | Hurt | +1.67 | 1.00 | 0.39 | +1.67 | 3 / 5 |
| **joy** | `uni-lyric/joy-somber-lm-uni-lyric` | uni-lyric | Somber | +0.17 | 1.00 | 0.34 | +0.17 | 2 / 5 |
| **live** | `uni-lyric/live-lm-uni-lyric` | uni-lyric | Live | +0.66 | 0.92 | 0.88 | +0.71 | 2 / 6 |
| **rapslow** | `uni-lyric/rapslow-lm-uni-lyric` | uni-lyric | Rap | +0.60 | 0.96 | 0.32 | +0.63 | 3 / 5 |
| **rhyme** | `uni-lyric/rhyme-prose-lm-uni-lyric` | uni-lyric | Prose | +0.42 | 1.00 | 0.87 | +0.42 | 2 / 6 |
| **sexy** | *none* | — | — | — | — | — | — | 0 / 4 |
| **tempo** | `uni-lyric/tempo-lm-uni-lyric` | uni-lyric | Fast | +1.00 | 1.00 | 0.67 | +1.00 | 3 / 7 |
| **tender** | `uni-lyric/tender-fierce-lm-uni-lyric` | uni-lyric | Fierce | +0.09 | 0.92 | 0.36 | +0.10 | 2 / 4 |
| **triphop** | `uni-lyric/triphop-lm-uni-lyric` | uni-lyric | Trip-hop | +1.21 | 1.00 | 0.48 | +1.21 | 3 / 6 |
| **yearn** | `uni-v2/yearn-lm-uni-v2` | uni-v2 | Yearn | +0.37 | 1.00 | 0.49 | +0.37 | 3 / 4 |

`sexy` is the only axis with **no** passer anywhere in the corpus.

Reading the E column: `hurt-lm-uni-prefix` tops the board at +1.67 on a
`uni-v3` prefix variant that has never been listened to, while the two
ears-approved catalog entries that *are* labeled sit at +1.11
(`gender-lm-uni-v2`) and +0.68 (`tempo-lm-uni-v2`). Nothing validates that
ordering. Treat the column as a queue for listening, not a ranking of quality.

## 2. Axes where nothing passes, and the dominant failure mode

| axis | scored | passers | dominant fired gate | what the numbers say |
|---|---|---|---|---|
| **sexy** | 4 | **0** | U4 same-song, 4/4 | every `sexy` checkpoint drifts off its own song at ±1: `sexy-plain-lm-uni-lyric` 1.31 × anchor, `sexy-lm-uni-v2` 1.25, `sexy-lm-uni-lyric` 0.99, `sexy-lm-v9` 0.96 — with lyric recall 1.00 on three of the four. It is not garble; the render becomes a different recording that happens to sing the same words. `sexy-lm-v9` adds garble on top (+1 recall 0.23) |

No other axis is empty, but three are close, and the near-misses are worth as
much as the empty one:

| axis | passers / scored | note |
|---|---|---|
| `breath` | 2 / 7 | 4 of 5 failures are U4 (`breath-lm-v9` 1.22 × anchor, `breath-lm-uni-lyric` 1.07). The shipped `breath-lm-uni-lyric` is one of them |
| `gender` | 4 / 11 | 7 U4 failures, and the axis carries both the corpus's clearest garble (`gender-lm-uni-lyric` +1 recall **0.15**) and its one `UNSCOREABLE` ladder |
| `distortion` | 2 / 7 | 4 of 5 failures are U2 garble — the only axis where garble, not drift, is the dominant mode |
| `tempo` | 3 / 7 | all 4 failures are U4 — one `uni-v3` prefix variant plus `v20`, `v21` and `v9`; every `uni-lyric` and `uni-v2` tempo checkpoint passes |

Family-level census, for orientation:

| family | PASS | FAIL | UNSCOREABLE | fired gates (count) |
|---|---|---|---|---|
| `uni-lyric` | 21 | 7 | — | U4 ×6, U2 ×3 |
| `uni-v2` | 8 | 6 | — | U2 ×5, U4 ×2 |
| `uni-v3` | 4 | 4 | — | U4 ×4 |
| `v16` | 1 | 3 | — | U4 ×2, U2 ×1, U3 ×1 |
| `v17` | 0 | 2 | — | U4 ×2 |
| `v18` | 1 | 0 | — | — |
| `v19` | 2 | 2 | — | U2 ×1, U4 ×1 |
| `v20` | 1 | 8 | — | U4 ×7, U2 ×3, U3 ×1 |
| `v21` | 0 | 6 | — | U4 ×5, U2 ×3 |
| `v22` | 2 | 0 | — | — |
| `v9-ritual` | 2 | 12 | — | U4 ×9, U2 ×7, U3 ×1 |
| `v3-era` | 1 | — | 1 | U0 ×1, U4 ×1 |

The single dominant failure mode across the whole corpus is **U4 song drift**
(39 folders), not garble (23). That inverts the campaign's working assumption,
which came from `docs/lm-uni-v2-garble.md` and named garble as the LM failure
mode. Garble dominates only in `uni-v2`; from `uni-v3` onward the sliders keep
the words and lose the song. Two caveats on that reading: U4's threshold rests
on a seven-point gap (`LM-SCORING.md`), and "drift" in this instrument means
whisper-embedding cosine distance, which cannot distinguish "a bolder
arrangement of the same song" from "a different recording" — the exact
confusion that produced the one documented false accept.

## 3. The shipped `uni-lyric` catalog: which axes hold the lyrics at +1

The user-facing question. `rec@+1` is whisper-large-v3-turbo bag-of-words recall
of the folder's own yaml line; **U2 = GARBLE** means the frozen lyric gate fired.

| axis / pole | folder | rec@0 (base) | rec@+1 | rec@REF+ | U2 | verdict |
|---|---|---|---|---|---|---|
| joy / Joy | `joy-lm-uni-lyric` | 0.61 | **0.00** | 1.00 | GARBLE | FAIL |
| gender / Female | `gender-lm-uni-lyric` | 1.00 | **0.15** | 0.77 | GARBLE | FAIL |
| distortion / Distorted | `distortion-lm-uni-lyric` | 0.62 | **0.31** | 0.54 | GARBLE | FAIL |
| grit / Grit | `grit-lm-uni-lyric` | 0.62 | **0.77** | 1.00 | holds | PASS |
| hurt / Hurt | `hurt-lm-uni-lyric` | 1.00 | **0.77** | 0.92 | holds | PASS |
| rhyme / Rhyme | `rhyme-lm-uni-lyric` | 0.92 | **0.77** | 1.00 | holds | FAIL |
| distortion / Clean | `distortion-clean-lm-uni-lyric` | 0.62 | **0.92** | 0.85 | holds | PASS |
| gender / Male | `gender-male-lm-uni-lyric` | 1.00 | **0.92** | 1.00 | holds | PASS |
| grit / Smooth | `grit-smooth-lm-uni-lyric` | 0.62 | **0.92** | 1.00 | holds | PASS |
| live / Live | `live-lm-uni-lyric` | 1.00 | **0.92** | 1.00 | holds | PASS |
| live / Studio | `live-studio-lm-uni-lyric` | 1.00 | **0.92** | 1.00 | holds | PASS |
| tender / Fierce | `tender-fierce-lm-uni-lyric` | 1.00 | **0.92** | 1.00 | holds | PASS |
| tender / Tender | `tender-lm-uni-lyric` | 1.00 | **0.92** | 1.00 | holds | PASS |
| rapslow / Rap | `rapslow-lm-uni-lyric` | 1.00 | **0.96** | 1.00 | holds | PASS |
| breath / Breathy | `breath-lm-uni-lyric` | 1.00 | **1.00** | 0.92 | holds | FAIL |
| energy / Loud | `energy-lm-uni-lyric` | 1.00 | **1.00** | 0.92 | holds | PASS |
| energy / Quiet | `energy-quiet-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | PASS |
| hurt / Numb | `hurt-numb-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | PASS |
| joy / Somber | `joy-somber-lm-uni-lyric` | 1.00 | **1.00** | 0.92 | holds | PASS |
| rhyme / Prose | `rhyme-prose-lm-uni-lyric` | 0.92 | **1.00** | 1.00 | holds | PASS |
| sexy / Sexy | `sexy-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | FAIL |
| sexy / Plain | `sexy-plain-lm-uni-lyric` | 1.00 | **1.00** | 0.85 | holds | FAIL |
| tempo / Fast | `tempo-lm-uni-lyric` | 1.00 | **1.00** | 0.92 | holds | PASS |
| tempo / Slow | `tempo-slow-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | PASS |
| triphop / Trip-hop | `triphop-lm-uni-lyric` | 0.92 | **1.00** | 1.00 | holds | PASS |
| triphop / Pop | `triphop-pop-lm-uni-lyric` | 0.92 | **1.00** | 1.00 | holds | PASS |
| yearn / Yearn | `yearn-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | PASS |
| yearn / Settled | `yearn-settled-lm-uni-lyric` | 1.00 | **1.00** | 1.00 | holds | PASS |

**Three of the 28 shipped `uni-lyric` checkpoints garble at +1**, and in every
case it is the "forward" pole that breaks while its opposite-pole twin holds:

| axis | garbling pole | rec@+1 | opposite pole | rec@+1 |
|---|---|---|---|---|
| **joy** | `joy-lm-uni-lyric` (Joy) | **0.00** — transcript is not the sheet at all | `joy-somber-lm-uni-lyric` (Somber) | 1.00 |
| **gender** | `gender-lm-uni-lyric` (Female) | **0.15** | `gender-male-lm-uni-lyric` (Male) | 0.92 |
| **distortion** | `distortion-lm-uni-lyric` (Distorted) | **0.31** | `distortion-clean-lm-uni-lyric` (Clean) | 0.92 |

That pattern — the pole that adds energy/intensity breaks AR, the pole that
removes it does not — is consistent across the catalog and is the same failure
mode `docs/lm-uni-v2-garble.md` recorded for `grit`, `distortion` and `joy` in
the previous wave. It is a per-pole checkpoint problem, not a per-axis one:
half of each pair is fine.

Four more `uni-lyric` checkpoints keep the words but fail **U4** (the render
drifts off its own song at +1), which is a different complaint the ear may or
may not agree with: `breath-lm-uni-lyric` (1.07 × anchor),
`rhyme-lm-uni-lyric` (1.60), `sexy-lm-uni-lyric` (0.99),
`sexy-plain-lm-uni-lyric` (1.31). None of these four has an ear label.

**21 of 28 uni-lyric checkpoints pass all vetoes.** Again: that means they do
not break the song, not that they move the concept.

## 4. Every scored folder

Columns: `rec@0` base render recall, `rec@unit±` recall at the gated rung,
`proj+` whisper-embedding projection at the unit plus rung (**logged, and known
to disagree with ears — see LM-SCORING.md finding 2**), `dsp±` DSP-space
projection at each unit rung (the sign channel U3 actually gates on),
`same_song ÷ anchor` U4's statistic against its 0.90 threshold, `E` the
uncertified ranking heuristic. `ears` is the locked human label from
`eval/lm_score_labels.json` where one exists — the scorer agrees with 9 of the
11 PASS/FAIL labels; the two it does not are `v16/energy-lm-v16` (ears FAIL,
scorer PASS) and `gender-lm-20s` (ears PASS, scorer UNSCOREABLE because the
ladder has no ±1 rung; at its ±2 rungs it would fail U4 at 1.17 × anchor).
Blank `rec@unit−` / `dsp−` cells are uni ladders, which have no minus pole.

### uni-lyric

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `uni-lyric/breath-lm-uni-lyric` | breath / Breathy | uni | FAIL | U4 | 1.00 | 1.00 |  | +0.59 | +0.27 |  | 1.07 | +0.27 |  |
| `uni-lyric/distortion-clean-lm-uni-lyric` | distortion / Clean | uni | **PASS** | - | 0.62 | 0.92 |  | +0.50 | +0.21 |  | 0.58 | +0.21 |  |
| `uni-lyric/distortion-lm-uni-lyric` | distortion / Distorted | uni | FAIL | U2 | 0.62 | 0.31 |  | +0.39 | +0.24 |  | 0.65 | +0.12 |  |
| `uni-lyric/energy-lm-uni-lyric` | energy / Loud | uni | **PASS** | - | 1.00 | 1.00 |  | +0.39 | +1.30 |  | 0.75 | +1.30 |  |
| `uni-lyric/energy-quiet-lm-uni-lyric` | energy / Quiet | uni | **PASS** | - | 1.00 | 1.00 |  | +0.50 | -0.38 |  | 0.70 | -0.38 |  |
| `uni-lyric/gender-male-lm-uni-lyric` | gender / Male | uni | **PASS** | - | 1.00 | 0.92 |  | +0.55 | +0.62 |  | 0.70 | +0.57 |  |
| `uni-lyric/gender-lm-uni-lyric` | gender / Female | uni | FAIL | U2,U4 | 1.00 | 0.15 |  | +0.92 | +0.42 |  | 1.21 | +0.06 |  |
| `uni-lyric/grit-lm-uni-lyric` | grit / Grit | uni | **PASS** | - | 0.62 | 0.77 |  | +0.36 | +0.19 |  | 0.64 | +0.19 |  |
| `uni-lyric/grit-smooth-lm-uni-lyric` | grit / Smooth | uni | **PASS** | - | 0.62 | 0.92 |  | +0.52 | +0.77 |  | 0.61 | +0.77 |  |
| `uni-lyric/hurt-numb-lm-uni-lyric` | hurt / Numb | uni | **PASS** | - | 1.00 | 1.00 |  | +0.66 | -0.62 |  | 0.77 | -0.62 |  |
| `uni-lyric/hurt-lm-uni-lyric` | hurt / Hurt | uni | **PASS** | - | 1.00 | 0.77 |  | +0.16 | +0.31 |  | 0.71 | +0.24 |  |
| `uni-lyric/joy-lm-uni-lyric` | joy / Joy | uni | FAIL | U2,U4 | 0.61 | 0.00 |  | +0.89 | +1.09 |  | 2.06 | +0.00 |  |
| `uni-lyric/joy-somber-lm-uni-lyric` | joy / Somber | uni | **PASS** | - | 1.00 | 1.00 |  | +0.10 | +0.17 |  | 0.34 | +0.17 |  |
| `uni-lyric/live-lm-uni-lyric` | live / Live | uni | **PASS** | - | 1.00 | 0.92 |  | +0.71 | +0.71 |  | 0.88 | +0.66 |  |
| `uni-lyric/live-studio-lm-uni-lyric` | live / Studio | uni | **PASS** | - | 1.00 | 0.92 |  | +0.40 | +0.27 |  | 0.46 | +0.25 |  |
| `uni-lyric/rapslow-lm-uni-lyric` | rapslow / Rap | uni | **PASS** | - | 1.00 | 0.96 |  | +0.39 | +0.63 |  | 0.32 | +0.60 |  |
| `uni-lyric/rhyme-lm-uni-lyric` | rhyme / Rhyme | uni | FAIL | U4 | 0.92 | 0.77 |  | +1.06 | -0.72 |  | 1.60 | -0.60 |  |
| `uni-lyric/rhyme-prose-lm-uni-lyric` | rhyme / Prose | uni | **PASS** | - | 0.92 | 1.00 |  | +0.43 | +0.42 |  | 0.87 | +0.42 |  |
| `uni-lyric/sexy-plain-lm-uni-lyric` | sexy / Plain | uni | FAIL | U4 | 1.00 | 1.00 |  | +0.63 | +0.79 |  | 1.31 | +0.79 |  |
| `uni-lyric/sexy-lm-uni-lyric` | sexy / Sexy | uni | FAIL | U4 | 1.00 | 1.00 |  | +0.77 | +0.40 |  | 0.99 | +0.40 |  |
| `uni-lyric/tempo-lm-uni-lyric` | tempo / Fast | uni | **PASS** | - | 1.00 | 1.00 |  | +0.62 | +1.00 |  | 0.67 | +1.00 |  |
| `uni-lyric/tempo-slow-lm-uni-lyric` | tempo / Slow | uni | **PASS** | - | 1.00 | 1.00 |  | +0.12 | +0.07 |  | 0.57 | +0.07 |  |
| `uni-lyric/tender-fierce-lm-uni-lyric` | tender / Fierce | uni | **PASS** | - | 1.00 | 0.92 |  | +0.16 | +0.10 |  | 0.36 | +0.09 |  |
| `uni-lyric/tender-lm-uni-lyric` | tender / Tender | uni | **PASS** | - | 1.00 | 0.92 |  | +0.49 | +0.01 |  | 0.51 | +0.00 |  |
| `uni-lyric/triphop-lm-uni-lyric` | triphop / Trip-hop | uni | **PASS** | - | 0.92 | 1.00 |  | +0.52 | +1.21 |  | 0.48 | +1.21 |  |
| `uni-lyric/triphop-pop-lm-uni-lyric` | triphop / Pop | uni | **PASS** | - | 0.92 | 1.00 |  | +0.36 | +0.50 |  | 0.33 | +0.50 |  |
| `uni-lyric/yearn-lm-uni-lyric` | yearn / Yearn | uni | **PASS** | - | 1.00 | 1.00 |  | +0.32 | -0.75 |  | 0.52 | -0.75 |  |
| `uni-lyric/yearn-settled-lm-uni-lyric` | yearn / Settled | uni | **PASS** | - | 1.00 | 1.00 |  | -0.15 | +0.01 |  | 0.57 | +0.01 |  |

### uni-v2

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `uni-v2/breath-lm-uni-v2` | breath / Breathy | uni | **PASS** | - | 1.00 | 0.92 |  | +0.49 | +0.31 |  | 0.87 | +0.28 |  |
| `uni-v2/distortion-lm-uni-v2` | distortion / Distorted | uni | FAIL | U2 | 0.62 | 0.00 |  | +0.27 | +0.45 |  | 0.69 | +0.00 | FAIL |
| `uni-v2/energy-lm-uni-v2` | energy / Loud | uni | **PASS** | - | 1.00 | 0.92 |  | +0.66 | +0.21 |  | 0.65 | +0.20 | BORDERLINE |
| `uni-v2/gender-lm-uni-v2` | gender / Female | uni | **PASS** | - | 1.00 | 0.92 |  | +0.57 | +1.20 |  | 0.77 | +1.11 | PASS |
| `uni-v2/grit-lm-uni-v2` | grit / Grit | uni | FAIL | U2,U4 | 0.62 | 0.00 |  | +0.59 | +0.48 |  | 1.01 | +0.00 | FAIL |
| `uni-v2/hurt-lm-uni-v2` | hurt / Hurt | uni | FAIL | U2 | 1.00 | 0.46 |  | +0.80 | +0.88 |  | 0.86 | +0.41 | BORDERLINE |
| `uni-v2/joy-lm-uni-v2` | joy / Joy | uni | FAIL | U2 | 1.00 | 0.31 |  | +0.26 | +0.27 |  | 0.53 | +0.08 | FAIL |
| `uni-v2/rapslow-lm-uni-v2` | rapslow / Rap | uni | **PASS** | - | 0.75 | 0.75 |  | +0.29 | +0.23 |  | 0.38 | +0.23 | BORDERLINE |
| `uni-v2/rhyme-lm-uni-v2` | rhyme / Rhyme | uni | **PASS** | - | 0.92 | 1.00 |  | +0.53 | -2.79 |  | 0.58 | -2.79 |  |
| `uni-v2/sexy-lm-uni-v2` | sexy / Sexy | uni | FAIL | U4 | 1.00 | 1.00 |  | +1.01 | +0.96 |  | 1.25 | +0.96 |  |
| `uni-v2/tempo-lm-uni-v2` | tempo / Fast | uni | **PASS** | - | 1.00 | 1.00 |  | +0.56 | +0.67 |  | 0.80 | +0.67 | PASS |
| `uni-v2/tender-lm-uni-v2` | tender / Tender | uni | FAIL | U2 | 1.00 | 0.62 |  | +0.58 | +0.85 |  | 0.61 | +0.52 |  |
| `uni-v2/triphop-lm-uni-v2` | triphop / Dusty | uni | **PASS** | - | 0.92 | 0.92 |  | +0.52 | +0.82 |  | 0.58 | +0.82 |  |
| `uni-v2/yearn-lm-uni-v2` | yearn / Yearn | uni | **PASS** | - | 1.00 | 1.00 |  | +0.09 | +0.37 |  | 0.49 | +0.37 |  |

### uni-v3

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `uni-v3/distortion-lm-uni-prefix` | distortion / Distorted | uni | **PASS** | - | 0.62 | 0.62 |  | +0.46 | +0.20 |  | 0.76 | +0.20 |  |
| `uni-v3/energy-lm-uni-prefix` | energy / Loud | uni | **PASS** | - | 1.00 | 1.00 |  | +0.54 | +0.62 |  | 0.53 | +0.62 |  |
| `uni-v3/gender-lm-uni-prefix` | gender / Female | uni | FAIL | U4 | 1.00 | 1.00 |  | +0.69 | +0.91 |  | 0.95 | +0.91 |  |
| `uni-v3/grit-lm-uni-prefix` | grit / Grit | uni | FAIL | U4 | 0.62 | 0.62 |  | +0.57 | +0.50 |  | 1.18 | +0.50 |  |
| `uni-v3/hurt-lm-uni-prefix` | hurt / Hurt | uni | **PASS** | - | 1.00 | 1.00 |  | +0.17 | +1.67 |  | 0.39 | +1.67 |  |
| `uni-v3/joy-lm-uni-prefix` | joy / Joy | uni | FAIL | U4 | 1.00 | 0.92 |  | +0.79 | -0.13 |  | 1.37 | -0.12 |  |
| `uni-v3/rapslow-lm-uni-prefix` | rapslow / Rap | uni | **PASS** | - | 0.75 | 1.00 |  | +0.43 | +0.36 |  | 0.59 | +0.36 |  |
| `uni-v3/tempo-lm-uni-prefix` | tempo / Fast | uni | FAIL | U4 | 1.00 | 1.00 |  | +0.66 | +0.62 |  | 1.05 | +0.62 |  |

### v16

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v16/energy-lm-v16` | energy / Loud→Quiet | bipolar | **PASS** | - | 1.00 | 1.00 | 1.00 | +0.18 | +0.15 | +0.15 | 0.69 | -0.15 | FAIL |
| `v16/gender-lm-v16` | gender / Female→Male | bipolar | FAIL | U4 | 1.00 | 1.00 | 1.00 | -0.20 | +0.04 | +0.45 | 1.09 | -0.45 | FAIL |
| `v16/gender-lm-v16-v9kl-20260825-1501` | gender / Female→Male | bipolar | FAIL | U2,U4 | 1.00 | 0.38 | 1.00 | +0.02 | -0.00 | -0.06 | 0.97 | -0.00 |  |
| `v16/live-lm-v16` | live / Live→Studio | bipolar | FAIL | U3 | 1.00 | 1.00 | 0.92 | -0.07 | -0.41 | -0.46 | 0.47 | -0.41 |  |

### v17

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v17/gender-lm-v17` | gender / Female→Male | bipolar | FAIL | U4 | 1.00 | 0.92 | 0.92 | -0.16 | +0.16 | -0.27 | 1.34 | +0.15 |  |
| `v17/live-lm-v17` | live / Live→Studio | bipolar | FAIL | U4 | 1.00 | 0.92 | 1.00 | +0.08 | -0.94 | -0.83 | 1.03 | -0.86 |  |

### v18

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v18/energy-lm-v18` | energy / Loud→Quiet | bipolar | **PASS** | - | 1.00 | 0.92 | 1.00 | -0.02 | +0.57 | +0.07 | 0.70 | -0.07 | PASS |

### v19

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v19/breath-lm-v19` | breath / Breathy→Clean | bipolar | FAIL | U2 | 1.00 | 0.62 | 0.62 | +0.25 | +1.38 | +0.53 | 0.87 | -0.33 |  |
| `v19/energy-lm-v19` | energy / Loud→Quiet | bipolar | **PASS** | - | 1.00 | 1.00 | 0.92 | +0.14 | +0.06 | -0.28 | 0.80 | +0.06 |  |
| `v19/gender-lm-v19` | gender / Female→Male | bipolar | **PASS** | - | 1.00 | 0.92 | 1.00 | -0.01 | +0.01 | +0.24 | 0.76 | -0.24 |  |
| `v19/live-lm-v19` | live / Live→Studio | bipolar | FAIL | U4 | 1.00 | 1.00 | 0.92 | -0.18 | +0.57 | -0.40 | 0.93 | +0.37 |  |

### v20

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v20/breath-lm-v20` | breath / Breathy→Clean | bipolar | FAIL | U2,U4 | 1.00 | 1.00 | 0.62 | +0.04 | +1.20 | +0.87 | 1.03 | -0.54 |  |
| `v20/distortion-lm-v20` | distortion / Heavy→Clean | bipolar | FAIL | U4 | 0.92 | 1.00 | 1.00 | -0.18 | +0.19 | -0.25 | 1.43 | +0.19 |  |
| `v20/energy-lm-v20` | energy / Loud→Quiet | bipolar | **PASS** | - | 1.00 | 1.00 | 1.00 | +0.25 | +0.45 | -0.69 | 0.76 | +0.45 |  |
| `v20/gender-lm-v20` | gender / Female→Male | bipolar | FAIL | U4 | 1.00 | 0.85 | 0.77 | +0.42 | +0.09 | +0.13 | 1.73 | -0.10 |  |
| `v20/live-lm-v20` | live / Live→Studio | bipolar | FAIL | U2,U4 | 1.00 | 1.00 | 0.23 | +0.01 | -0.16 | -0.68 | 1.76 | -0.16 |  |
| `v20/rapslow-lm-v20` | rapslow / Rap→Slow | bipolar | FAIL | U3 | 0.75 | 0.70 | 1.00 | -0.16 | -0.25 | -0.19 | 0.88 | -0.24 |  |
| `v20/rhyme-lm-v20` | rhyme / Rhyme→Prose | bipolar | FAIL | U4 | 0.92 | 0.92 | 1.00 | -0.07 | +0.96 | -1.57 | 1.02 | +0.96 |  |
| `v20/tempo-lm-v20` | tempo / Fast→Slow | bipolar | FAIL | U4 | 1.00 | 1.00 | 0.69 | +0.05 | +0.24 | -0.21 | 1.34 | +0.14 |  |
| `v20/triphop-lm-v20` | triphop / Trip-hop→Pop | bipolar | FAIL | U2,U4 | 1.00 | 0.62 | 0.92 | +0.31 | +1.64 | +1.27 | 1.72 | -1.17 |  |

### v21

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v21/breath-lm-v21` | breath / Breathy→Clean | bipolar | FAIL | U4 | 1.00 | 0.92 | 0.92 | +0.18 | +0.97 | +1.39 | 1.00 | -1.29 |  |
| `v21/distortion-lm-v21` | distortion / Heavy→Clean | bipolar | FAIL | U2 | 0.92 | 0.31 | 0.77 | -0.11 | +0.00 | -0.18 | 0.68 | +0.00 |  |
| `v21/energy-lm-v21` | energy / Loud→Quiet | bipolar | FAIL | U4 | 1.00 | 0.92 | 0.92 | +0.21 | +0.17 | -1.14 | 0.95 | +0.15 |  |
| `v21/rhyme-lm-v21` | rhyme / Rhyme→Prose | bipolar | FAIL | U2,U4 | 0.92 | 1.00 | 0.62 | -0.13 | +0.31 | -1.77 | 1.55 | +0.31 |  |
| `v21/tempo-lm-v21` | tempo / Fast→Slow | bipolar | FAIL | U4 | 1.00 | 1.00 | 1.00 | +0.14 | +0.58 | -0.02 | 1.34 | +0.02 |  |
| `v21/triphop-lm-v21` | triphop / Trip-hop→Pop | bipolar | FAIL | U2,U4 | 1.00 | 0.77 | 0.54 | +0.27 | +0.97 | +0.53 | 1.36 | -0.28 |  |

### v22

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v22/breath-lm-v22` | breath / Breathy→Clean | bipolar | **PASS** | - | 1.00 | 0.92 | 0.77 | +0.34 | +1.02 | +1.31 | 0.88 | -1.01 |  |
| `v22/energy-lm-v22` | energy / Loud→Quiet | bipolar | **PASS** | - | 1.00 | 0.85 | 1.00 | +0.11 | +0.37 | -0.21 | 0.64 | +0.21 |  |

### v9-ritual

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v9-ritual/breath-lm-v9` | breath / Breathy→Clean | bipolar | FAIL | U4 | 0.38 | 1.00 | 0.69 | +0.31 | +0.76 | +0.73 | 1.22 | -0.73 |  |
| `v9-ritual/distortion-lm-v9` | distortion / Metal→Acoustic | bipolar | FAIL | U2 | 0.53 | 0.47 | 1.00 | -0.31 | +0.04 | -0.39 | 0.85 | +0.04 |  |
| `v9-ritual/energy-lm-v9` | energy / Loud→Quiet | bipolar | FAIL | U2 | 1.00 | 0.08 | 1.00 | -0.22 | +0.21 | +0.13 | 0.80 | -0.13 |  |
| `v9-ritual/gender-lm-v9` | gender / Female→Male | bipolar | **PASS** | - | 0.45 | 0.91 | 1.00 | -0.01 | +0.33 | -0.49 | 0.66 | +0.33 | PASS |
| `v9-ritual/grit-lm-v9` | grit / Grit→Smooth | bipolar | FAIL | U2,U4 | 0.92 | 1.00 | 0.00 | +0.20 | -0.43 | -1.39 | 1.02 | -0.43 |  |
| `v9-ritual/hurt-lm-v9` | hurt / Hurt→Composed | bipolar | FAIL | U4 | 0.08 | 0.62 | 0.92 | +0.47 | +0.15 | -0.02 | 1.61 | +0.02 |  |
| `v9-ritual/joy-lm-v9` | joy / Joy→Somber | bipolar | **PASS** | - | 0.15 | 0.92 | 0.85 | -0.02 | +0.45 | -0.01 | 0.88 | +0.01 |  |
| `v9-ritual/rapslow-lm-v9` | rapslow / Rap→Slow | bipolar | FAIL | U2,U4 | 1.00 | 0.90 | 0.35 | +0.21 | +0.29 | +0.28 | 1.02 | -0.10 |  |
| `v9-ritual/rhyme-lm-v9` | rhyme / Rhyme→Prose | bipolar | FAIL | U3 | 0.57 | 0.80 | 0.76 | -0.43 | -0.01 | -0.93 | 0.69 | -0.01 |  |
| `v9-ritual/sexy-lm-v9` | sexy / Sexy→Plain | bipolar | FAIL | U2,U4 | 0.92 | 0.23 | 0.54 | +0.35 | -0.33 | -0.32 | 0.96 | -0.08 |  |
| `v9-ritual/tempo-lm-v9` | tempo / Fast→Slow | bipolar | FAIL | U4 | 0.80 | 1.00 | 1.00 | +0.26 | -0.55 | -0.71 | 1.46 | -0.55 |  |
| `v9-ritual/tender-lm-v9` | tender / Tender→Fierce | bipolar | FAIL | U2,U4 | 1.00 | 0.62 | 0.92 | +0.49 | +2.02 | +0.07 | 1.18 | -0.06 |  |
| `v9-ritual/triphop-lm-v9` | triphop / Triphop→Pop | bipolar | FAIL | U2,U4 | 0.85 | 0.92 | 0.62 | +0.07 | -0.47 | +0.33 | 0.96 | -0.47 |  |
| `v9-ritual/yearn-lm-v9` | yearn / Yearning→Content | bipolar | FAIL | U4 | 0.00 | 0.85 | 1.00 | -0.58 | +0.20 | +0.75 | 1.83 | -0.75 |  |

### v3-era

| folder | axis / pole | kind | verdict | gates fired | rec@0 | rec@unit+ | rec@unit− | proj+ | dsp+ | dsp− | same_song ÷ anchor | E | ears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `energy-v3-lm-20s` | energy / Loud→Quiet | bipolar | **PASS** | - | 0.62 | 1.00 | 0.92 | -0.30 | +0.44 | -0.06 | 0.58 | +0.06 | PASS |
| `gender-lm-20s` | gender / female→male | bipolar | *UNSCOREABLE* | U0,U4 | 0.46 | 0.85 | 0.92 | -0.08 | +0.57 | +0.71 | 1.17 | -0.71 | PASS |

## What this board cannot tell you

- Whether any of the 43 PASS folders **moves its concept**. No channel in the
  frozen instrument certifies that; the two that tried both reject ears-approved
  sliders.
- Whether a candidate holds up on a second seed. Everything here is seed 7.
- Whether a candidate ends properly. Every ladder is 20 s, so U5 only warns.
- Whether `v16/energy-lm-v16`-style failures are present among the unlabeled
  PASSes. That folder is an ears-FAIL that reads PASS here ("alternates between
  the two songs", with recall 1.00 and `same_song` 0.61 × anchor). Any of the 43
  could be the same thing. The only cure is ears.

The productive use of this board is as a **listening queue**: the gates have
removed 50 folders that demonstrably break the song or the words, which is worth
roughly that many listens; ordering the remainder by E is a guess.

## Derate sweep (2026-09-02)

**Single-seed provisional. Nothing here promotes or ships anything.** Every
number is seed 7, 20 s, prompt row 0, one render per rung. Promotion still
requires the **≥ 3 seeds + 60–90 s ending check** of `LM-SCORING.md` (where U5
becomes a veto instead of a warn), and PASS remains a no-harm certificate, not
an efficacy certificate.

### The question

Seven shipped `uni-lyric` catalog poles FAIL the frozen gates at +1, and all
seven fail in ways that *look* dose-shaped — three garble the lyric
(`joy` 0.00, `gender-Female` 0.15, `distortion-Distorted` 0.31 recall) and four
keep the lyric but leave the song (U4 `same_song ÷ cross_anchor`: `breath` 1.07,
`rhyme` 1.60, `sexy` 0.99, `sexy-plain` 1.31). Precedent said dose was the
variable: `rapslow-lm-uni-v2` is clean at +1 and dead at +2, and the shipped
distortion **transformer** half ships derated
(`distortion-tf-v10-traj25_derate055.safetensors`, alpha 4.4 = 8.0 × 0.55).

Hypothesis under test: **each failing pole has a sub-unit scale that clears
U1/U2/U4 while still moving the concept, so the ship fix is a sidecar derate,
not a retrain.**

### Method

New ladders in `eval/listen/uni-lyric-derate/<name>/` at scales
`0, 0.25, 0.5, 0.75, 1`. The scale-0 clip, the +1 clip and both REF clips are
**byte-identical copies** of the shipped `eval/listen/uni-lyric/<name>/` ladder,
renamed to the index `generate_listen.py` emits for this scale list; only the
0.25 / 0.5 / 0.75 rungs are new renders (21 renders total, ~13 s each). The
resume path confirmed the copies were skipped, and the scored +1 rung of every
derate folder reproduces the shipped folder's +1 numbers exactly. The null pool
globs (`uni-v2/*`, `uni-lyric/*`) do not match `uni-lyric-derate/*`, so
`cross_anchor` is unchanged from the shipped scoring.

Scored with the frozen instrument into `eval/lm_score_derate_board.tsv`. `s*` is
the **largest** rung in `{0.25, 0.5, 0.75, 1}` that clears U1/U2/U4 *and* has
`dsp_proj > 0` (the concept still points at the + REF). U5 warns on all 28 rungs
because every clip is 20 s; that warn is not a defect signal. U1 never fired
anywhere — the widest swing in the sweep is `|d_ln_rms| = 0.94` against a 1.15
limit.

### Per-pole ladders

`same/anch` = `same_song ÷ cross_anchor`; U4 needs **≤ 0.90**. `recall` needs
**≥ 0.40** and **≥ max(base, REF+) − 0.35**; the resulting floor is in each
caption. `dsp_proj > 0` is the direction requirement, marked `dsp<0` when it
fails.

**`breath-lm-uni-lyric`** — Breathy. anchor 0.0851, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 1.000 | 1.109 | +0.597 | **U4** |
| 0.50 | 1.000 | 0.764 | +0.308 | pass |
| 0.75 | 1.000 | 0.784 | +0.563 | pass |
| 1.00 | 1.000 | 1.075 | +0.265 | **U4** (shipped) |

**`rhyme-lm-uni-lyric`** — Rhyme. anchor 0.0851, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 0.923 | 1.035 | −1.890 | **U4**, dsp<0 |
| 0.50 | 0.615 | 0.988 | +1.480 | **U2, U4** |
| 0.75 | 1.000 | 0.811 | −1.871 | gates pass, **dsp<0** |
| 1.00 | 0.769 | 1.597 | −0.716 | **U4**, dsp<0 (shipped) |

**`sexy-lm-uni-lyric`** — Sexy. anchor 0.0831, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 1.000 | 0.906 | +0.942 | **U4** |
| 0.50 | 0.923 | 1.512 | +0.875 | **U4** |
| 0.75 | 1.000 | 0.809 | +0.727 | pass |
| 1.00 | 1.000 | 0.989 | +0.404 | **U4** (shipped) |

**`sexy-plain-lm-uni-lyric`** — Plain. anchor 0.0831, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 1.000 | 1.391 | +0.604 | **U4** |
| 0.50 | 1.000 | 1.114 | +0.844 | **U4** |
| 0.75 | 1.000 | 1.496 | +0.980 | **U4** |
| 1.00 | 1.000 | 1.307 | +0.785 | **U4** (shipped) |

**`joy-lm-uni-lyric`** — Joy. anchor 0.0777, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 1.000 | 1.743 | +1.050 | **U4** |
| 0.50 | 1.000 | 1.509 | +0.932 | **U4** |
| 0.75 | 0.722 | 2.523 | +0.951 | **U4** |
| 1.00 | 0.000 | 2.057 | +1.088 | **U2, U4** (shipped) |

**`gender-lm-uni-lyric`** — Female. anchor 0.0877, recall floor 0.650.

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 0.923 | 0.797 | −0.443 | gates pass, **dsp<0** |
| 0.50 | 1.000 | 0.633 | +0.734 | pass |
| 0.75 | 1.000 | 0.738 | +0.181 | pass |
| 1.00 | 0.154 | 1.206 | +0.419 | **U2, U4** (shipped) |

**`distortion-lm-uni-lyric`** — Distorted. anchor 0.0821, recall floor 0.400
(this ladder's own scale-0 base transcribes at 0.615, so the drop allowance
bottoms out on the 0.40 absolute floor).

| s | recall | same/anch | dsp_proj | gates |
|---:|---:|---:|---:|---|
| 0.25 | 0.615 | 0.442 | +0.156 | pass |
| 0.50 | 1.000 | 1.363 | +0.125 | **U4** |
| 0.75 | 0.000 | 0.975 | −0.426 | **U2, U4**, dsp<0 |
| 1.00 | 0.308 | 0.651 | +0.235 | **U2** (shipped) |

### s\* per pole and the derate recommendation

Following the `derate055` precedent — a **separate** sidecar checkpoint whose
`alpha` is the trained alpha times the derate factor, `unit_scale` left at 1.0 —
the recommendation for each rescuable pole is a new
`models/<name>/<name>_derate<NN>.safetensors` at `alpha = 8.0 × s*`. All seven
checkpoints are rank 8 / alpha 8.0.

**Recommendation only. No sidecar, no `app/sliders.json` entry, and no
checkpoint was written by this sweep.**

| pole | s\* | recall @ s\* | same/anch @ s\* | dsp_proj @ s\* | recommended sidecar | new alpha |
|---|---:|---:|---:|---:|---|---:|
| `breath-lm-uni-lyric` | **0.75** | 1.000 | 0.784 | +0.563 | `breath-lm-uni-lyric_derate075` | 6.0 |
| `sexy-lm-uni-lyric` | **0.75** | 1.000 | 0.809 | +0.727 | `sexy-lm-uni-lyric_derate075` | 6.0 |
| `gender-lm-uni-lyric` | **0.75** | 1.000 | 0.738 | +0.181 | `gender-lm-uni-lyric_derate075` | 6.0 |
| `distortion-lm-uni-lyric` | **0.25** | 0.615 | 0.442 | +0.156 | `distortion-lm-uni-lyric_derate025` | 2.0 |
| `rhyme-lm-uni-lyric` | — | — | — | — | UNRESCUABLE-BY-DERATE | — |
| `sexy-plain-lm-uni-lyric` | — | — | — | — | UNRESCUABLE-BY-DERATE | — |
| `joy-lm-uni-lyric` | — | — | — | — | UNRESCUABLE-BY-DERATE | — |

**UNRESCUABLE-BY-DERATE (retrain candidates): `rhyme-lm-uni-lyric`,
`sexy-plain-lm-uni-lyric`, `joy-lm-uni-lyric`.** Two different reasons:

- `sexy-plain` and `joy` fail **U4 at every dose down to 0.25** — 1.11–1.50 and
  1.51–2.52 × anchor respectively. There is no sub-unit setting at which these
  renders are still the same song; the failure is not a dose.
- `rhyme` has a rung whose *gates* pass (0.75: recall 1.000, 0.811 × anchor) but
  whose `dsp_proj` is **−1.871** — the strongest negative in the sweep. At 0.75
  the render holds the song and the words while pointing away from the Rhyme
  REF, which is a working no-harm clip of a slider doing the opposite of its
  label. Marked unrescuable under the `dsp_proj > 0` requirement rather than
  shipped as a derate. (`rhyme`'s DSP sign is negative at three of four doses
  and +1.480 at 0.50 — see the monotonicity finding below before trusting either
  sign.)

`distortion`'s `s* = 0.25` deserves a flag of its own: it clears the gates only
because this ladder's *base* transcribes at 0.615, which drops the U2 bar to the
0.40 absolute floor. Its `recall` at 0.25 (0.615) merely ties the base, and
`dsp_proj` is +0.156 — the smallest positive in the table. A `derate025` here is
the weakest recommendation on the page.

### Finding: the dose hypothesis is falsified for `same_song`

The sweep was designed on the premise that both failure channels are monotone in
dose. **They are not.** Spearman ρ of each channel against dose over
`{0.25, 0.5, 0.75, 1}` (`const` = the channel does not vary across the ladder):

| pole | ρ(recall) | ρ(same/anch) | ρ(dsp_proj) |
|---|---:|---:|---:|
| `breath` | const (1.000 at every dose) | −0.20 | −0.80 |
| `rhyme` | +0.00 | +0.20 | +0.40 |
| `sexy` | +0.26 | +0.00 | −1.00 |
| `sexy-plain` | const (1.000 at every dose) | +0.00 | +0.40 |
| `joy` | −0.95 | +0.60 | +0.40 |
| `gender` | −0.32 | +0.40 | +0.40 |
| `distortion` | −0.60 | +0.20 | +0.20 |

Read across:

- **`same_song` is not dose-shaped on any pole.** No |ρ| reaches 0.60; the median
  is 0.20. Concretely, `breath` fails U4 at 0.25 (1.109) and 1.00 (1.075) while
  passing comfortably at 0.50 (0.764) and 0.75 (0.784), and `sexy` fails at 0.25
  (0.906) and 0.50 (1.512) while passing at 0.75 (0.809). A U-shaped or
  sawtooth `same_song` curve is not a dose response — on a 20 s single-seed AR
  render it is the render lottery (whether the model repeats the chorus, drops a
  section, or ends early), which is exactly the magnitude `LM-SCORING.md`
  refuses to treat as a unit.
- **`dsp_proj` is worse.** It is non-monotone on all seven poles and changes
  *sign* across the ladder on `rhyme` (−1.89, +1.48, −1.87, −0.72) and `gender`
  (−0.44, +0.73, +0.18, +0.42). Since `dsp_proj > 0` is the requirement that
  gates `s*`, three of the four `s*` values in this sweep rest on a channel that
  flips sign between adjacent 0.25-wide rungs of the same ladder.
- **Lyric recall is the one channel that behaves.** It is dose-shaped where it
  is the failure mode: `joy` ρ = −0.95 (1.000, 1.000, 0.722, 0.000) and `gender`
  goes 0.923 / 1.000 / 1.000 / **0.154**, a cliff between 0.75 and 1.00 rather
  than a slope. The transcripts make the mechanism plain — at +1 `joy` sings
  `"Yeah. Yeah. Yeah. …"` and `gender` produces `"I can"` and stops, while both
  sing the full sheet at 0.75. `distortion` breaks the pattern the other way
  (ρ = −0.60 but recall 0.000 at 0.75 against 0.308 at 1.00: `"Oh, oh, oh,
  oh, …"` at three-quarter dose, partial words at full dose).

**Contradiction against the shipped board.** `distortion-lm-uni-lyric` fires
`U2` only in `eval/listen/uni-lyric/`; in the derate ladder it fires `U2, U4`,
because the *added* 0.5 rung fails U4 at 1.363 while the +1 rung it shares with
the shipped folder is 0.651. Same checkpoint, same seed, same anchor — the
folder verdict moved because a sub-unit rung was rendered. `rhyme` behaves the
same way (`U4` shipped, `U2, U4` here, from the 0.5 rung at recall 0.615). A
ladder's gate set is therefore sensitive to which rungs it contains, and the
shipped `gates_fired` column is a statement about the shipped scale list only.

**Confound worth naming before any retrain.** The two poles that fail U4 at
every dose have the two shortest scale-0 anchors in the set: `joy` 12.61 s and
`sexy-plain` 13.72 s against 20 s slider rungs. `same_song` is a cosine distance
to that anchor, and a truncated anchor inflates it for every rung equally —
which is precisely the flat, dose-independent U4 failure both poles show.
`joy`'s anchor also transcribes at only 0.611. `sexy` shares the 13.72 s anchor
and still finds a passing rung, so the anchor length is not sufficient on its
own, but no U4 verdict on `joy` or `sexy-plain` should be trusted until a
same-caption re-rolled zero render exists — the resolution `LM-SCORING.md`
already schedules for the ≥3-seed promotion stage.

### What this sweep does and does not establish

- It does **not** show that any of `breath` / `sexy` / `gender` / `distortion`
  works at `s*`. `s*` is a no-harm certificate at one seed, on a channel set that
  the calibration pass measured to have no efficacy power.
- It does **not** establish `s* = 0.75` as a boundary. The passing band is
  interior on `breath` (0.50–0.75) and `gender` (0.50–0.75) and a lone island on
  `sexy` (0.75); with `same_song` non-monotone the next seed can move it.
- The honest reading of `breath` / `sexy` / `gender` is that **a sub-unit derate
  is worth a ≥3-seed confirm**, not that a derate fixes them. `distortion` at
  0.25 and all three UNRESCUABLE poles should go to the retrain queue instead.

Artifacts: `eval/listen/uni-lyric-derate/*/` (49 wavs, 21 new), per-folder
`lm_scores.json`, board `eval/lm_score_derate_board.tsv`.

## Partial multi-seed confirm (2026-09-03, stopped early)

The ≥3-seed confirm matrix was cut short after 6 of ~38 folders (user call:
no more promotion renders). What did land — seeds 11/13 minimal ladders
(`0,<u>` + REFs, 20 s) under `eval/listen/lm-confirm/`, scored with the frozen
gates (`eval/lm_confirm_board.tsv`), aggregated with the seed-7 rung from the
derate sweep:

| candidate @ u | s7 | s11 | s13 | status |
|---|---|---|---|---|
| `sexy-lm-uni-lyric` @0.75 | pass (0.81×) | pass (0.48×) | pass (0.73×) | **3/3 no-harm pass**, dsp_proj + on all seeds (+0.73/+0.26/+0.56). First passer ever on the sexy axis. Still owed: 60 s ending check, ears |
| `gender-lm-uni-lyric` @0.75 | pass (0.74×) | pass (0.39×) | *(folder incomplete)* | 2/2 scored seeds pass, recall 1.00 both (shipped +1 garbles at 0.15) |
| `breath-lm-uni-lyric` @0.75 | pass (0.78×) | **FAIL U4 (0.96×)** | **FAIL U4 (2.23×)** | **REFUTED** — the derate sweep's s* was seed-7 luck, exactly the sawtooth it warned about |

(parenthesized: same_song ÷ cross_anchor at u.)

One-in-three derate picks evaporating under re-roll is the multi-seed rule
earning its keep; no single-seed s* should move a sidecar without this stage.
Remaining owed for any promotion: seeds 11/13 for the other 16 candidates,
s13 for gender, and 60 s ending checks (U5 veto) — all rendering, all paused.
