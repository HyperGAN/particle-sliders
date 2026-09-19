# LM slider scoring spec (2026-09-02)

The contract for **LM halves**. `SCORING.md` is the transformer contract and
stays frozen; this document is the thing its Amendment 6 promised and did not
build. Instrument: `scripts/lm_score.py` (`selftest` runs the attack suite on
CPU in a second; `tests/test_lm_score.py` runs the same cases under pytest).

Thresholds below are **FROZEN 2026-09-02** by the single calibration pass
recorded in "Validation against locked verdicts" and "What the calibration pass
decided". They were placed once, on the 11 PASS/FAIL folders of
`eval/lm_score_labels.json`; the 3 BORDERLINE folders were reported and never
used to place a threshold. They are not tuned on candidates again.

**Read this before reading a PASS.** The calibration pass killed the two
channels that were supposed to certify that the concept *moved* — U6 (the null)
and the rank half of U3 — because both reject ears-approved sliders. What
survived is a **no-harm certificate**: a PASS means the render still sings the
line, is still the same song, is not silent, and does not point backwards on
either the embedding or the DSP axis. It does **not** mean the slider works. The
efficacy question is still open and `docs/lm-render-board.md` says so on every
row.

## Why LM halves need their own contract

SCORING.md, Amendment 6, measured on a control run:

> shipped, ears-approved `energy-lm-v4`, rendered through the identical paired
> 12 s ladder bed, fails G3/G4/G6 exactly like seven deliberately varied dust LM
> candidates (onset corr 0.05–0.13 at every scale). The mechanism: LM halves
> change the arrangement by design, so same-seed onset/envelope identity against
> the scale-0 clip is ~0 for any working LM slider, and the G6 null is built from
> the same seed-noise structure. **Do not gate or rank LM halves with this
> instrument** — it measures the host, not the slider.

The train-time numbers are not a substitute. `docs/lm-uni-v2-garble.md`: over
eight finished plus axes, last-50 `c+`, `p%` and `‖h+−h0‖` all correlate with
the ears verdict at ρ ≈ +0.52, and `grit-lm-uni-v2` prints `c+` 0.945 — a clean
hit — while its +1 render sings no lyrics at all. Whisper lyric recall on the
rendered clip matched ears on all eight. This instrument scores the **render
path**, and its channels are the ones that page prescribes ("What to measure
instead of ears").

## What "works" means, as a product

At user setting **+1**, on the neutral caption with the yaml lyrics:

1. the render still **sings the yaml line** (the caption is not the variable —
   the LoRA is);
2. it audibly **leans toward the + REF** on the concept;
3. it is still **one coherent song**, not a different recording;
4. at long durations it **ends on its own** instead of being cut off hot.

Slider clips encode the neutral caption + yaml lyrics, LoRA scale only; REF
clips change the prompt with the slider off. If +1 garbles and the + REF still
sings the line, the caption is singable and the LoRA broke AR
(`docs/lm-uni-v2-garble.md`, failure mode 2).

## Measurement

Per wav, cached by content hash (`--cache_dir`), so thresholds are free to
re-evaluate and a byte-identical clip shared by two folders is measured once:

- **rms**, the repo's one convention: `sqrt(mean(all samples^2))` over the
  de-interleaved array (SCORING.md Amendment 2 — mono downmix mixes level with
  stereo width, up to 40 % on real renders).
- **duration**, natural-end flag (`duration < requested − 0.5 s`), tail ratio
  `rms(last 1.5 s) / rms(overall)`.
- **transcript**: `openai/whisper-large-v3-turbo`, greedy, English. Lyric word
  recall against the folder's own `LISTEN.md` sheet, section tags stripped —
  the same normalization as `scripts/blindspot_whisper.py`.
- **embedding**: whisper encoder last hidden state, mean-pooled over the real
  (unpadded) frames.
- **DSP**: 6 log band energies (`scripts/ref_fidelity.py` edges), ln rms,
  magnitude-spectrum centroid. Drafted as secondary and never gated; the
  2026-09-02 calibration pass promoted its **sign** into U3 (as one half of an
  `emb OR dsp` disjunction) and made it the direction term of the uncertified
  ranking scalar. Its **magnitude** is still never a gate.

Derived per scale `s`, against the folder's **own same-seed scale-0 clip**:

    v        = emb(REF+) − emb(REF−)      (bipolar)
             = emb(REF+) − emb(zero)      (uni)
    proj[s]  = (emb(s) − emb(zero)) · v̂ / ‖v‖
    offaxis  = ‖(emb(s) − emb(zero)) − (proj·‖v‖)·v̂‖ / ‖v‖
    same_song[s] = cosine distance(emb(s), emb(zero))

**`‖v‖` is not a unit of anything.** It is a single-seed caption-swap
magnitude, and SCORING.md ruling 2 measured that magnitude to be unmeasurable:
over 10 seeds the dust caption swap's centroid runs −50 %, −25 %, −11 %, −9 %,
−5 %, +31 %, +42 %, +48 %, +55 %, +92 %. Only its **direction** is certifiable.
So every gate below uses **sign only** — never a raw `proj` magnitude and never
a "fraction of the caption swap" threshold. (Rank and null-calibrated units were
the draft's other two answers; calibration demoted both. The ranking scalar E
does use a caption-swap fraction, which is exactly why E is marked uncertified
and can never become a gate.)

**Null calibration.** Pool = every `01_slider_neutral_base_zero.wav` and
`05_REF_prompt_Off_no_slider.wav` under `eval/listen/uni-v2` and
`eval/listen/uni-lyric`, deduped by content hash and minus the scored folder's
own clips (13 distinct clips today: same seed, same lyrics, different neutral
captions). From it:

- `cross_anchor` = median cosine distance between distinct pool clips — the
  "different song, same-ish caption" scale (U4's yardstick).
- `null95` = 95th percentile of `proj` computed with each pool clip substituted
  in place of the +1 clip. Same construction as SCORING.md G6: pseudo-candidates
  built from renders the slider never touched. `null95_paired` (both ends
  re-rolled, `p_i − p_j`) and `null95_dsp` (the same substitution null taken in
  DSP space) are logged. **None of the three is a gate** — see U6 below.

The pool is the **mixed** pool (`--pool_same_lyrics` OFF). Calibration measured
both: restricting the pool to clips singing the scored folder's own sheet keeps
a U4 gap (ears-PASS top out at 0.870 of the anchor, the replacement FAILs sit at
1.127 and 1.225) but **empties** the pool for every folder whose sheet the pool
does not contain — `rapslow-lm-uni-v2`, the ears-PASS `v9-ritual/gender-lm-v9`,
and the whole v9-ritual wave. U4 and the null would then silently cease to exist
on exactly the ladders that most need checking. The flag stays available for a
future pool that covers every sheet.

## The rung a gate is read at

The product claim is stated at **user setting ±1**, so each gate reads the rung
closest to 1 from below on each pole. A ladder with no rung in `0 < |s| ≤ 1` was
never rendered at the product setting and the contract has no data on it:
**U0_shape** fires and the verdict is `UNSCOREABLE`, not FAIL. One folder in the
corpus is in that state (`eval/listen/gender-lm-20s`, rendered ±2 only).

## Gates — vetoes, plus two demoted diagnostics

| gate | rule (FROZEN 2026-09-02) | veto? | designed to catch |
|---|---|---|---|
| **U0 shape** | a rung exists at `0 < \|s\| ≤ 1` on every pole the ladder claims | yes → `UNSCOREABLE` | a ladder the contract cannot reach. Not a slider failure |
| **U1 level** | `rms(s) ≥ 0.02 × rms(zero)` at every scale; `\|Δln rms\| ≤ 1.15` (≈10 dB) at the unit rung | yes | SCORING.md's `D-pole-uni` silence (rms 0.0019 vs 0.104 scores +6.86 on any effect-size scalar); gain is the cheapest rank-8 delta and it compounds |
| **U2 lyric hold** | at `\|s\| ≤ 1`: `recall ≥ 0.40` **and** `recall ≥ max(recall_base, recall_REF+) − 0.35` | yes | the garble that train `c+` cannot see. Verified: fires on 3/3 ears-FAIL garbles, 0/6 ears-PASS |
| **U3 direction** | **plus pole only**, at the unit rung: `proj > 0` **OR** `dsp_proj > 0` | yes | a flagrantly reversed slider. **No measured power**: 0 of 5 ears-FAIL folders fire it |
| **U4 same-song** | `same_song[s] ≤ 0.90 × cross_anchor` at `\|s\| ≤ 1` | yes | song replacement. SCORING.md G4's `D-pop-uni` discarded the song and rendered a new one while reading healthy on every spectral feature. The LM analogue is the v4 "two-song yaml" |
| **U5 ending** | at `\|s\| ≤ 1`: natural end **or** `tail_ratio ≤ 0.5`. **WARN, not veto, below 45 s** | yes ≥45 s | the endreg failure. Every listen ladder here is 20 s, so U5 only ever warns in this corpus. The real gate runs on 60–90 s renders |
| **U6 null** | `proj ≥ null95` at the unit rung | **NO — diagnostic** | nothing, as measured. It rejects **all 6** ears-PASS folders. Status `not_applicable` (uni) / `no_power` (bipolar); logged with `would_pass` |
| *ρ (Spearman)* | *was: rank monotonicity ≥ 0.8* | **NO — diagnostic** | it anti-correlates with ears on this corpus. Logged as `spearman` |
| *`ref_axis_dist`* | *proposed U0 "close REF pair"* | **NO — refuted** | ears-PASS `energy-v3-lm-20s` has the smallest pair in the corpus (0.030) and ears-FAIL `energy-lm-v16` the largest (0.144) |

Verdict: **PASS** if every veto passes (a U5 warn is allowed); **UNSCOREABLE**
if U0 fires; otherwise **FAIL** with the list of fired gates. The full gate
vector — including the demoted channels — is written to `lm_scores.json` in the
folder for every candidate, passing or not, so a Goodhart walk is visible at the
boundaries before the scalar lies.

## Ranking scalar, among gate-passers — **UNCERTIFIED**

The drafted spec ranked by `proj` in null units. Calibration destroyed both
halves of that unit: the numerator (`proj`, whisper-embedding) contradicts ears
at the unit rung on 4 of 6 ears-PASS folders, and the denominator (`null95`) is
the statistic U6 was demoted for. So E is rebuilt on the channel whose **sign**
survived, and is demoted with it:

    E_side       = (±dsp_proj at the unit rung) × lyric_factor_side
    lyric_factor = min(1, recall(±1) / max(recall_base, 0.40))
    E            = min over sides        # a dead pole cannot be averaged away
    reported as E, the squashed E/(E+2), and the old E_emb for comparison

`dsp_proj` is a **fraction of the same-seed caption swap** — exactly the kind of
raw magnitude this document forbids as a *gate*, for the reason in ruling 2, and
that prohibition still stands. E is therefore:

- never a gate, never a veto, never a certification;
- not comparable across seeds (ruling 2's ±50 % denominator wobble is not
  cancelled by taking a ratio);
- partly a loudness match on energy-like axes, because `ln rms` is one of the 8
  DSP components and the Loud/Quiet caption swap moves it. **Stated Goodhart
  risk:** a slider that only turns up the gain scores well on `energy`;
- unvalidated in rank order — the labeled corpus has 6 PASS folders and no ear
  ranking *among* them, so nothing here says E orders good sliders correctly.

It exists to put gate-passers in a deterministic order on the board. The lyric
factor is a **shaping term, not the gate** — U2 is the gate.

*Deviation from the drafted spec:* the draft defines `lyric_factor` from
`recall(+1)` for both sides. Here each side is graded by its own clip, so a
garbled minus pole cannot hide behind a clean plus pole. On a uni ladder the two
definitions are identical.

## Policy, inherited from SCORING.md

- **Paired, same-seed design.** Every clip in a ladder is the same seed; every
  comparison is against that seed's own scale-0 clip. Unpaired ranking is
  unfalsifiable here (between-seed sd of the +1 rms delta ≈ 1.0 ln vs a ≤ 0.25
  ln recipe effect).
- **The scorer never sees optimizer-authored captions.** Captions swung the
  level channel 68 points on an identical recipe; the measurement spec is frozen
  before caption search starts, and caption diffs need human sign-off.
- **Single seed = provisional.** These listen ladders are one seed;
  `lm_scores.json` records `"seeds": 1` and every verdict from them is
  provisional. Promotion requires **≥ 3 seeds** plus a **60–90 s ending check**
  (where U5 is a veto).
- **Thresholds frozen after one calibration pass** against the labeled corpus
  below, then never tuned on candidates. Done, 2026-09-02.
- **Gates are vetoes, never terms.** The ranking scalar exists only to order
  gate-passers, and after calibration it is explicitly uncertified.
- **A channel that does not separate PASS from FAIL is logged, not gated.**
  Applied here at cost: U6, Spearman ρ and `ref_axis_dist` were all demoted by
  this rule, and U3 survives only as a reversal floor with 0/5 measured power.
  The resulting blind spots are asserted as tests, not written out of the spec.

## Validation against locked verdicts

Labeled by ears (`eval/lm_score_labels.json`; sources: `docs/lm-uni-v2-garble.md`
uni-v2 whisper-tiny recall table, `docs/lm-pair-exam.md` /
`docs/lm-2d-scoreboard.md` v16/v18 live listens, `eval/listen/README.md` v3
table, MUSIC3.md catalog flips). The scorer column is the **frozen** scorer,
2026-09-02. BORDERLINE rows are reported; no threshold was placed using them.

| folder | ears | scorer | gates fired | agree? |
|---|---|---|---|---|
| `uni-v2/grit-lm-uni-v2` | FAIL | **FAIL** | U2, U4 | ✔ |
| `uni-v2/distortion-lm-uni-v2` | FAIL | **FAIL** | U2 | ✔ |
| `uni-v2/joy-lm-uni-v2` | FAIL | **FAIL** | U2 | ✔ |
| `uni-v2/gender-lm-uni-v2` | PASS | **PASS** | — | ✔ |
| `uni-v2/tempo-lm-uni-v2` | PASS | **PASS** | — | ✔ |
| `v18/energy-lm-v18` | PASS | **PASS** | — | ✔ |
| `v16/gender-lm-v16` | FAIL | **FAIL** | U4 | ✔ |
| `v9-ritual/gender-lm-v9` | PASS | **PASS** | — | ✔ |
| `energy-v3-lm-20s` | PASS | **PASS** | — | ✔ |
| `v16/energy-lm-v16` | FAIL | **PASS** | — | ✘ false accept |
| `gender-lm-20s` | PASS | **UNSCOREABLE** | U0 (+U4 at ±2) | ✘ no data at ±1 |
| `uni-v2/hurt-lm-uni-v2` | *BORDERLINE* | FAIL | U2 | (not calibrated on) |
| `uni-v2/energy-lm-uni-v2` | *BORDERLINE* | PASS | — | (not calibrated on) |
| `uni-v2/rapslow-lm-uni-v2` | *BORDERLINE* | PASS | — | (not calibrated on) |

**Agreement: 9 of 11 PASS/FAIL folders.** Two do not agree, and neither was
tuned away:

- **`v16/energy-lm-v16` — a false accept, and the instrument's worst hole.**
  Ears: "random words; alternates between the two songs; no audible swing." The
  frozen scorer sees lyric recall **1.00 at every scale** (whisper-large-v3-turbo
  disagrees with the "random words" note, as it also does on `gender-lm-v16`),
  `same_song[+1] = 0.61 × cross_anchor` (well inside U4), and a correctly signed
  plus pole. Nothing in the instrument sees "alternates between two songs" when
  both songs sing the same words at the same level. The one channel that did
  reject it — U6, `proj[+1] = 0.175 < null95 = 0.244` — rejects ears-PASS
  `energy-lm-v18` on the identical REF pair and null (`proj[+1] = −0.016`). U6
  cannot be kept for this one catch at the price of every PASS.
- **`gender-lm-20s` — UNSCOREABLE, not FAIL.** The folder was rendered at ±2
  only, so the contract's `|s| = 1` claim has no clip to read. Stated in full:
  had the gates been read at its nearest rung (±2) it would have **failed U4** —
  `same_song` is 0.92 × anchor at +2 and **1.17 × anchor** at −2. That is not
  evidence the slider is bad; ±2 is a rung U4's threshold was never placed at.
  The fix is a re-render at ±1, not a threshold.

A `D-pop-uni`-style song replacement (SCORING.md G4) has no LM listen folder; it
is covered by the synthetic `song_replace` case in `lm_score.py selftest`.

## What the calibration pass decided (2026-09-02)

The smoke run raised three questions. All three are now answered against the
full labeled set, and two of the answers cost the instrument power.

### 1. U6 is a diagnostic, on **both** ladder kinds

The smoke run found U6 structurally broken on uni ladders and proposed keeping
it as a bipolar veto. The full labeled set refutes the second half.

*Uni, as measured before:* with `v = emb(REF+) − emb(zero)`, both the candidate
statistic and the substitution null are anchored at the zero clip, so both carry
`‖emb(zero) − pool mean‖²`. The **median** foreign clip projects **+0.84**
(grit) and **+0.46** (gender) along `v`, against candidate `proj[+1]` of 0.59
and 0.57. A uni slider would have to out-run the + REF itself. Status:
`not_applicable`. The paired null (`p_i − p_j`, both ends re-rolled) is **not**
substituted in — it is unmatched to the candidate statistic — it stays logged.
This resolves when same-caption re-rolled zero renders exist, i.e. at the ≥3-seed
promotion stage.

*Bipolar:* the null **is** centred (median +0.09 v18, −0.11 v16), so the
construction is sound — but the statistic it thresholds is the embedding `proj`,
and that channel is the one finding 2 discredits. Measured at the unit rung:

| folder | ears | `proj` | `null95` | U6 |
|---|---|---|---|---|
| `v18/energy-lm-v18` | PASS | −0.016 | 0.244 | reject |
| `v9-ritual/gender-lm-v9` | PASS | −0.009 | 0.577 | reject |
| `energy-v3-lm-20s` | PASS | −0.296 | 0.121 | reject |
| `gender-lm-20s` | PASS | −0.080 | 0.185 | reject |

Four of four ears-PASS bipolar ladders, six of six overall. Taken in **DSP**
space instead (`null95_dsp`, also logged) it still rejects 4 of 6. U6 is
therefore computed, logged with `would_pass`, and **never a veto**. Its status
field says which failure mode applies.

The cost is stated rather than hidden: **the frozen instrument cannot veto a
no-op slider.** The `noop` case in `selftest` and
`tests/test_lm_score.py::test_noop_is_a_known_blind_spot_after_the_2026_09_02_freeze`
now assert that a no-op *passes the gates* and is separated only by the ranking
scalar. That is a regression test on the blind spot, not an endorsement of it.

### 2. U3 gates the plus pole, on an `emb OR dsp` sign, with no rank term

The smoke run saw the embedding sign contradict ears on `energy-lm-v18`. Over
the full labeled set it contradicts ears on **4 of 6** ears-PASS folders, while
the DSP sign agrees with **6 of 6**:

| folder | ears | `proj` at unit + | `dsp_proj` at unit + |
|---|---|---|---|
| `uni-v2/gender-lm-uni-v2` | PASS | **+0.570** | **+1.197** |
| `uni-v2/tempo-lm-uni-v2` | PASS | **+0.556** | **+0.675** |
| `v18/energy-lm-v18` | PASS | −0.016 | **+0.566** |
| `v9-ritual/gender-lm-v9` | PASS | −0.009 | **+0.326** |
| `energy-v3-lm-20s` | PASS | −0.296 | **+0.440** |
| `gender-lm-20s` | PASS | −0.080 | **+0.574** |

Neither channel is trusted alone. DSP is blind where the concept is not
spectral — on `gender-lm-20s` the DSP minus pole reads **+0.715**, i.e. toward
*Female* at −2 — and the whisper embedding is a speech-content space (Amendment
6's finding, restated). So the frozen rule is the disjunction, and only on the
pole the product contract actually claims:

    U3 passes iff  proj(unit +) > 0  OR  dsp_proj(unit +) > 0

Three sub-decisions, each with its cost:

- **Plus pole only.** Section "What 'works' means" states the claim at `+1`.
  Gating the minus pole with the same disjunction would reject ears-PASS
  `gender-lm-20s` (emb +0.156 **and** DSP +0.715 at −2, both toward Female). The
  minus sign is logged as `minus_sign_ok`. Cost: a dead or reversed minus pole is
  not vetoed; it is priced by `min()` over sides in E.
- **Rank monotonicity removed.** Over the labeled set Spearman ρ of `proj`
  *anti-correlates* with ears — ears-PASS ρ = +1.0, +0.5, +0.90, −0.40, −0.30,
  −1.0 against ears-FAIL ρ = +1.0, +1.0, +1.0, +1.0, −0.70. DSP ρ is no better
  (PASS 0.5, 1.0, 0.8, −0.1, 0.8, −0.5). Both are logged, neither gates. Cost:
  the `nonmonotone` attack is no longer vetoed, and the test now says so.
- **Honest power statement.** With this rule U3 fires on **0 of 5** ears-FAIL
  folders. It has no measured discriminative power; it is a floor against a
  flagrantly reversed slider (both channels wrong-signed), which is what the
  `reversed` synthetic is. Across the 94-folder discovery sweep it fires on 3
  folders — `live-lm-v16`, `rapslow-lm-v20`, `rhyme-lm-v9` — whose validity is
  unknown. That is unvalidated rejection and is flagged as such on the board.

### 3. `ref_axis_dist` as a U0 "close REF pair" is **refuted**

The smoke run proposed certifying axes by REF-pair separation, on two points
(`energy-lm-v18` 0.144 vs `gender-lm-v16` 0.059). The full labeled set inverts
it: ears-PASS `energy-v3-lm-20s` has the **smallest** REF pair in the corpus
(**0.030**) and ears-FAIL `energy-lm-v16` the **largest** (**0.144**, the same
REF pair as ears-PASS v18 — identical captions, identical value, opposite ears
verdicts). PASS range 0.030–0.144, FAIL range 0.059–0.144: no separation. It is
logged, not gated, and the U0 slot went to ladder *shape* instead.

### 4. U2 verified as drafted; U4 re-derived to 0.90

**U2 (0.40 floor / 0.35 drop) stands unchanged.** It fires on 3 of 3 ears-FAIL
garbles (+1 recall 0.00 grit, 0.00 distortion, 0.31 joy — whisper-large reads
distortion at 0.00 where the older whisper-tiny table read 0.08) and on 0 of 6
ears-PASS folders. The tightest PASS margin is `v9-ritual/gender-lm-v9`: +1
recall 0.909 against `hold_ref = max(base 0.4545, REF+ 0.4545) = 0.4545`. The
`max(base, REF+)` anchoring is what saves it — that folder's own base render
transcribes badly, and a base-only anchor would have been meaningless.

**U4 moves 0.85 → 0.90.** Over all labeled folders, `max(same_song / cross_anchor)`
at `|s| ≤ 1`:

    PASS: energy-v3 0.578 | gender-v9 0.659 | v18 0.702 | gender-uni 0.771 | tempo-uni 0.800
    FAIL: joy 0.533 | distortion 0.687 | v16-energy 0.687 || grit 1.012 | v16-gender 1.086
                                       ears-PASS ceiling 0.800 ─┤ gap ├─ 1.012 first FAIL

The gap is `(0.800, 1.012)`; its midpoint is 0.906, so the frozen threshold is
**0.90** — 11 % of headroom above the highest ears-PASS and 11 % below the
lowest replacement-style ears-FAIL. The three garble FAILs sit *below* the
threshold, which is correct: U2 is the gate for garble, U4 is the gate for song
replacement, and they are not meant to overlap. `gender-lm-v16` fails U4 with
**perfect lyric recall at every scale** — whisper-large contradicts the
`docs/lm-2d-scoreboard.md` note "garbled lyrics" for that checkpoint and agrees
with `docs/lm-pair-exam.md`'s "no audible swing".

BORDERLINE landings, reported and not calibrated on: `hurt-lm-uni-v2` +1 recall
**0.4615** clears the 0.40 floor but is 0.538 below its `hold_ref` of 1.00, so
U2's drop rule rejects it (ears: "partly wrong words" — the scorer is harsher
than the ear). `energy-lm-uni-v2` +1 recall **0.9231** and `rapslow-lm-uni-v2`
+1 recall **0.7500** both pass; the ear notes for those two ("first line only",
"+2 dead") describe things at rungs or granularities this instrument does not
read — bag-of-words recall cannot see "first line only" when the chorus words
reappear, and `rapslow` is graded at +1 by contract, which is exactly where the
ear passed it.

## Known blind spots, stated plainly

The first four are the ones the calibration pass **created or confirmed**, and
they are the reason a PASS from this instrument is a no-harm certificate rather
than an efficacy certificate.

- **No certified efficacy channel exists.** Neither the whisper-embedding
  projection nor the DSP projection separates ears-PASS from ears-FAIL on the
  labeled corpus, in sign, in rank, or against a null. Every "does the concept
  move?" gate the draft proposed was demoted. Of the 94 folders in the discovery
  sweep, 43 read PASS; that number means "43 renders did not break the song",
  and nothing more.
- **A no-op slider passes.** U6 was the only veto that could see one, and it
  rejects every ears-PASS folder. Asserted as a regression test, not fixed.
- **A dead or reversed minus pole passes.** U3 reads the plus pole only; the
  minus pole is priced by E, which is itself uncertified.
- **One documented false accept.** `v16/energy-lm-v16`, ears "alternates between
  the two songs", reads PASS. When two different songs sing the same words at
  the same level, no channel here can tell them apart.
- **Concept versus proxy.** The whisper embedding is a speech-content
  representation; "female voice" and "different singer" are the same vector in
  it, and finding 2 shows a real arrangement ride can land off-axis. The DSP
  vector is blind wherever the concept is not spectral (F0, BPM).
- **Single seed.** Every listen folder here is seed 7. `"seeds": 1` in the JSON;
  nothing promotes on it. Every verdict on `docs/lm-render-board.md` is
  provisional in that exact sense.
- **20 s window.** U5 cannot veto at this length — it warns on 89 of 94
  folders — and one denoise window of arrangement is not a song.
- **No tempo channel.** BPM rides are visible to ears and to nothing here.
- **Whisper recall is noisy on singing.** It is a ~13-word bag-of-words recall;
  the top and bottom of the table are not close calls, the middle is. It
  disagrees with the older whisper-tiny table on `distortion-lm-uni-v2` (0.00 vs
  0.08) and with two listening notes that called `v16` lyrics "garbled" (it
  reads 1.00).
- **The null pool is 13–14 clips**, and it is only used for `cross_anchor` now
  (U4) plus the demoted null. A 95th percentile from 13 samples is coarse.
  `--pool_same_lyrics` is OFF in the frozen contract for the reason given under
  "Null calibration"; on the sheets it does cover it moves `cross_anchor`
  0.088 → 0.078 and every U4 ratio up by ~12 %.
- **Threshold provenance is thin.** U4's 0.90 rests on a gap defined by five
  PASS and two FAIL folders. It is frozen because re-tuning it on candidates is
  the failure mode this document exists to prevent, not because seven points is
  a lot of evidence.
