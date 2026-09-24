# Prompt-file catalog

All files are under `conceptmod/textsliders/data/` unless noted.
Axes and constraints below are taken from the YAML headers and the
trainer that loads them — not from listen folklore.

Pass the path with `--prompts_file` (upstream SD-era trainers read it
from `--config_file` instead; see below). Do not prefix yaml
`attributes` onto captions unless that backend says so (Anima / Supra /
Krea / LTX / H3 chiaro: attributes are unused-token pins only). Music
and YuE2 poles must describe **sound** only — no real artist, band,
songwriter, producer or album names.

## YuE2 / particle smokes

| file | trainer | axis | notes |
|---|---|---|---|
| `prompts-yue2-metal-arm-b.yaml` | `train_lora_yue2_arm_b.py` | metal | Default `--prompts_file` for that trainer and `scripts/train_yue2_arm_b_campaign.py`. Four sound-only rows; one lyric sheet per row, shared by both poles. Used by `--recipe unipolar_gan` (default) / `gan_plus_neu` / `particle_bridge`. |
| `prompts-yue2-metal-arm-b-eval.yaml` | `scripts/evaluate_yue2_arm_b.py` (default), campaign `--eval_prompts_file` | metal holdout | Two held-out listening rows, verses disjoint from training. Not training rows. |
| `prompts-yue2-metal-arm-b-x16.yaml` | `train_lora_yue2_arm_b.py` | metal | 16-row metal pack; first four rows match the default file. Not the CLI default; nothing in-tree loads it. |
| `prompts-yue2-female.yaml` / `prompts-yue2-female-eval.yaml` | `train_lora_yue2.py` / `train_lora_yue2_fresh.py` | female voice | Historical UNI16 / fresh-3400. Flattened Music 3 layout. Four rows each; `-eval` is the held-out sheet. `prompts-yue2-female.yaml` is the `train_lora_yue2_fresh.py` default and the `config-yue2-female-uni16.yaml` target. |
| `prompts-yue2.yaml` | `train_lora_yue2.py` | breath | CLI default (`--recipe uni16` default, `hidden` opt-in). Two rows. Historical example. |
| `prompts-tiny-llm.yaml` | `train_lora_tiny_llm.py` | garden / sea / street | CLI default. Three plain captions. `--recipe particle_bridge` only. `unconditional` is an unscored canary, never a teacher. |
| `prompts-bonsai-gguf.yaml` | `train_lora_bonsai_gguf.py` | same three rows | CLI default. Text-only smoke. Same recipe and canary rule. |

Campaign copies under `analysis/yue2_uni16_1200_20260917/prompts/`
(16 sliders × `-train` / `-eval`, written by
`scripts/queue_yue2_particle_catalog.py`, read by
`analysis/yue2_gmix_catalog_20260918/onoff_listen.py`) are dated
research artifacts, not the trainer defaults.

## Image / video (opt-in UNI)

| file | trainer | axis | notes |
|---|---|---|---|
| `prompts-krea-happy.yaml` | `train_lora_krea.py` | smile / happy | Bare captions. TE-only embed UNI card (`--lora_targets te --lm_target embed`). |
| `prompts-krea-detailed.yaml` | `train_lora_krea.py` | fine detail / texture | Same-subject as neu (no style jump). detail-krea-v3. |
| `prompts-krea.yaml` | `train_lora_krea.py` | age | CLI default. plus = old, neu = person. Unused gender pinned. |
| `prompts-anima.yaml` | `train_lora_anima.py` | smile / happy | CLI default. Bare captions. `concept_words: smiling, smile, happy, joyful, teeth`. |
| `prompts-anima-canary.yaml` | `train_lora_anima.py` | same, minus left in | Minus is a canary, not a teacher. |
| `prompts-supra.yaml` | `train_lora_supra.py` | lighting (warm sun ↔ neutral daylight) | CLI default. Bare captions. `concept_words: warm, golden, sunlit, glow`. Student +1 on **neu**; + is teacher only. |
| `prompts-supra-canary.yaml` | `train_lora_supra.py` | same, minus left in | One row. Minus (moonlight) is a canary, not a teacher. |
| `prompts-sana.yaml` | `train_lora_sana.py` | happy | CLI default. Train/infer +1 on **neu**. Fruit-bowl control. Stronger smile positives (weak "a happy person" did not move stock Sana). |
| `prompts-zimage.yaml` | `train_lora_zimage.py` | age | CLI default. plus = old, neu = person. Infer +1 on neu + LoRA. |
| `prompts-ltx25-smile.yaml` | `train_lora_ltx25.py` | smile / happy | CLI default. Bare captions. Same subject / clothes / framing / motion / light / sound. Hold PRE-connector. Default `--recipe ltx25_uni_embed`. |
| `prompts-ltx25-chiaroscuro.yaml` | `train_lora_ltx25.py` | lighting | Same-subject Rembrandt vs soft fill (person + still life). Live v1: lighting YES, composition drift WEAK. |
| `prompts-minimax-h3.yaml` | `train_lora_minimax_h3.py` | age | CLI default. FL2VA / t2va. Negative is a canary. |
| `prompts-minimax-h3-chiaroscuro.yaml` | `train_lora_minimax_h3.py` | lighting | Same concrete subject on neu and plus (person / dining room / scholar jar). Default `--hold_mode non_concept`. Recommended live card is chiaro-v5 (`--hold_weight 3.0`, 2500 steps). |

Upstream SD / XL / SD3 / Flux / Cascade yamls (`prompts.yaml`,
`prompts-xl.yaml`, `prompts-sd3.yaml`, `prompts-flux.yaml`,
`prompts-cascade.yaml`) belong to the original Baulab trainers
(`train_lora.py`, `train_lora_xl.py`, `_sd3`, `_flux`, `_cascade`), not
the Music 3 or UNI cards above. Those trainers take the yaml from the
config's `prompts_file`, not `--prompts_file`.

Upstream image-slider yamls moved with their trainer to
`legacy/trainscripts/imagesliders/data/`. `legacy/prompts/*.csv`
(car / food / person / room / sky) are upstream image eval prompts for
`legacy/eval-scripts/` (`--prompts_path`).

## Config cards

`prompts_file` in each card names the yaml above (`config-yue2.yaml`
has none; the CLI default `prompts-yue2.yaml` applies).

| file | read by | notes |
|---|---|---|
| `config-krea.yaml` | `train_lora_krea.py` default `--config_file` | Age card. `config-krea-happy.yaml` / `config-krea-detailed.yaml` point at the happy / detailed yamls; pass them explicitly. |
| `config-sana.yaml` / `config-zimage.yaml` | `train_lora_sana.py` / `train_lora_zimage.py` default `--config_file` | |
| `config-ltx25.yaml` / `config-ltx25-chiaroscuro.yaml` | `train_lora_ltx25.py` (smile card is the default) | `recipe: ltx25_uni_embed`, `hold_mode: non_concept`. |
| `config-minimax-h3.yaml` / `config-minimax-h3-chiaroscuro.yaml` | `train_lora_minimax_h3.py` (age card is the default) | Chiaro card is chiaro-v5: 2500 iterations, `hold_weight: 3.0`, `non_concept`. |
| `config-supra.yaml` / `config-anima.yaml` | no trainer (neither has `--config_file`) | Record the live card. Supra CLI defaults match its card; `tests/test_supra_slider.py` checks it. The Anima card says `lm_target: direct`; the trainer default is `trajectory`. |
| `config-tiny-llm.yaml` / `config-bonsai-gguf.yaml` | optional `--config_file` (default none) | Game numbers are echoed for readability; the trainer reads `particle_bridge_gan.REFERENCE`. |
| `config-yue2.yaml` / `config-yue2-female-uni16.yaml` | `train_lora_yue2.py --config_file` | Historical UNI16 (`recipe: uni16`, lr 5e-4, 600 steps). Not the Arm B / particle path. |
| `config-music3.yaml` | `train_lora_music3.py` default `--config_file` | Points at `prompts-music3.yaml`. `config-music3-encoder.yaml` is not loaded by anything in-tree. |
| `config.yaml` / `config-xl.yaml` / `config-cascade.yaml` | upstream SD / XL / Cascade trainers | `train_lora.py` defaults to `data/config.yaml`; the others require `--config_file`. No SD3 / Flux card ships. |

## Music 3 — shipped / current axes

v4 files are Structured Caption poles (Global Metadata / Vocal
Details / Arrangement). v3 files are the earlier flat
`Genre: … BPM: …` poles — off-distribution for the studio rewriter.
TF halves stay on the v2 / single-row files named in MUSIC3.md.
`prompts-<axis>-tf.yaml` files are single-row TF pairs ("multi-row
averaging kills transformer sliders"); `scripts/train_v6_*.sh` trains
them as `<axis>-tf-v6`.

| axis | v2 / TF | v3 LM | v4 LM | leftover `ê` (v4 header) |
|---|---|---|---|---|
| energy (quiet ↔ loud) | `prompts-energy.yaml` | `prompts-energy-v3.yaml` | `prompts-energy-v4.yaml` | mix / BPM / genre restates `a` — see MUSIC3.md |
| distortion (clean ↔ metal) | `prompts-distortion.yaml` | `prompts-distortion-v3.yaml` | `prompts-distortion-v4.yaml` | leftover = genre + BPM; `--lm_target pair_odd_sub_e` |
| tempo (slow ↔ fast) | `prompts-tempo.yaml` | `prompts-tempo-v3.yaml` | `prompts-tempo-v4.yaml` | leftover = genre; BPM is the slider |
| space (dry ↔ wet) | `prompts-space.yaml` | — | — | TF only in-repo |
| gender (male ↔ female) | `prompts-gender-tf.yaml` (failed TF) | `prompts-gender-v3.yaml` | `prompts-gender-v4.yaml` | clean pair; hold 0 |
| trip-hop ↔ pop | `prompts-triphop.yaml` / `prompts-triphop-v3-single.yaml` | `prompts-triphop-v3.yaml` | `prompts-triphop-v4.yaml` | leftover = BPM only; genre/dusty/glossy is the slider |
| rap ↔ slow sung | `prompts-rapslow-tf.yaml` | `prompts-rapslow-v3.yaml` | `prompts-rapslow-v4.yaml` | leftover = genre + mix; delivery is the slider; row 1 tempo-matched |
| rhyme | `prompts-rhyme-tf.yaml` | `prompts-rhyme-v3.yaml` | `prompts-rhyme-v4.yaml` | leftover = genre + BPM; axis lives in Vocal Details + Arrangement |
| breath | `prompts-breath-tf.yaml` | `prompts-breath-v3.yaml` | `prompts-breath-v4.yaml` | leftover = genre + BPM; air / inhales; needs whole-record gestalts |
| live (room bleed) | `prompts-live-tf.yaml` | `prompts-live-v3.yaml` | `prompts-live-v4.yaml` | no `leak_*`; people / leakage; tempo and energy pinned |
| dust (dusty ↔ glossy) | `prompts-cand-dust-v1.yaml` | — | — | **only TF slider that passed all six render gates** |
| grit (smooth ↔ rasp) | `prompts-grit-tf.yaml` | — | `prompts-grit-v4.yaml` | no `leak_*`; vocal texture, not amp distortion |
| hurt (numb ↔ wounded) | `prompts-hurt-tf.yaml` | — | `prompts-hurt-v4.yaml` | no `leak_*` |
| joy (somber ↔ joyful) | `prompts-joy-tf.yaml` | — | `prompts-joy-v4.yaml` | no `leak_*` |
| sexy (plain ↔ close) | `prompts-sexy-tf.yaml` | — | `prompts-sexy-v4.yaml` | no `leak_*`; genre / BPM / kit move with the axis |
| tender (fierce ↔ tender) | `prompts-tender-tf.yaml` | — | `prompts-tender-v4.yaml` | no `leak_*`; genre / BPM / kit move with the axis |
| yearn (settled ↔ longing) | `prompts-yearn-tf.yaml` | — | `prompts-yearn-v4.yaml` | no `leak_*` |

`prompts-pop-v3-single.yaml` is `prompts-triphop-v3-single.yaml` with
poles swapped (unidirectional glossy-pop LoRA). `prompts-gender.yaml`
/ `prompts-music3.yaml` are older generic rows; prefer the axis files
above. `prompts-music3.yaml` (one energy row) is still the
`train_lora_music3.py` CLI default.

Leaky v4 axes train with `--lm_target pair_odd_sub_e` (or default
`v9` + declared `leak_positive` / `leak_negative`). Gender stays
`v9` with no `ê`.

## Music 3 — later pair families

Same axes, newer files. None is a trainer default
(`train_lm_slider_music3.py` requires `--prompts_file`; defaults stay
`--lm_target v9` / `--pole_mode hidden`). The launcher column names
the scripts that load each family.

| suffix | axes | header contract | launcher |
|---|---|---|---|
| `-tf-v7` | breath, distortion, dust, energy, grit, hurt, joy, live, rapslow, rhyme, sexy, space, tempo, tender, triphop, yearn | single-row TF pair, BPM pinned across poles (space: no BPM; tempo: two rows, BPM is the axis) | `scripts/tf_rollout_axis.sh` (`<axis>-tf-v10`), `scripts/tf_gate_metric.py` |
| `-v5` | breath, distortion, energy, tempo | pinned bipolar, same room both poles; omit `leak_*`; header says `--lm_target faithful` | none in-tree |
| `-v6` | breath, grit, hurt, live, tender, yearn | v6-noendreg rewrite of lopsided, weak or failed v4 pairs | `scripts/retrain_weak_lm.sh` (`<axis>-lm-v6b`); fallback in `train_v11_lm.sh` / `train_v12_lm.sh` |
| `-v7` | every uni-v1 axis except live | BPM pinned per row (tempo: BPM is the axis); `slider_positive` / `slider_negative` declared for `--lm_target v9`. v7b rewrites (breath, hurt, joy, rhyme, sexy, yearn) let the non-BPM wording diverge; rapslow is one row | `scripts/train_v11_lm.sh` (`v9`), `train_v12_lm.sh` (`symmetric`); preferred over v6. `scripts/combined_ladder.py` default |
| `-uni-v1` | breath, distortion, energy, gender, grit, hurt, joy, live, rapslow, rhyme, sexy, tempo, tender, triphop, yearn | plus-only same-room rows; `negative := target` is a canary, never taught; `minus_label: Off`, range `[0, 2]` | `scripts/train_uni_v1_lm.sh` (`faithful_plus`), `train_uni_v2_plus.sh` (`faithful_plus_neu`), `train_uni_v3_prefix.sh` (`faithful_plus_neu_prefix`); axes with no uni-v2 file in `train_uni_lyric_poles.sh` / `train_uni_lyric_adv.sh` |
| `-uni-v2` | distortion, energy, gender, grit, hurt, joy, rapslow, tempo | plus-only; teacher is +1, "+2 is shred"; range `[0, 1]` | `scripts/train_uni_lyric.sh` (`faithful_plus_neu_lyric`); `train_uni_lyric_poles.sh` prefers it over uni-v1; `train_uni_lyric_adv.sh`. Gender is the default for `train_lm_gan_bcap.sh` / `train_lm_gan_v2.sh` |
| `-minus-uni-v1` | same as uni-v1 | the old minus pole as its own plus-only UNI (`plus_label` = old minus) | `scripts/train_uni_v1_minus.sh` (`faithful_plus_neu`), `train_uni_lyric_poles.sh <axis>-minus`, `train_uni_lyric_adv.sh` |

Write-ups: [lm-uni-v1.md](lm-uni-v1.md),
[lm-uni-v2-garble.md](lm-uni-v2-garble.md),
[lm-lyric-hold.md](lm-lyric-hold.md).
`prompts-energy-v7-edge.yaml` / `prompts-energy-tf-v7-edge.yaml` are
single-direction controls (`negative := target`; only + moves).
`prompts-distortion-tf-v7-swap.yaml` is `prompts-distortion-tf-v7.yaml`
with poles swapped (plus = Acoustic).

## Music 3 — candidate pairs (pipeline onboard)

Certify with `scripts/slider_pipeline.py onboard` **before** training.
These are production-axis candidates at fixed genre/BPM/arrangement
(except energy, which puts level on the odd axis). Gate notes are in
each file header (revised 2026-08-21).

| file | intended axis | pipeline note |
|---|---|---|
| `prompts-cand-dust-v1.yaml` | dust / sheen | passing TF baseline (`dust-tf-v1`) |
| `prompts-cand-space-v1.yaml` | dry / cavernous | `onboard` (G0) this pair first |
| `prompts-cand-energy-v1.yaml` | density / level | next energy pair after the caption-BPM leak (BPM pinned 110; [tf-leak.md](tf-leak.md)); do **not** add `--attributes` to TF |
| `prompts-cand-grit-v1.yaml` | grit / clean | `onboard` (G0) this pair first |
| `prompts-cand-vintage-v1.yaml` | vintage / modern | `G-vintage` failed G0 at p=0.75 (SCORING.md) |

`prompts-cand-dust-lm-v2.yaml` (structured, fixed genre) and
`prompts-cand-dust-lm-divergent.yaml` are dust **LM** drafts, not TF.
Headers: probe to sep ≳ 0.20 and get sign-off before any train; v2
probed at 0.146.

## Music 3 — probe / try cells

Not shipped. Used by `scripts/probe_lm_axis_signal.py` when hunting
a separable pole pair.

| prefix | what the headers test |
|---|---|
| `prompts-breath-try-*.yaml` | divergent-record cells (gestalt, energy-shaped hard / punchy, R&B) vs lexical minimal pair; caption shape; nameless copy; identity-as-who; long neutral |
| `prompts-rhyme-try-*.yaml` | country AABB, hook-pop sprawl, artsong minus, mantra, overflow, recitative |

Keep artist names out of new Music 3 poles (MUSIC3.md).
