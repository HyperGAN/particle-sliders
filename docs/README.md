# Docs index

Operator map for this repo. Train cards live next to the trainer that
owns them. Do not mix Music 3 defaults into a YuE2 or image/video env
(and the reverse).

The [root README](../README.md) is the public YuE2 / Music 3 landing
page: getting started ([YuE2](../README.md#yue2)) and
[Tests](../README.md#tests). Math, training evidence and evaluation now
live in [math.md](math.md). This index is the in-repo map of cards,
prompt files, and 2-D exams.

## Start here

| if you need | read |
|---|---|
| installable shared core for product repos (`winning_formulation()`, `FormulationGame`) | [particle-sliders-core](../packages/particle-sliders-core/README.md) · [winning-formulation.md](winning-formulation.md) · [shared-core.md](shared-core.md) |
| YuE2 / Music 3 published weights, install, current train command, CPU tests | [README.md](../README.md) |
| YuE2 math, training evidence, Music 3 / image objectives, evaluation | [math.md](math.md) |
| YuE2 recipes, render, resume | [yue2-slider.md](yue2-slider.md) |
| Music 3 trainer defaults, shipped sliders, GPU pitfalls | [MUSIC3.md](../MUSIC3.md) |
| TF-only gate contract (G0–G7) | [SCORING.md](../SCORING.md) |
| LM-half contract (`scripts/lm_score.py`; PASS is no-harm only) | [LM-SCORING.md](../LM-SCORING.md) |
| paired recipe-comparison runbook | [slider_pipeline/README.md](../slider_pipeline/README.md) |
| listen folders / play order | [eval/listen/README.md](../eval/listen/README.md) |
| which `prompts-*.yaml` to pass | [prompts.md](prompts.md) |
| CPU formulation leaderboards (bipolar / unipolar) | [FORMULATION_LEADERBOARD_BIPOLAR.md](FORMULATION_LEADERBOARD_BIPOLAR.md) · [FORMULATION_LEADERBOARD_UNIPOLAR.md](FORMULATION_LEADERBOARD_UNIPOLAR.md) |

The pipeline scores **transformer sliders only**. A working LM half
changes the arrangement on purpose; do not gate LM checkpoints with
it.

## Shared core

`packages/particle-sliders-core` installs as `particle-sliders-core`
(import `particle_sliders`). Product repos pin a full commit of it and
train `winning_formulation()`: gmix architecture plus the provisional
`particle-gmix-1600-v2` overlay until
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38)
crowns a full-board winner. `FormulationGame` is the shared
feature-space D/G step; `EndpointGame` stays the bipolar teacher/predict
path. Cap, RpGAN loss and particle VIC come from `particlegan` (develop),
a transitive dependency. `concept-slider-core` / `concept_slider_core`
are deprecated aliases; `packages/concept-slider-core` is a retired
pointer.

| page | what it covers |
|---|---|
| [particle-sliders-core README](../packages/particle-sliders-core/README.md) | install pin, stamp API, product train step |
| [winning-formulation.md](winning-formulation.md) | gmix architecture, current overlay, how to plug in the #38 winner, product follow-ups |
| [shared-core.md](shared-core.md) | core vs product split, routing and the ordinary LoRA fit, adding a product |

The in-tree YuE2 / Music 3 / image trainers do not import the stamp.
`tests/test_particle_sliders_formulation.py` pins it against
`V2_SPEC` in `analysis/slider2d/yue2_gmix_v2_exam.py`.

## Music and particle backends

These are separate adapters. Formats are not interchangeable.

| backend | trainer | card | student +1 / notes |
|---|---|---|---|
| **YuE2 particles** (preferred) | `train_lora_yue2_arm_b.py --recipe particle_bridge` | [yue2-slider.md](yue2-slider.md) | Neutral caption; routed 128×4 cloud on AR q/k/v/o. Published 1,200-update release ≠ current CLI defaults. Math: [math.md](math.md#yue2-the-lead-formulation). |
| YuE2 original bridge audit | same entry point | [yue2-particle-bridge.md](yue2-particle-bridge.md) | Absolute-whitening MLP proof (HIT 3400/8000). Live native now uses paired-edit whitening. |
| YuE2 gmix-v2 on 2-D | `analysis/slider2d/yue2_gmix_v2_exam.py` | [yue2-gmix-v2-2d-scoreboard.md](yue2-gmix-v2-2d-scoreboard.md) | Released-v2 critic + paired-edit + T=1600. UNI toys PASS at the native 1600 stop (EMA). |
| Music 3 LM | `train_lm_slider_music3.py` | [MUSIC3.md](../MUSIC3.md) | Live default `--lm_target v9` / `--pole_mode hidden`. Published Hub recipe is a later [span-GAN](math.md#the-released-music-3-span-gan-objective). |
| Music 3 flow | `train_lora_music3.py` | [MUSIC3.md](../MUSIC3.md) | Dust-recipe TF defaults (`--targets full`, NMSE). Pipeline-only for comparisons. |
| Music 3 particles | `train_lora_music3_particle.py` | module `music3_particle_bridge.py` | Propose-only port of the YuE2 game onto Qwen3 last-hidden. Default critic `bottleneck`. Does not change `--lm_target v9`. |
| Tiny LLM smoke | `train_lora_tiny_llm.py` | [tiny-llm-slider.md](tiny-llm-slider.md) | Qwen3-0.6B-Base. Same shared game **without** paired-edit neutrals. |
| Bonsai GGUF smoke | `train_lora_bonsai_gguf.py` | [bonsai-gguf-slider.md](bonsai-gguf-slider.md) | Frozen PTQ1_0 readout + torch residual head. Same shared game, no neutrals. |

`train_lora_yue2_arm_b.py` defaults `--recipe unipolar_gan`; pass
`--recipe particle_bridge`. That recipe still defaults `--critic mlp`.
The Sept 18 gmix catalog (hip-hop training evidence,
`analysis/yue2_gmix_catalog_20260918/controller.py`) passes
`--critic gmix` explicitly; the gmix-v2 2-D arm builds `critic='gmix'`.
Do not assume the argparse default is the Hub release.

## Opt-in image / video trainers

Each backend is a separate script. None of them change
`train_lora_music3.py` / `train_lm_slider_music3.py` (`--lm_target v9`,
`--pole_mode hidden`). `--dummy` is the CI path: no Hub, no GPU.

| backend | trainer | card | live box | student +1 |
|---|---|---|---|---|
| **Krea** (Raw → Turbo) | `train_lora_krea.py` | [krea-slider.md](krea-slider.md) | A100 (~48–80 GB) | plus (velocity) or **neu** (TE embed UNI) |
| **Anima** (2B DiT) | `train_lora_anima.py` | [anima-slider.md](anima-slider.md) | 4090 / 512 px | **neu** (plus is teacher) |
| **Supra2-IMG** (~104M DiT) | `train_lora_supra.py` | [supra-slider.md](supra-slider.md) | not recorded (256² card) | **neu** (plus is teacher) |
| **Sana** 0.6B | `train_lora_sana.py` | [sana-slider.md](sana-slider.md) | cheap GPU (Modal / RunPod) | **neu** |
| **Z-Image Turbo** | `train_lora_zimage.py` | [zimage-slider.md](zimage-slider.md) | 6B DiT box | **neu** |
| **LTX-2.5** video | `train_lora_ltx25.py` | [ltx25-slider.md](ltx25-slider.md) | **dual RTX A6000** (validated 2026-09-02): TE+connectors `cuda:1`, DiT `cuda:0` for sample. CLI default `--encoder_device cpu`. Never `pipe.to` TE+DiT on one GPU. | **neu** (embed-match; plus is teacher). DiT stays frozen. |
| **MiniMax-H3** t2va | `train_lora_minimax_h3.py` | [minimax-h3-slider.md](minimax-h3-slider.md) | **B200 / B300** (~135 GB); 2×H100 needs `--encoder_device` | plus pack (velocity UNI) |

Shared pitfalls (verified in the trainers, not folklore):

- **Zero-init LoRA-up** is UNI identity on H3 / LTX (`loss 0.0000`).
  Default is `N(0, 0.02)`.
- **Train +1 on the plus caption, infer on neu** misses the concept
  (Sana age dud / H3 caption coupling). Sana / ZiT / LTX / Anima /
  Supra / Krea TE-embed train +1 on **neu**.
- **`--hold_mode attributes`** on H3 / LTX only pins yaml unused
  tokens. Shared subject tokens stay free and LoRA rewrites clothes /
  props. Default is `non_concept`.
- **LTX velocity UNI is a dead teacher** (plus/neu DiT velocity
  cos ~0.9999). Default is `--recipe ltx25_uni_embed` (post-connector
  video, MSE + rel-L2). `--recipe ltx25_uni_velocity` is ablation
  only. Do not switch to `transformer_full/` because velocity cos ≈ 1.
- **LTX hold is PRE-connector.** Tokenize → TE features →
  `apply_unused_hold` → left-pad `seq_len` to a multiple of 128 →
  then `pipe.connectors(...)`. After connectors, T is not 1:1 with
  prompt tokens.
- **Backend wheels stay on their boxes.** H3 on B300 needs
  `torch 2.13.0+cu130`; LTX needs a git Diffusers with LTX-2.5. Do not
  install them, or `legacy/requirements.txt`, into the Music 3 or YuE2
  env.
- PEFT `set_adapter_scale` often no-ops (Krea #74). Continuous
  scales write `LoraLayer.scaling = (alpha/r) * scale`.

`--diag` (no train) on LTX / H3 writes a JSON gap table. LTX:
`post_cos` is the working embed gap (~0.68 live); velocity cos ≈ 1
is the negative control. H3: read lighting match and `s1_s0` drift
together before changing `hold_weight`.

H3 recommended live card is still **chiaro-v5** (`--hold_weight 3.0`,
rank 16 × 2500). Do not treat open chiaro-v6
([#88](https://github.com/HyperGAN/particle-sliders/pull/88)) as
shipped.

## Music 3 target formulas

CPU-pure copies of the live losses live in
`conceptmod/textsliders/slider_targets.py`. The module docstring is
the contract: Music 3 `--lm_target` recipes (`v9`, `pair_odd_sub_e`,
`faithful_*`), SD enhance/erase, and the image UNI helpers (ZiT / Krea
embed). 2-D fixture write-ups that import it:

| page | what it grades |
|---|---|
| [FORMULATION_LEADERBOARD_BIPOLAR.md](FORMULATION_LEADERBOARD_BIPOLAR.md) / [FORMULATION_LEADERBOARD_UNIPOLAR.md](FORMULATION_LEADERBOARD_UNIPOLAR.md) | `run_formulation_leaderboard.py --polarity bi` / `uni` boards; metrics and plots in [formulation-leaderboard/](formulation-leaderboard/) |
| [2d-analysis.md](2d-analysis.md) | method table on a synthetic energetic×gender field |
| [tf-leak.md](tf-leak.md) | energy/distortion caption leak (BPM in `pos−neg`) |
| [lm-live-cells.md](lm-live-cells.md) | live `v9` / hold-ê cells |
| [lm-2d-scoreboard.md](lm-2d-scoreboard.md) | compiled 2-D / high-D / sheet board |
| [lm-2d-adv.md](lm-2d-adv.md) | RpGAN + `b_cap` vs leftover-gated supervised baselines (CPU; not a live `--pole_mode`) |
| [lm-sheet-goodhart.md](lm-sheet-goodhart.md) | why `c+` / `collapse` are not the success metric |
| [lm-highd-leftover.md](lm-highd-leftover.md) | leftover mix under live width |
| [lm-roles.md](lm-roles.md) | `faithful_plus_neu_roles` role split |
| [lm-lyric-hold.md](lm-lyric-hold.md) / [lm-lyric-orth.md](lm-lyric-orth.md) | lyric-token hold / last-delta lyric-span projection UNI |
| [lm-lyric-recall.md](lm-lyric-recall.md) | which existing metric would have flagged the `faithful_plus_neu` lyric garble (OOD hunt) |
| [lm-plus-exam.md](lm-plus-exam.md) / [lm-plus-neu-exam.md](lm-plus-neu-exam.md) / [lm-pair-exam.md](lm-pair-exam.md) | plus-only / plus+neu / pair exams |
| [lm-even-leftover.md](lm-even-leftover.md) / [lm-hold-overlap.md](lm-hold-overlap.md) | even leftover / hold overlap |
| [lm-v9-2d.md](lm-v9-2d.md) / [lm-v9-mismatch.md](lm-v9-mismatch.md) / [lm-faithful-2d.md](lm-faithful-2d.md) / [lm-rich-2d.md](lm-rich-2d.md) | older 2-D cells |

Default LM recipe is still `--lm_target v9` / `--pole_mode hidden`.
The `faithful_*` and `semantic_kl` cards are opt-in — see MUSIC3.md
“Current LM trainer defaults”. Those defaults do **not** reproduce
the published Music 3 adversarial release.

## Adversarial 2-D (CPU only)

[#94](https://github.com/HyperGAN/particle-sliders/pull/94) ported
ParticleGAN's `rp_g_loss` + one-sided `b_cap` onto the existing sheet /
exam fixtures. Real samples are leftover-gated hidden-state deltas, not
audio. Defaults in `analysis/slider2d/run_lm_adv.py`: teacher
`faithful_guard_e`, `b_cap=1.0`, `cover_weight=1.5`, feature matching
off, 1200 GAN steps. The Gaussian smoke is
`analysis/gan_bcap/gaussian_repro.py` (CI: 8 modes). This is **not** a
new live `--pole_mode`.

[#128](https://github.com/HyperGAN/particle-sliders/pull/128) is a
separate YuE2 cell: the released-v2 gmix head on the UNI PairField
toys. Shared losses come from `particle_bridge_gan`; it does not
change Music 3 `--lm_target`.

## Music 3 LM GAN and listens

| page | what it records |
|---|---|
| [lm-gan-bcap.md](lm-gan-bcap.md) | LM GAN with the Gaussian reference `b_cap`; current listening pick is the 300-update constant-LR smoke |
| [lm-gan-bcap-reference.md](lm-gan-bcap-reference.md) | exact ParticleGAN `b_cap` reference game (commit `b5ee35f`) and LM transfer audit |
| [music-arm-b-gates.md](music-arm-b-gates.md) | Music Arm B (#94 RpGAN + `b_cap`, leftover-gated): Field3D honesty gates vs what CPU can assert |
| [lm-uni-v1.md](lm-uni-v1.md) | uni-v1 plus-only prompts: raw `h+`, no leftover-gate, one concept per yaml |
| [lm-uni-v2-garble.md](lm-uni-v2-garble.md) | uni-v2 `faithful_plus_neu` live listens: last-token hit, lyric garble |
| [lm-render-board.md](lm-render-board.md) | `scripts/lm_score.py` verdicts on every scored listen ladder (2026-09-02) |

## YuE2 audits

Unipolar GAN-only recipes on `train_lora_yue2_arm_b.py`
(`--recipe unipolar_gan` / `gan_plus_neu`) and the unipolar CPU toy
boards behind them.

| page | what it records |
|---|---|
| [yue2-arm-b-verification.md](yue2-arm-b-verification.md) | exact objective of `unipolar-rpgan-bcap-yue2-v3` (`--recipe unipolar_gan`) |
| [yue2-gan-toy-audit.md](yue2-gan-toy-audit.md) | 600-step GAN-only recipe fails the unipolar toy gates; the same update passes at 3400 |
| [yue2-gan-stability.md](yue2-gan-stability.md) | `--propose_only_lr_scale .2`: both GAN recipes finish fresh 600-step native runs without the late collapse |
| [yue2-c9-native-trial.md](yue2-c9-native-trial.md) | matched `--propose_only_c9_g4x` metal trial; native stability not established |
| [unipolar-gan-plus-neu.md](unipolar-gan-plus-neu.md) | `--recipe gan_plus_neu`: passes the unipolar toy gates at 400 updates |
| [unipg-c-scoreboard.md](unipg-c-scoreboard.md) | UniPG-C: ParticleGAN schedule / optimizer clones over the production YuE2 GAN loop |
| [unipolar-gan-sweep.md](unipolar-gan-sweep.md) | UniPG-E: ParticleGAN toy sweep, GAN-only |
| [MUSIC_UNIPOLAR_GAN_NEXT.md](MUSIC_UNIPOLAR_GAN_NEXT.md) | handoff at `94fc990`: which unipolar GAN recipes already exist and what to try next |

## Selection / quality

| page | what it records |
|---|---|
| [slider-selection-metric.md](slider-selection-metric.md) | proposed single objective: usable-slider rate over a fixed ladder (LM first) |
| [slider-quality-pilot-results.md](slider-quality-pilot-results.md) | first listening pilot: composite not ready to optimize; `optimizer_score` stays null |

Workflow: [slider_selection/README.md](../slider_selection/README.md).

## Reward-model research

| page | what it records |
|---|---|
| [reward-model-sliders-handoff.md](reward-model-sliders-handoff.md) | spec: reward-derived Music 3 sliders, starting with Audiobox Content Enjoyment |
| [reward-slider-game-handoff.md](reward-slider-game-handoff.md) | handoff: a repeatable reward-slider game and the search loop around it |

Code: [reward_sliders/](../conceptmod/textsliders/reward_sliders/README.md),
[reward_game/](../conceptmod/textsliders/reward_game/README.md).

## Release-card sources

Local copies of the Music 3 Hub cards and their publish records.

| page | what it is |
|---|---|
| [hub-readme-fresh-selected.md](hub-readme-fresh-selected.md) / [hub-formulation-fresh-selected.md](hub-formulation-fresh-selected.md) | current card (Sept 16 fresh-continuation release) and its method page |
| [hub-native-usage.md](hub-native-usage.md) | loading the native adapters with the release's `source/lora.py` |
| [hub-readme-uni16-gan.md](hub-readme-uni16-gan.md) / [hub-formulation-uni16-gan.md](hub-formulation-uni16-gan.md) | earlier uni16-GAN card (16 voice & genre controls + reward slider) and its method page |
| [hub-readme-uni-lyric.md](hub-readme-uni-lyric.md) | earlier unidirectional lyric-hold LM catalog card |
| [hub-readme-v1.md](hub-readme-v1.md) | earlier bipolar v1 LM catalog card |
| [hub-release-20260909.json](hub-release-20260909.json) · [hub-reward-restored-20260909.json](hub-reward-restored-20260909.json) · [hub-showcase-off-on-20260909.json](hub-showcase-off-on-20260909.json) · [hub-card-refresh-20260909.json](hub-card-refresh-20260909.json) · [hub-card-voice-first-20260910.json](hub-card-voice-first-20260910.json) | publication and remote/browser verification records |

## Operator notes

- On the shared music workstation, train and render on physical GPU 1
  while the studio uses GPU 0. `CUDA_VISIBLE_DEVICES=1` exposes that
  card as logical `cuda:0`; use `--device 0` or `--device cuda:0`
  according to the script.
- CPU test deps are in `requirements-dev.txt` (see
  [Tests](../README.md#tests)). Each backend's runtime comes from its
  card. Never install `legacy/requirements.txt` (upstream
  diffusion-era pins) into a Music 3, YuE2 or other modern env.
- [legacy/](../legacy/README.md) holds the upstream Concept Sliders
  notebooks, eval scripts and paired-image trainer. Unmaintained.
