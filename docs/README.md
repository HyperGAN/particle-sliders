# Docs index

Operator map for this repo. Train cards live next to the trainer that
owns them. Do not mix Music 3 defaults into a YuE2 or image/video env
(and the reverse).

The [root README](../README.md) is the public YuE2 / Music 3 landing
page. This index is the in-repo map of cards, prompt files, and 2-D
exams.

## Start here

| if you need | read |
|---|---|
| YuE2 math, published particles, current train command | [README.md](../README.md) |
| YuE2 recipes, render, resume | [yue2-slider.md](yue2-slider.md) |
| Music 3 trainer defaults, shipped sliders, GPU pitfalls | [MUSIC3.md](../MUSIC3.md) |
| TF-only gate contract (G0–G7) | [SCORING.md](../SCORING.md) |
| paired recipe-comparison runbook | [slider_pipeline/README.md](../slider_pipeline/README.md) |
| listen folders / play order | [eval/listen/README.md](../eval/listen/README.md) |
| which `prompts-*.yaml` to pass | [prompts.md](prompts.md) |

The pipeline scores **transformer sliders only**. A working LM half
changes the arrangement on purpose; do not gate LM checkpoints with
it.

## Music and particle backends

These are separate adapters. Formats are not interchangeable.

| backend | trainer | card | student +1 / notes |
|---|---|---|---|
| **YuE2 particles** (preferred) | `train_lora_yue2_arm_b.py --recipe particle_bridge` | [yue2-slider.md](yue2-slider.md) | Neutral caption; routed 128×4 cloud on AR q/k/v/o. Published 1,200-update release ≠ current CLI defaults. |
| YuE2 original bridge audit | same entry point | [yue2-particle-bridge.md](yue2-particle-bridge.md) | Absolute-whitening MLP proof (HIT 3400/8000). Live native now uses paired-edit whitening. |
| YuE2 gmix-v2 on 2-D | `analysis/slider2d/yue2_gmix_v2_exam.py` | [yue2-gmix-v2-2d-scoreboard.md](yue2-gmix-v2-2d-scoreboard.md) | Released-v2 critic + paired-edit + T=1600. UNI PASS at native 1600 EMA. |
| Music 3 LM | `train_lm_slider_music3.py` | [MUSIC3.md](../MUSIC3.md) | Live default `--lm_target v9` / `--pole_mode hidden`. Published Hub recipe is a later span-GAN. |
| Music 3 flow | `train_lora_music3.py` | [MUSIC3.md](../MUSIC3.md) | Dust-recipe TF defaults (`--targets full`, NMSE). Pipeline-only for comparisons. |
| Music 3 particles | `train_lora_music3_particle.py` | module `music3_particle_bridge.py` | Propose-only port of the YuE2 game onto Qwen3 last-hidden. Default critic `bottleneck`. Does not change `--lm_target v9`. |
| Tiny LLM smoke | `train_lora_tiny_llm.py` | [tiny-llm-slider.md](tiny-llm-slider.md) | Qwen3-0.6B-Base. Same shared game **without** paired-edit neutrals. |
| Bonsai GGUF smoke | `train_lora_bonsai_gguf.py` | [bonsai-gguf-slider.md](bonsai-gguf-slider.md) | Frozen PTQ1_0 readout + torch residual head. Same shared game, no neutrals. |

CLI `--recipe particle_bridge` still defaults `--critic mlp`. The
hip-hop catalog and the published gmix-v2 2-D arm pass `--critic gmix`
explicitly. Do not assume the argparse default is the Hub release.

## Opt-in image / video trainers

Each backend is a separate script. None of them change
`train_lora_music3.py` / `train_lm_slider_music3.py` (`--lm_target v9`,
`--pole_mode hidden`). `--dummy` is the CI path: no Hub, no GPU.

| backend | trainer | card | live box | student +1 |
|---|---|---|---|---|
| **Krea** (Raw → Turbo) | `train_lora_krea.py` | [krea-slider.md](krea-slider.md) | A100 (~48–80 GB) | plus (velocity) or **neu** (TE embed UNI) |
| **Anima** (2B DiT) | `train_lora_anima.py` | [anima-slider.md](anima-slider.md) | 4090 / 512 px | **neu** (plus is teacher) |
| **Sana** 0.6B | `train_lora_sana.py` | [sana-slider.md](sana-slider.md) | cheap GPU (Modal / RunPod) | **neu** |
| **Z-Image Turbo** | `train_lora_zimage.py` | [zimage-slider.md](zimage-slider.md) | 6B DiT box | **neu** |
| **LTX-2.5** video | `train_lora_ltx25.py` | [ltx25-slider.md](ltx25-slider.md) | **dual RTX A6000** (validated 2026-09-02): TE+connectors `cuda:1`, DiT `cuda:0` for sample. CLI default `--encoder_device cpu`. Never `pipe.to` TE+DiT on one GPU. | **neu** (embed-match; plus is teacher). DiT stays frozen. |
| **MiniMax-H3** t2va | `train_lora_minimax_h3.py` | [minimax-h3-slider.md](minimax-h3-slider.md) | **B200 / B300** (~135 GB); 2×H100 needs `--encoder_device` | plus pack (velocity UNI) |

Shared pitfalls (verified in the trainers, not folklore):

- **Zero-init LoRA-up** is UNI identity on H3 / LTX (`loss 0.0000`).
  Default is `N(0, 0.02)`.
- **Train +1 on the plus caption, infer on neu** misses the concept
  (Sana age dud / H3 caption coupling). Sana / ZiT / LTX / Anima / Krea
  TE-embed train +1 on **neu**.
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
- **Do not** `pip install -r requirements.txt` into the Music 3 or
  YuE2 env. H3 on B300 needs `torch 2.13.0+cu130`; LTX needs a git
  Diffusers with LTX-2.5. Those wheels stay on their boxes.
- PEFT `set_adapter_scale` often no-ops (Krea #74). Continuous
  scales write `LoraLayer.scaling = (alpha/r) * scale`.

`--diag` (no train) on LTX / H3 writes a JSON gap table. LTX:
`post_cos` is the working embed gap (~0.68 live); velocity cos ≈ 1
is the negative control. H3: read lighting match and `s1_s0` drift
together before changing `hold_weight`.

H3 recommended live card is still **chiaro-v5** (`--hold_weight 3.0`,
rank 16 × 2500). Do not treat open chiaro-v6 as shipped.

## Music 3 target formulas

CPU-pure copies of the live losses live in
`conceptmod/textsliders/slider_targets.py`. The module docstring is
the contract: Music 3 `--lm_target` recipes (`v9`, `pair_odd_sub_e`,
`faithful_*`), SD enhance/erase, and the image UNI helpers (ZiT / Krea
embed). 2-D fixture write-ups that import it:

| page | what it grades |
|---|---|
| [2d-analysis.md](2d-analysis.md) | method table on a synthetic energetic×gender field |
| [tf-leak.md](tf-leak.md) | energy/distortion caption leak (BPM in `pos−neg`) |
| [lm-live-cells.md](lm-live-cells.md) | live `v9` / hold-ê cells |
| [lm-2d-scoreboard.md](lm-2d-scoreboard.md) | compiled 2-D / high-D / sheet board |
| [lm-2d-adv.md](lm-2d-adv.md) | RpGAN + `b_cap` vs leftover-gated supervised baselines (CPU; not a live `--pole_mode`) |
| [lm-sheet-goodhart.md](lm-sheet-goodhart.md) | why `c+` / `collapse` are not the success metric |
| [lm-highd-leftover.md](lm-highd-leftover.md) | leftover mix under live width |
| [lm-roles.md](lm-roles.md) | `faithful_plus_neu_roles` role split |
| [lm-lyric-hold.md](lm-lyric-hold.md) / [lm-lyric-orth.md](lm-lyric-orth.md) / [lm-lyric-recall.md](lm-lyric-recall.md) | lyric-span UNI variants |
| [lm-plus-exam.md](lm-plus-exam.md) / [lm-plus-neu-exam.md](lm-plus-neu-exam.md) / [lm-pair-exam.md](lm-pair-exam.md) | plus-only / plus+neu / pair exams |
| [lm-even-leftover.md](lm-even-leftover.md) / [lm-hold-overlap.md](lm-hold-overlap.md) | even leftover / hold overlap |
| [lm-v9-2d.md](lm-v9-2d.md) / [lm-v9-mismatch.md](lm-v9-mismatch.md) / [lm-faithful-2d.md](lm-faithful-2d.md) / [lm-rich-2d.md](lm-rich-2d.md) | older 2-D cells |

Default LM recipe is still `--lm_target v9` / `--pole_mode hidden`.
The `faithful_*` and `semantic_kl` cards are opt-in — see MUSIC3.md
“Current LM trainer defaults”. Those defaults do **not** reproduce
the published Music 3 adversarial release.

## Adversarial 2-D (CPU only)

[#94](https://github.com/mikkel/sliders-conceptmod/pull/94) ported
ParticleGAN's `rp_g_loss` + one-sided `b_cap` onto the existing sheet /
exam fixtures. Real samples are leftover-gated hidden-state deltas, not
audio. Defaults in `analysis/slider2d/run_lm_adv.py`: teacher
`faithful_guard_e`, `b_cap=1.0`, `cover_weight=1.5`, feature matching
off, 1200 GAN steps. The Gaussian smoke is
`analysis/gan_bcap/gaussian_repro.py` (CI: 8 modes). This is **not** a
new live `--pole_mode`.

[#128](https://github.com/mikkel/sliders-conceptmod/pull/128) is a
separate YuE2 cell: the released-v2 gmix head on the UNI PairField
toys. Shared losses come from `particle_bridge_gan`; it does not
change Music 3 `--lm_target`.
