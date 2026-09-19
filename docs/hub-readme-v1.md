---
license: mit
base_model: MiniMaxAI/MiniMax-Music-3
tags:
  - music
  - text-to-music
  - lora
  - concept-sliders
  - minimax-music-3
library_name: diffusers
pipeline_tag: text-to-audio
---

# MiniMax Music 3 concept sliders

Bipolar LoRA sliders for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).
One scalar moves one musical property while the prompt stays fixed: `0` is off,
`+2` leans one way, `−2` the other.

**Current studio catalog is v1, language-model only.** Each shipped file is
`weights/<axis>-lm-v1/<axis>-lm-v1_last.safetensors` (unit-normalized, drop in at
strength 1).

Space is the exception: reverb is a rendering property, so `space-tf-v6` remains
the only transformer slider. Dust stays the faithful-KL LM winner
(`dust-lm-v1-faithful-kl-pole03`).

## Objective (honest)

v1 is a bare `--lm_target v9` train, **not** the old Hub-leash writeup.

```
--lm_target v9          # default
# projected-odd teacher onto a declared slider_positive / slider_negative
# orthogonal hold (hold_weight=1)
# κ = 0  (no even blend-back)
--endreg_weight 1.0     # default; keep <|audio_end|> margins
```

What this is **not**:

```
--lm_target hub
# --symmetric + --anchor_weight 0.3 --anchor_autocal --leakage_floor -0.9
```

The Hub floor still leaks unused attributes that live inside `(pos−neg)/2`.
v1 projects that odd teacher onto a declared axis and holds the residual.
Do not omit `slider_positive` / `slider_negative` and expect `(pos−neg)` to
be the axis — that is the leak.

Earlier `*-lm-v9` weights stay under `weights/` as the Hub-leash archive.
Ladders for v1 are under [`samples/v1/`](samples/v1/).

## Catalog

| id | minus | plus | weights |
|---|---|---|---|
| gender | Male | Female | `weights/gender-lm-v1/` |
| rapslow | Slow | Rap | `weights/rapslow-lm-v1/` |
| triphop | Pop | Trip-hop | `weights/triphop-lm-v1/` |
| energy | Quiet | Loud | `weights/energy-lm-v1/` |
| distortion | Acoustic | Metal | `weights/distortion-lm-v1/` |
| tempo | Slow | Fast | `weights/tempo-lm-v1/` |
| live | Studio | Live | `weights/live-lm-v1/` |
| breath | Clean | Breathy | `weights/breath-lm-v1/` |
| rhyme | Prose | Rhyme | `weights/rhyme-lm-v1/` |
| sexy | Plain | Sexy | `weights/sexy-lm-v1/` |
| tender | Fierce | Tender | `weights/tender-lm-v1/` |
| grit | Smooth | Grit | `weights/grit-lm-v1/` |
| joy | Somber | Joy | `weights/joy-lm-v1/` |
| yearn | Content | Yearning | `weights/yearn-lm-v1/` |
| hurt | Composed | Hurt | `weights/hurt-lm-v1/` |
| space | Dry | Wet | `weights/space-tf-v6/` (transformer) |
| dust | Glossy | Dusty | `weights/dust-lm-v1-faithful-kl-pole03/` |

Same prompt, same lyrics, same seed. Only the LoRA scale changes. Play
`samples/v1/<axis>-lm-v1/` in filename order (−2 −1 0 +1 +2, then caption REFs).
