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
widget:
  - text: "Same prompt and lyrics. Energy slider Off then Loud +1. Unidirectional lyric-hold LM."
    example_title: "Energy Off → Loud +1"
    output:
      url: samples/demo/energy-ab.mp3
  - text: "Same prompt and lyrics. Energy slider Off then Loud +2. Unidirectional lyric-hold LM."
    example_title: "Energy Off → Loud +2"
    output:
      url: samples/demo/energy-ab-2.mp3
  - text: "Same prompt and lyrics. Gender slider Off then Female +1. Unidirectional lyric-hold LM."
    example_title: "Gender Off → Female +1"
    output:
      url: samples/demo/gender-ab.mp3
  - text: "Same prompt and lyrics. Gender slider Off then Female +2. Unidirectional lyric-hold LM."
    example_title: "Gender Off → Female +2"
    output:
      url: samples/demo/gender-ab-2.mp3
  - text: "Same prompt and lyrics. Distortion slider Off then Distorted +1. Unidirectional lyric-hold LM."
    example_title: "Distortion Off → Distorted +1"
    output:
      url: samples/demo/distortion-ab.mp3
  - text: "Same prompt and lyrics. Distortion slider Off then Distorted +2. Unidirectional lyric-hold LM."
    example_title: "Distortion Off → Distorted +2"
    output:
      url: samples/demo/distortion-ab-2.mp3
  - text: "Same prompt and lyrics. Joy slider Off then Joy +1. Unidirectional lyric-hold LM."
    example_title: "Joy Off → Joy +1"
    output:
      url: samples/demo/joy-ab.mp3
  - text: "Same prompt and lyrics. Joy slider Off then Joy +2. Unidirectional lyric-hold LM."
    example_title: "Joy Off → Joy +2"
    output:
      url: samples/demo/joy-ab-2.mp3
---

# MiniMax Music 3 concept sliders

Unidirectional LoRA sliders for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).
One scalar adds one musical property while the prompt stays fixed: `0` is off,
`+1` is one trained unit of the concept. Opposite poles are **separate**
LoRAs (Loud and Quiet are two files, not ± on one file).

**Current studio catalog is uni-lyric, language-model only.** Each shipped
file is `weights/<name>-lm-uni-lyric/<name>-lm-uni-lyric_last.safetensors`
(unit-normalized, drop in at strength 1), except the four v24 lyric-hold
refreshes (`gender-lm-v24-lyrichold`, `distortion-lm-v24-uni`,
`breath-lm-v24-uni`, `joy-lm-v24-uni`). Space stays the transformer
exception (`space-tf-v6`); dust stays `dust-lm-v1-faithful-kl-pole03`.

Earlier bipolar `*-lm-v9` weights remain under `weights/` for comparison.

## Listen (uni-lyric)

Same prompt, same lyrics, same seed. Only the LoRA scale changes. Player
clips below are uni-lyric at +1 and +2. Full ladders (0 / +1 / +2 plus
caption REFs) live under [`samples/uni-lyric/`](samples/uni-lyric/).

### Energy — Off → Loud

**+1 loud**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/energy-loud.mp3" type="audio/mpeg"></audio>

**+2 loud**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/energy-loud-2.mp3" type="audio/mpeg"></audio>

### Quiet — Off → Quiet

**+1 quiet**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/energy-quiet.mp3" type="audio/mpeg"></audio>

**+2 quiet**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/energy-quiet-2.mp3" type="audio/mpeg"></audio>

### Gender — Off → Female

**+1 female**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/gender-female-v24.mp3" type="audio/mpeg"></audio>

**+2 female**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/gender-female-v24-2.mp3" type="audio/mpeg"></audio>

### Male — Off → Male

**+1 male**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/gender-male.mp3" type="audio/mpeg"></audio>

**+2 male**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/gender-male-2.mp3" type="audio/mpeg"></audio>

### Distortion — Off → Distorted

**+1 distorted**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/distortion-metal-v24.mp3" type="audio/mpeg"></audio>

**+2 distorted**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/distortion-metal-v24-2.mp3" type="audio/mpeg"></audio>

### Clean — Off → Clean

**+1 clean**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/distortion-acoustic.mp3" type="audio/mpeg"></audio>

**+2 clean**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/distortion-acoustic-2.mp3" type="audio/mpeg"></audio>

### Joy — Off → Joy

**+1 joy**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/joy-joy-v24.mp3" type="audio/mpeg"></audio>

**+2 joy**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/joy-joy-v24-2.mp3" type="audio/mpeg"></audio>

### Somber — Off → Somber

**+1 somber**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/joy-somber.mp3" type="audio/mpeg"></audio>

**+2 somber**

<audio controls preload="none"><source src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/demo/joy-somber-2.mp3" type="audio/mpeg"></audio>

Training code: [ntc-ai/sliders-conceptmod](https://github.com/ntc-ai/sliders-conceptmod)
(`conceptmod/textsliders/train_lm_slider_music3.py`).

## Objective (uni-lyric)

Plus-only LoRA on `Qwen3Attention`. The generation prompt is always the
**neutral** caption. At LoRA scale `s = +1` the prompt-last hidden `h(s)`
is pulled toward the frozen encode of the + caption `h+`; at `s = 0` it
must reproduce the neutral encode `h0`. A lyric-token hold keeps the yaml
`lyrics` span pinned to encode(neu):

$$
\mathcal{L}_{\mathrm{pole}} = \|h(+1) - h_+\|^2 + \|h(0) - h_0\|^2
$$

$$
\mathcal{L}_{\mathrm{lyric}} = \|h_{\mathcal{L}}(+1) - h_{0,\mathcal{L}}\|^2
$$

$$
\mathcal{L} = \mathcal{L}_{\mathrm{pole}} + \mathcal{L}_{\mathrm{lyric}} + \lambda_{\mathrm{end}}\,\mathcal{L}_{\mathrm{end}}
$$

Teacher is raw `h+` at +1 (last token), `h0` at scale 0, and encode(neu)
on the lyrics span only (subscript `L`). Vocal Details / Global Metadata /
Arrangement are not held. No minus pole, no leftover-gate, no pair-odd. Infer
with the neutral caption + LoRA — not the + caption.

The end term teacher-forces the LoRA'd LM over a frozen base-model
composition and penalizes drift of the stop margin:

$$
m = \mathrm{logit}(\langle\mathrm{audio\_end}\rangle) - \mathrm{logsumexp}(\text{semantic band}), \qquad \mathcal{L}_{\mathrm{end}} = \|m(+1) - m_{\mathrm{base}}\|^2
$$

The slider may move the musical plan; it must not move the stop decision.

Flags:

```
--lm_target faithful_plus_neu_lyric --pole_mode hidden --endreg_weight 1.0
```

Opposite concepts ship as separate weights (e.g. `energy-lm-uni-lyric`
and `energy-quiet-lm-uni-lyric`). There is no trained −1 on a plus pole.

## Catalog

| id | concept | weights |
|---|---|---|
| gender | Female | `weights/gender-lm-v24-lyrichold/` |
| male | Male | `weights/gender-male-lm-uni-lyric/` |
| energy | Loud | `weights/energy-lm-uni-lyric/` |
| quiet | Quiet | `weights/energy-quiet-lm-uni-lyric/` |
| distortion | Distorted | `weights/distortion-lm-v24-uni/` |
| clean | Clean | `weights/distortion-clean-lm-uni-lyric/` |
| tempo | Fast | `weights/tempo-lm-uni-lyric/` |
| slow | Slow | `weights/tempo-slow-lm-uni-lyric/` |
| live | Live | `weights/live-lm-uni-lyric/` |
| studio | Studio | `weights/live-studio-lm-uni-lyric/` |
| breath | Breathy | `weights/breath-lm-v24-uni/` |
| rapslow | Rap | `weights/rapslow-lm-uni-lyric/` |
| triphop | Trip-hop | `weights/triphop-lm-uni-lyric/` |
| pop | Pop | `weights/triphop-pop-lm-uni-lyric/` |
| rhyme | Rhyme | `weights/rhyme-lm-uni-lyric/` |
| prose | Prose | `weights/rhyme-prose-lm-uni-lyric/` |
| sexy | Sexy | `weights/sexy-lm-uni-lyric/` |
| plain | Plain | `weights/sexy-plain-lm-uni-lyric/` |
| tender | Tender | `weights/tender-lm-uni-lyric/` |
| fierce | Fierce | `weights/tender-fierce-lm-uni-lyric/` |
| grit | Grit | `weights/grit-lm-uni-lyric/` |
| smooth | Smooth | `weights/grit-smooth-lm-uni-lyric/` |
| joy | Joy | `weights/joy-lm-v24-uni/` |
| somber | Somber | `weights/joy-somber-lm-uni-lyric/` |
| yearn | Yearning | `weights/yearn-lm-uni-lyric/` |
| settled | Settled | `weights/yearn-settled-lm-uni-lyric/` |
| hurt | Hurt | `weights/hurt-lm-uni-lyric/` |
| numb | Numb | `weights/hurt-numb-lm-uni-lyric/` |
| dust | Tape | `weights/dust-lm-v1-faithful-kl-pole03/` |
| space | Wet | `weights/space-tf-v6/` (transformer) |

Play `samples/uni-lyric/<name>-lm-uni-lyric/` in filename order (0, +1, +2,
then caption REFs).

### ComfyUI

Each shipped file has a sibling `*_comfyui.safetensors` (LM files use
`text_encoders.model.layers.*.self_attn.*`). Drop the Comfy file in
`ComfyUI/models/loras/` and load it with **Load LoRA**. Strength is the slider
scale (`0` off, `+1` the baked unit).

ComfyUI support is untested.

## Which stage a slider attaches to

MiniMax Music 3 generates in two stages:

1. A **Qwen3 language model** writes a plan — who is singing, melody, rhythm, arrangement.
2. A **flow transformer** renders that plan — timbre, loudness, tone, space.

uni-lyric attaches to stage 1. Space is the catalog exception on stage 2
(reverb is a rendering property).
