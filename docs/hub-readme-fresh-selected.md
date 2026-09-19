---
license: mit
base_model: MiniMaxAI/MiniMax-Music-3
tags:
- music
- text-to-music
- lora
- concept-sliders
- minimax-music-3
- comfyui
library_name: diffusers
pipeline_tag: text-to-audio
---

# MiniMax Music 3 concept sliders

**[Try the MiniMax Music 3 Space](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)** · **[YuE2 slider weights](https://huggingface.co/ntc-ai/yue2-concept-sliders)** · **[YuE2 Space](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders)**

## NEW — September 16, 2026

**16 newly retrained voice and genre LoRAs, with new matching samples and ComfyUI exports.** The downloads below are the selected checkpoints from the new four-prompt training runs.

![Same seed. Different sound. Off and On waveforms from the selected Female checkpoint.](assets/same-seed.svg)

**New voice and genre controls · Selected checkpoints · Native and ComfyUI downloads**

Steer the lead toward a female vocal, a piano-led arrangement toward heavy riffs, or a small band toward a club pulse. These LoRA adapters change the language model that plans music in [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).

**September 16, 2026:** all sixteen retrains are complete. This release uses the checkpoints selected by the final per-slider audit: steps 1,000, 2,000, 3,000 or 3,400 depending on the control. All voice and genre downloads and On samples below refer to those selected checkpoints.

[Choose a control](#choose-a-control) · [ComfyUI](#comfyui) · [Native loading](#native-loading) · [Selection criteria](#how-these-checkpoints-were-chosen) · [Training](#training) · [How it works + math](#how-the-sliders-learn)

## Press play

**Off** uses the base model. **On** adds the named adapter at strength **+1**. The prompt, lyrics and seed match within each pair; composition and phrasing can still change. Recordings retain their original levels and lengths.

### Female

Same prompt · same lyrics · seed **1709** · selected step **1000**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/female/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/female/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/female/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/female/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/female/row2-seed1709-reference.mp3)

### Metal

Same prompt · same lyrics · seed **1709** · selected step **2000**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/metal/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/metal/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/metal/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/metal/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/metal/row2-seed1709-reference.mp3)

### House

Same prompt · same lyrics · seed **1709** · selected step **1000**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/house/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/house/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/house/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/house/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/house/row2-seed1709-reference.mp3)

### Disco Funk

Same prompt · same lyrics · seed **1709** · selected step **2000**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/disco-funk/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/disco-funk/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/disco-funk/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/disco-funk/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/disco-funk/row2-seed1709-reference.mp3)

### Pop Punk

Same prompt · same lyrics · seed **1709** · selected step **3400**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/pop-punk/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/pop-punk/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/pop-punk/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/pop-punk/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/pop-punk/row2-seed1709-reference.mp3)

### Lo-fi

Same prompt · same lyrics · seed **1709** · selected step **3000**

<table style="display:table;width:100%;table-layout:fixed">
<thead><tr><th>OFF · 0</th><th>ON · +1</th></tr></thead>
<tbody><tr><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/lofi/row2-seed1709-off.mp3"></audio></td><td><audio controls preload="none" style="width:100%;min-width:0;max-width:280px" src="https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/lofi/row2-seed1709-on.mp3"></audio></td></tr></tbody>
</table>

[Off WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/lofi/row2-seed1709-off.wav) · [On WAV](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/lofi/row2-seed1709-on.wav) · [Target-caption reference](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/resolve/main/samples/uni16-fresh-selected-v2/lofi/row2-seed1709-reference.mp3)

[Hear all 64 Off/On comparisons](samples/uni16-fresh-selected-v2/README.md). Every selected checkpoint has two arrangements × two seeds. Each featured example uses the first fixed arrangement and seed; examples were not chosen by their individual scores. Target-caption references are also included.

## Choose a control

**All sixteen downloads in this table are NEW as of September 16, 2026.** Each file adds one target sound. All are language-model attention adapters with rank 8 and alpha 8. Female and Male are separate controls; negative strength is not a trained opposite.

| New control | Target sound | Selected step | Native | ComfyUI | New samples |
|---|---|---:|---|---|---|
| Female | Encourages one adult female lead vocal | 1,000 | [Weights](weights/uni16-fresh-selected-v2/female/female_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/female/female_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/female_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/female/README.md) |
| Lo-fi | Soft swung drums, mellow keys and gentle tape warmth | 3,000 | [Weights](weights/uni16-fresh-selected-v2/lofi/lofi_step3000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/lofi/lofi_step3000.json) | [Weights](comfyui/uni16-fresh-selected-v2/lofi_step3000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/lofi/README.md) |
| Male | Encourages one adult male lead vocal | 3,400 | [Weights](weights/uni16-fresh-selected-v2/male/male_step3400.safetensors) · [Settings](weights/uni16-fresh-selected-v2/male/male_step3400.json) | [Weights](comfyui/uni16-fresh-selected-v2/male_step3400_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/male/README.md) |
| Pop | Clear hooks, crisp drums and a polished chorus | 2,000 | [Weights](weights/uni16-fresh-selected-v2/pop/pop_step2000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/pop/pop_step2000.json) | [Weights](comfyui/uni16-fresh-selected-v2/pop_step2000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/pop/README.md) |
| Hip-Hop | Rapped verses, deep sub bass and nimble hats | 3,000 | [Weights](weights/uni16-fresh-selected-v2/hiphop/hiphop_step3000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/hiphop/hiphop_step3000.json) | [Weights](comfyui/uni16-fresh-selected-v2/hiphop_step3000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/hiphop/README.md) |
| R&B | Warm keys, deep pocket and fluid vocal phrasing | 3,400 | [Weights](weights/uni16-fresh-selected-v2/rnb/rnb_step3400.safetensors) · [Settings](weights/uni16-fresh-selected-v2/rnb/rnb_step3400.json) | [Weights](comfyui/uni16-fresh-selected-v2/rnb_step3400_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/rnb/README.md) |
| Indie Rock | Chiming guitars, moving bass and a human drum kit | 1,000 | [Weights](weights/uni16-fresh-selected-v2/indie-rock/indie-rock_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/indie-rock/indie-rock_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/indie-rock_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/indie-rock/README.md) |
| Pop Punk | Palm-muted power chords and driving chorus drums | 3,400 | [Weights](weights/uni16-fresh-selected-v2/pop-punk/pop-punk_step3400.safetensors) · [Settings](weights/uni16-fresh-selected-v2/pop-punk/pop-punk_step3400.json) | [Weights](comfyui/uni16-fresh-selected-v2/pop-punk_step3400_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/pop-punk/README.md) |
| Metal | Heavy guitar riffs, tight kicks and big melodic choruses | 2,000 | [Weights](weights/uni16-fresh-selected-v2/metal/metal_step2000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/metal/metal_step2000.json) | [Weights](comfyui/uni16-fresh-selected-v2/metal_step2000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/metal/README.md) |
| Country | Acoustic strum, twangy fills and an easy backbeat | 1,000 | [Weights](weights/uni16-fresh-selected-v2/country/country_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/country/country_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/country_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/country/README.md) |
| Acoustic Folk | Fingerpicked strings and a warm small-room performance | 2,000 | [Weights](weights/uni16-fresh-selected-v2/acoustic-folk/acoustic-folk_step2000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/acoustic-folk/acoustic-folk_step2000.json) | [Weights](comfyui/uni16-fresh-selected-v2/acoustic-folk_step2000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/acoustic-folk/README.md) |
| House | Steady club kick, offbeat hats and a rolling bass line | 1,000 | [Weights](weights/uni16-fresh-selected-v2/house/house_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/house/house_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/house_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/house/README.md) |
| Disco Funk | Elastic bass, clipped guitar and bright dance-floor strings | 2,000 | [Weights](weights/uni16-fresh-selected-v2/disco-funk/disco-funk_step2000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/disco-funk/disco-funk_step2000.json) | [Weights](comfyui/uni16-fresh-selected-v2/disco-funk_step2000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/disco-funk/README.md) |
| K-pop | Sharp synth hooks, tight edits and a big chorus lift | 2,000 | [Weights](weights/uni16-fresh-selected-v2/kpop/kpop_step2000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/kpop/kpop_step2000.json) | [Weights](comfyui/uni16-fresh-selected-v2/kpop_step2000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/kpop/README.md) |
| Reggaeton | Dembow drums, rounded sub bass and clipped melodic hooks | 1,000 | [Weights](weights/uni16-fresh-selected-v2/reggaeton/reggaeton_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/reggaeton/reggaeton_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/reggaeton_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/reggaeton/README.md) |
| Afrobeats | Interlocking percussion, melodic bass and buoyant guitar | 1,000 | [Weights](weights/uni16-fresh-selected-v2/afrobeats/afrobeats_step1000.safetensors) · [Settings](weights/uni16-fresh-selected-v2/afrobeats/afrobeats_step1000.json) | [Weights](comfyui/uni16-fresh-selected-v2/afrobeats_step1000_comfyui.safetensors) | [4 pairs](samples/uni16-fresh-selected-v2/afrobeats/README.md) |

[Machine-readable catalog](catalog.json) · [Selection scores and checkpoint comparisons](evidence/uni16-fresh-selected-v2/README.md) · [Release file hashes](release-manifest.json)

## ComfyUI

1. Download the **ComfyUI** file for your chosen control and place it in `ComfyUI/models/loras/`.
2. Use **Load LoRA** with your Music 3 text encoder connected to **CLIP**.
3. Set **strength_model = 0** and **strength_clip = 1** for the published On treatment. Compare with CLIP strength 0; start with one adapter.

Use a Music 3 text encoder with separate `q_proj`, `k_proj`, `v_proj` and `o_proj` layers. Text encoders with merged `qkv_proj` layers cannot apply the separate query/key/value factors in these files. See the [ComfyUI loading guide](comfyui/README.md).

All sixteen ComfyUI files were produced by the upstream conversion script from the selected native exports. Factors are BF16 and alpha scalars are FP32. The published recordings use the native adapters; ComfyUI precision and generation settings can change the resulting audio.

## Native loading

Download the native weight and its matching JSON settings file. Use the included [loading example](usage.md) with your existing Music 3 pipeline.

```python
from huggingface_hub import snapshot_download

folder = snapshot_download(
    "ntc-ai/minimax-music3-concept-sliders",
    allow_patterns=[
        "catalog.json", "usage.md", "source/lora.py",
        "weights/uni16-fresh-selected-v2/female/*",
        "prompts/uni16-fresh-selected-v2/*",
    ],
)
```

Replace `female` with a control ID, or use `weights/uni16-fresh-selected-v2/*/*` for all native files. For ComfyUI only, download `comfyui/uni16-fresh-selected-v2/*`.

Start at direct multiplier **1** and compare with **0** using the same caption, lyrics and seed. The studio normalizes slider mixtures by host energy, so its fader position may differ from the direct multiplier used here.

## How these checkpoints were chosen

The final audit scored **80 checkpoint candidates**, with four matched clips per candidate. It excluded checkpoints with technical flags, then evaluated **Content Enjoyment (CE)** and **Production Quality (PQ)** separately using Audiobox Aesthetics.

For each control, a checkpoint qualifies when its mean CE is within **0.2 points** of the best clean CE and its mean PQ is within **0.2 points** of the best clean PQ. Among qualifying checkpoints, the latest checkpoint from the new training run is preferred. The two best values can come from different checkpoints. This is the `quality-later-v2` policy.

**Description similarity has no influence on selection.** CLAP measurements are kept as a separate diagnostic. Cached lyric checks are informational only. There is no combined style/quality score in this selection rule.

The 0.2-point tolerance is a preference rule, not a calibrated audible threshold. The [selection report](evidence/uni16-fresh-selected-v2/README.md) shows how picks change at 0.1, 0.2 and 0.3 points. Four short clips support a shortlist; they do not establish universal quality, broad musical diversity, full-song reliability or behavior when stacking adapters.

All sixteen selected native exports contain finite tensors and match their saved full training states. Original-waveform checks cover silence, clipping, shortening, high-frequency excess, stereo issues, interior gaps and repeated or near-identical audio. These checks do not rule out subtle musical defects.

## Training

Each slider uses all **four existing prompt pairs**. A fresh rank-8 adapter first trains for 600 updates with all four rows per batch. Further updates use one row at a time in balanced shuffled passes, with a fresh frozen-base continuation on every update. Continuations supply ending supervision; the prompt-state style targets remain fixed per row.

Eight runs reached 3,400 updates, and the other eight stopped at the approved 2,000-update budget. The published checkpoints were selected afterward. Generator and critic learning rates stay at 0.0005 and 0.00075, with adversarial, feature-matching and ending terms each weighted 1. Milestone samples retain their first draws without seed retries.

[Method and equations](evidence/uni16-fresh-selected-v2/formulation.md) · [Training record](evidence/uni16-fresh-selected-v2/training.json) · [Checkpoint integrity](evidence/uni16-fresh-selected-v2/integrity.json) · [Conversion record](evidence/uni16-fresh-selected-v2/conversion.json)

## How the sliders learn

The **new September 16 voice and genre adapters** use LoRA to change the language model that plans the music.

### One strength control

For an adapted weight matrix, the slider applies:

$$
W(s) = W_0 + s\,\frac{\alpha}{r}BA.
$$

Here \\(W_0\\) is the frozen base weight, \\(A\\) and \\(B\\) are the learned low-rank factors, and \\(s\\) is the strength. The new exports use \\(r=8\\) and \\(\alpha=8\\), so \\(s=0\\) switches the adapter off and \\(s=1\\) applies its exported change. Intermediate strengths scale the weight change smoothly; the resulting music need not change linearly. Negative strength is not a trained opposite sound.

### Voice and genre: learn the change between descriptions

Each of the sixteen new controls learns from **four caption pairs**. One caption describes a starting arrangement and the other adds the requested voice or genre. Lyrics, tempo and section tags stay fixed within each pair.

**1. Build a target change.** The frozen language model encodes the starting and target captions. We compare corresponding lyric-token states and the audio-start state. Let \\(H_0\\) be the starting-caption sequence, \\(H_+\\) the target-caption sequence, and \\(H_\theta\\) the adapted model's sequence on the starting caption:

$$
x_+ = \frac{H_+ - H_0}{\sigma}, \qquad
x_\theta = \frac{H_\theta - H_0}{\sigma}.
$$

The fixed scale \\(\sigma\\) is calibrated from the teacher changes using root mean square (RMS). It puts the target change \\(x_+\\) and the adapter's change \\(x_\theta\\) in the same units. Padding is excluded.

**2. Train a critic to recognize that change.** A small two-layer transformer \\(D\\) learns to score target changes above adapted changes. It combines the mean of valid token features with the final audio-start feature. Its relativistic pairing objective is:

$$
\mathcal L_D =
\mathbb E\!\left[\operatorname{softplus}\big(D(x_\theta)-D(x_+)\big)\right]
+ \mathcal R_{\mathrm{cap}}.
$$

Here \\(\mathbb E\\) denotes a batch average, and \\(\operatorname{softplus}(u)=\log(1+e^u)\\) penalizes the wrong ordering smoothly. The critic is used during training only.

**3. Train the adapter for the target change, its features and stopping behavior.** The adapter minimizes three terms, each with coefficient 1:

$$
\begin{aligned}
\mathcal L_G ={}&
\mathbb E\!\left[\operatorname{softplus}\big(D(x_+)-D(x_\theta)\big)\right] \\
&+ \operatorname{MSE}\!\left(\mathbb E[\phi(x_\theta)],\mathbb E[\phi(x_+)]\right)
+ \mathcal L_{\mathrm{end}}.
\end{aligned}
$$

| Term | What it teaches |
|---|---|
| Adversarial comparison | Make the adapted change resemble the target-caption change. |
| Feature matching | Match the batch means of the critic's learned features \\(\phi\\). MSE is mean squared error over feature coordinates. |
| Ending supervision | Preserve the base model's balance between continuing and ending on a supplied token history. |

Warm-up uses four rows per batch. Fresh-continuation training uses one row per update, so feature matching then compares that update's student and teacher features. The adapter has no separate explicit lyric-hold term.

**4. Preserve the stopping margin on fresh histories.** Let \\(\ell\\) denote token logits and \\(\mathcal S\\) the semantic audio-token vocabulary. On the same supplied history, compare the audio-end logit with the total semantic-continuation mass:

$$
\begin{aligned}
m &= \ell_{\mathrm{audio\_end}} - \log\sum_{j\in\mathcal S}e^{\ell_j}, \\
\mathcal L_{\mathrm{end}} &= \operatorname{MSE}(m_\theta,m_0).
\end{aligned}
$$

The first 600 updates use fixed base histories. **From update 601 onward, each update samples a fresh frozen-base continuation with the adapter disabled**, cycling through all four training rows in balanced shuffled passes. Seeds do not repeat; duplicate continuation tensors are rejected without substituting a new seed. These histories diversify ending supervision. The prompt-state style targets remain fixed per row.

The ending term preserves a stopping decision on the supplied histories. It does not establish that every newly generated full song will end naturally.

**5. Keep the critic's gradients bounded.** In the calibrated input coordinates, the cap penalizes gradient norms only above 1:

$$
\mathcal R_{\mathrm{cap}} =
\frac{1}{2}\sum_{z\in\{x_+,x_\theta\}}
\mathbb E\!\left[
\max\!\left(0,\lVert\nabla_zD(z)\rVert_2-1\right)^2
\right].
$$

The cap coefficient and threshold are both 1. Small gradients are allowed, and the initially zero adapter change remains included.

| Training setting | New September 16 release |
|---|---|
| Adapter location | Language-model attention; 36 layers × four projections |
| Starting point | A fresh zero-output rank-8, alpha-8 adapter and critic per control |
| Warm-up | 600 updates; all four prompt pairs per batch |
| Further training | One row per update; balanced shuffled passes; one fresh base continuation per update |
| Learning rates | Adapter 0.0005; critic 0.00075; constant |
| Continuation step bound | Adapter parameter-update L2 norm at most 2 |
| Training budgets | Eight runs to 3,400; eight runs to 2,000 |
| Published checkpoints | Selected per control at 1,000, 2,000, 3,000 or 3,400 |
| Published samples | 64 Off/On pairs: two arrangements × two seeds for each of all sixteen controls |

The critic, optimizers, sampler and RNG state carry across milestones. Every selected native export was checked tensor-for-tensor against its saved full state. The full [training record](evidence/uni16-fresh-selected-v2/training.json) gives the recipe and budgets.

### Checkpoint selection: quality bounds, then later steps

Selection happens after training. For one slider, let \\(\mathcal C\\) contain checkpoints that pass the technical screen, with four-clip mean enjoyment \\(E_c\\), mean production quality \\(P_c\\), and step \\(t_c\\). The quality anchors and eligible set are:

$$
\begin{aligned}
E^* &= \max_{c\in\mathcal C} E_c, \qquad
P^* = \max_{c\in\mathcal C} P_c, \\
\mathcal Q &= \left\{c\in\mathcal C :
E^*-E_c\leq0.2\;\text{ and }\;P^*-P_c\leq0.2\right\}.
\end{aligned}
$$

Among qualifying checkpoints from the new run, choose:

$$
c_{\mathrm{release}} =
\operatorname*{arg\,max}_{c\in\mathcal Q_{\mathrm{new}}} t_c.
$$

If no new-run checkpoint qualifies, the legacy release can be considered under the same quality bounds; an empty eligible set requires listening to resolve the tradeoff. All sixteen current selections are from the new runs. Description similarity and cached lyric checks have zero influence on this selection. The 0.2-point tolerance is a preference rule, not statistical equivalence or a proven audible boundary. [Selection policy and sensitivity](evidence/uni16-fresh-selected-v2/README.md).

## Earlier releases

The update-660 UNI16 release and older uni-lyric, v24 and bipolar weights remain at their existing paths. The root catalog and download table now point to the selected fresh-continuation release. [Previous model card](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders/blob/cb5b47832362354aa35aa1b4104f3dcf60bbb6fd/README.md).
