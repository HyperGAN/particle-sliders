# Product repositories

Use the product's README for model installation, local weights, prompts, recipes,
training, rendering and supported slider strengths. Core supplies common
interfaces; the product decides which ones it implements and which source pins
it supports. Keep conflicting native dependencies in separate environments.

Use the [shared `--vibe` instructions](vibe.md) when documenting a product’s route from matched teachers to a released RAW adapter and tuned strength.

For multi-step generation, use the
[scheduled strength guide](strength-schedules.md) to qualify the product's
native denoising, generation-token or refinement stages. Find the exact API
or flag in the [product entry points](strength-schedules.md#product-entry-points). Check the product
guide for the schedule and RAW profile that have been measured together.

| Modality | Product | Where to test strength |
| --- | --- | --- |
| Image | [Anima](https://github.com/HyperGAN/anima-particle-sliders) | Denoising intervals. Use early for the measured Candlelit RAW256 profile. |
| Image | [Supra](https://github.com/HyperGAN/supra-particle-sliders) | Denoising intervals. |
| Image | [Krea2](https://github.com/HyperGAN/krea2-particle-sliders) | Denoising intervals. |
| Image/RGBA | [Ming](https://github.com/HyperGAN/ming-particle-sliders) | Native image or layer generation stage carrying the adapter. |
| Image | [Iris](https://github.com/HyperGAN/iris-particle-sliders) | Denoising intervals. |
| Image/RGBA | [Qwen Image](https://github.com/HyperGAN/qwen-image-particle-sliders) | Denoising intervals in the selected image or layer workflow. |
| Image | [Z-Image](https://github.com/HyperGAN/zimage-particle-sliders) | Denoising intervals. |
| Image | [P-H3 Image](https://github.com/HyperGAN/p-h3-image-particle-sliders) | Native image denoising intervals. |
| Image | [MiniMax H3 Image](https://github.com/HyperGAN/minimax-h3-image-particle-sliders) | Native PDD solver intervals. |
| Image | [MOSS](https://github.com/HyperGAN/moss-image-sliders) | Denoising intervals. |
| Image | [FLUX](https://github.com/HyperGAN/flux-particle-sliders) | Native packed flow intervals. |
| Image | [Klein](https://github.com/HyperGAN/klein-particle-sliders) | Native flow intervals. |
| Image | [Stable Diffusion](https://github.com/HyperGAN/sd-particle-sliders) | Retired scaffold without a native renderer. |
| Video/audio | [LTX 2.5](https://github.com/HyperGAN/ltx25-particle-sliders) | Joint native denoising intervals. Preserve current video/audio routing. |
| Video/audio | [MiniMax H3](https://github.com/HyperGAN/minimax-h3-particle-sliders) | Native audiovisual flow intervals. |
| Music | [Music3](https://github.com/HyperGAN/music3-particle-sliders) | Prefill and semantic audio frames before the original acoustic decoder. |
| Music | [YuE2](https://github.com/HyperGAN/yue2-particle-sliders) | Causal music-generation positions where the adapter is attached. |
| Speech | [Sophon](https://github.com/HyperGAN/sophon-particle-sliders) | Native speech generation stage carrying the slider. Check input-only controls at their native input site. |
| Speech | [VoxCPM2](https://github.com/HyperGAN/voxcpm2-particle-sliders) | Acoustic patch positions through native end-of-speech or the fixed patch limit. |
| Text | [Glimmer](https://github.com/HyperGAN/glimmer-particle-sliders) | Causal language-generation positions. |
| Text | [Meta Glimmer](https://github.com/HyperGAN/meta-glimmer-particle-slider) | Causal language-generation positions. |
| Text | [Qwen3.5](https://github.com/HyperGAN/qwen35-particle-sliders) | Causal generation positions. |
| Text | [Ripple](https://github.com/HyperGAN/ripple-particle-sliders) | Causal generation positions. |
| Text | [Arete](https://github.com/HyperGAN/arete-particle-sliders) | Causal generation positions. |
| Text | [Tiny Aya](https://github.com/HyperGAN/tiny-aya-particle-sliders) | Causal generation positions. |
| Text | [Gemma 4](https://github.com/HyperGAN/gemma-4-particle-sliders) | Causal generation positions. |
| Text | [Inkling 2 Mini](https://github.com/HyperGAN/inkling-particle-slider) | Fixed refinement passes when the adapted component runs on multiple passes. |
| Text decisions | [GLiDE](https://github.com/HyperGAN/glide-particle-sliders) | Scalar for one-shot choice scoring. |

Meta Glimmer provides a separately maintained interface around Glimmer.
The MOSS and Meta Glimmer names
are exceptions to the usual `*-particle-sliders` naming pattern.

A repository link is a route to its current documentation, not a claim of native
training or generation support. Check the product's capabilities and source
cohort before using a shared-core example. Historical integrations retained in
this repository are [reproduction material](history.md), not the usage entry point.
