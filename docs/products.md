# Product repositories

Use the product's README for model installation, local weights, prompts, recipes,
training, rendering and supported slider strengths. Core supplies common
interfaces; the product decides which ones it implements and which source pins
it supports. Keep conflicting native dependencies in separate environments.

| Modality | Product |
|---|---|
| Image | [Anima](https://github.com/HyperGAN/anima-particle-sliders) |
| Image | [Supra](https://github.com/HyperGAN/supra-particle-sliders) |
| Image | [Krea2](https://github.com/HyperGAN/krea2-particle-sliders) |
| Image | [Ming](https://github.com/HyperGAN/ming-particle-sliders) |
| Image | [Iris](https://github.com/HyperGAN/iris-particle-sliders) |
| Image | [Qwen Image](https://github.com/HyperGAN/qwen-image-particle-sliders) |
| Image | [Z-Image](https://github.com/HyperGAN/zimage-particle-sliders) |
| Image | [P-H3](https://github.com/HyperGAN/p-h3-image-particle-sliders) |
| Image | [MOSS](https://github.com/HyperGAN/moss-image-sliders) |
| Music/audio | [Music3](https://github.com/HyperGAN/music3-particle-sliders) |
| Music/audio | [YuE2](https://github.com/HyperGAN/yue2-particle-sliders) |
| Text | [Glimmer](https://github.com/HyperGAN/glimmer-particle-sliders) |
| Text | [Qwen3.5](https://github.com/HyperGAN/qwen35-particle-sliders) |
| Text | [Glide](https://github.com/HyperGAN/glide-particle-sliders) |

[Meta Glimmer](https://github.com/mikkel/meta-glimmer-particle-slider) provides a
separately maintained interface around Glimmer. The MOSS and Meta Glimmer names
are exceptions to the usual `*-particle-sliders` naming pattern.

A repository link is a route to its current documentation, not a claim of native
training or generation support. Check the product's capabilities and source
cohort before using a shared-core example. Historical integrations retained in
this repository are [reproduction material](history.md), not the usage entry point.
