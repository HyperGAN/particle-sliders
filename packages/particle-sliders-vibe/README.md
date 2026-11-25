# Particle Sliders Vibe

Persistent best-of-N training-data curation with model-specific workflows.
Keep successful prompts, generate replacements for weaker ones, and retain
every finished output and judgment as data accumulates.

This optional add-on has no dependency on Torch or a particular core version.
It can coexist with a product's pinned Particle Sliders training runtime.
The separate `particle_sliders_vibe` namespace leaves that core's source closure
unchanged. Install `[nanoodle]` for the workflow adapters.

Use `particle-slider-vibe --vibe ... --workflow product.module:factory --out ...`,
or `particle-slider collect` in a core version with collection dispatch.
Full protocol, persistence and scoring details are in
[VIBE-COLLECTION.md](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/VIBE-COLLECTION.md).
Anima supplies the first product renderer and frozen-target manifest handoff. The [shared `--vibe` instructions](https://github.com/HyperGAN/particle-sliders/blob/main/docs/vibe.md) take a qualified collection through training, alpha selection and release.
