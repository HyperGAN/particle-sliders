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

The collector stores its provider request namespace in `request-scope.json`. Keep
this file with the collection when resuming or moving it. Each fresh collection
has its own request IDs, including when two collections use the same settings.
The `request_id` in each call receipt identifies the provider request. When an
older collection resumes, recorded calls keep their original IDs and completed
rounds keep their original identity.

If the proposer returns an invalid batch, the collector saves a failed receipt
and stops. Running the same command again requests a replacement batch under a
fresh ID. Completed renders and scores are reused. A pending provider call still
requires reconciliation before `--retry-pending` retries its original ID.


Image products that implement a native `VibeHost` can call `image_cli.entry` to train a
character slider from its phrase. Install the `image` extra alongside the host's CUDA and
ParticleGAN dependencies. The workflow keeps per-scene image judgments and structural
checks, retries unqualified scenes, freezes matched fields on both trajectories, and
searches a constant strength after the 448-update RAW export. Product READMEs give
their model locations, sampler and replay commands.
