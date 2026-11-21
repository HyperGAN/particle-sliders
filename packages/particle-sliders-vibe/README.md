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
Anima supplies the first product renderer and frozen-target manifest handoff.

### Audit a saved collection

Run `python -m particle_sliders_vibe.audit /path/to/collection.json` to check
the saved equal-case ranking and best-of-N choices. Add `--verify-media` when
the original output files are available. The report includes counts and mean
effect gain for both the full pool and retained best-of-N contexts, plus the
number of candidates tied at the top of each round. It uses saved scores and
makes no model or judge calls.

The retained subset is selected using the same measurements shown in the
report. Its mean describes the selected contexts. Estimate calibration or
held-out effectiveness only with separately labelled, matched evidence for
that use. Reported judge confidence has no role in the audit or selection.
