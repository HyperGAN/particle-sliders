# Recipe entry-point audit, 2026-11-04

New training must resolve a named upstream preset with `particlegan.get_recipe`.
Bare or partial `Recipe(...)` construction must not select an older recipe's
defaults. Restoring a recorded recipe is a separate operation: use
`Recipe.from_dict(saved_fields)` when that upstream API is available. The
historical fixture test retains a complete-field constructor fallback only for
the currently pinned official 0.9.4 release, which predates that API.

The following default branches were inspected directly, including their
shared factory calls. No active partial upstream `Recipe(...)` construction
was found in these eight repositories.

| Repository | Audited default commit | Current entry point |
| --- | --- | --- |
| HyperGAN/sd-particle-sliders | `ee27f14a008c8f590cb8f56e0bcc312510705406` | Retired scaffold; no trainer. |
| HyperGAN/cascade-particle-sliders | `f6035d289a1d92ec01a8e7f853ce1c863cd51fdd` | `backend.py:359` calls `ConditionalPG09`. Adapter-only; no particle bank. |
| HyperGAN/music3-particle-sliders | `aedbaa1d0000229c74ef43d906bc1c45fd4e2943` | `music3/lifecycle.py:424` and optional HNDL `binding.py:227` call `ConditionalPG09`. |
| HyperGAN/yue2-particle-sliders | `2d6fdf411f4996f2d14caff7a53de3c4d32dbe45` | `yue2_sliders/lifecycle.py:75` and optional HNDL `hndl_response.py:203` call `ConditionalPG09`. |
| HyperGAN/tinyqwen-particle-sliders | `90ca7131fadb8b21468076974dfea15ebebcd61f` | `binding.py:404` calls `ConditionalPG094`, the explicit-version alias of `ConditionalPG09`. |
| HyperGAN/glimmer-particle-sliders | `7c8e60a6761e913f73ce9ecb202b453430e20458` | `glimmer_modern/lifecycle.py:113` calls `ConditionalPG09`. |
| mikkel/meta-glimmer-particle-slider | `c0f074f0ae682f782e695a1c12744fef716c952a` | New training is redirected to standalone modern Glimmer. Optional PG82 reproduction remains isolated and version-pinned. |
| HyperGAN/particle-sliders | `a4cfe080bc895a7eab1bcdcb6c617ad8f883bda8` | `pg09.py:360`, `bispf_game.py:76`, and `formulation.py:398` use `get_recipe`. |

`ConditionalPG09` resolves the requested upstream preset, then selects an
upstream-supported dense optimizer profile through a complete-field
`dataclasses.replace`. It does not change the loss, regularizers, EMA, or LR
schedule as part of that profile selection. Product geometry and any admitted
explicit rate overrides are recorded in the recipe and checkpoint identity.

The shared `textsliders/gan_v2` research `Recipe` and historical `SliderRecipe`
are distinct local dataclasses, not aliases for `particlegan.Recipe`. Existing
historical implementations, checkpoints, and published weights are not
converted by this audit.

## EMA and schedule ownership

For the audited official PG 0.9.4 `bispf_optimized` preset, `ema_decay` is zero
and both LR floors are one. Although the metadata names a cosine schedule,
its effective G, D, and particle rates are constant. CPU checks queried the
actual optimizer groups at indices 0, 199, 200, 799, 800, 959, 960, 1024, and
1600: G `0.012`, D `0.018`, and particle bank `0.03` throughout. These queries
perform no optimizer updates and are not native training evidence.

The separate response lifecycle's output EMA defaults to disabled (`None`).
Callers can explicitly request an export EMA with `ema_decay`; it is not the
critic's regularizer or a replacement training recipe. Glimmer's bounded
`cpu_config` helper explicitly chooses `0.8` for its synthetic width-12 test
and export demonstration. Its pretrained native training path is disabled.
That fixture choice must not be described as an implicit native-model default.

Current product pins and source checks remain unchanged. A future upstream
release needs its own explicit source admission and verification; this audit
does not silently substitute a checkout for an admitted release.

## Verification

- Existing focused CPU update, state validation, restore, and optimizer-profile
  suite: 248 passed, 1 optional-dependency skip.
- Revised recipe/snapshot contract tests on the pinned release: 67 passed,
  1 optional-dependency skip.
- Final focused update, restore, state-validation, and contract suite with the
  new regression: 249 passed, 1 optional-dependency skip.
- Historical restoration and optimizer-profile checks against the separate
  upstream constructor-guard source: 15 passed, 43 unrelated tests deselected.
  This is API compatibility evidence only; the source was not installed or
  admitted for product training.
- CUDA was hidden; no model download, live runtime change, GPU training,
  checkpoint modification, or historical-source repinning was performed.
