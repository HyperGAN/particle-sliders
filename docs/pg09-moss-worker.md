# Guardian-invoked MOSS worker

Run only after the resource owner admits the reviewed recipe. This module does
not obtain leases or admit GPU work. Importing it and validating a recipe use
only the standard library. Its CPU tests use tiny native MOSS fixtures and never
load pretrained weights or initialize CUDA.

```text
python -m conceptmod.textsliders.pg09_moss_worker \
  --recipe /reviewed/case.json --output /reviewed/new-run \
  --mode probe --lease-fd <inherited admission FD> --lease-fd <inherited GPU FD>
```

The guardian supplies both already-held lease descriptors, one physical card in
`CUDA_VISIBLE_DEVICES`, the offline environment and kernel monitor. The worker
checks descriptor targets without acquiring/releasing locks. It checks its own
source, all listed source bytes, and the *actual imported* helper/backend/core/
native module origins against `source_pins` before loading weights. ResourceGuard
is loaded from its exact separate pinned path and attached to base, critic and
VAE. Final recipes must pin the checkout from which Python actually imports.

`BoundRecipe.validate` defines schema `moss-pg09-worker-v1`. Required fields are:

- `objective=pg09_gan_only`, `concept=bad-intent|afterimage`, `pg_version=0.9.0`,
  `preset=bispf_optimized`, `response_domain=continuous_field`, `sampling=balanced`.
- `model_revision=fc5de0d56fc910cbca6f6625a70f5f582b26deb1` and its local official
  `model_snapshot`; original case `cache_directory` and preserved
  `old_mse_adapter=.../checkpoints/step-001600/adapter`.
- `total_steps=3840`, `batch_size=8`, `training_seed=1000`, `checkpoint_every=80`,
  `wall_seconds=604800`, `signed_rows=15360`; optional `critic_width` is fixed32.
- Frozen training `response_scale`: Bad Intent `0.18700603425178916`, Afterimage
  `0.19194795196490175`. `cache_identity_sha256` binds the lazy provider policy;
  `resolved_recipe_sha256` binds `dataclasses.asdict(game.recipe)` from the actual
  public PG preset, including explicit reviewed `recipe_overrides` if supplied.
- `old_adapter_sha256`, `old_manifest_sha256`, `source_pins` absolute file/hash
  mapping, `resource_guard={path,sha256}`, two distinct `lease_paths`, and separate
  `output_directory` under slider runs or qualification runs.

Probe uses the **same 3840-step recipe**, stopping after three completed updates.
Before any update, two actual cache/teacher states emit relative target/base RMS
and MSE with thresholds; failure preserves those diagnostics and stops. A passing
probe replays the third update from step2, proving exact loss/full state/sample,
then proves adapter export/reload and streams the complete frozen parameter SHA
again. No entire base clone is kept. Gradient hooks and parameter hashes witness
actual adapter and critic updates; shared `ConditionalPG09` owns every loss,
optimizer, protected-gradient operation and training RNG. Functional MSE calls
are forbidden; teacher diagnostics use detached square/mean only.

Train resumes only this recipe's full GAN checkpoint, continues to3840, and saves
every80 completed updates. SIGINT/SIGTERM request stopping after a completed step.
Immutable checkpoint directories include full game state and base/recipe hashes.
Same-step reuse requires exact packet equality; divergence is rejected before
`LATEST` changes. A completed directory published just before interruption is
recoverable even if `LATEST` was not updated. Partial `.pending-*` directories
are never resumed. MSE checkpoints and changed source/recipe/base are rejected.

Evaluation is a separate phase after step3840. It loads the local FP32 Flux2 VAE,
uses cached contexts and the official MOSS scheduler, and renders both new GAN
and preserved old MSE1600 exports under the common protocol: two heldouts,
seed29001,512px,50 steps, five variants (off/direct-positive/direct-positive-CFG1/
minus/plus). CFG uses a frozen empty-prompt branch and no old skip22 rule.
Teacher contrast remains positive-minus-neutral; those roles are explicit in
receipts. Original images are preserved. Latent nonfinites fail before VAE;
float VAE nonfinites fail before clamp/PIL conversion and image writes.

Progress is atomic `progress.json` plus `events.jsonl`; immutable checkpoints,
adapter exports and per-image provenance stay in the new run directory. Records
include per-step timing, current/peak RAM and VRAM, observed backward norms,
actual parameter changes, source/cache/condition provenance and D/G balanced
coverage summaries. Passing probe or completing CPU tests does not qualify
pretrained visual quality. Full model probe, sustained training and matched
heldout review remain resource-owner actions.
