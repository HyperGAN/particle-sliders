# ParticleGAN compatibility

Core 0.15.0 runs only on published `particlegan==0.10.1`. Adopting a later ParticleGAN
release means regenerating `engine_release.json` and its pinned identity; module
and symbol names stay unchanged. The shared engine is `particle_sliders.engine`
(`ConditionalParticleGAN`, `validate_engine_domain`, `engine_options`). Its source
manifest, `engine_release.json`, binds every installed ParticleGAN Python file to
the published wheel SHA256
`b1fd33133356d2ac0584db23188ad116075f4bf0be4cf3f0ed53c083e000f845` and release-tag
commit `d7bc20166c022fe037cfb9266272d2cdd7624399`. Any other installed
ParticleGAN version, source profile or modified source file is refused before a
model factory runs. `requirements-engine.txt` hash-pins that wheel; after
installing PyTorch and NumPy, verify it with
`pip install --no-deps --require-hashes -r requirements-engine.txt`.

## Continuous features

Use `response_domain="continuous_vector", penalty_input_domain="continuous"`
for native continuous features, including widths 170, 1024, 2048 and larger.
This uses the released upstream domain declaration; no response is reshaped or
reinterpreted as categorical. The default `auto` setting retains its width gates.
Image/field responses continue using `auto`. Old unreleased continuous overlays
are not aliases for this release.

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.10.1 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## Checkpoints and earlier core lines

Core 0.15.0 is a fresh training cohort. The engine checkpoint schema is
`conditional-engine-v1` (adapter export: `conditional-engine-adapter-v1`), and
the signature binds the source hash of `particle_sliders.engine`. Checkpoints,
adapter exports and cached responses written by core 0.14.x or earlier resume
only with their original package/source pins; there is no automatic migration.
Version-named entrypoints of earlier lines were removed rather than aliased, and
no old ParticleGAN release is rebound to this engine. Historical release
manifests, receipts and scientific results remain in the repository history and
in their original pinned core commits.

Training-state source/version/domain checks and transactional restore remain in
place. CPU tests cover actual updates, exact split continuation, inference export,
wide-vector declaration, and rejection of mismatched checkpoint identities.
These tests do not establish improvement on a pretrained model. Products adopt
this source explicitly and evaluate fresh runs separately.
