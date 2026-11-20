# Pinned ParticleGAN 0.10.1 compatibility

Core 0.14.2 uses published `particlegan==0.10.1`.
The `ConditionalPG09`, backend, lifecycle and command APIs remain unchanged.
`particle_sliders.pg0101` supplies the explicit versioned entrypoint. Its source
manifest binds every installed ParticleGAN Python file to the published wheel
SHA256 `b1fd33133356d2ac0584db23188ad116075f4bf0be4cf3f0ed53c083e000f845`
and release-tag commit `d7bc20166c022fe037cfb9266272d2cdd7624399`.

## Continuous features

Use `response_domain="continuous_vector", penalty_input_domain="continuous"`
for native continuous features, including widths 170, 1024, 2048 and larger.
This uses the released upstream domain declaration; no response is reshaped or
reinterpreted as categorical. The default `auto` setting retains its width gates.
Image/field responses continue using `auto`. Old unreleased continuous overlays
are not aliases for the new release.

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.10.1 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## Checkpoints and compatibility

A new ParticleGAN version is a fresh training cohort. Existing PG0.9.5/earlier
checkpoints require their original package/source pins. The `pg095`, `pg094` and `pg093`
explicit entrypoints refuse rather than silently rebinding an old version.
Historical release manifests and scientific results are preserved unchanged.
Training-state source/version/domain checks and transactional restore remain in
place. CPU tests cover actual updates, exact split continuation, inference export,
wide-vector declaration, and rejection of mismatched checkpoint identities.
These tests do not establish improvement on a pretrained model.

Optional HNDL/HyperGAN pins are unchanged. This pin does not qualify a native model result. Products adopt this source
explicitly and evaluate fresh runs separately.
