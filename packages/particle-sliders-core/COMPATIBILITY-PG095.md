# Core 0.13.0 and ParticleGAN 0.9.5

This release updates the shared slider engine to published `particlegan==0.9.5`.
The `ConditionalPG09`, backend, lifecycle and command APIs remain unchanged.
`particle_sliders.pg095` supplies the explicit versioned entrypoint. Its source
manifest binds every installed ParticleGAN Python file to the published wheel
SHA256 `c1c0de7a635ac1b1634ba158eed050b6f6aed993e01f9b463435430c77d846ae`
and release-tag commit `1b8355b184b21bb7d81104ee590813c262f6df0a`.

## Continuous features

Use `response_domain="continuous_vector", penalty_input_domain="continuous"`
for native continuous features, including widths 170, 1024, 2048 and larger.
This uses the released upstream domain declaration; no response is reshaped or
reinterpreted as categorical. The default `auto` setting retains its width gates.
Image/field responses continue using `auto`. Old unreleased continuous overlays
are not aliases for the new release.

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.9.5 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## Checkpoints and compatibility

A new ParticleGAN version is a fresh training cohort. Existing PG0.9.4/earlier
checkpoints require their original package/source pins. The `pg094` and `pg093`
explicit entrypoints refuse rather than silently rebinding an old version.
Historical release manifests and scientific results are preserved unchanged.
Training-state source/version/domain checks and transactional restore remain in
place. CPU tests cover actual updates, exact split continuation, inference export,
wide-vector declaration, and rejection of mismatched checkpoint identities.
These tests do not establish improvement on a pretrained model.

Optional HNDL/HyperGAN pins are unchanged. This change does not qualify a new
native Fibo, audio, image or language-model result. Products adopt this source
explicitly and evaluate fresh runs separately.
