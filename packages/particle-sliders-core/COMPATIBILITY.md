# ParticleGAN compatibility

Core 0.15.3 runs on the published `particlegan==0.12.0` wheel. The shared
engine is `particle_sliders.engine` (`ConditionalParticleGAN`,
`validate_engine_domain`, `engine_options`). Its `engine_release.json` manifest
binds the installed ParticleGAN Python files to wheel SHA-256
`8ae6ea792d61f7d3e0a6d19b90cd403b50b7eba54d6c5bac48e09324eeac418d` and
release commit `3256a05867688ab7ec0e5b4f50354c14d6972e6f`. The engine checks
the version, source profile and file hashes before opening a model. After
installing PyTorch and NumPy, install the verified wheel with
`pip install --no-deps --require-hashes -r requirements-engine.txt`.

## Continuous features

Use `response_domain="continuous_vector", penalty_input_domain="continuous"`
for native continuous features, including widths 170, 1024, 2048 and larger.
This uses the released upstream domain declaration; no response is reshaped or
reinterpreted as categorical. The default `auto` setting retains its width gates.
Image/field responses continue using `auto`. Old unreleased continuous overlays
are not aliases for this release.

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.12.0 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## ParticleGAN 0.12.0

Upstream selects Unified v12 when a caller omits the recipe. The conditional
slider engine selects its recipe explicitly. Use `bispf_optimized` for the
existing default or `preset="empf", optimizer_dense_profile="recipe"` for stock EMPF.
[ParticleGAN release](https://github.com/HyperGAN/ParticleGAN/releases/tag/v0.12.0)

## Checkpoints and earlier core lines

Core 0.15.x is a fresh training cohort. The engine checkpoint schema is
`conditional-engine-v1` (adapter export: `conditional-engine-adapter-v1`), and
the signature binds the source hash of `particle_sliders.engine`. Resume and
load adapter exports with their original core source and ParticleGAN pins,
including packets from the earlier BISPF-only core 0.15.2 revision. The package
version alone does not match source identity. Reuse earlier cached responses
with their original package/source pins. Historical release manifests, receipts
and scientific results remain in the repository history and in their original
pinned core commits.

Training-state source/version/domain checks and transactional restore remain in
place. CPU tests cover actual updates, exact split continuation, inference export,
wide-vector declaration, and rejection of mismatched checkpoint identities.
These tests do not establish improvement on a pretrained model. Products adopt
this source explicitly and evaluate fresh runs separately.
