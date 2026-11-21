# ParticleGAN compatibility

Core 0.15.2 runs on the published `particlegan==0.11.1` wheel. The shared
engine is `particle_sliders.engine` (`ConditionalParticleGAN`,
`validate_engine_domain`, `engine_options`). Its `engine_release.json` manifest
binds the installed ParticleGAN Python files to wheel SHA-256
`ce9dd51cf2bf0067751c9e6409d05d149ce41c1df98fe3edb0412e3df430de52` and
release commit `d5dec2fe005305082acf382f0939a4bdb51c7131`. The engine checks
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

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.11.1 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## ParticleGAN 0.11.1

The release updates Unified's critic, loss, penalty and optimizer, and fixes
critic hooks that could retain GPU memory. The shared engine explicitly selects
`bispf_optimized` by default and admits `bispf` as an explicit recipe. Their
upstream source and recipe fields are unchanged across the patch release. The
scoped initializer retains the same upstream file digest. Select stock EMPF
explicitly with `preset="empf", optimizer_dense_profile="recipe"`.
[Release notes](https://github.com/HyperGAN/ParticleGAN/releases/tag/v0.11.1)

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
