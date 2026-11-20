# ParticleGAN compatibility

Core 0.15.1 runs only on published `particlegan==0.11.0`. Adopting a later ParticleGAN
release means regenerating `engine_release.json` and its pinned identity; module
and symbol names stay unchanged. The shared engine is `particle_sliders.engine`
(`ConditionalParticleGAN`, `validate_engine_domain`, `engine_options`). Its source
manifest, `engine_release.json`, binds every installed ParticleGAN Python file to
the published wheel SHA256
`218ac69efd42e092367b4c7a5c4abb4685dc63a73ccc23223a8bb4dd1543e327` and release-tag
commit `b86f0379b4d887f4cde88d686aedc04661a2b61b`. Any other installed
ParticleGAN version (including 0.10.1), source profile or modified source file is refused before a
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

The shared engine calls `get_recipe()` for fresh runs. ParticleGAN 0.11.0 rejects
bare/partial `Recipe(...)`; the existing optimizer-profile helper passes complete
recipe fields through dataclass replacement. Public losses, penalties, prior
regularizers and optimizer mathematics remain upstream.

## Changes from ParticleGAN 0.10.1 to 0.11.0

The 0.11.0 wheel keeps all 87 Python files of 0.10.1, changes five
(`__init__.py`, `recipes.py`, `training.py`, `recipe_schedules.py`,
`optim/dualnorm.py`) and adds 76 files for new critic formulations and
optimizer families. `init.py` is unchanged, so scoped initialization keeps its
recorded initializer digest. For `bispf_optimized` and `bispf`, every recipe field
that 0.10.1 resolves keeps its value, and the shared dense policy resolves the same
way. 0.11.0 adds 16 recipe fields (`t1_*`, `t1c_*`, `rgdt_*`). They are read only by
the new `t1_coupling`/`rgdt` formulations and by the deconfliction branch when
`t1c_grad_deconflict > 0`, which is 0 here. The new optimizer paths are inactive
for these recipes. CPU parity runs of the engine produced bitwise-identical
network, critic, prior, optimizer state and metrics on both releases. Checkpoints
from 0.15.0 (ParticleGAN 0.10.1) still do not resume on 0.15.1: the signature
binds the ParticleGAN version and source digest, and the recorded recipe now has
16 more keys.

## Checkpoints and earlier core lines

Core 0.15.x is a fresh training cohort. The engine checkpoint schema is
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
