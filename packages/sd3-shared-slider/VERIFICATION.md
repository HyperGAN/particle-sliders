# SD3 shared preview verification

Bounded software qualification on Pop, Python3.12.15/Torch2.14.1+cpu. No model
weights were downloaded. All native models were tiny randomly initialized actual
`diffusers.SD3Transformer2DModel` objects with two joint-attention blocks, native
pooled/text conditioning and a nonsquare `[1,2,8,6]` latent. No GPU was initialized.
These results do not qualify full/pretrained SD3, rendering, mixed precision or
native training campaigns.

Source base: `3c25f785d14073e6f627c0fa9d2297a041c79913`.
The original `conceptmod/textsliders/train_lora_sd3.py` remains byte-identical.
Only the additive shared entry point and this separate package change. Historical
training dependencies, prompt-pair loss, objectives and filenames remain untouched.
The preview declares a different positive-response adversarial objective and
`adapter_gan` arm, with no routed bank or automatic checkpoint conversion.

Exact common cohort:

- Core source `7a8a321d04323d49319d0b08e6093a0ddce36b27` (draft PR148 lifecycle follow-up).
- All49 installed shared `.py`/`.json`/`.hndl` files checked by `core_lock.json`.
- Official PG0.9.4/release-0.9.4/auto, with unchanged native-field admission.
- HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean VCS build provenance.
- HNDL0.10.0 via `verified_builder()`, not the legacy direct0.8.2 builder.
- Torch2.14.1+cpu, Diffusers0.38.0, Transformers4.57.6, safetensors0.8.0.

The new projection executes the actual HyperGAN builder, which returns an actual
`hndl.torch.Network`. Projection source, resolved plan, dimensions and runtime
identity are recorded in inspectable adapter representation/export provenance.
The shared `HNDLFieldCritic` executes its spatial graph. No GAN losses, penalty
formula, optimizer, update loop, EMA or packet implementation is copied here.

Installed wheel and installed sdist suite: **40 passed** per installation.
Tests cover matched-weight LoRA outputs/input-and-parameter VJPs, native SD3
joint-attention outputs/VJPs, spatial-critic outputs/VJPs and penalty-relevant
second derivatives, genuine shared updates, exact full/split resume and EMA,
strength0/0.5/1, checksum/cohort/optimizer/shape/EMA rejection, late load-hook
rollback, partial attach and every-cleanup-on-failure, borrowed modes/gradients/
checkpoint-function restoration and failing observation cleanup. Actual source
file changes for core/HNDL/HyperGAN/native Diffusers/projection are rejected before
native hooks. Changed cache/model/rank/geometry gates also run before mutation.

Fresh-process raw and EMA v2 inference passes after removing the row cache and
training checkpoint. The process constructs native model/adapter response only;
the training factory is replaced by an assertion and no game/critic is created.
Installed CLI train/split/resume writes real shared training/v1/v2 receipts.
A representative historical native tensor-state envelope is rejected before the
native factory; it is not represented as a trained historical artifact.

Commands, run from the isolated task workspace (`$PWD` is task-25):

```sh
.venv/bin/python -m build --no-isolation particle-sliders/packages/sd3-shared-slider --outdir evidence/sd3-dist
.venv/bin/python -m pip install --no-deps --force-reinstall evidence/sd3-dist/particle_sliders_sd3_shared-0.1.0-py3-none-any.whl
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -c /dev/null --import-mode=importlib particle-sliders/packages/sd3-shared-slider/tests -q
.venv/bin/python -m pip install --no-deps --no-build-isolation --force-reinstall evidence/sd3-dist/particle_sliders_sd3_shared-0.1.0.tar.gz
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -c /dev/null --import-mode=importlib evidence/sdist-extracted/particle_sliders_sd3_shared-0.1.0/tests -q
.venv/bin/python -m pip check
```

Both test commands use installed package sources. Repository root pytest.ini
otherwise inserts the historical core source ahead of the installed modern wheel;
`-c /dev/null` deliberately disables that incompatible source-path override.
`test_installed_source_identity` asserts both packages resolve in site-packages.

Selected installed common-core regressions: **100 passed, 1 skipped**. The skip is
the optional Fibo derivative fixture, not an SD3 fixture. Shared tests require
`--import-mode=prepend` for their existing sibling-test helper imports, with no
core src override:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -c /dev/null --import-mode=prepend .deps/packages/particle-sliders-core/tests/test_lifecycle_restore_failures.py .deps/packages/particle-sliders-core/tests/test_pg094_admission.py .deps/packages/particle-sliders-core/tests/test_pg094_contract.py .deps/packages/particle-sliders-core/tests/test_fibo_hndl.py .deps/packages/particle-sliders-core/tests/test_conditional_pg09.py .deps/packages/particle-sliders-core/tests/test_image_lifecycle.py .deps/packages/particle-sliders-core/tests/test_compact_inference.py -q
```

Clean public CPU resolution (`--ignore-installed`) succeeds with39 distributions,
including the exact publicly reachable core and HyperGAN VCS commits above:

```sh
.venv/bin/python -m pip install --dry-run --ignore-installed --report evidence/public-cpu-resolution.json --cache-dir /tmp/task25-pip-cache --extra-index-url https://download.pytorch.org/whl/cpu particle-sliders/packages/sd3-shared-slider
```

`pip check`: **No broken requirements found**. The core wheel and this package's
wheel/sdist were built locally, not released. Raw logs, resolution JSON, installed
freeze, archive hashes and publication receipts remain in task-25/evidence.

Merge remains gated on exact central core admission plus root's actual-diff Claude
review, fixes and return review. This draft adds no release/CI workflow, dependency
update to an existing environment, model/package release, deployment or GPU use.
