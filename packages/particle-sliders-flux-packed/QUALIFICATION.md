# Installed CPU prerequisite qualification

This is the bounded correction to reviewed source `3d1aefd77038d9c67becd7df5ff5acda7e3e872d` in draft PR157. Shared training remains unavailable. No native/model/GPU qualification or game/resume/EMA/export claim is made.

## Actual admission evidence

`preflight_training(shape, model=staged_model)` validates the optional model read-only, then calls the existing shared `validate_pg09_domain(shape, response_domain="continuous_field")` with the **unchanged** native packed rank3 shape. It preserves the actual `continuous_field requires genuine native 4D/5D shape` rejection as its cause. This is an unsupported-geometry probe, not a declaration that packed tokens are supported fields. No new shared domain string or shape conversion exists. A separate, unrelated genuine-field positive control proves that the same existing API accepts its documented rank4 input.

Wrong installed PG versions or Python source bytes raise `FluxSourceMismatch`, retaining the original runtime/source failure; they are not mislabeled as a packed-rank refusal. Tests pass an actual tiny Flux model and compare every state_dict tensor, gradient/flag, mode, existing forward/pre-forward hook and CPU Torch RNG across refusal.

`FluxLoRA` constructs all branch graphs/initializations within one private CPU Torch RNG fork. The caller's RNG is restored on success and graph-build failure, while the private stream advances between branches. Tests check both paths and distinct equal-shape branch initializations. The low-level `HNDLLoRA` constructor also forks/restores CPU Torch RNG on success/failure. A bank-private generator carries sequential initialization state between these isolated branch forks.

## Installed receipts

These checked-in files are literal command outputs:

- [Torch2.14.1 CPU installed wheel](qualification/installed-wheel-torch214.txt):43 passed in4.65s.
- [Torch2.6.0 CPU installed wheel](qualification/installed-wheel-torch26.txt):43 passed in5.41s.
- [Torch2.14.1 CPU installed sdist](qualification/installed-sdist-torch214.txt):43 passed in4.56s.
- [Torch2.14 pip check](qualification/pip-check-torch214.txt) and [Torch2.6 pip check](qualification/pip-check-torch26.txt):No broken requirements found.
- [Torch2.14 installed identity/payload](qualification/runtime-torch214.json) and [Torch2.6 installed identity/payload](qualification/runtime-torch26.json):the exact VCS origins, versions, verified builder identity and hashes of all five installed preview `.py`/`.json`/`.hndl` payload files, byte-matched to repository source.

The cohort remains core `7a8a321d04323d49319d0b08e6093a0ddce36b27`/PG0.9.4, actual clean HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`/HNDL0.10.0, Diffusers0.38.0 and Transformers4.57.6. All model API tests use randomly initialized tiny native FluxTransformer2DModel fixtures. They do not qualify pretrained execution.

Literal commands, run from the task workspace (the checkout is `flux-shared/`):

```sh
.venv-flux/bin/python -m build --no-isolation flux-shared/packages/particle-sliders-flux-packed --outdir evidence/return-dist
.venv-flux/bin/python -m pip install --no-deps --no-index --force-reinstall evidence/return-dist/particle_sliders_flux_packed_preview-0.1.0-py3-none-any.whl
.venv-flux26/bin/python -m pip install --no-deps --no-index --force-reinstall evidence/return-dist/particle_sliders_flux_packed_preview-0.1.0-py3-none-any.whl
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-flux/bin/python -m pytest -c flux-shared/packages/particle-sliders-flux-packed/pyproject.toml flux-shared/packages/particle-sliders-flux-packed/tests -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-flux26/bin/python -m pytest -c flux-shared/packages/particle-sliders-flux-packed/pyproject.toml flux-shared/packages/particle-sliders-flux-packed/tests -q
.venv-flux/bin/python -m pip install --no-cache-dir --no-index --no-deps --no-build-isolation --force-reinstall evidence/return-dist/particle_sliders_flux_packed_preview-0.1.0.tar.gz
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-flux/bin/python -m pytest -c flux-shared/packages/particle-sliders-flux-packed/pyproject.toml flux-shared/packages/particle-sliders-flux-packed/tests -q
.venv-flux/bin/python -m pip check
.venv-flux26/bin/python -m pip check
```

## Exact49-file shared check

`checked_builder` compares `sources.json:core_sources` against every file beneath the installed **`particle_sliders/` package root** whose suffix is `.py`, `.json` or `.hndl`:35 Python files,12 JSON files and2 HNDL graphs. The dictionary comparison rejects missing/extra files of those suffixes and differing bytes. It does **not** check the core LICENSE, `.dist-info`/wheel metadata, bytecode or other suffixes as part of this49-file check. Historical manifest files in this list are identity-bound data; checking them does not admit their source profiles. The actual HyperGAN/HNDL runtime closure is verified separately by the shared `verified_builder`.

Exact names relative to installed `particle_sliders/`:

```text
__init__.py
_training.py
audio_closure_provenance.json
bispf_game.py
bispf_provenance.json
distillation.py
endpoint_game.py
fibo_critic.hndl
fibo_hndl.py
fibo_hndl_010_sources.json
fibo_hndl_sources.json
fibo_native.py
fibo_projection.hndl
fibo_provenance.json
field_critic.py
formulation.py
formulation_game.py
grad_regularizers.py
inference.py
legacy_compat.py
legacy_stamp.json
lifecycle.py
locked_provenance.json
music3_native.py
native_lora.py
optimizer_policy.py
peft_scaling.py
pg09.py
pg093.py
pg093_release.json
pg094.py
pg094_release.json
pg09_audio.py
pg09_audio_cli.py
pg09_backends.py
pg09_continuous_profile.json
pg09_hndl.py
pg09_images.py
pg09_readout_critic.py
pg09_release_sources.json
provenance.json
qwen_native.py
recipe.py
reference.py
response.py
routed_projection.py
session_cli.py
teachers.py
yue2_native.py
```

Six separately checked native Diffusers source paths:

```text
models/attention.py
models/attention_dispatch.py
models/embeddings.py
models/modeling_utils.py
models/normalization.py
models/transformers/transformer_flux.py
```

The product's `lora.hndl` graph is checked separately. Root owns Claude actual-diff fixes and RETURN review before any landing decision; this follow-up adds no training registration, native admission, release, deployment or GPU work.
