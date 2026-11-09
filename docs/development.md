# Developing the shared core

Use Python 3.11+ and an environment compatible with PyTorch 2.6–2.x and exact
`particlegan==0.9.5`. Native model dependencies belong in their
[product repositories](products.md).

## Install and run tests

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './packages/particle-sliders-core[test,serialization]'
CUDA_VISIBLE_DEVICES='' python -m pytest
```

The default suite targets the installed shared core. Do not add `src/` to
`PYTHONPATH`: that exposes excluded historical product code and defeats the
packaging checks. Some tests require optional native dependencies and skip when
they are absent. A CPU fixture checks software behavior, not GPU fit or media
quality.

## Build and verify the distribution

```sh
python -m pip install build
python -m build packages/particle-sliders-core
```

The wheel uses an explicit module allowlist in
[`_core_build.py`](../packages/particle-sliders-core/_core_build.py). Strict editable
installs use the same boundary. Adding source does not automatically add a
public module. Keep model loaders, prompts and campaign code out of that list.
`tests/test_build_boundary.py` fails when a packaged module imports a
`particle_sliders` module the wheel omits.

For a release check, install the built wheel in a clean environment and run the
core tests from outside the checkout. Verify that excluded native products and
legacy `conceptmod` aliases cannot be imported. The packaging tests also cover
strict editable installation, uninstall and coexistence with a separately owned
Conceptmod installation.

## Where changes belong

- Shared game and response contracts: `packages/particle-sliders-core/src/particle_sliders`.
- Product models, target selection, recipes and native generation: the product repo.
- Shared command planning and worker ownership: `particle_sliders.execution`.
- Search protocols and arithmetic: `particle_sliders.parameter_search`.
- Current explanations and API navigation: this docs directory and package guides.

Keep the [math](math.md), [engine contract](../packages/particle-sliders-core/CONDITIONAL-PG09-CONTRACT.md)
and [distribution boundary](../packages/particle-sliders-core/SHARED-CORE-BOUNDARY.md)
aligned with the implementation. Use the upstream ParticleGAN APIs for optimizer
and loss behavior rather than copying a local formula. Fresh configs explicitly
use `ema_decay=None`; no model-weight EMA is permitted.

Source-bound checkpoint tests deliberately reject changed identities. Preserve
old fixtures and receipts; use a new cohort for a changed implementation rather
than weakening validation. Historical tests outside the default suite apply to
their original complete checkout; see [history](history.md).
