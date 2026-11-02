# Python namespace migration

`particle_sliders` is the canonical Python namespace. The existing
`particle-sliders-core` distribution now includes the research backends alongside
the shared core. There is no second distribution competing to own the same
regular Python package.

```bash
pip install -e './packages/particle-sliders-core[serialization]'
```

```python
from particle_sliders.textsliders.particle_bridge_gan import update
from particle_sliders.textsliders.lora import LoRAModule
```

For a Git pin, retain the package subdirectory:

```text
particle-sliders-core[serialization] @ git+https://github.com/HyperGAN/particle-sliders.git@<full-reviewed-commit>#subdirectory=packages/particle-sliders-core
```

The import change is `conceptmod.textsliders` → `particle_sliders.textsliders`
and `conceptmod.notrigger` → `particle_sliders.notrigger`. Shared-core imports
such as `particle_sliders.lifecycle` already have their canonical name.
Code, backend prompt files and optional research modules now live under
`packages/particle-sliders-core/src/particle_sliders/` and ship in both wheel and
source archive. Native libraries are still installed per backend; this does not
make research modules standalone native model products. Campaign scripts that
read repository `analysis/`, `scripts/`, caches or local app files still require
a source checkout and those original artifacts.

Prefer module commands after installation, for example:

```bash
python -m particle_sliders.textsliders.distribution_probe --help
```

Legacy `python -m conceptmod.textsliders.<command>` is also supported. The old
`python conceptmod/textsliders/<file>.py` filesystem paths have moved; update
launchers to module commands or the new source path. Historical campaign
receipts, source pins, snapshots and external release cards are not rewritten.

## Compatibility

The legacy `conceptmod` package installs a narrowly scoped import finder for
its `textsliders` and `notrigger` trees. Each alias returns the canonical module
object. It does not execute the same source twice or create wrapper subclasses.
Imports work in either order, including nested packages and private attributes.
Monkeypatches and a Glimmer adapter subclassing `LoRAModule` see one identity.
The root API exports remain available lazily, so a LoRA/bridge import does not
require initializing ParticleGAN, HyperGAN/HNDL or Diffusers.

Old qualified pickle globals (`conceptmod.textsliders.lora.LoRAModule`, for
example) resolve through the alias. New pickles record canonical names and need
this revision or newer; an older checkout cannot necessarily load them.
Tensor state-dict keys, safetensors metadata keys, sidecars and existing format
labels containing `conceptmod` stay unchanged. Existing identity/hash checks
still apply: this source migration does **not** bless a source-bound training
checkpoint for resume under a different source pin. Preserve its exact original
environment and source cohort unless that product qualifies a migration.
PyTorch's restricted `weights_only` allowlist is separate from import resolution;
this change does not relax checkpoint-loading policy.

The historical research bridge's existing `analysis.slider2d.adv` and
`grad_regularizers` implementations move unchanged apart from import names into
`particle_sliders.textsliders._slider2d`, so the bridge works outside a checkout.
The old analysis imports alias these modules. They remain the historical bridge
losses; they are not silently switched to the current shared-core formulation.

## Downstream adoption

The immediate consumer is
[HyperGAN/glimmer-particle-sliders](https://github.com/HyperGAN/glimmer-particle-sliders):
its Trainer calls the research bridge's `update` directly and its adapter
subclasses `LoRAModule`. Adopt a separately reviewed pin, replace those two
import prefixes, and run Glimmer's own CPU and source/checkpoint identity checks.
Do not change an active submission release merely to follow this rename.

Additional concrete adoption surfaces found in existing local product revisions:

| Product and inspected revision | Surface | Effect when adopting this source |
|---|---|---|
| [Anima](https://github.com/HyperGAN/anima-particle-sliders/tree/36ad126761056fbb06d6b02574149f0634f50033) | `scripts/anima_verify_normalization.py`, `anima_verify_reference.py`, `anima_verify_update.py` refer to old source paths. | These are reference/source checks. Preserve their historical pin; update paths only in a separately qualified adoption. |
| [Supra](https://github.com/HyperGAN/supra-particle-sliders/tree/a569e00aa78ea8afeb83c75482f6d7a83cf2d376) | `supra/runtime.py:MODEL_SOURCE` loads `conceptmod/textsliders/supra_model.py` by filename and checks `backend.lock.json`. | Import aliases do not restore this filename. A new pin needs explicit path and source-lock qualification. |
| [Krea2](https://github.com/HyperGAN/krea2-particle-sliders/tree/360708fbb20601efebefb07be627eb60724e3061) | CPU source guards assert the old research import/path is absent. | No legacy runtime import found in this inspected revision; existing shared-core names remain canonical. |

Other model products in HyperGAN include Iris, Qwen Image, Fibo, Z-Image,
TinyQwen, Music3, YuE2, Cascade, SD, MOSS and Ming. Their individual runtime pins
were not exhaustively audited here. Products using only `particle_sliders` core
APIs need no import rename; backend users, vendored copies and source-path
launchers should check their pinned sources. This PR changes none of these
repositories, their pins, releases, weights or running services. Repository-name
discovery alone is not proof of a legacy import.

## Verification

Run the focused import/checkpoint tests and the repository CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q \
  packages/particle-sliders-core/tests/test_research_namespace.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m build --no-isolation packages/particle-sliders-core
```

Run the focused file again from a temporary directory in a fresh environment
with the built wheel installed and no repository on `sys.path`. It covers both
import orders, nested modules, reload, monkeypatch identity, old pickle globals,
a legacy full-object checkpoint, exact state-dict restoration, packaged prompts
and license, optional-runtime isolation and legacy module execution.

This is naming/packaging work. The original Concept Sliders research credit,
Rohit Gandikota's copyright and MIT notices remain in the repository and wheel.
See [Lineage and license](../README.md#lineage-and-license).
