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

The wheel owns no files under `conceptmod/`. A uniquely named, RECORD-tracked
`particle_sliders_legacy_imports.pth` registers a narrowly scoped import finder
at normal Python startup. It imports only the lightweight package root and
standard-library alias helpers; it does not load Torch, ParticleGAN, native
backends, access the network, or write application files. Each research alias returns the
canonical module object after importlib initializes a disposable alias module,
so it never temporarily changes the canonical module's `__spec__`.
Imports work in either order, including nested packages and private attributes.
Monkeypatches and a Glimmer adapter subclassing `LoRAModule` see one identity.
The root API exports remain available lazily, so a LoRA/bridge import does not
require initializing ParticleGAN, HyperGAN/HNDL or Diffusers.

This startup registration is needed for legacy-first imports when the separate
`conceptmod` distribution is installed: putting a shim in its package would
make both distributions own the same file, and a namespace subdirectory cannot
extend an unrelated regular package. Existing regular, namespace and editable
providers of the `conceptmod` root get first refusal. Only when no provider
supplies that root does the final fallback finder synthesize a package in
memory. The separate project's root and native `conceptmod.backends` API remain
owned by that project. The research `textsliders`/`notrigger` names are reserved
compatibility routes; installing an unrelated provider of those same legacy
subtrees is not supported. The repository-root shim remains for source use and
is deliberately excluded from wheels.

The small PEP 517 backend delegates to setuptools and adds the hook at the wheel
root with a verified RECORD hash/size. This avoids setuptools `data-files`,
which would put a `.pth` under the environment prefix rather than site-packages.
It handles regular and PEP 660 editable wheels alike, and the source archive
contains the backend and hook. Pip uninstall removes the hook with its owning
distribution. Tests cover both install orders, legacy-first imports, native
API preservation, independent uninstalls, and editable installation/removal.

Python `-S` disables site processing, and adding a `pip --target` directory with
`PYTHONPATH` alone does not process its `.pth` files. In those environments,
use canonical imports or explicitly call
`particle_sliders._legacy_namespace.install()` before a legacy import. Restart
Python after installs or uninstalls; already-running interpreters retain their
module and importer state. No global Git or interpreter settings are changed.

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
In a checkout, the old analysis imports alias these modules; the wheel does not
own the generic `analysis` namespace. Old full-object pickles naming those
repository-only classes still require a checkout. These remain the historical bridge
losses; they are not silently switched to the current shared-core formulation.

Preset config YAMLs that embed
`packages/particle-sliders-core/src/particle_sliders/...` paths remain
source-checkout presets. Their relative prompt paths are not portable when a
wheel is installed elsewhere. Wheel users should use the backend's packaged
default prompt path or an explicit absolute `--prompts_file`, and separately
qualify model-specific config adoption. For example, `config-sana.yaml` embeds
a checkout-relative prompt path, while its packaged `data/prompts-sana.yaml`
resource is available from `importlib.resources.files('particle_sliders.textsliders')`.
This migration does not claim that arbitrary research YAMLs or workstation
campaigns are runnable from an installed wheel.

`locked_provenance.json` records the canonical teacher file relative to the
package and its verified hash, preserving the historical repository path,
hash and commit separately. The endpoint source check runs in a checkout and
an installed wheel: a missing source fails rather than skips, and its six
teacher-function ASTs must match the extracted teachers exactly.

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
  packages/particle-sliders-core/tests/test_research_namespace.py \
  packages/particle-sliders-core/tests/test_namespace_installation.py \
  packages/particle-sliders-core/tests/test_endpoint_game.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m build --no-isolation packages/particle-sliders-core
```

Run the focused file again from a temporary directory in a fresh environment
with the built wheel installed and no repository on `sys.path`. It covers both
import orders, nested modules, reload, monkeypatch identity, old pickle globals,
a legacy full-object checkpoint, exact state-dict restoration, packaged prompts
and license, optional-runtime isolation and legacy module execution.
For the installation matrix outside a checkout, set `PARTICLE_SLIDERS_TEST_WHEEL`
to the built wheel and `PARTICLE_SLIDERS_TEST_PROJECT` to the source package
directory for its separate editable-install check. The fixture owns the same
root file as the separate Conceptmod distribution and exposes a native backend;
no files in either installed distribution may overlap.

This is naming/packaging work. The original Concept Sliders research credit,
Rohit Gandikota's copyright and MIT notices remain in the repository and wheel.
See [Lineage and license](../README.md#lineage-and-license).

## Publishing this migration for review

Check the push destination before publishing a task branch. This checkout
inherited `push.default=tracking`; an unqualified source branch can therefore
target its tracked `main` branch. Inspect the setting without changing it, use
an explicit destination refspec for both the dry run and the push, and verify
the remote refs afterward:

```bash
git config --get push.default
git ls-remote origin refs/heads/main refs/heads/codex/particle-sliders-namespace
git push --dry-run origin HEAD:refs/heads/codex/particle-sliders-namespace
git push --set-upstream origin HEAD:refs/heads/codex/particle-sliders-namespace
git ls-remote origin refs/heads/main refs/heads/codex/particle-sliders-namespace
```

The dry-run destination must be `codex/particle-sliders-namespace`, and the
remote `main` SHA must remain the pre-push SHA. For this draft, the reviewed
initial branch head was `bbda8ed17f1b3ebbe82e13608cef9db57326f00b`; use the
draft PR's current exact reviewed head after any review fixes. The restored
`main` is `e4a9273dce38bb47ec9787123c8b1faa38e1bf1c`. This procedure does not
require changing global or unrelated Git settings.
