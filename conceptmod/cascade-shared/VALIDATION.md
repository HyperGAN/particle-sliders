# Candidate validation receipt

Base: `3c25f785d14073e6f627c0fa9d2297a041c79913`. Only the isolated Cascade
package is added. The CPU CI workflow remains a local proposal in original
commit `6be54817efbfd43087b45f347878e8be2d240298`, not part of this source-only
publication. Core candidate:
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` (draft #148). Exact dependency
admission remains pending with the core owner. Root must run Claude actual-diff
review, corrections, and return review before landing. No merge was requested.

On authorized Pop, in a new checkout/branch `codex/cascade-shared-cohort`, Python
3.14.4, public CPU PyTorch 2.9.0+cpu, Diffusers 0.36.0, Transformers 4.57.1.
No applicable `AGENTS.md` or software-engineering skill was present in the
checkout or exposed selected-environment skill catalog. No sibling worktree,
shared core contract, default environment, GPU or full model was modified.

Commands below were run from the task workspace (the checkout is
`cascade-cohort`). All build/install/test commands returned 0:

```bash
python3 -m venv .venv-cascade
.venv-cascade/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv-cascade/bin/python -m pip install --force-reinstall --no-deps \
 'particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231#subdirectory=packages/particle-sliders-core'
.venv-cascade/bin/python -m build --no-isolation ./cascade-cohort/conceptmod/cascade-shared --outdir ./cascade-dist
.venv-cascade/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation cascade-dist/particle_sliders_cascade-0.1.0-py3-none-any.whl
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .venv-cascade/bin/python -m pytest --noconftest -c /dev/null --import-mode=importlib cascade-cohort/conceptmod/cascade-shared/tests -q
# 13 passed, 2 warnings in 4.96s
.venv-cascade/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation cascade-dist/particle_sliders_cascade-0.1.0.tar.gz
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .venv-cascade/bin/python -m pytest --noconftest -c /dev/null --import-mode=importlib cascade-cohort/conceptmod/cascade-shared/tests -q
# 13 passed, 2 warnings in 4.26s
.venv-cascade/bin/python -m pip check
# No broken requirements found.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .venv-cascade/bin/particle-slider train --interface response \
 --config cascade-cohort/conceptmod/cascade-shared/tiny-config.json --out cascade-cli-smoke
# 3 actual shared PG updates; training.pt, adapter.pt, manifest.json, metrics.jsonl
python3 -m venv .venv-resolution
.venv-resolution/bin/python -m pip install --dry-run --report cascade-resolution.json \
 --extra-index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu ./cascade-cohort/conceptmod/cascade-shared
# 39 distributions resolve from public indexes/exact Git commits; no NVIDIA packages
```

Initial unisolated pytest discovered the repository `pytest.ini`, which puts its
old core source on `sys.path`; it correctly rejected HNDL 0.10 versus old 0.9.
The commands above disable that source override and assert both package/core
imports come from site-packages. The seven key installed core source/manifest
files were also compared byte-for-byte to `git show 4abc0eb:...`. The installed
backend was compared byte-for-byte to this candidate source. The shared builder
verified HNDL source closure and HyperGAN clean immutable build provenance.
The two warnings are upstream CPU CUDA-autocast disabling and a TF32 API warning;
no CUDA execution occurred. A sandboxed build-isolation attempt could not resolve
the public index; installing public build tools via authorized network access and
using local no-isolation builds resolved it.

`validation.json` records installed versions, exact Git receipts, installed
source hashes, public dependency URLs/hashes, PG domain admission, builder
identity, wheel/sdist hashes, and the untouched historical trainer SHA256.
Raw local build/install/test output is retained as `cascade-verify.log` in the
task workspace; its hash is recorded there. CLI metrics have three rows and the
final row has `step=3`, `g_adv=0.689228355884552`, `d_pen=1.8908452830146416e-06`.
These values demonstrate execution, not model quality.

Coverage: real tiny native Stage C spatial response; exact zero baseline and
finite signed strengths; HNDL LoRA matched outputs and input/weight VJPs; shared
HNDL/native critic matched outputs, first gradients and second derivatives;
actual adapter changes; exact interrupted resume including EMA; raw/EMA export;
native-only inference with game/critic/cache factories patched to fail; local
tiny safetensors loading and exact cache/base identity; pre-allocation cohort and
unsupported-render rejection; rejected checkpoint leaves session unchanged;
borrowed modes/gradient objects/hook counts restored; injected failures in graph,
teacher, critic, game and callback/forward phases unwind owned hooks.

The historical trainer/config/model helpers and shared core tree have an empty
diff from the base. This adds no package/weight release or deployment. The local
CI proposal builds/tests installed wheel and sdist on public CPU PyTorch;
it is not published or dispatched by this source-only change.
Full pretrained Stage C, GPU memory/quality, prompt-to-cache capture, native
DDPMWuerstchen sampling, Stage B decoder and Stage A codec qualification remain
outside these CPU fixtures. Numerical responses never claim complete images.

Core-owner coordination message to task
`01a192cb-212c-7688-8418-ba7406670007` was denied by automatic approval review:
it considered authorization to contact that task unclear under the external-agent
restriction. The message was not sent and was not retried. Core admission remains
pending; root must provide an admitted revision or authorize that coordination.
