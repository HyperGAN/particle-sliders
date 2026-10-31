# CPU software qualification, 2026-10-31

Repository base: `3c25f785d14073e6f627c0fa9d2297a041c79913`.
The isolated opt-in package uses core candidate
`7a8a321d04323d49319d0b08e6093a0ddce36b27` from draft148, released
ParticleGAN0.9.4, actual HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0 and Diffusers0.35.1.
Central post-fix review/admission and root's Claude review remain required
before landing. No merge, package release or native training is included.

Python3.12.15/Linux, Torch2.6.0+cpu: **21 installed-wheel tests pass**, and the
same21 tests pass after installing the sdist into a second initially empty
virtual environment with public CPU dependency resolution. Both environments
pass `pip check`. The package checks all49 shared core Python/JSON/HNDL files
and the exhaustive Diffusers Python source closure before attaching adapters.
Actual installed product source files were also compared byte-for-byte with
the checkout. No source `PYTHONPATH` or editable install is used by these tests.

The tests execute actual tiny randomly initialized Diffusers UNets, the actual
verified HyperGAN-built `hndl.torch.Network` projections/critic, and the shared
released-PG game/session. They cover:

- DDIM/DDPM raw epsilon and v-prediction, native spatial geometry and CFG.
- Matched routed weights through the full native UNet: outputs and input,
  embedding, particle-bank and every projection/router/bridge parameter VJP.
- Matched native-field critic outputs and input/parameter VJPs; actual public
  BISPF penalty values and critic-weight second derivatives.
- Both public presets: actual generator/particle updates, optimizer ownership,
  full versus split exact resume, EMA, strengths0/.5/1 and observation isolation.
- Exact raw/EMA native-only output parity after training data deletion; a new
  Python process with a different hash seed; no game/teacher/critic construction.
- Rejection of old MSE packets, changed cohorts, sources, base weights,
  scheduler/shape, nonfinite state, unsupported precision and checkpointing.
  Invalid restore packets do not reach live load hooks or alter owned state.
- Teacher, critic, game, inference-spec and update failure cleanup; restoration
  of borrowed modes, original gradient objects/flags and native forwards;
  nested strength scopes; cleanup continues after another action fails and
  preserves the original exception.

The historical trainer is byte-identical to its base after removing the nine
new opt-in dispatch lines. Original trainer SHA256:
`591177cc32c98d152ed586917670223c4f3c782281b1d9d427cefff5f81a3628`.
No historical requirements, objectives, defaults or checkpoint writers change.

Reproduction (from repository root, in a new environment):

```sh
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' 'build' 'setuptools>=84.2.1'
python -m build --no-isolation --outdir /tmp/sd-dist conceptmod/sd-shared-game
python -m pip install '/tmp/sd-dist/conceptmod_sd_shared_game-0.1.0-py3-none-any.whl[test]'
OMP_NUM_THREADS=1 python -m pytest conceptmod/sd-shared-game/tests -q
python -m pip check
OMP_NUM_THREADS=1 python conceptmod/textsliders/train_lora.py --shared-game --fixture --output /tmp/sd-fixture
# In another new environment, resolve/install the sdist and repeat the tests:
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' '/tmp/sd-dist/conceptmod_sd_shared_game-0.1.0.tar.gz[test]'
```

The dispatcher smoke completed4 shared updates and emitted distinct training,
raw and EMA packets with native response shape `[1,4,8,8]` and the declared new
cohort. The subprocess regression also confirms native-only loading without
reconstructing training data or its factory.

These are bounded **CPU API fixtures**, not SD1/SD2 pretrained qualification,
quality evidence, GPU tests or a production training run. No model weights were
downloaded. The modern cohort accepts CPU float32 only; it exposes numerical
raw prediction responses and does not render images or implement a sampler.
No wide-vector or domain-admission guard is changed. There is no new shared API
gap for this spatial SD binding. The repository has no hosted unit-test workflow;
local test receipts are not hosted CI.
