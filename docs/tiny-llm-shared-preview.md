# TinyQwen shared ParticleGAN / HyperGAN / HNDL preview

This separate opt-in **CPU fixture cohort** executes the installed shared core,
actual HyperGAN develop builder and HNDL0.10 graphs. Native
`Qwen/Qwen3-0.6B-Base` remains **blocked**: its genuine final hidden width1024
cannot enter the reviewed release0.9.4/auto continuous-vector game. No weights
are loaded, reshaped, projected or converted to evade that gate.

The existing `train_lora_tiny_llm.py`, `tiny_llm_particle.py`, archived
`conceptmod-tiny-llm-particle-v1` checkpoints and their Rp paired-error/noise/VIC
recipe remain unchanged. Continue using their original source/runtime for old
artifacts. This preview is a new objective/cohort, with a neutral-caption student
and frozen positive-caption final-hidden teacher, shared optimized BISPF recipe,
and exact common lifecycle state. It is not an objective-preserving replacement
for the historical trainer and does not import `particle_bridge_gan` at runtime.

## Exact isolated environment

Build the separate source-tree wheel under `packages/tinyqwen-shared-preview`.
Its package metadata requires core0.12 with the `hypergan-hndl` extra; it does not
combine an old pinned base dependency with a contradictory extra. The checked-in
`tiny_llm_shared.lock.json` binds the exact installed core Python/JSON/HNDL source closure:

- core candidate `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, wheel SHA256
  `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`;
- official ParticleGAN0.9.4;
- clean HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`;
- HNDL0.10.0, shared manifest SHA256
  `69e65e3daf18ac5c99154fd735d0b6fc8e8c95433c5b0618a074cdd203ed6251`.

This is a source-qualified preview, not a published core0.12 package-release
claim. The candidate branch is publicly fetchable as of the remote receipt;
use the reviewed exact wheel/source, not an assumed PyPI0.12 release. The product
is a source-tree **wheel-only preview**; no standalone sdist is supplied. Keep
this environment separate from archived TinyQwen/Bonsai or HNDL0.8.2 environments.
Python>=3.11,<3.15 and float32 CPU fixtures are the admitted product profile.

```bash
python3 -m venv .venv-tinyqwen-shared
. .venv-tinyqwen-shared/bin/activate
# Install the exact reviewed core candidate and its declared hypergan-hndl
# dependencies first. Verify the candidate wheel against the SHA256 above.
python -m pip install /path/to/particle_sliders_core-0.12.0-py3-none-any.whl
python -m pip install 'particle-sliders-core[hypergan-hndl]==0.12.0'
python -m pip wheel --no-deps packages/tinyqwen-shared-preview -w /tmp/tinyqwen-wheels
python -m pip install /tmp/tinyqwen-wheels/tinyqwen_shared_preview-0.1.0-py3-none-any.whl
```

No model download occurs on import or fixture opening. No new token sampler,
text renderer, waveform renderer, GPU qualification or pretrained training is
advertised. The text modality describes a numerical hidden-state probe only.
The optional fixture flag selects a distinct random model, never native approval.

## Shared lifecycle usage

```python
from conceptmod.textsliders.tiny_llm_shared import config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

rows = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]
with open_response(config(fixture=True, steps=4, ema_decay=.9), rows=rows) as session:
    session.train()  # Bounded software fixture updates, not pretrained training.
    session.checkpoint('/tmp/tinyqwen-resume.pt')
    session.export_inference('/tmp/tinyqwen-native.pt', selection='ema')
    hidden = session.evaluate(strength=.5)
# The native-only loader needs neither rows, teacher cache, critic nor game.
with open_inference('/tmp/tinyqwen-native.pt', operation='response') as native:
    hidden = native.response(ids=[1, 2], strength=.5)
```

The core owns all GAN losses, penalties, regularization, optimizers, update loops,
EMA, exact resume and inference packets. Product code owns q/k/v/o targets,
model identity, native final-token selection, one actual128x4 bank, graph shape
binding and cleanup. It reuses `HNDLParticleProjection` (rank8/alpha8) and
`ARLastHiddenResponseAdapter`; no routed network or GAN mathematics are copied.
The product critic is the exact Linear16/SiLU/Linear1 readout architecture via
`verified_builder()`, with source/plan/native shapes in its inspectable identity.
Projection/critic identities include the product lock and binding digests.

The response registry rejects unknown/profile/precision options and wide native
responses before model loading or hooks. Builder/core tampering rejects before
attachment. Graphs are all constructed before any native forward is replaced;
failed preparation removes owned hooks and restores borrowed modes/gradient
flags. Native-only inference compares the full reconstructed model/source/shape
identity using **unattached** graphs before attachment or parameter loading.
Changed training-packet configuration is checked before native preparation by
the core; deeper signature/tensor validation happens after preparation, before
parameter mutation, with cleanup on rejection. This distinction is intentional.

## Verification and remaining shared requirements

Run modern tests against the installed wheel from outside the checkout:

```bash
mkdir -p /tmp/tinyqwen-installed-tests
cp tests/test_tiny_llm_shared.py /tmp/tinyqwen-installed-tests/
TINYQWEN_EXPECT_INSTALLED=1 TINYQWEN_SOURCE_ROOT="$PWD" \
  python -m pytest -q /tmp/tinyqwen-installed-tests/test_tiny_llm_shared.py
python -m pytest -q tests/test_tiny_llm_slider.py
```

The historical-source path is used only to compare its exact archived projection
class against the new HNDL graph; the installed product never imports the old game.
The checks cover actual HNDL module execution, matched weights/output/input/bank
VJPs, input and parameter second derivatives including the actual shared BISPF
penalty, shared updates/ownership, exact uninterrupted versus split resume,
raw/EMA strengths0/.5/1 native-only reload, zero bypass, source/profile/model/shape
rejection, cleanup and genuine wide-vector refusal.

The executor has no `/dev/nvidia*`; GPU occupancy is unknown and GPU0 is reserved.
No pretrained weights, GPU gradients, memory fit, quality, training/convergence or
checkpoint conversion were tested. CPU fixtures do not qualify native TinyQwen.

Central follow-ups, not implemented here:

1. A separately reviewed explicit continuous-domain PG source profile and shared
   export/signature admission for native width1024. Never bypass the current guard.
2. Canonical shared modern conditional vector critic API through the same verified
   builder. This product-bound graph can later be replaced only as a new reviewed
   source/cohort; Fibo critic hashes do not identify this vector graph.
3. If strict rejection of every training source/model/shape mismatch **before any
   hook** is required, a shared packet-aware pre-prepare validation seam. The current
   shared factory receives config/staging, not the resume packet; its full signature
   check occurs after preparation. This preview guarantees cleanup and no loaded
   parameter mutation at that boundary.

No package release workflow is introduced. Publishing a source draft PR does not
publish model weights, deploy a service, release a package or qualify native use.
