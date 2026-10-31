# TinyQwen shared ParticleGAN / HyperGAN / HNDL preview

This separate opt-in **CPU fixture v2 cohort** executes the installed shared core,
actual HyperGAN develop builder and HNDL0.10 graphs. Native
`Qwen/Qwen3-0.6B-Base` remains **blocked**: its genuine final hidden width1024
cannot enter the reviewed release0.9.4/auto continuous-vector game. No model
weights are loaded, reshaped, projected or converted to evade that gate.

The historical `train_lora_tiny_llm.py`, `tiny_llm_particle.py`, archived
`conceptmod-tiny-llm-particle-v1` checkpoints and Rp paired-error/noise/VIC recipe
remain unchanged. This preview has a new neutral-caption student / frozen
positive-caption final-hidden objective and shared optimized BISPF lifecycle.
It is not historical objective parity or checkpoint conversion. The earlier0.1/v1
preview also requires its original source/environment; no artifact is relabeled.

## Namespace and exact environment

The0.2 wheel installs only **`tinyqwen_shared_preview`**, never `conceptmod`, its
initializers or umbrella/LTX module paths. The builder packages the existing
`conceptmod/textsliders/tiny_llm_backend.py` source byte-for-byte as
`tinyqwen_shared_preview/backend.py`; the repository's historical file is unchanged.
The binding, initializer and complete packaged host source are checked against
`src/tinyqwen_shared_preview/sources.lock.json` before native mutation.

The separate source-tree wheel requires core0.12 with the `hypergan-hndl` extra.
It does not combine an old pinned base with a contradictory extra. Admission
checks exact PEP610 provenance plus the entire core Python/JSON/HNDL source closure:

- core Git commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, repository
  `https://github.com/HyperGAN/particle-sliders.git`, subdirectory
  `packages/particle-sliders-core`, noneditable; **or** reviewed core wheel SHA256
  `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`;
- official ParticleGAN0.9.4 with its common source-closure check;
- clean HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`;
- HNDL0.10.0, common manifest SHA256
  `69e65e3daf18ac5c99154fd735d0b6fc8e8c95433c5b0618a074cdd203ed6251`.

Install origin enters identity without local paths. Missing/version-only, wrong
wheel/Git-origin or changed-source installations are refused. Metadata is not a
cryptographic attestation. Torch/stdlib sources are not independently hashed here;
the common game records runtime versions in its continuation signature.

This is a source-qualified **wheel-only preview**, not a core0.12 package-index
release claim. No standalone sdist is supplied. Use a separate environment from
historical TinyQwen/Bonsai and HNDL0.8.2 cohorts. Python>=3.11,<3.15 and float32
CPU fixtures are the admitted product profile.

```bash
python3 -m venv .venv-tinyqwen-shared
. .venv-tinyqwen-shared/bin/activate
# Verify the reviewed core wheel SHA256 above before installation.
python -m pip install /path/to/particle_sliders_core-0.12.0-py3-none-any.whl
python -m pip install 'particle-sliders-core[hypergan-hndl]==0.12.0'
python -m pip wheel --no-deps packages/tinyqwen-shared-preview -w /tmp/tinyqwen-wheels
python -m pip install /tmp/tinyqwen-wheels/tinyqwen_shared_preview-0.2.0-py3-none-any.whl
```

## Lifecycle and actual strength semantics

```python
from tinyqwen_shared_preview.binding import config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

rows = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]
with open_response(config(fixture=True, fixture_width=16, steps=4, ema_decay=.9), rows=rows) as session:
    session.train()  # Bounded fixture updates, not pretrained training.
    session.checkpoint('/tmp/tinyqwen-resume.pt')
    session.export_inference('/tmp/tinyqwen-native.pt', selection='ema')
    hidden = session.evaluate(strength=.5)
with open_inference('/tmp/tinyqwen-native.pt', operation='response') as native:
    hidden = native.response(ids=[1, 2], strength=-1.)
```

`fixture_width` selects only16 or32 and constructs a random dummy of that actual
config. It cannot change native Qwen's1024 domain. The critic response width comes
from `native.model.config.hidden_size`, with condition width from actual common
`audio_conditions` output. Graph text/hash, response width, condition width and
condition-layout identity enter model/critic packet signatures. Training packets
also retain the common `conditions_sha256` value.

Training uses scale+1. Runtime probes support **finite signed and extrapolated
scalars**: each routed projection implements `base(x) + strength * branch(x)`.
Scale0 bypasses exactly; negative values reverse each local residual and values
above1 extrapolate it. The whole nonlinear model response need not interpolate
linearly. No quality calibration is claimed beyond the training scale. Nonfinite
strengths and generated responses are rejected. Tests check local affine scaling
and inference/evaluation parity at -1,0,.5,1,2; there is no inferred0..1 cap.
Inference and evaluation use `torch.no_grad()`; training keeps adapter gradients.
No token/media renderer, model downloads, pretrained/GPU or quality qualification
is advertised.

## Ownership and cleanup limits

The core owns all GAN losses, penalties, regularizers, optimizers, updates, EMA,
resume and inference packets. Product code owns q/k/v/o targets, final-token
selection, one actual128x4 bank, graph/shape identity and native cleanup. It reuses
`HNDLParticleProjection` rank8/alpha8 and `ARLastHiddenResponseAdapter`. The small
Linear16/SiLU/Linear1 product critic runs through `verified_builder()`. No GAN
mathematics are copied.

Graphs are constructed before hooking. Cleanup attempts **all** owned forwards
and every saved heterogeneous mode/gradient restoration even after a failure.
A foreign forward is retained rather than overwritten; other hooks and saved
modes/gradient flags are still restored. Unresolved forward/state actions stay in
`slider.cleanup_pending`; failed checkpoint handles stay in the common adapter's
ledger. Native `close()` independently attempts checkpoint and slider cleanup,
reports the first failure and can retry remaining actions after a conflict resolves.

Product factory failures preserve the initiating exception. If cleanup fails too,
the exception retains `cleanup_owner`, `cleanup_errors` and a note; retry through
`exception.cleanup_owner.close()`. This is best-effort cleanup with an explicit
ledger, not guaranteed removal of every foreign hook or failing restoration.
Common sessions mark themselves closed before `Binding.close`; after a failed
session close, use retained `session.binding.close()` or the native owner to retry.
Common outer error paths may still surface a cleanup failure. Preserving the
original exception and retry state at every outer lifecycle boundary is a central
follow-up, not a product guarantee.

Native-only inference compares full reconstructed model/source/shape identity on
unattached real graphs before hooks or parameter loading. The common training
loader checks config before preparation; deeper signature/tensor checks happen
after preparation, before parameter mutation. Preparation failures attempt cleanup
as described above. These are different guarantees.

## Tests and requested central contracts

Run installed-wheel tests outside the checkout:

```bash
mkdir -p /tmp/tinyqwen-installed-tests
cp tests/test_tiny_llm_shared.py /tmp/tinyqwen-installed-tests/
TINYQWEN_EXPECT_INSTALLED=1 TINYQWEN_SOURCE_ROOT="$PWD" \
TINYQWEN_WHEEL=/tmp/tinyqwen-wheels/tinyqwen_shared_preview-0.2.0-py3-none-any.whl \
  python -m pytest -q /tmp/tinyqwen-installed-tests/test_tiny_llm_shared.py
python -m pytest -q tests/test_tiny_llm_slider.py
```

Tests cover actual HG/HNDL execution, historical projection VJPs, actual common
BISPF penalty derivatives, updates/ownership, exact resume, raw/EMA data-independent
inference, signed strengths, autograd state, source/origin/model/shape rejection,
width16/32 critic sizing, native1024 refusal, cleanup failures/retries and a wheel
installation beside an umbrella sentinel.

Requested central APIs, **not existing imports or product implementations**:

1. Reviewed explicit-continuous PG source/profile plus shared1024 admission,
   signature and export provenance. The current guard stays unchanged.
2. `Backend.packet_preflight(config, request, staged, *, packet, kind) -> None`,
   called with a checksum/config-validated packet before native `prepare` or hooks,
   to reject source/model/shape mismatches. The core keeps all packet, optimizer,
   tensor and game validation/math ownership.
3. `HNDLConditionalVectorCritic(response_width, condition_width, *, hidden=16,
   profile='hypergan-hndl-v2')`, a canonical verified Linear16/SiLU/Linear1 graph,
   raw `[B]` logits, and `identity()`/`extra_repr()` binding graph/runtime/input/
   output shapes. This proposal grants no native response-domain admission.
4. Common lifecycle cleanup should attempt every owner, retain the initiating
   exception as primary, and expose retryable cleanup after close failure instead
   of hiding it behind `session.closed`.

The executor has no `/dev/nvidia*`; occupancy is unknown and GPU0 is reserved.
No pretrained model/GPU, memory-fit, quality or convergence checks ran. No weights,
checkpoint conversion, deployment or automatic package release is included.
