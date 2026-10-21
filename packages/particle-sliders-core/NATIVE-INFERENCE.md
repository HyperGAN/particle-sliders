# Native-only inference, core 0.8.0

Dependency direction is product -> particle-sliders -> ParticleGAN/HyperGAN.
Products register `particle_sliders.inference_backends` separately from
`particle_sliders.image_backends`. Core never falls back to a training factory.
Native base models and model libraries remain local product dependencies.

```python
from particle_sliders.inference import open_inference

# After installing one reviewed product and its native dependencies:
with open_inference("ema.pt", options={"model_dir": "/local/model"},
                    operation="response") as session:
    response = session.response(strength=0.5, seed=7)
    print(session.selection, session.manifest["origin"])
```

The product documents its local runtime keys and response arguments. For a
render-qualified product, omit operation="response" and call session.render.
Default output is the first declared native format (a latent-only fixture needs
no PIL override). Unsupported rendering fails before allocation. No generic
solver substitutes for native image/audio sampling.

CLI: `particle-slider infer --artifact ema.pt --runtime local.json --mode
response --native request.json --out response.pt`. This accepts no training
configuration, resume packet, teacher cache or dataset. It saves a raw/EMA and
origin receipt beside the output.

## Factory contract

`InferenceBackend(name, capabilities, prepare, validate, response=True,
configure_capabilities=None, preflight=None)` declares native operations.
`validate(local_options, spec)` checks local paths/dependencies without allocation.
`preflight(local_options, request, spec)` checks native request geometry before
allocation and before each observation. Request contains operation, strength,
references, output_type and native arguments. These hooks must not mutate or
allocate models. `prepare(local_options, spec)` returns
`InferenceBinding(network, identity, close, render=None, response=None,
observation=None)`. It constructs only native base/adapter objects; no game,
critic, optimizer, provider or training rows. Partial construction cleanup stays
inside the factory until a binding transfers ownership. `observation()` is an
optional context manager for restoring online/native provider state on success
and failure. Core preserves adapter buffers/modes and CPU/explicit CUDA RNG.

During training the product creates a descriptor using
`INFERENCE_BACKEND.specification(native_backend, model_identity=verified_model,
protocol="product-native-v1", configuration=architecture_options)` and places
it in `Binding.inference`. The descriptor contains no training/cache identity.
`session.export_inference("ema.pt", selection="ema")` validates that descriptor
against the actual trained native adapter. Source/model/architecture/protocol,
factory hashes and every tensor schema are validated on native loading before
state mutation. Artifacts retain cohort, training-manifest digest, PG version,
completed steps and raw/EMA selection as receipts, without requiring training
resources. Loading a v2 export cannot train, checkpoint or resume.

Legacy v1 export/load remains explicit and still rebuilds the training game.
Its session now rejects train/step/checkpoint/restore immediately after inference
loading; PG already rejected training from an inference export underneath.
No old packet is silently promoted to v2. Exact training resume retains the old
strict game/config/data/source/cohort validation. Core source changes require a
new cohort; preserve all original checkpoint pins and result receipts.

## Preparation and update ownership

`Backend.preflight(options, request, staged)` runs before model allocation.
`Backend.open(..., native_request={...})` supplies native request details.
Nested native device arguments and staged tensor/module devices participate in
preparation RNG isolation; CPU device indices do not accidentally select CUDA.
Factories must declare precision/device truth and reject unsupported accelerators.

`Binding.before_update(request)` runs immediately before each PG update. Request
is a detached dictionary with completed_steps and manifest. It must be a pure
admission decision over the factory's existing native evidence. Return False or
raise to reject. Core restores PG state, RNG and network flags on rejection;
no update/EMA/on_step occurs and the session remains usable. External I/O or
arbitrary native mutation is forbidden in admission. Accepted hooks do not alter
GAN mathematics. Existing Anima non-CPU admission gates remain until adopted
and tested by that derivative.

`native_cleanup(module)` captures borrowed training modes, requires-grad flags
and gradient-checkpoint settings before mutation. Factories compose its
idempotent callback with removal of owned hooks. `new_directory(path)` owns a
new capture directory and removes partial files on any failure; it refuses an
existing directory. These facilities require factory adoption; core cannot
transactionally reverse arbitrary opaque product loading side effects.

Capabilities have `operations` and `unavailable_reasons` (pairs of operation and
specific reason). An unimplemented backend can declare operations=(),
base_precisions=(). Do not advertise response/native rendering or precision
support that does not exist. The existing `render` gate remains compatible.

HyperGAN owns run presentation. Use on_step plus detached observations and
manifest/artifact references with its supported external RunReporter, in a
separate reporting subdirectory. Core does not depend on HyperGAN Execution
internals or introduce another renderer. HNDL matched-weight network migrations
remain separate. Native pretrained backbones remain native.
