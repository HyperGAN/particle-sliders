# Numerical response sessions

Use `particle_sliders.response` to train and evaluate a product's native
numerical responses with the shared lifecycle. It supports text/audio hidden
vectors and other declared continuous responses. It supplies no token, image or
waveform renderer. Model setup and generation live in the
[product repositories](../../docs/products.md).

Response sessions use the same `ConditionalPG09`, updates, checkpointing,
observation and cleanup as image sessions. They do not add another optimizer or
training loop. Current core pins ParticleGAN 0.9.5.

## Register a complete product factory

```python
from particle_sliders.response import ResponseBackend, ResponseCapabilities

CAPS = ResponseCapabilities(
    generator_arm="adapter_gan",  # use routed_particle_gan only for a real bank
    modality="text_hidden", response_kind="native_final_hidden",
    response_domain="continuous_vector", response_shape=(1, 2048),
    pg_version="0.9.5", pg_source_profile="release-0.9.5",
    penalty_input_domain="continuous", base_precisions=("float32",),
)
BACKEND = ResponseBackend("my-response", CAPS, prepare, validate,
                          preflight=preflight)
```

`prepare`, `validate` and `preflight` are product functions:

- `validate(options)` and optional `preflight(options, request, staged)` perform
  allocation-free option and native-request checks.
- `prepare(options, **staged)` owns the frozen loader, adapter, target/provider,
  conditions, critic and game. It returns a `lifecycle.Binding` and unwinds
  partial construction failures before transferring ownership.

```toml
[project.entry-points."particle_sliders.response_backends"]
my-response = "my_product.lifecycle:BACKEND"
```

Declare the actual native shape and pass the same profile/domain to the game.
Use `continuous` for flat wide vectors; do not reshape them to look like images.
Base precision is separate from FP32 trainable leaves. The prepared game is
checked against advertised shape, domain, profile and actual generator arm.

## Use a session

After installing the product, supply its documented options:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig("my-response", "new-reviewed-cohort",
                       steps=4, ema_decay=None, **product_options)
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint("training.pt")
    response = run.evaluate(index=0, strength=.5, seed=7)
with open_response(config, resume="training.pt") as run:
    run.train()  # completes the original four-step horizon
    run.export("adapter.pt", selection="raw")
    run.export_inference("adapter-v2.pt", selection="raw")
```

The last operation requires a product-provided `InferenceSpec` in the binding.
V1 export needs the matching prepared session to load; v2 uses a separate
[native-only factory](NATIVE-INFERENCE.md) without teacher cache or game.
Neither is a training resume checkpoint. Fresh sessions explicitly disable
weight EMA; only raw selection is supported.

```sh
particle-slider capabilities --interface response --config local-product.json
particle-slider train --interface response --config local-product.json \
  --out new-run --export-format both --selection raw
particle-slider evaluate --interface response --config local-product.json \
  --resume new-run/training.pt --strength .5 --out response.pt
particle-slider infer --artifact new-run/adapter-v2.pt --runtime local-native.json \
  --mode response --strength .5 --out response.pt
```

These commands require the registered product. The standalone
[`examples/response_fixture`](../../examples/response_fixture) demonstrates the
factory seams on CPU without pretrained weights.

## Observation and lifecycle ownership

`Binding.observation()` encloses provider length, row fetch, scale substitution
and native generation. Product hooks restore native LRU/cache/buffer/sampler
state on success and failure. Core preserves ambient RNG, adapter buffers/modes
and game state; it does not copy an entire native base. Optional
`evaluation_rows` supplies read-only splits with explicit stable identities.

The [image session guide](IMAGE-LIFECYCLE.md) defines shared failure, restore and
cleanup semantics. [Activation checkpoint ownership](CHECKPOINT-OWNERSHIP.md)
explains how to retain the forward's strength during recomputation.

Response registrations use `particle_sliders.response_backends`, separate from
`image_backends`. V1 packets use `particle-sliders-response-session-v1-training`
and `...-inference`; descriptors use `particle-sliders-response-capabilities-v1`.
There is no image fallback or packet relabelling. Native-only v2 artifacts use
`particle-sliders-native-inference-v2` for either family.
