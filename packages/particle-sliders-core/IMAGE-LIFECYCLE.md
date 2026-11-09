# Image sessions

`particle_sliders.lifecycle` owns preparation through a product factory,
updates, evaluation, exact resume, raw exports, rendering dispatch and cleanup.
The product owns its frozen model, teacher capture, adapter, native response
and renderer. For model-specific usage, use the
[product repositories](../../docs/products.md); core has no built-in pretrained
Iris, Qwen, Krea or other model loader.

## Session use

After installing a product that registers `my-image`:

```python
from particle_sliders.lifecycle import SessionConfig, open_session

config = SessionConfig("my-image", "new-reviewed-cohort",
                       steps=4, ema_decay=None, **product_options)
with open_session(config) as run:
    run.train(stop_after=2)
    run.checkpoint("training.pt")
with open_session(config, resume="training.pt") as run:
    run.train()  # continues to the original four-step horizon
    response = run.evaluate(index=0, seed=11, strength=.5)
    run.export("adapter.pt", selection="raw")
    # Only when the product provides a native-only inference specification:
    run.export_inference("adapter-v2.pt", selection="raw")
```

`product_options` comes from the product. `evaluate` returns a numerical response.
`render(...)` calls a product callback and is available only when the capability
allows it. There is no generic sampler substitution. Products define prompts,
references, output formats, strength ranges and native precision support.

## Product factory

Register a `Backend` under `particle_sliders.image_backends`:

```python
from particle_sliders.lifecycle import Backend, Binding, Capabilities

CAPS = Capabilities(generator_arm="adapter_gan", render=False)
BACKEND = Backend("my-image", CAPS, prepare, validate, preflight=preflight)
```

`validate(options)` and optional `preflight(options, request, staged)` must check
inputs without allocating weights. `prepare(options, **staged)` builds the
complete game and returns `Binding(game, close, ...)`. It must unwind its own
partial failures. `configure_capabilities(options)` can provide allocation-free
configuration-dependent descriptors.

A binding may also supply `render`, `inference`, `before_update`, `observation`,
`evaluation_rows` and `response` hooks. Keep native scheduler, packing, decoder,
base identity and adapter target selection inside the product. Cleanup must
restore owned hooks and borrowed native settings.

## State and failure rules

`SessionConfig(backend, cohort, **options)` snapshots finite JSON; a nonempty
cohort is required. Opening checks operation, strength, references, precision,
options, budgets, released PG sources and packet identity before preparation.
The factory's actual game must match the declared generator arm.

Training uses the adapter scale context through backward. Each successful
`step()` completes both critic and generator phases before publishing metrics.
Training failures close the session; reopen from a full checkpoint. Direct
engine use poisons interrupted partial updates until a valid full restore.

Admission via `Binding.before_update(request)` must be a pure decision over
existing native evidence. Rejection restores game state, native flags and RNG;
no update or `on_step` event occurs. Failed rollback quarantines the session
until cleanup. Inference-load failure similarly restores the preceding training
state, or quarantines when restoration fails. Callers use the engine's public
restore/transaction APIs, not private state flags.

Observation restores RNG and all registered adapter buffers/modes, including
nonpersistent buffers. A product `observation()` hook owns native state outside
the adapter, including provider fetches. Cleanup is explicit and idempotent on
success; ownership conflicts must be resolved rather than overwritten.

## Training checkpoints and exports

| Artifact | Contents | Purpose |
|---|---|---|
| `training.pt` | Raw game weights, optimizers, streams, counters, recipe and exact identity | Continue the original training cohort |
| V1 `adapter.pt` | Raw adapter/bank with matching session manifest | Inference through the matching prepared session |
| V2 `adapter-v2.pt` | Raw adapter/bank with native-only specification and origin receipt | Inference without training resources |

Image v1 schemas are `particle-sliders-image-session-v1-training` and
`...-inference`. Manifests bind configuration/cohort, capabilities, factory/core
sources, model, data and game signature. Content checksums detect corruption;
they are not cryptographic authorship signatures. Packets use
`torch.load(weights_only=True)` and same-filesystem atomic replacement.

Invalid state is rejected before mutation. Full and adapter restores are
transactional. Source, runtime, model, cache or recipe changes do not silently
migrate a checkpoint. V1 inference-loaded sessions reject training, step,
checkpoint and restore. V2 has its own [inference interface](NATIVE-INFERENCE.md).

Weight EMA is disabled for adapter, critic and bank. The retained training
packet's disabled EMA envelope has `decay=None, values=None`; nonempty EMA state
is rejected. All exports and inference use raw current weights. Upstream scalar
optimizer statistics are unchanged.

See [response sessions](RESPONSE-SESSIONS.md) for numerical audio/text bindings,
[execution](TRAINING-COMMAND.md) for training commands and
[engine contracts](CONDITIONAL-PG09-CONTRACT.md) for the shared game.
