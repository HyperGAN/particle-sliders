# Public image lifecycle, version 1

Use `particle_sliders.lifecycle` for new image integrations. It owns preparation
through a registered product factory, updates, EMA, resume, export, evaluation,
native rendering dispatch and cleanup. Callers supply a local configuration;
they do not construct a slider, response adapter, provider, critic or optimizer.
`ConditionalPG09` still owns the actual GAN mathematics, private random streams,
coverage, optimizers and strict source-bound state. No reconstruction loss,
particle bank or generic solver is added to force models into the same shape.

## Small local examples

Install the core plus ONE derivative in its compatible backend environment, at
the reviewed exact pins. Installation never implies permission to fetch models.
The examples use random/synthetic CPU fixtures and need no model downloads.
Set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` for bounded tests. These fixtures establish software
contracts, not pretrained convergence, accelerator fit or image quality.

Iris and Qwen take the same locally understandable paired-prompt input:

```python
import torch
from particle_sliders.lifecycle import SessionConfig, open_session

torch.set_num_threads(1)
rows = [dict(neutral='a red cup', positive='a shiny red cup'),
        dict(neutral='a blue bowl', positive='a shiny blue bowl')]
config = SessionConfig('iris', 'my-new-cpu-cohort', dummy=True, rows=rows,
    steps=4, height=16, width=16, trajectory_steps=4, records_per_pair=2,
    ema_decay=.9)
# For Qwen use backend='qwen', height=32, width=32 in its own environment.
with open_session(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()  # continues to the ORIGINAL fixed four-update budget
    session.export('new-run/adapter.pt', selection='ema')
with open_session(config, artifact='new-run/adapter.pt', operation='evaluate') as session:
    response = session.evaluate(index=0, seed=11, strength=.5)
```

Krea's tiny fixture creates an actual native tiny transformer and PEFT adapters,
then prepares its edit/preservation cache internally:

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('krea2', 'my-new-krea-cpu-cohort', dummy=True, steps=4)
with open_session(config) as session:
    session.train()
    response = session.evaluate(strength=-.5)
    session.export('krea-new-run/adapter.pt')
```

The Krea tiny fixture is response-only. Its native factory accepts `cache_file`
(the existing frozen local teacher cache) and `native_args` (the product's
existing CLI option list, including local transformer/skeleton paths). It loads
the native backend itself, verifies the cache against actual frozen weights,
and builds the complete adapter GAN. `--allow_hub`, legacy resume and adapter
loading arguments are rejected; use the session's resume/artifact interface.
The native factory uses BF16 base weights and FP32 trainables. This path requires
separate resource admission; the CPU examples do not run it.

Fibo's complete installed staged mock example is one command:

```bash
python -m fibo_particle_slider.example new-fibo-cpu-run
```

Or call `fibo_particle_slider.lifecycle.synthetic_config('new-local-cache')`,
then use exactly the session calls above. The factory loads the staged mock,
attaches the actual routed bank, captures teachers, writes/loads the cache,
constructs the critic/game and undoes attachment on exit. Its mock EDIT pipeline
is explicitly not the pretrained/native pipeline.

For already staged native Fibo, use backend `fibo` and options `model_dir`,
`manifest` (the original verified staging receipt), `cache_dir`, `index_sha256`,
`response_scale`, `steps`, `base_precision='bfloat16'`. Instead of an existing
cache SHA256, `pairs` may contain dictionaries with `neutral`, `positive`, local
`images` paths, optional local `mask`, and `seed`; the factory performs capture
and cache construction. Ordered references/masks and the native four-step
sampler remain in Fibo. No local manifest is fabricated for pretrained weights.

Native rendering uses the same loaded session/adapter:

```python
# Qwen, in its backend environment; tiny fixture supports output_type='latent'.
images = session.render(prompt='a red cup', strength=.5, seed=7,
    height=32, width=32, steps=4, output_type='latent')
# Fibo: session.render(prompt='keep jacket', images=[local_pil_image],
#     strength=.5, seed=7, output_type='latent')
# Krea native: session.render(prompt=grounded_prompt, strength=-.5,
#     seed=7, resolution=768)
```

Iris `render` raises `UnsupportedOperation` before model allocation through
`open_session(..., operation='render')`, or before any sampler call on an open
session. Raw Iris velocity evaluation never claims image output. Native
FlowDPMSolver parity is still required.

## Contracts and capability preflight

`SessionConfig(backend, cohort, **options)` snapshots finite JSON input.
`cohort` is mandatory. `backend(name).describe(config)` reports effective
capabilities without loading weights. `open_session(config, operation=...,
strength=..., references=...)` checks capability, options, budgets, exact
released PG sources and packet kind/config before invoking the factory.

| Backend | Generator | Response strengths | Native rendering | References |
|---|---|---|---|---|
| Iris | routed particle GAN | [0,1] | disabled pending solver parity | unsupported |
| Qwen Image 2.1 | routed particle GAN | [0,1] | native pipeline; synthetic fixture latent only | unsupported |
| Krea2 | adapter GAN, particles=None | all finite signed values | native pipeline; tiny fixture response only | unsupported |
| Fibo Edit Turbo | routed particle GAN | [0,1] | native EDIT pipeline | required for rendering |

Strength ranges are executable interface support, not quality qualification.
All trainables remain FP32; base precision support is declared separately.
Native Qwen and Fibo pin incompatible Diffusers Git implementations: use separate
environments. This API does not promise co-installation or cross-model weight
compatibility. Manifests share a protocol, not a frozen-model identity.

Product packages register `particle_sliders.image_backends` entry points. A
registration exposes `Backend(name, capabilities, prepare, validate, optional
configure_capabilities)`. `validate(options)` performs allocation-free local
checks. `prepare(options)` owns all construction and returns
`Binding(game, close, render=None)`; it must unwind partial failures. Callers
may supply an already staged runtime to a factory for local integration tests.
`close()` is idempotent through the session and restores owned adapter hooks and
checkpoint settings. The factory owns native packing, conditioning, decoder,
scheduler, base identity and genuine formulation differences.

`session.step()` and `train(stop_after=..., on_step=...)` use the common scale
context through backward. EMA includes actual network ownership, so the same
particle parameter is averaged exactly once. Training failures close the
session; recover by opening a fresh session and loading a qualified checkpoint.
Observation restores global RNG and adapter modes/buffers and does not consume
training streams. Native factories preserve their sampler state.

## Checkpoints, artifacts and errors

Training packet schema `particle-sliders-image-session-v1-training` contains
full PG state, EMA configuration/values and last metrics. Inference schema
`particle-sliders-image-session-v1-inference` contains the selected raw/EMA
adapter export and cannot resume training. Both include a manifest binding
configuration/cohort, capabilities, factory/lifecycle bytes and the PG source,
backend, model, conditioning, architecture and recipe signature, plus a SHA256
content checksum. These checksums detect corruption; they are not signatures.
Packets use `torch.load(weights_only=True)` and same-filesystem atomic replace.
Invalid metadata, EMA, sampler or optimizer state is rejected before mutation.
Inference loading intentionally requires the same prepared configuration and
frozen cache; independent new-prompt rendering then uses the native callback.
A compact inference-only loader without preparation is outside this first slice.

Existing product envelopes keep their original entrypoints and schema. Their
writers reuse shared stepping/EMA/atomic-save facilities. Existing cdf/f797
runs, active environments and result receipts remain unchanged. New pins are
new source cohorts; even when an old schema is retained, source checks may reject
old packets. No automatic migration or rebinding is provided. The new CLI rejects
legacy packets with a diagnostic directing callers to their original loader.

Catch `UnsupportedOperation` for capability gates, `BackendUnavailable` for
missing/ambiguous registration or dependency isolation, and `CheckpointError`
for packet kind/version/checksum/source/cohort mismatch. They derive from
`LifecycleError`. Invalid configuration uses `ValueError`; native errors keep
their original cause. A rejected restore leaves a live session usable.

## Installed CLI

Write `config.as_dict()` as JSON and use the same common commands in each
backend environment:

```bash
particle-slider capabilities --config config.json
particle-slider train --config config.json --stop-after 2 --out new-run
particle-slider train --config config.json --resume new-run/training.pt --out continuation
particle-slider evaluate --config config.json --artifact continuation/adapter.pt --out response.pt
```

Training writes full state, inference artifact, metrics and a manifest with file
hashes. Use a new output directory to preserve existing receipts. `--selection
ema` requires configured EMA. Render uses `--native render.json`, whose keys are
the local backend arguments above; reference images are local paths. No command
implicitly enables downloads, patches a dependency or schedules GPU work.
