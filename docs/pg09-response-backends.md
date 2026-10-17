# Frozen response backends for the conditional PG0.9 core

These adapters own model forwards and adapter state. `particle_sliders.pg09.ConditionalPG09` owns the single GAN implementation. No adapter calls the legacy MSE trainers. The generator objective is the PG adversarial loss plus its required regularizer for an **actual participating** routed bank; the critic uses its PG adversarial loss and required penalty. A paired residual is a response coordinate, not a reconstruction loss.

The isolated backend branch adds no shared-core files. The coordinated core contract is `integration/pg09-conditional-core`, initial contract commit `97ec394`. The initial backend milestone can run its model/CFG/export tests before the core implementation exists; six integration cases explicitly skip until `particle_sliders.pg09` imports. That milestone does not qualify GAN training.

## Interface

`prepare(ResponseRow)` returns `PreparedResponse(row, target, neutral)`, with frozen cloned conditioning and targets. `generated(prepared)` returns the differentiable native response `[1,C,H,W]` or `[1,C,T,H,W]`. `network` is an FP32 adapter-only module with reversible `scaled()`; frozen pretrained weights never enter its optimizer or export. Conditions supplied to core must encode the declared row context, sign and time. `response_signature()` binds exact conditioning/latent/schedule/target bytes and staged-model fingerprint for provenance.

The three base backends reuse existing adapters and model loaders:

- `MossResponseAdapter(transformer, slider)`: native `moss_sliders.Slider` and vendored MOSS transformer. Target is frozen neutral velocity plus `sign * strength * (positive-negative)` at the same latent/time. Raw conditional training; CFG rendering applies the adapter to the conditional branch, keeps the negative branch at zero and disables negative-layer skipping.
- `AnimaResponseAdapter(backend)`: existing `LiveAnimaBackend` or CPU stand-in. Conditioning is freshly evaluated on every student forward when conditioner LoRA is trained. Supports direct responses or explicitly supplied trajectories.
- `SupraResponseAdapter(backend)`: existing `LiveSupraBackend` or stand-in. Bypasses the legacy `predict_v` method's internal reset to scale1, while reusing native context and model forwards.

Product runtime bindings maximize reuse:

- `AnimaRuntimeResponseAdapter(runtime, particle_adapter, name)` reuses `lumen_studio.backends.anima.TurboRuntime`, its exact `ParticleAdapter` and `Mixer`. Attach it to `runtime.mixer` first. Pass **that same** `backend.particles` to core. The adapter's existing bank participates in the actual hooked model forward; no new prior is invented. Runtime decoding, model identity, scheduler times and BF16 Euler rounding are retained.
- `SupraRuntimeResponseAdapter(runtime, model_identity=verified_identity)` reuses `supra.runtime.SupraRuntime`'s text cache (including stored unconditional embeddings/masks), native DiT, LoRA and VAE scale. It preserves CUDA BF16 autocast when later admitted, while separating adapted conditional and frozen negative CFG forwards.

Every non-CPU binding requires a verified staged-model identity. Loading/downloading models and GPU/process admission remain the coordinator's responsibility. These modules perform neither on import.

## Time and rendering contracts

`FlowSchedule.supra(K)` implements the Supra Hub convention: `t=i/K`, starting at noise `t=0`, `x += v/K` toward data `t=1`. It is **not** a descending-sigma scheduler.

For Anima/MOSS, configure a fresh copy of the actual Euler scheduler and call `FlowSchedule.from_scheduler(scheduler)`; this preserves shifted/snapped timesteps and all sigma intervals. `FlowSchedule.linear(K,time_scale=1000)` is an explicit unshifted trajectory fixture, not a substitute for a product scheduler. Anima live forward normalizes raw scheduler timesteps by its training-timestep count; MOSS consumes raw `1000*sigma`.

`render(row,scale=...)` uses the same response/CFG loop as training and an injected native decoder. The Turbo runtime binding preserves model-output BF16 rounding before and after each Euler step. For custom MOSS/Anima/Supra bindings, supply the exact staged model's decoder; no fake image fallback exists. The CPU fixtures use tensor decoders only to check the loop contract, without image-quality claims.

Install `checkpoint_with_adapter_scale(block,network)` instead of legacy MOSS checkpoint wrappers. The new wrapper captures each forward's scale for backward recomputation; signed rows may share a batch. Native enabled Diffusers checkpointing is rebound through its public checkpoint-function hook. Double installation of the explicit wrapper is rejected. Nested local/PEFT scales restore exactly without changing frozen-base or adapter `requires_grad` flags.

## CPU verification

Use the isolated released PG0.9 environment, with CUDA hidden. `requirements-tiny-pg09.txt` contains the released wheel hash. Do not install the legacy package's older PG pin into that environment.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/path/to/core/packages/particle-sliders-core/src:/path/to/moss/src:.
/path/to/pg090/bin/python -m pytest tests/test_pg09_backends.py -q -o pythonpath='' -p no:cacheprovider
/path/to/pg090/bin/python scripts/qualify_response_backends_pg09_cpu.py --family all --output /tmp/pg09-backend-qualification
```

The CPU command uses one shared `ConditionalPG09` game for all model fixtures, explicitly selects `get_recipe('bispf_optimized')` through core, verifies exact resume/reload, and writes a JSON receipt plus adapter-only exports. The base adapters are labeled **adapter-GAN only**: they do not contain a routed particle bank. The native Anima runtime binding exposes its genuine participating bank separately for coordinator/QA qualification. Native fields must reach the PG penalty without flattening; flat continuous vectors of width170+ must fail closed until a qualified upstream domain API is installed.

No GPU launch, cancellation, queue dispatch, merge, publication or model-quality qualification is performed by this branch.
