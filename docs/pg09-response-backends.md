# Frozen response backends for the conditional PG0.9 core

These adapters own model forwards and adapter state. `particle_sliders.pg09.ConditionalPG09` owns the single GAN implementation. No adapter calls the legacy MSE trainers. The generator objective is the PG adversarial loss plus its required regularizer for an **actual participating** routed bank; the critic uses its PG adversarial loss and required penalty. A paired residual is a response coordinate, not a reconstruction loss.

The isolated backend branch adds no shared-core files. The coordinated core contract is `integration/pg09-conditional-core`, initial contract commit `97ec394`. The initial backend contract milestone was `063814e`; the follow-up qualifies these adapters against the implemented native-field game. All GAN tests call that single core directly. Tests fail on import errors once the integration is installed; an uninstalled optional product fixture may skip its dedicated native-bank case.

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

`render(row,scale=...)` uses the same response/CFG loop as training and an injected native decoder. The Turbo runtime binding preserves the official FP32 tensor scalar `dt` multiplied into BF16 model output, followed by FP32 sample addition and BF16 output cast; Python scalar multiplication or prematurely upcasting the velocity changes rounding. For custom MOSS/Anima/Supra bindings, supply the exact staged model's decoder; no fake image fallback exists. The CPU fixtures use tensor decoders only to check the loop contract, without image-quality claims.

Install `checkpoint_with_adapter_scale(block,network)` instead of legacy MOSS checkpoint wrappers. The new wrapper captures each forward's scale for backward recomputation; signed rows may share a batch. Native enabled Diffusers checkpointing is rebound through its public checkpoint-function hook. Double installation of the explicit wrapper is rejected. Nested local/PEFT scales restore exactly without changing frozen-base or adapter `requires_grad` flags.

## CPU verification

Use the isolated released PG0.9 environment, with CUDA hidden. `requirements-tiny-pg09.txt` contains the released wheel hash. Do not install the legacy package's older PG pin into that environment.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/path/to/core/packages/particle-sliders-core/src:/path/to/moss/src:/path/to/anima-product:.
/path/to/pg090/bin/python -m pytest tests/test_pg09_backends.py -q -o pythonpath='' -p no:cacheprovider
/path/to/pg090/bin/python scripts/qualify_response_backends_pg09_cpu.py --family all --output /tmp/pg09-backend-qualification
/path/to/pg090/bin/python scripts/qualify_response_backends_pg09_cpu.py --family anima-particles --output /tmp/pg09-native-bank-qualification
```

The CPU command uses one shared `ConditionalPG09` game for all model fixtures, explicitly selects `get_recipe('bispf_optimized')` through core, verifies exact resume/reload, and writes a JSON receipt plus adapter-only exports. The base adapters are labeled **adapter-GAN only**: they do not contain a routed particle bank. The `anima-particles` example and dedicated native-bank test use the actual product ParticleAdapter/Mixer with a real CosmosTransformer3DModel at tiny random dimensions. They verify adapter/router/bank updates, a nonzero response gradient to the same128×4bank, frozen base weights, native5Dfield shape, and exact checkpoint/export continuation. This is software qualification, not pretrained image-quality evidence. Native fields must reach the PG penalty without flattening; flat continuous vectors of width170+ must fail closed until a qualified upstream domain API is installed.

No GPU launch, cancellation, queue dispatch, merge, publication or model-quality qualification is performed by this branch.

## Final frozen CPU qualification

Adapter implementation: `182db8e885dec28f062f0fda4e882abe708de89b`.
Shared core implementation: `7884c8dd87ad510be6cad2bdf88c0c8ba23bf0c4` (`particle-sliders-core`0.5.0).

**92 tests passed with no skips or failures in5.33seconds**, combining this branch's21backend tests, the56core package tests and15independentQA core/runtime tests. All three base examples and the native Anima routed-bank example also pass exact resume and portable adapter reload. Lint passes. No CUDA runtime was initialized.

Exact source hashes, resolved recipes, metrics and JUnit output are retained under `evidence/pg09-backends/`. The evidence-only follow-up commit changes no implementation. The core owner subsequently integrated the adapter commits into its checkout; the core module is byte-identical to7884c8d. GPU/full-model image-quality validation remains for parent admission; none was attempted here.
