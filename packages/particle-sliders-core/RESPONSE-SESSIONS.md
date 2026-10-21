# Numerical response sessions: source proposal

This additive contract is **source-only and unqualified** until the parent
admits serial CPU validation. Reviewed core0.8.1/PG0.9.1 (`32f6830`) and
core0.9.0/PG0.9.3 (`da4bae3`) remain unchanged. This proposal is a new source
cohort on each exact dependency line, not a checkpoint migration or a new GAN law.

## Ownership and bounded implementation

Products depend on particle-sliders, which depends on exact upstream ParticleGAN.
Native model libraries remain isolated backend dependencies. No core import of
`music3`, `yue2` or another derivative is allowed. HyperGAN owns run presentation;
`on_step`, immutable manifests and detached responses remain the reporting seams.
Existing raw network definitions in AudioResponseCritic, LoRA and RoutedMLP are
preserved pending separate matched-weight HNDL parity; native pretrained models
remain explicit external backbones.

The bounded slice adds an independent numerical-response descriptor and registry
over the existing complete session. It shares preparation, budgets, native request
preflight, update admission, PG updates, EMA, checkpoints, strict restore,
observation and cleanup. It does not add a second training loop or optimizer.
A separate descriptor records modality, native response kind/domain/shape, actual
routed-particle versus adapter-only arm, exact PG profile and operation refusals.
Response-session v1 exposes numerical evaluation and native-only response inference;
it deliberately has no image, token, acoustic or waveform rendering operation.

Image registrations remain `particle_sliders.image_backends` and image packet
schemas remain `particle-sliders-image-session-v1-{training,inference}`. New response
registrations use `particle_sliders.response_backends` and packets use
`particle-sliders-response-session-v1-{training,inference}`. Neither registry falls
back to the other, and the other family's packet is rejected before preparation.
New source hashes intentionally prevent implicit resume of old source cohorts.
Native-only v2 keeps `particle-sliders-native-inference-v2`; response descriptors
carry `particle-sliders-response-capabilities-v1`. Existing image v2 descriptors
keep their fields. Older readers must reject the new capability discriminant;
no relabeling or automatic migration is supported.

## Minimal product integration

A product already owning a complete factory exposes this object:

```python
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend, ResponseCapabilities

CAPS = ResponseCapabilities(
    generator_arm='routed_particle_gan',  # actual participating bank, not a dummy
    modality='audio_ar_hidden', response_kind='native_final_ar_hidden',
    response_domain='continuous_vector', response_shape=(1, 4096),
    pg_version='0.9.1', pg_source_profile='alz-continuous-v1',
    penalty_input_domain='continuous', base_precisions=('bfloat16',),
    unavailable_reasons=(('render', 'Music3 acoustic/token sampling is unqualified'),))

# validate(options) and optional preflight(options, request, staged) allocate no weights.
# prepare(options, **staged) owns native loader, real slider, teachers/provider,
# conditions, critic and ConditionalPG09; returns Binding(game, close, ...).
# Pass the identical declared response/profile pair to ConditionalPG09.
BACKEND = ResponseBackend('music3', CAPS, prepare, validate, preflight=preflight)
```

```toml
[project.entry-points."particle_sliders.response_backends"]
music3 = "music3.lifecycle:BACKEND"
```

The adapter and native model may differ in precision: `base_precisions` describes
the actual native base; trainable PG leaves remain FP32. Capabilities may be
configured allocation-free for tiny versus staged native models. A flat native
state is never projected, truncated, spatially reshaped or classified as tokens
in order to pass admission. The actual prepared game is checked against the
advertised shape, domain, profile and generator arm, then closed on contradiction.
A product's existing native admission/observation hooks keep native provider,
checkpoint, mode, gradient, sampler and buffer ownership explicit.

Consumers need only the registered product and its documented local options:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig('music3', 'new-reviewed-source-cohort', **local_product_options)
with open_response(config) as run:
    run.train(stop_after=2, on_step=report_completed_metrics)
    run.checkpoint('training.pt')
    response = run.evaluate(index=0, strength=.5, seed=7)
    run.export('adapter.pt', selection='ema')              # v1: matching session
    run.export_inference('adapter-v2.pt', selection='ema') # product InferenceSpec
with open_response(config, resume='training.pt') as run:
    run.train()
```

```sh
particle-slider capabilities --interface response --config local-product.json
particle-slider train --interface response --config local-product.json \
  --out new-run --export-format both --selection ema
particle-slider evaluate --interface response --config local-product.json \
  --resume new-run/training.pt --strength .5 --out hidden-response.pt
particle-slider infer --artifact new-run/adapter-v2.pt --runtime local-native.json \
  --mode response --strength .5 --out hidden-response.pt
```

Configure EMA in local options before selecting it. For v2, the product registers
its `InferenceBackend` through existing `particle_sliders.inference_backends` and
supplies its `InferenceSpec` on Binding. Its native-only factory builds base and
adapter only, without teacher rows/cache, data, game, critic or optimizer.
`local-native.json` is that product's documented runtime input, never a training
configuration. Missing support fails before CLI updates/output creation. A response
artifact carries raw/EMA and origin receipts, never a waveform qualification claim.
The standalone `examples/response_fixture` plugin and its source-free commands document
the same
complete factory seams without native dependencies or a pretrained model.
Its test fixture lives at `tests/fixtures/response_backend.py`.

## Exact profile boundaries

| Core line | Source/profile pair | Native flat continuous response |
|---|---|---|
| PG0.9.1 | `release` / `auto` | Width below170 only |
| PG0.9.1 | `alz-continuous-v1` / `continuous` | Wide states admitted only when every installed PG source matches the reviewed overlay catalog |
| PG0.9.3 | `release-0.9.3` / `auto` | Width below170 only; native Music3 4096 and YuE2 2048 refused |

These gates inspect installed versions and source hashes before preparation. They
never install/apply an overlay, mutate an environment, alias ALZ develop to an
upstream release, change dependency ranges or upgrade an existing research cohort.
PG0.9.3 lacks the reviewed continuous-vector input override needed for wide states.
A supported wide0.9.3 path requires separate upstream/profile qualification; it is
an actual native adoption blocker, not resolved by this API. Tiny32 response tests
cannot qualify native width2048/4096 or acoustic quality. Existing PG0.9.3 research
field/recipe parity remains evidence solely for its reviewed source/profile.

## Minimal audio package closure and historical compatibility

Only `pg09_audio`, `music3_native`, `native_lora`, `yue2_native` and
`pg09_audio_cli` are restored. Their original and packaged hashes are recorded in
`particle_sliders/audio_closure_provenance.json`. The legacy CLI source is from the
previous lock-verified PG0.9.1 installation, not inferred from an incomplete git
snapshot. No campaign, worker, render stack or optional native dependency is added
to mandatory package requirements. Native source/license attribution is retained;
the package MIT license applies to the shared snapshots.

Historical `audio-pg091-*` entrypoints refuse any other PG line before constructing
models or loading packets. Music3's old bounded random-native route is retained.
YuE2 automatic construction previously imported its product/native namespace;
this new source requires product ownership through `build_audio_run` with supplied
backend/slider or the response registry. Thus **the restored YuE2 legacy random
CLI is not activated**. Existing old-pin receipts/loaders stay unchanged. The
new response route, not the compatibility module, owns future common lifecycle
adoption. Neither legacy source restoration nor generic registration means either
product has migrated or passed qualification under this proposal.

## Pending validation and handoff

Authored tests cover exact response split resume/EMA, invalid restore, cleanup,
zero/fractional strengths and refusals, profile-aware preallocation gates,
registry/packet isolation, native-only raw/EMA loading in a fresh process without
training data, image defaults and explicit CLI artifact names. They have **not run**.
Only stdlib source parsing, dependency import analysis and git whitespace checks
are permitted in this phase. Build/install/source-free CPU qualification needs a
parent-admitted slot on each exact PG environment; no GPU/model jobs are needed.

After CPU qualification, the parent routes exact proposed source/wheel hashes to
Music3/YuE2 owners. Each owner changes its own registration, source lock, complete
factory and native-only inference factory, retains old pins/results, and tests a
new cohort. Shared interfaces do not waive their native loader/history/token,
actual-bank, response width, source, admission or HNDL/HyperGAN requirements.
