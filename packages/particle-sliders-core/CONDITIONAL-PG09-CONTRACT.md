# Conditional ParticleGAN engine

`ConditionalPG09` owns the shared adversarial response game, optimizer state,
private sampling streams and exact continuation. Current core 0.14.1 uses exact
released ParticleGAN 0.9.5. For the explanation and equations, see
[teacher/student training](../../docs/math.md); for model usage, see the
[product repositories](../../docs/products.md).

## Build a game

A product supplies a frozen native backend, prepared teacher/student rows and a
conditional critic. Most consumers use a [session factory](RESPONSE-SESSIONS.md)
rather than constructing this engine directly.

```python
from particle_sliders.pg09 import ConditionalPG09

game = ConditionalPG09(
    backend, prepared_rows, critic,
    conditions=conditions, particles=None,
    response_domain="continuous_field", response_scale=1.0,
    preset="bispf_optimized", batch_size=2, seed=7, total_steps=100,
    sigma=1.0, sampling="balanced", recipe_overrides=None,
    pg_source_profile="release-0.9.5", penalty_input_domain="auto",
)
metrics = game.step()
state = game.state_dict()
game.load_state_dict(state)
```

The names `backend`, `prepared_rows`, `critic` and `conditions` above are
product-owned objects. `backend.network` exposes only the intended adapter
trainables. The frozen host model is outside the optimizer. The backend supplies:

- `prepare(ResponseRow) -> PreparedResponse(row, target, neutral)` for capture.
- `generated(prepared)` returning a differentiable response of the target's
  exact native shape, with the row's sign/scale preserved through backward and
  activation recomputation.
- `metadata()` binding the native model, response coordinates, conditioning,
  target convention and scale/checkpoint behavior.

The critic is `critic(response, condition)` and returns raw scalar logits.
Conditions are detached numeric `[rows, K]` coordinates shared across real,
fake and penalty evaluations. Native rendering belongs to the product.

## Response geometry and source profile

| Response | Native shape | Declaration |
|---|---|---|
| Continuous image/video field | `[1,C,H,W]` or `[1,C,T,H,W]` | `continuous_field`, penalty domain `auto` |
| Continuous vector | `[1,D]` | `continuous_vector`, penalty domain `continuous` |

The released `continuous` declaration supports wide native vectors, including
widths 1024, 2048 and 4096. `auto` retains upstream width gates and refuses flat
continuous vectors of width 170 or more. Do not project, truncate, or spatially
reshape a vector to get through admission. Pass the same shape/profile/domain
pair to the descriptor and actual game. `validate_pg09_domain(...)` performs the
source and geometry check before a native forward.

The core verifies installed ParticleGAN sources against its release manifest.
It does not install patches, mutate an environment or alias historical profiles
to the current release. See [compatibility](COMPATIBILITY-PG095.md).

## Loss and trainable ownership

The critic sees paired Gaussian reference and reference plus
`(generated - target) / response_scale`. The default recipe uses the upstream
Sinkhorn-relativistic adversarial loss. Critic penalty, particle regularizer,
constraint-geometry backward and optimizer rules remain upstream. There is no
added MSE, reconstruction or auxiliary distillation objective.

`particles=None` means an adapter-only game. Otherwise `particles` is the actual
trainable `nn.Parameter[N,Z]` bank used by generation. A registered but unused
bank does not qualify. Core validates optimizer ownership and bank dimensions;
products must not add a disconnected bank to satisfy an interface.

## Providers, sampling and observation

An eager tuple can hold small datasets. A lazy provider supplies `len(provider)`,
`response_shape` including leading 1, `identity()`, `provider[index]` returning a
`PreparedResponse`, and `conditions(indices)` returning detached `[batch,K]`.
Pass `conditions=None` for a provider. Construction retains it without reading
all response rows; each phase fetches selected unique rows. The provider owns
checksums and immutable cache identity. Response memory stays bounded by the
active batch and provider cache; sampler metadata is proportional to row count.

Default `sampling="balanced"` independently shuffles complete row cycles for
critic and generator. `replacement` is a separate explicit policy. Permutations,
cursors, row exposure counts, row/noise streams and model RNG are checkpointed.
`evaluate(index, seed=...)` preserves training RNG; session observation also
restores adapter buffers/modes and product-owned native state.

## State transitions

A step completes only after both critic and generator updates. An interrupted
partial step, including `KeyboardInterrupt` or `SystemExit`, poisons the game:
updates, checkpointing and exports refuse until a valid full training restore.
Sessions close on training failure; reopen from a qualified full checkpoint.

`state_dict()` includes raw adapter/critic/bank weights, optimizers, regularizer,
streams, counters, resolved recipe and source/model/data/runtime identity.
`load_state_dict()` validates before mutation and restores atomically; a failed
rollback poisons the game. `rollback_on_error()` is the public transaction for
reversible admission checks. `game.signature` returns an independent snapshot;
callers must not manipulate engine private flags.

`export_state_dict()` produces adapter/bank inference weights without the frozen
base. `load_export_state_dict()` validates and applies them transactionally.
Inference-loaded state cannot resume training. Use a full training checkpoint
for continuation and preserve its original source cohort. Weight EMA is disabled;
ParticleGAN scalar optimizer statistics are unchanged.
