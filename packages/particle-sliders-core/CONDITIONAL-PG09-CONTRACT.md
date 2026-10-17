# Conditional ParticleGAN 0.9 integration contract

Canonical checkout: `/ml2/codex-storage/future-workspaces/2026-10-16/task-6/particle-sliders-pg09-core`, branch `integration/pg09-conditional-core`.

Implementation API:

```python
from particle_sliders.pg09 import ConditionalPG09
game = ConditionalPG09(
    backend, prepared_rows, critic,
    conditions=conditions, particles=particles,
    response_domain="continuous_field", response_scale=1.0,
    preset="bispf_optimized", batch_size=2, seed=7,
    total_steps=100, sigma=1.0, recipe_overrides=None,
)
metrics = game.step()
state = game.state_dict()
game.load_state_dict(state)
```

`backend.network` owns only generator adapter trainables. `backend.prepare(ResponseRow)` returns frozen `PreparedResponse(row, target, neutral)`; `backend.generated(prepared)` returns a differentiable tensor of exactly the target shape. `backend.metadata()` describes model identity, CFG, timestep coordinates, target convention, and checkpoint scale capture. No outer `network.scaled(1)` may override the row sign. Backend owns conditional-only CFG and rendering through the same response loop. Backend owns no loss or optimizer.

Native continuous field shapes must remain `[1,C,H,W]` or `[1,C,T,H,W]` through the public PG penalty. Critic may flatten internally. Generic continuous vectors `[1,D]` with D >= 170 are rejected: released PG 0.9 interprets these as categorical simplex data. Reshaping generic vectors to impersonate images is prohibited. Conditions are fixed detached numeric `[rows,K]` tensors containing the declared timestep/sign/context coordinates; critic signature is `critic(response, condition)` and produces raw scalar logits. The same row conditions bind real, fake and penalty evaluations.

`particles` is either None (adapter GAN arm, explicitly no particle-bank claim) or the actual shared routed `nn.Parameter[N,Z]` used by `generated`. Mere registration of an otherwise unused bank does not qualify. Core validates optimizer ownership and recipe dimensions against this bank. Dense routing binds all participating bank rows through the public optimizer API. A reusable routed-bank module belongs in core; adapter owners must not invent a disconnected prior to satisfy the signature.

The core owns independent D/G row and Gaussian-reference streams. Critic coordinates are real Gaussian reference and fake reference plus `(generated-target)/response_scale`; this paired residual is not an optimized MSE loss. Generator objective is solely the public recipe adversarial loss plus its required prior regularizer. Critic objective is the public adversarial loss plus public critic penalty. Optimized recipe uses public constraint-geometry backward and learning-rate policy. No MSE, reconstruction, distillation or separately invented auxiliary objective is admitted.

Checkpoints bind resolved recipe, verified PG version, model/protocol/conditioning/target digests, response domain and scale, network/critic/actual bank, both optimizer states, regularizer state, D/G row/noise and model RNG states and completed step. Restore must validate transactionally before mutating live state. Export includes adapter and actual routed bank plus metadata; frozen base weights are excluded. Evaluation/render RNG must not advance training streams. The pending game implementation must reject detached generator callbacks under every supported preset, including original `bispf`.

The implementation verifies all 49 installed PG Python sources against the released wheel, and binds actual core/backend/critic/native-adapter submodule source hashes into checkpoints. Patched PG sources are rejected until a separate reviewed dependency is admitted. It supports unchanged-release native fields and continuous vectors below width170. ALZ's explicit-continuous proposal is not installed by this integration.

`evaluate(index, seed=...)` preserves training RNG; `export_state_dict()` and `load_export_state_dict()` transfer adapter-only inference payloads. An inference export cannot resume training; use the full training checkpoint. Backend rendering remains on `backend.render` using its declared response loop.

No GPU admission until objective, gradient, row binding, conditioning, domain, sampling, export/reload and exact-resume CPU gates pass. Prior MSE jobs remain held and are preserved as wrong-objective baselines. CPU qualification is not model-quality or live-memory qualification.
