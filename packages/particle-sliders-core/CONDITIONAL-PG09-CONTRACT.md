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
    total_steps=100, sigma=1.0, recipe_overrides=None, sampling="balanced",
    pg_source_profile="release", penalty_input_domain="auto",
)
metrics = game.step()
state = game.state_dict()
game.load_state_dict(state)
```

`backend.network` owns only generator adapter trainables. `backend.prepare(ResponseRow)` returns frozen `PreparedResponse(row, target, neutral)`; `backend.generated(prepared)` returns a differentiable tensor of exactly the target shape. `backend.metadata()` describes model identity, CFG, timestep coordinates, target convention, and checkpoint scale capture. No outer `network.scaled(1)` may override the row sign. Backend owns conditional-only CFG and rendering through the same response loop. Backend owns no loss or optimizer.

Native continuous field shapes must remain `[1,C,H,W]` or `[1,C,T,H,W]` through the public PG penalty. Critic may flatten internally. Generic continuous vectors `[1,D]` with D >= 170 are rejected: released PG 0.9 interprets these as categorical simplex data. Reshaping generic vectors to impersonate images is prohibited. Conditions are fixed detached numeric `[rows,K]` tensors containing the declared timestep/sign/context coordinates; critic signature is `critic(response, condition)` and produces raw scalar logits. The same row conditions bind real, fake and penalty evaluations.

`particles` is either None (adapter GAN arm, explicitly no particle-bank claim) or the actual shared routed `nn.Parameter[N,Z]` used by `generated`. Mere registration of an otherwise unused bank does not qualify. Core validates optimizer ownership and recipe dimensions against this bank. Dense routing binds all participating bank rows through the public optimizer API. A reusable routed-bank module belongs in core; adapter owners must not invent a disconnected prior to satisfy the signature.

The core owns independent D/G row and Gaussian-reference streams. Critic coordinates are real Gaussian reference and fake reference plus `(generated-target)/response_scale`; this paired residual is not an optimized MSE loss. Generator objective is solely the public recipe adversarial loss plus its required prior regularizer. Critic objective is the public adversarial loss plus public critic penalty. Optimized recipe uses public constraint-geometry backward and learning-rate policy. No MSE, reconstruction, distillation or separately invented auxiliary objective is admitted.

Large caches use a lazy provider in place of an eager tuple. Its API is `len(provider)`, `provider.response_shape` including the leading1, `provider.identity()` returning a stable immutable cache/index/conditioning-policy descriptor, `provider[index] -> PreparedResponse`, and `provider.conditions(indices) -> detached [batch,K]`. Pass `conditions=None`; core reads no response rows during construction, retains the provider, and fetches only selected unique rows per phase. Native target tensors are never concatenated across the whole cache. Small CPU sampler metadata is O(number of rows); response memory is bounded by the active batch/provider cache. Provider source and identity are checkpoint-bound. Selected targets/neutral/conditions are validated before use; providers must checksum their immutable assets and supply generation inputs on the backend's device.

Default `sampling="balanced"` independently shuffles complete D and G row cycles. Permutations, cursors, row exposures and four private row/noise RNG states are checkpointed and validated before live restore. Exact row counts/coverage are reported on each completed step. `sampling="replacement"` remains an explicit distinct bound policy. For MOSS,15360signedrows require1920updates at batch8 per complete G cycle; two cycles require3840updates. Actual admission must first measure model memory/runtime. Useful context, timestep, sign, strength and protocol coordinates belong in conditions; row IDs alone do not qualify held-out transfer.

Checkpoints bind resolved recipe, verified PG version, model/protocol/conditioning/target digests, response domain and scale, network/critic/actual bank, both optimizer states, regularizer state, D/G row/noise and model RNG states and completed step. Restore must validate transactionally before mutating live state. Export includes adapter and actual routed bank plus metadata; frozen base weights are excluded. Evaluation/render RNG must not advance training streams. The pending game implementation must reject detached generator callbacks under every supported preset, including original `bispf`.

The implementation verifies all 49 imported PG Python sources against an explicit profile, and binds actual core/backend/critic/native-adapter submodule source hashes, native architecture trees, Python/Torch/CUDA/numpy runtime and numeric flags into checkpoints. Default `pg_source_profile="release", penalty_input_domain="auto"` verifies the released wheel, supports native fields and continuous vectors below170, and rejects patched sources. No package is installed or modified by this integration.

The separately qualified pair `pg_source_profile="alz-continuous-v1", penalty_input_domain="continuous"` accepts native rank-2 continuous vectors of any width ONLY when the imported sources exactly match `pg09_continuous_profile.json`. This manifest records the single ALZ patch SHA256 `c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571`, base released wheel and all49 modified-overlay Python hashes. The overlay changes the public penalty's input-domain declaration, not host loss math. The pair is rejected for image fields; the default profile never falls back. Independent CPU tests cover actual G/D/bank updates, positive continuous penalties, fresh-process optimizer/RNG continuation for both presets at1024/2048/4096/5120,42 default numeric parity comparisons, and real cross-profile checkpoint rejection before mutation.

Thin backends can call `validate_pg09_domain(shape, response_domain=..., pg_source_profile=..., penalty_input_domain=...)` before their first model forward, then supply those same options to core. It validates actual sources and native shape and never reshapes data. Training checkpoints bind declared/resolved domains, profile, base and patch/source identities; cross-mode, cross-source and cross-runtime restores fail closed.

`evaluate(index, seed=...)` preserves training RNG; `export_state_dict()` and `load_export_state_dict()` transfer adapter-only inference payloads. An inference export cannot resume training; use the full training checkpoint. Backend rendering remains on `backend.render` using its declared response loop.

No GPU admission until objective, gradient, row binding, conditioning, domain, sampling, export/reload and exact-resume CPU gates pass. Prior MSE jobs remain held and are preserved as wrong-objective baselines. CPU qualification is not model-quality or live-memory qualification.
