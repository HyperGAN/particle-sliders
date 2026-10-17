# particle-sliders-core

The installable training and native-response API for HyperGAN slider products.
Version **0.6.0** requires released **ParticleGAN 0.9.1**, PyTorch 2.6+ and Python
3.10+. Optional model packages load only when a native backend is constructed.

## Install

Install your tested PyTorch build, then pin a reviewed full Git commit:

```bash
python -m pip install "particle-sliders-core[serialization] @ git+https://github.com/HyperGAN/particle-sliders.git@<reviewed-commit>#subdirectory=packages/particle-sliders-core"
```

An installed wheel works outside this repository. No `conceptmod` checkout,
workspace path variable or copied backend is needed. The `serialization` extra
provides safetensors for native adapter/cache IO. `hndl` provides the optional
HNDL critic on Python 3.11–3.14; model-specific dependencies remain explicit
product/runtime dependencies. Imports never load weights or initialize CUDA.

## Public API

| Import | Purpose |
|---|---|
| `particle_sliders.ConditionalPG09` | One shared conditional GAN update and full training/inference state |
| `particle_sliders.pg09.validate_pg09_domain` | Pre-forward native shape and exact dependency-source gate |
| `particle_sliders.pg09_backends` | `AdapterParameters`, response rows, `FlowSchedule`, signed checkpointing, `lora_scale_context`, Anima/Supra/MOSS adapters, native adapter IO |
| `particle_sliders.pg09_images` | `ImageResponseAdapter`, `IrisResponseAdapter`, `QwenImage21ResponseAdapter`, `image_condition_features` |
| `particle_sliders.pg09_audio` | `AudioRow`, YuE2/Music3 AR response adapters, `AudioResponseCritic`, `audio_conditions` |
| `particle_sliders.pg09_readout_critic` | `ConditionalReadoutCritic` with raw scalar logits |
| `particle_sliders.pg09_moss_campaign` | Checksummed lazy cache provider, native field critic and MOSS run builder |
| `particle_sliders.pg09_remaining_campaigns` | Shared Supra/final-form cache bindings and teacher-parity gate |
| `particle_sliders.music3_native`, `native_lora`, `supra_model` | Reusable native architectures with lazy explicit loaders |
| `particle_sliders.tiny_llm_backend`, `tiny_native`, `bonsai_native`, `pg09_text_video` | Native Tiny/Bonsai architectures and text/video response bindings |

The repository compatibility paths re-export the same classes. The deprecated
`concept_slider_core` import warns and re-exports the public root API;
`packages/concept-slider-core` is a retired directory, not an install target.

## Training, resume and inference

```python
import torch
from particle_sliders import ConditionalPG09

# The product supplies already attached frozen native components, frozen rows,
# and detached context/time/sign coordinates. Backend.network owns only the
# actual active FP32 adapter leaves. Its particles must be the real routed bank.
game = ConditionalPG09(
    backend, prepared_rows, critic, conditions=conditions, particles=particles,
    response_domain="continuous_field", response_scale=1.0,
    preset="bispf_optimized", batch_size=4, total_steps=1000, seed=7,
)
metrics = game.step()
torch.save(game.state_dict(), "training.pt")
# Reconstruct the same source/runtime/model/rows/config before continuation:
game.load_state_dict(torch.load("training.pt", weights_only=True))
adapter_packet = game.export_state_dict()
# Reconstruct an equivalent game for source-bound inference:
inference_game.load_export_state_dict(adapter_packet)
response = inference_game.evaluate(0, seed=23)
```

Generator training is exclusively public PG adversarial loss plus the public
prior regularizer; the critic uses public adversarial loss plus public penalty.
No MSE/reconstruction term is optimized. Conditions bind native context, time
and sign; the backend retains the row scale through checkpoint recomputation.
D/G row and Gaussian-reference streams are independent and checkpointed.

Full checkpoints include actual network/bank, critic, optimizers, observers,
private RNG streams, sampler coverage and completed steps. Restore validates
before live mutation. Failed partial updates cannot be saved or exported.
Inference exports preserve completed steps and cannot become training packets.
Evaluation preserves adapter buffers/modes and training RNG; frozen model eval
mode remains the backend's responsibility. An AR hidden-state probe is not a
waveform render or autoregressive token sampler.

## Dependency and domain choices

The caller-owned game explicitly binds `training_precision="float32"`.
Upstream 0.9.1 defaults also enable BF16/channels-last in its own supported CUDA
trainer and batched convolution optimizer execution; selecting the dependency
alone does not qualify that precision policy in a native host loop.
[Release details](https://github.com/255BITS/ParticleGAN/pull/450).

Native fields stay `[1,C,H,W]` or `[1,C,T,H,W]` through the penalty. Released
0.9.1 still interprets rank-2 vectors of width >=170 as categorical: native
continuous vectors at those widths fail closed under the default release
profile. The explicit `alz-continuous-v1`/`continuous` pair requires the exact
separately staged 0.9.1 source profile recorded in
`pg09_continuous_profile.json`. Installing the package does not install or
modify that overlay, project/reshape responses, or select another loss.

See [the complete protocol](CONDITIONAL-PG09-CONTRACT.md). All 49 dependency
Python sources, implementation classes, runtime, architecture, frozen model and
conditioning are checkpoint-bound. Relocation and 0.9.0→0.9.1 change identity:
active old cohorts keep their original pins; migration needs explicit review
or a new cohort. CPU tests establish software contracts, not native GPU memory,
rendering parity, convergence or perceptual quality.

## Historical arms

`winning_formulation()` remains the historical gmix stamp, not the new game.
`FormulationGame`, `EndpointGame` and `BISPFGame` retain named historical
objectives; EndpointGame includes MSE terms, and the old wide-vector BISPF path
has no new domain gate. A stamp import or dependency update does not migrate
an actual trainer. New GAN-only products call `ConditionalPG09.step()`.
Historical provenance JSON describes the original arms, not current dependency
selection. See [shared architecture](../../docs/shared-core.md).
