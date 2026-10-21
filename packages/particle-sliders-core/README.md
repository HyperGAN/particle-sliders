# particle-sliders-core 0.8.3 — official PG0.9.1 lifecycle

Model product -> particle-sliders -> HyperGAN / ParticleGAN. Products own native
loading, packing, schedulers, adapter attachment and their complete factory;
particle-sliders owns session preparation, admission, train/resume, EMA,
evaluation, export/native-only inference and cleanup. Upstream ParticleGAN owns
GAN mathematics. HyperGAN/HNDL reporting and matched-weight network migration
remain separate integration work; native pretrained backbones stay native.

This source integration includes the reviewed image and numerical response
lifecycles plus a narrow legacy compatibility layer. It is not a package release.
Install a full reviewed Git pin using `#subdirectory=packages/particle-sliders-core`
or its exact wheel; retain that pin, source map and receipts. Dependency is exact
`particlegan==0.9.1`, Torch>=2.6,<3 (also required by upstream), Python>=3.10.
Core0.8.1/0.8.2 source pins remain immutable. See [LEGACY-MIGRATION.md](LEGACY-MIGRATION.md)
for preserved constructors/stamp fields, explicit retired-kernel refusals and
runnable old/core0.3 instructions; this upgrade does not migrate checkpoints.

Start with [IMAGE-LIFECYCLE.md](IMAGE-LIFECYCLE.md) for the complete product factory
and common image session. [RESPONSE-SESSIONS.md](RESPONSE-SESSIONS.md) documents
numerical response registration; a hidden response is not an image or waveform.
The independent installed `examples/response_fixture` plugin needs only local
scalar files and demonstrates the entire train/resume/export/data-free inference
flow. Core imports do not load pretrained models or initialize CUDA.

- [NATIVE-INFERENCE.md](NATIVE-INFERENCE.md): loading base and adapter without
  teacher data, critic, optimizer or training configuration.
- [CLI-ARTIFACTS.md](CLI-ARTIFACTS.md): v1 default versus explicit v2/both exports.
- [PROVIDER-OBSERVATION.md](PROVIDER-OBSERVATION.md): observation covers provider
  fetch as well as forward, with state/RNG restoration.
- [CHECKPOINT-OWNERSHIP.md](CHECKPOINT-OWNERSHIP.md): reversible native wrappers.
- [CONDITIONAL-PG09-CONTRACT.md](CONDITIONAL-PG09-CONTRACT.md): strict game/resume.

Software evidence covers synthetic CPU contracts, not pretrained convergence,
accelerator fit, native image quality or acoustic rendering. Iris rendering stays
disabled pending native solver parity; Krea2 is adapter-only (`particles=None`).
Qwen/Fibo native dependencies remain in separate environments. Official PG0.9.1
wide continuous audio states remain refused; no unofficial upstream proposal is
included or activated. MOSS uses its separate officialPG0.9.3 line. Mere backend
registration does not qualify a product migration.

Optional `[serialization]` adds safetensors for native adapter/cache IO;
`[hndl]` adds the distinct optional conditional readout critic. No campaign
launchers or reverse product imports enter this package closure.
