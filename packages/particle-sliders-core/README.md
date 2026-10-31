# particle-sliders-core0.11.0 — modern PG0.9.3/Gram target

Read [MODERN-GRAM.md](MODERN-GRAM.md) first for the new exact-release, optimizer-policy, split-evaluation and adoption contract. This source candidate requires officialPG0.9.3; old091/0.5 overlays and packets are not silently admitted. Current runtime qualification is recorded only in the exact candidate receipt.

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
`particlegan==0.9.3`, Torch>=2.6,<3 (also required by upstream), Python>=3.11 (Pillow 13 security baseline for the test extra).
Executed qualification uses Python3.12.15/Torch2.14.1+cpu; the declared lower
bounds and other runtime combinations are untested. Modern HNDL extras require
Python>=3.11,<3.15. Earlier core0.8/0.10 source pins remain immutable. See [LEGACY-MIGRATION.md](LEGACY-MIGRATION.md)
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
- [CUT-POINTS.md](CUT-POINTS.md): native attachment/readout boundaries, masks,
  scale and gradient ownership, extension examples and provenance limits.
- [PEFT-SCALING.md](PEFT-SCALING.md): opt-in exact-name scaling that preserves
  fixed companions, strict scope admission and captured checkpoint replay.
- [CONDITIONAL-PG09-CONTRACT.md](CONDITIONAL-PG09-CONTRACT.md): strict game/resume.

Software evidence covers synthetic CPU contracts, not pretrained convergence,
accelerator fit, native image quality or acoustic rendering. Iris rendering stays
disabled pending native solver parity; Krea2 is adapter-only (`particles=None`).
Qwen/Fibo native dependencies remain in separate environments. Official PG0.9.1
wide continuous audio states remain refused; no unofficial upstream proposal is
included or activated. MOSS requires a freshly qualified product target on this exact modern line. Mere backend
registration does not qualify a product migration.

Optional `[serialization]` adds safetensors for native adapter/cache IO;
`[hndl]` adds the distinct optional conditional readout critic. No campaign
launchers or reverse product imports enter this package closure.
