# particle-sliders-core 0.8.0 clean image lifecycle candidate

Product -> particle-sliders -> ParticleGAN/HyperGAN. Native model dependencies
remain isolated in each product environment. The exact released ParticleGAN
0.9.1 wheel/source gate owns GAN mathematics. Existing named legacy root APIs
remain explicitly available for compatibility; new integrations use lifecycle.

Start with [IMAGE-LIFECYCLE.md](IMAGE-LIFECYCLE.md) for a complete registered
product factory and common train/resume/EMA/response/render lifecycle. Use
[NATIVE-INFERENCE.md](NATIVE-INFERENCE.md) for native-only inference artifacts,
CLI examples, capability refusals and preparation/update/observation hooks.
Install a reviewed commit using the packages/particle-sliders-core subdirectory,
or its built wheel; test fixtures run offline outside the source checkout.

This main-based candidate transplants only the necessary image package closure,
contracts, tests, licenses and provenance. It does not import historical MOSS,
audio, text or campaign orchestration. Those integrations retain their original
pins until separately transplanted and qualified. Passive response adapters in
pg09_backends are reusable definitions, not evidence of completed migration.

Optional [serialization] installs safetensors for native adapter/cache IO;
[hndl] installs the distinct optional conditional readout critic. Matched-weight
HNDL migrations of existing networks and native pretrained quality remain
separate work. Importing the package never loads models or initializes CUDA.
