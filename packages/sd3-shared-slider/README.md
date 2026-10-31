# Opt-in SD3 shared game preview

This separate distribution leaves `conceptmod/textsliders/train_lora_sd3.py`, its
`prompt_pair.loss`, environments and artifact names byte-for-byte unchanged.
The new backend is `sd3-shared-hndl010`, using a fresh cohort name and the shared
PG0.9.4 adversarial response game. It optimizes frozen positive flow responses
from a neutral-caption student; it does not claim to preserve the historical
signed prompt-pair objective. There is no automatic checkpoint conversion.

Only staged CPU float32 `diffusers.SD3Transformer2DModel` objects are admitted.
All joint attention image/text q/k/v/output projections use rank/alpha LoRA
residuals executed by the verified HyperGAN develop/HNDL0.10 builder. Native
attention processors, pooled projections, full token encodings, raw scheduler
times and `[1,C,H,W]` latent geometry are preserved. The shared HNDL spatial
critic and native response game retain that geometry at the PG penalty boundary.
The critic's own spatial readout is the shared, parity-tested architecture.
No routed particle bank is claimed: this is the shared `adapter_gan` arm.

The core dependency is the lifecycle-correction candidate `7a8a321` in draft
PR148, on top of `4abc0eb`. Exact source hashes reject other same-version cores.
That central candidate is held pending further lifecycle corrections and review;
public availability is not admission. This downstream draft must not land on the
strength of this pin or the bounded tests below. HyperGAN is
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0 and official PG0.9.4/auto.
There is no wide-vector guard bypass or explicit-domain overlay.

Install this directory in a separate Python>=3.11,<3.15 environment. To request
CPU Torch explicitly, first install a compatible CPU wheel from the public
PyTorch CPU index. The install follows immutable public VCS pins, including
`#subdirectory=packages/particle-sliders-core`. Do not combine this with old-core
or legacy HNDL0.8.2 environments.

Use a local SD3 **transformer** directory (`save_pretrained` layout), never a Hub
identifier. Obtain `base_sha256` from
`particle_sliders_sd3_shared.runtime.model_identity(staged_transformer)`.
The identity hashes actual base weights, semantic model config and native
Diffusers module/attention-processor source files. Only the native
`JointAttnProcessor2_0` is admitted. Torch2.14.1, Diffusers0.38.0,
Transformers4.57.6 and safetensors0.8.0 are the bounded software cohort.

Bind the fixed cache bytes with `rows_sha256` (SHA256 of the rows file).
Also set `row_content_sha256` using
`particle_sliders_sd3_shared.backend.row_content_digest(rows)` for the actual
`ResponseRow` sequence. This semantic hash is required for file-backed and
in-memory rows. For a cache, use `row_content_digest(load_rows(cache_path))`.
The factory hashes and decodes the same cache bytes, captures independent row
copies, checks the semantic identity before binding, and records it in the shared
game signature. Caller mutation of staged tensors cannot alter those captured rows.
Training rows are a weights-only-loadable list of dictionaries saved by
`torch.save`. Each dictionary has exactly `neutral`, `positive`, `negative`,
`latent`, `timestep`. Each condition has exactly `embeds` `[1,T,joint_attention_dim]`
and `pooled` `[1,pooled_projection_dim]`; tensors are finite CPU float32.
The latent is `[1,in_channels,H,W]` on the genuine native patch lattice, and the
timestep is one finite raw scheduler coordinate in `[0,1000]`. Rows are fixed
raw-velocity samples, with CFG/strength/sign1 and one spatial geometry per cohort.
Frozen encodings stay product-owned; this preview does not train text encoders.

A session JSON has this form (replace paths and the hash):

```json
{
  "backend": "sd3-shared-hndl010",
  "cohort": "my-new-sd3-shared-review2",
  "options": {
    "model_path": "/staged/sd3-transformer",
    "rows_path": "/staged/sd3-rows.pt",
    "row_content_sha256": "ACTUAL_SEMANTIC_HASH_OF_RESPONSE_ROWS",
    "rows_sha256": "ACTUAL_SHA256_OF_ROWS_PT",
    "base_sha256": "ACTUAL_64_CHARACTER_SHA256",
    "base_precision": "float32",
    "rank": 4,
    "alpha": 4.0,
    "steps": 4,
    "batch": 2,
    "seed": 7,
    "ema_decay": 0.9
  }
}
```

```sh
sd3-shared-slider train --config cohort.json --out fresh-run --export-format both
sd3-shared-slider train --config cohort.json --resume fresh-run/training.pt --out another-fresh-run
sd3-shared-slider evaluate --config cohort.json --artifact fresh-run/adapter.pt --out response.pt
```

The additive repository entry point is
`python -m conceptmod.textsliders.train_lora_sd3_shared` with those same arguments.
The installed CLI directly uses the shared lifecycle CLI. Use `--stop-after` for a
split training budget, and `--selection ema` for an explicit EMA export.

Native-only inference uses `particle_sliders.inference.open_inference` with the
v2 artifact and `options={"model_path": "/staged/sd3-transformer"}`,
`operation="response"`. Call `session.response(latent=..., embeds=...,
pooled=..., timestep=..., strength=0.5)`. Only the frozen transformer and exported
adapter are needed; no rows, critic, optimizer or game are reconstructed.
Strength0 returns the exact native base; strengths0.5 and1 execute native hooked
adapter responses, without assuming global linearity through the transformer.

No image renderer/sampler, pretrained model quality, full SD3 weights, GPU,
mixed precision or native training campaign is qualified. Tiny randomly
initialized CPU **native Diffusers SD3** tests establish executable architecture
and lifecycle contracts only. No package release or model artifact is published.

The reviewed update binds `core_lock.json` bytes in the package source lock and
records their digest in each projection's signed architecture provenance. Direct
`prepare` uses a local CPU `torch.Generator`; a scoped compatibility context feeds
HNDL/native constructors that lack a generator argument and restores the caller's
default Torch RNG on success and failure. Construction remains single-threaded;
this does not isolate concurrent users of the default RNG.

These are bounded product input/source gates. Deep packet-aware admission before
native preparation is an unresolved shared-core gap: a malformed optimizer or
other deeper packet field can be detected after preparation and hook attachment.
Cleanup/rollback tests do not establish universal pre-hook refusal. The new source
identity and required row hash reject automatic rebinding of artifacts from the
initial `32abf53` preview; retain that exact environment for its existing packets.
