# Embedded Z-Image shared-game preview

This is a separate opt-in **adapter GAN** cohort for the embedded
`conceptmod/textsliders/train_lora_zimage.py` family. The historical velocity UNI
trainer, helper equations, default environment and safetensors identities are
unchanged. This new game optimizes the existing shared BISPF adversarial response
residual objective, not UNI MSE. It has no routed bank or new GAN mathematics.

The ordinary native Attention `to_q/to_k/to_v/to_out.0` rank16/alpha16 LoRA delta
executes as a real HyperGAN-built HNDL Linear16/Linear graph. Native leading axes
are preserved. The shared `ImageResponseAdapter('zimage')` preserves native list
inputs, `(1000-t)/1000`, output negation and the `[1,C,H,W]` velocity field at the
shared game's penalty boundary. The shared HNDL spatial FieldCritic executes the
existing field/FiLM/SiLU/readout architecture. Its spatial mean is internal to the
critic; the game sees the entire native field.

This package owns preparation and reversible hook attachment. Installed shared
`ConditionalPG094`, response sessions and inference sessions own all losses,
penalties, optimizers, step budgets, RNG streams, EMA, checkpointing and export.

The frozen positive caption response is the target; the student uses the frozen
neutral caption encoding. Only raw velocity (historical `sample_guidance=0`,
shared `guidance=1`) and one native timestep are admitted. There is no negative
canary or control-hold loss in this new cohort. It never loads historical UNI
safetensors, and its exported artifacts cannot be relabeled as historical LoRA.

## Separate environment

Use Python3.11–3.14, a fresh environment, a public CPU Torch wheel, and install
this directory. Its unique distribution/namespace/entry-point names avoid sibling
preview collisions. Do not add this as an extra on the old core distribution.
The dependency pins the held draft core candidate
`7a8a321d04323d49319d0b08e6093a0ddce36b27`, official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10 and Diffusers0.42.1.
Core7a8 remains held pending further lifecycle corrections and review. Public
availability does not admit it for landing, native training or release; this
package uses it only for the bounded CPU software qualification reported here.
The core Python/graph/manifest closure is recursively hashed with relative
paths, so unexpected nested sources also reject before attachment. The verified
builder closure must match as well.

## Staged inputs and execution

Only an explicitly local Diffusers `ZImageTransformer2DModel.save_pretrained`
bundle with `config.json` and unsharded `diffusion_pytorch_model.safetensors` is
accepted. The preview currently admits finite FP32 CPU weights, image patch2 /
temporal patch1 and an even spatial lattice. It never downloads models. Native
production precision, resource use, pretrained quality and media rendering remain
unqualified. The CPU fixtures are small random instances of the real native model.

Create frozen preencoded rows in your existing trusted model-preparation workflow:

```python
from embedded_zimage_shared.backend import bundle_identity, PROTOCOL
packet = dict(schema=PROTOCOL, model_identity=bundle_identity(model_dir), rows=[
    dict(neutral={"embeds": neutral}, positive={"embeds": positive},
         negative={"embeds": unconditional}, latent=z, timestep=t),
    # At least two rows; z=[1,C,H,W], embeds=[tokens,cap_feat_dim], t=raw scalar.
])
torch.save(packet, rows_path)
```

No strings, tokenizer, text encoder or hidden pooling are substituted. The CLI
records `rows_sha256=file_sha256(rows_path)` in SessionConfig. Programmatic
callers must provide the same complete-file hash. Every byte, including metadata
or a trailing-byte change that torch.load might ignore, is checked before
deserialization and rechecked before native attachment. Native coordinates and
parsed cached content also participate in the shared game signature.

Model identity hashes exact config/safetensors file bytes with bounded1MiB reads.
A header/offset pass computes the matching native state digest without loading
weight tensors. The actual base is deserialized once by the native loader;
finite FP32 validation runs on that loaded model before attachment.

```sh
embedded-zimage-shared train --model-dir LOCAL_MODEL --rows frozen-rows.pt \
  --shape 1 4 4 4 --steps 4 --stop-after 2 \
  --checkpoint modern-resume.pt --export modern-ema.pt
# Continue with identical configuration/budget and --resume modern-resume.pt.
embedded-zimage-shared response --model-dir LOCAL_MODEL --artifact modern-ema.pt \
  --request native-request.pt --strength 0.5 --output native-velocity.pt
```

`native-request.pt` contains only `condition={"embeds": tensor}`, `latent` and
`timestep`. Native-only inference loads no training rows, critic or optimizer;
training files can be removed after export. Strength0 is exactly the frozen base;
intermediate strength changes the LoRA delta before native model execution.
Artifacts are source/model/geometry/profile bound. Declared model/shape/profile/source
requests and foreign cohort configs reject before native attachment. Deep same-config
training signature checks currently occur after reversible preparation and before
parameter restoration; they clean up on rejection. A packet-aware pre-prepare seam
is a shared-core follow-up, reported centrally. No universal pre-hook packet
admission claim is made here. Changed inputs fail closed;
old packets require their original loaders. No implicit migration is provided.

Training checkpoints bind the exact SessionConfig, including absolute model and
rows paths, rows file SHA256, source/runtime and total step budget. Moving a
training bundle changes that config and is refused even if files are identical;
there is no automatic path rebinding or checkpoint migration.

Native-only inference artifacts omit training paths/rows/optimizers. They can
use a different model directory containing byte-identical config and safetensors
files, under the same pinned package sources/runtime. Both raw and EMA exports
are tested after relocating the base and deleting original model/training paths.
Reserializing equal weights can change file bytes and is refused. This is path
relocation within the pinned runtime, not cross-version artifact portability.

Preview0.1.1 uses cohort/protocolv2 for the stricter complete-file identity.
Preview0.1.0/v1 artifacts remain tied to the original bca841e source pin; this
update refuses them and never silently relabels them.

Qualification commands and results are recorded in `QUALIFICATION.md`. No model
weights, package release, deployment, GPU run or merge is part of this preview.
