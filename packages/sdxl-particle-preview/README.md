# SDXL shared response preview

This opt-in distribution runs the installed shared ParticleGAN 0.9.4 lifecycle
and actual verified HyperGAN develop/HNDL 0.10 graphs. It uses ordinary rank-4,
alpha-1 attention LoRA (configurable rank/alpha), a shared HNDL spatial critic,
and the `adapter_gan` arm. It has no routed particle bank. It contains no copied
GAN loss, penalty, optimizer, checkpoint implementation or update loop.

The historical function bodies (`flush`, `train`, `main`, `train_lora`), including
`prompt_pair.loss`, defaults and safetensors names, remain unchanged. The trainer
**file does change**: an 11-line addition defines the explicit `train_shared`
dispatcher that calls `sdxl_particle_preview.cli.train_shared`. It adds a separate
objective/cohort; the historical CLI continues calling `main` as before.
Historical checkpoints are never interpreted as shared packets. No release is
made here.

Use a separate Python 3.11–3.14 environment. Install CPU Torch separately (the
public CPU index is https://download.pytorch.org/whl/cpu), then:

```sh
python -m pip install ./packages/sdxl-particle-preview
sdxl-particle --config session.json --checkpoint new.training.pt --export new.inference.pt
```

The VCS core pin is draft PR 148 revision
`7a8a321d04323d49319d0b08e6093a0ddce36b27`, including the lifecycle fixes
present in that draft revision. It remains **held pending
further lifecycle corrections and review**, and public availability is not
admission. The recursive core source/data closure, including nested files and
LICENSE, is checked before this product's adapter attachment; generated Python
bytecode is excluded.
It uses public PG 0.9.4/release-auto and HyperGAN
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. Do not combine this package with the
historical core/HNDL environment. There is no shared-core source edit here.
Root review and final core admission are required before promoting this preview.

Example `session.json`:

```json
{"backend":"sdxl-native-hndl-v1","cohort":"sdxl-native-hndl-v1:my-new-experiment",
 "options":{"model_dir":"/already-staged/sdxl","rows_path":"/local/frozen-rows.pt",
 "base_identity_sha256":"<64-lowercase-hex-digest-from-the-staged-base>",
 "shape":[1,4,64,64],"rank":4,"alpha":1.0,"steps":4,"batch":2,
 "seed":7,"ema_decay":0.9,"preset":"bispf_optimized"}}
```

`base_identity_sha256` is mandatory. Compute it from the exact staged frozen
base before writing the session configuration:

```python
from sdxl_particle_preview.backend import load_model
from sdxl_particle_preview.native import base_identity_sha256
print(base_identity_sha256(*load_model("/already-staged/sdxl")))
```

It binds every frozen UNet tensor, native model/scheduler configurations, prediction
type and inspected implementation sources. The digest participates in the shared
SessionConfig and native inference specification. Training preparation compares
the actual base with the configured digest before installing product hooks; a
changed digest in a resume configuration is rejected by the shared config check
before resolving the factory. Moving or changing model files without changing
`model_dir` cannot bypass the actual-base digest check. Existing preview packets
without this binding require their original source or a newly named cohort.

The model directory must contain local Diffusers `unet/` and `scheduler/`
components. No download, GPU selection, VAE or text-encoder load occurs. This
preview admits CPU float32 with the native math `AttnProcessor` and DDIM/DDPM
scheduler only. Supplied borrowed models must already use that processor;
checkpoint recomputation and pre-existing adapters are refused before mutation.

The rows file is a weights-only-loadable list of at least two dictionaries with
`neutral`, `positive`, `negative`, `latent`, `timestep`, `guidance` (optional,
default 1). Each condition dictionary must hold frozen float32 `embeds` with
shape `[1,tokens,cross_attention_dim]` (concatenated hidden states from both CLIP
encoders), `pooled` with `[1,projection_dim]` (the second encoder's pooled
projection), and `time_ids` with `[1,6]` ordered original height/width, crop
top/left, target height/width. Target dimensions equal latent height/width times
8. Every row is a genuine `[1,4,H,W]` noise-prediction field with an integral
scheduler timestep. `scale=strength=1` declares the new unipolar frozen positive
response target. The student uses neutral conditioning; CFG's negative branch
stays frozen. Epsilon and v-prediction are kept in their native coordinates;
there is no fake flow conversion, pooling of the game response, or scheduler
step/render claim. Use separate cohorts when any model, condition, timestep,
shape, source, scheduler or configuration differs.

For Python, use `particle_sliders.response.open_response(SessionConfig(...),
model=unet, scheduler=scheduler, rows=rows)` or the explicit `train_shared`
function in `train_lora_xl.py`. The shared session owns updates, EMA, evaluation,
exact resume and failure cleanup. Native v2 export uses
`particle_sliders.inference.export_inference`; reopen using `open_inference`
with `options={"model_dir": ...}`, `operation="response"`. Call its `response`
with a fresh validated `row` and `strength` in `[0,1]`. Inference reads no
training rows or critic/optimizer state. This exports a response adapter packet,
not historical LoRA safetensors and not rendered images.

Admission limits are explicit: the product validates its configuration, actual
base digest, runtime closure and native row contracts before attaching its hooks.
This is **not universal packet-aware pre-hook refusal**. Deep shared packet and
restore checks can run after backend preparation and attachment; packet-aware
pre-prepare admission remains a separate shared-core gap. Cleanup/rollback tests
do not establish that every invalid packet avoids native preparation/execution.

See [VALIDATION.md](VALIDATION.md) for exact commands, runtime and results.

CPU fixtures execute an actual small random Diffusers text_time UNet, ordinary
LoRA matched-weight output/VJP parity, the real HNDL field critic including
second derivatives, shared updates, exact split resume, EMA, zero/intermediate/
full strength, native-only inference after removing training data, admission
refusals and cleanup/borrowed-state restoration. These establish software
conformance only. Full pretrained SDXL, dual text-encoder extraction, mixed
precision, native image sampling, GPU/resource/quality qualification remain
unperformed. No model weights or package releases are part of this preview.
