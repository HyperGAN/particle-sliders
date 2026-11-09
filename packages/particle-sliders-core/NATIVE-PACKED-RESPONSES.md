# Native packed rank3 responses

The explicit `continuous_packed` response domain preserves native `[1,T,C]`
geometry in shared training, critic penalty, full checkpoints and response
inference. It does not flatten, project, subsample or reinterpret tokens as
spatial image axes. Products own their actual full native response contract,
position identity, conditioning, critic, loader and numerical sources.

Use ParticleGAN0.9.5's unchanged release profile and `penalty_input_domain="auto"`.
Its official BISPF penalty uses its rank>=3 branch, named `image` internally;
the shared signature records that resolution. This is the upstream rank law,
not a claim packed coordinates are spatial pixels. `input_domain="continuous"`
does not disable that upstream rank branch and is not a packed-domain option.
No GAN loss, optimizer, regularizer, shape gate or scalar statistic changes.
Existing vector and genuine4D/5D field contracts and refusals remain unchanged.

Declare `adapter_gan` for an ordinary LoRA without a particle bank. Actual routed
networks separately declare their real bank; this API invents none. A response
descriptor does not provide a media renderer or establish native/pretrained
qualification. Separate product inference/render bindings own finished media.

Prospective fresh components use the explicit scoped initialization API before
optimizers, preserving pretrained/zero-output/tuned host laws. Restores and
inference never reinitialize. Use RAW/no weight EMA; preserve source-bound FULL
state and exact current native tensors. Sole effect ranking and the separate
qualified catastrophic-only eligibility check remain unchanged. CPU/native tests
and product source admission must be reported as actual evidence, not inferred
from a descriptor or this documentation.
