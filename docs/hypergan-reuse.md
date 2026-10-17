# Optional HyperGAN response diagnostics

The reusable implementation is **HyperGAN/HyperGAN**, distinct from ParticleGAN.
Live `git ls-remote https://github.com/HyperGAN/HyperGAN.git refs/heads/develop`
and an isolated fetch on 2026-10-17 resolved develop to
[`a879c4fbff903ccefeba2194bcc7a13fe7b5abb7`](https://github.com/HyperGAN/HyperGAN/commit/a879c4fbff903ccefeba2194bcc7a13fe7b5abb7).
The pre-existing assessment checkout at `bff6de3d` was older.

`particle_sliders.hypergan_diagnostics.response_diversity` reuses the public
[`DiversityMoments`](https://github.com/HyperGAN/HyperGAN/blob/a879c4fbff903ccefeba2194bcc7a13fe7b5abb7/src/hypergan/diversity_metrics.py)
implementation. It computes streaming distinct-pair RMS in centered float64 CPU
moments, supports native response tensors, and avoids storing every response or
allocating an all-pairs matrix. The bridge adds explicit condition grouping,
portable provenance and undefined-value reporting. It contains no copied
variance or pairwise-distance implementation.

## Installation and use

The integration is optional and imported only when requested. The shared package
has no new required dependency. HyperGAN develop requires Python >=3.11,<3.15;
the base slider package retains its existing Python compatibility. Use a separate
evaluation environment and the reviewed **base** HyperGAN package:

```sh
python -m pip install 'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@a879c4fbff903ccefeba2194bcc7a13fe7b5abb7'
```

No `train` extra is needed for these diagnostics. Use the torch version already
qualified for the evaluated slider environment. Installation into active pinned
training environments is outside this change.

```python
from particle_sliders.hypergan_diagnostics import response_diversity

# game and prepared_rows are supplied by the existing qualified campaign.
# These indices must represent ONE fixed condition and multiple latent samples.
indices = [0, 1, 2, 3]
report = response_diversity({
    "prompt-A/scale+1/cfg3/schedule-v1": (
        {"generated": game.evaluate(i, seed=73),
         "reference": prepared_rows[i].target}
        for i in indices
    )
})
```

Each condition ID must bind the conditioning/prompts, adapter scale, guidance,
timestep or schedule and geometry. Labels are caller declarations, not verified
semantic identities. Changing these within a group can hide latent collapse.
The bridge emits no pooled score across conditions. Reference and generated
sample counts may differ; neither side is repeated or truncated. Singleton
batches accumulate into a group with at least two samples per side. Constant
references retain zero absolute spread and an unavailable ratio; insufficient
samples and mismatched native shapes also produce explicit unavailable reasons.
Invalid/nonfinite tensors reject the observation rather than publish a partial
result. `pool_size=4` explicitly requests image NCHW average pooling; leave it
unset for audio, text fields, video tensors and arbitrary latent responses.

Reports record the reviewed HyperGAN commit and actual diversity source SHA-256
`966a3aa4aab5cf833ff2632878e47a3aea2243781ff5c4699497159fada552f9`.
The source must match; an unreviewed implementation raises an actionable optional
dependency error. A matching module hash establishes the reviewed metric source,
not the identity of the entire installed HyperGAN repository. The report is JSON
serializable with `allow_nan=False`.

This shared API works with supplied tensors from every derivative family; each
family owner can call it after deterministic evaluation. No product entry point
is claimed to have adopted it in this change. It does not replace semantic
quality evaluation: independent noise can have high diversity, and equal spread
does not establish realism, intent or coverage.

## Other reuse opportunities and boundaries

HyperGAN's [`build_network` and `HNDLNetwork`](https://github.com/HyperGAN/HyperGAN/blob/a879c4fbff903ccefeba2194bcc7a13fe7b5abb7/src/hypergan/hndl_networks.py)
support native named input ports and approved local pretrained providers. Its
[`pretrained_providers`](https://github.com/HyperGAN/HyperGAN/blob/a879c4fbff903ccefeba2194bcc7a13fe7b5abb7/src/hypergan/pretrained_providers.py)
include local source-verified DINOv3 readouts. These are relevant to optional
image feature critics when a staged backbone is justified; the separate HNDL
owner handles that architecture work. They add no value to a tiny generic
readout merely by replacing its constructor.

HyperGAN's trainer, objective compiler and checkpoint serializer bind its own
graph, prior, EMA, data and optimizer layout. They cannot replace slider games or
strict source/runtime signatures without a separate reviewed migration. Existing
ParticleGAN release pins, continuous-domain overlays, active cohorts, training
losses, samplers, checkpoints and product implementations remain governed by
their existing owners. This change neither selects ParticleGAN develop nor
substitutes HyperGAN training for the shared GAN game.
