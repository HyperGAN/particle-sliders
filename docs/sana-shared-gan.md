# Opt-in Sana shared GAN response preview

The existing `train_lora_sana.py`, YAML, `slider_targets.py` UNI objective and
checkpoint names remain unchanged. Continue using that historical environment
for existing xattn/ordinary-LoRA UNI artifacts. Do not load or relabel them as
GAN artifacts.

`packages/particle-sliders-sana` is a separate installable backend distribution
inside this repository. It opts into the new cohort
`sana-neutral-cfg-velocity-hypergan-hndl-v1`. This is an objective change, not an
upgrade preserving the historical UNI loss. There is no automatic dispatch
from the historical trainer.

## Exact environment

The package declares Python 3.11–3.14 eligibility; the qualified CPU lane uses
Python3.12.15 and Torch2.14.1+cpu in a separate environment. The package pins
the exact shared
core0.12 source at `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` (draft core
[PR #148](https://github.com/HyperGAN/particle-sliders/pull/148)), official
ParticleGAN0.9.4, actual HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0 through the common
`hypergan-hndl` extra. These remain preview dependencies, not a published
core0.12 release or a pretrained Sana qualification.

```bash
python3.12 -m venv .venv-sana-gan
printf 'torch==2.14.1+cpu\n' > /tmp/sana-cpu-constraints.txt
.venv-sana-gan/bin/python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --constraint /tmp/sana-cpu-constraints.txt './packages/particle-sliders-sana[test]'
.venv-sana-gan/bin/python -m pip check
.venv-sana-gan/bin/python -m pytest -c /dev/null \
  --confcutdir=packages/particle-sliders-sana/tests packages/particle-sliders-sana/tests -q
```

For offline qualification, install the exact core candidate wheel (SHA256
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`)
and its locked PG/HyperGAN/HNDL dependencies, then install the Sana wheel with
`--no-deps`. Both the offline wheel lane and full public dependency resolution
have passed CPU checks. Every installed shared-core package source **and data**
file must match `core012.lock.json`, and installed `direct_url.json` must identify
the exact Git commit/subdirectory or exact candidate wheel SHA256; a version
string alone is refused. The shared verified builder also verifies the
HNDL sources, HyperGAN builder closure, clean build identity and graph files.
A different candidate requires a separately reviewed source lock/cohort.
Do not install the legacy HNDL0.8.2 extra in this environment.

## Native contract

The host stages the frozen Sana transformer and the real text embeddings and
attention masks. This distribution never downloads a model, encodes text,
starts training on import, or chooses a GPU. The live declaration requires the
exact `Efficient-Large-Model/Sana_600M_512px_diffusers` ID, an immutable model
revision and staged file SHA256 map. Actual base tensors, native model source,
config, target names/dimensions, product source and shared source lock are also
bound to the configuration. This backend never imports Diffusers itself; the
host owns its chosen native model/encoder environment. Hashing staged tensors
reads each base tensor; no
base weights enter an exported artifact.

Only float32 base weights are admitted in this preview. Native BF16 execution,
full pretrained forward/backward, CUDA, decoded rendering and visual quality
are unqualified. The tiny CPU fixture implements the Sana call signature; it is
not Sana0.6B. The runtime has no `/dev/nvidia*`; GPU occupancy is unknown, GPU0
is reserved, and no GPU checks were attempted.

| Component | Opt-in behavior |
|---|---|
| Student | Raw velocity with the neutral caption at every strength |
| Frozen positive teacher | `v(unconditional) + guidance * (v(positive) - v(unconditional))`; guidance defaults to 4.5 |
| Shared `ResponseRow.negative` | Explicit empty-prompt encoding/mask matching the frozen unconditional encoding declared in configuration; never a negative-concept teacher |
| Zero strength | Exact frozen native forward, including masks and timestep convention |
| Native response | `[1,C,H,W]` velocity, preserving rectangular geometry; no vector reshape or pooling before PG |
| Timestep | Raw scheduler timestep multiplied by native `config.timestep_scale` by the shared Sana adapter |
| Output | First native velocity channel half if the model emits doubled channels |
| Adaptation | Frozen base; `transformer_blocks.*.attn2.to_q/to_k/to_v/to_out.0` only |
| Routed network | Shared HyperGAN-built HNDL rank8/alpha8 projections with exactly one128x4 trainable bank |
| Critic | Shared HyperGAN-built HNDLFieldCritic, conditioned on exact capture-row identity |
| Game | Shared `ConditionalPG094`, official BISPF preset, shared penalties, optimizers, sampling and checkpoints |

This new path does not optimize `sana_uni_loss`, copy GAN mathematics, add a
negative teacher or reinterpret a wide vector as a field. The official shared
width>=170 continuous-vector refusal remains in force. Image fields use their
genuine native image-domain admission.

## Explicit staged API

The existing host owns model staging and conditioning. `rows` below is a list
of at least two shared `ResponseRow` captures, each with one frozen latent,
one raw scheduler timestep. `neutral` and `positive` are dictionaries with keys
`embeds` and `mask`. The `negative` slot is an unconditional dictionary with
exactly `embeds`, `mask` and `prompt`, where `prompt` must be the empty string.
Pass the same frozen empty-prompt encoding to `configuration(unconditional=...)`;
every row's negative encoding and mask must match its bound tensor fingerprint.
A missing/nonempty prompt declaration or a swapped encoding is refused before
forward hooks. The host remains responsible for actually encoding the declared
empty prompt: the binding cannot infer caption semantics from arbitrary tensors.
Keep `row.guidance=1`, `scale=1` and `strength=1`; declare **teacher** guidance in
`configuration`. Trajectory schedules are refused by this velocity cohort.

```python
from particle_sliders.lifecycle import open_session
from particle_sliders.inference import open_inference
from sana_particle_gan import configuration, staged_model

# transformer, rows and declaration come from an already-staged native host.
# declaration = {"model_id": ..., "revision": <40-hex>,
#                "staged_files_sha256": {<local relative filename>: <64-hex>, ...}}
transformer.requires_grad_(False)
config = configuration(
    transformer, model_identity=declaration,
    response_shape=tuple(rows[0].latent.shape), unconditional=rows[0].negative,
    steps=4, batch=2, seed=7, guidance=4.5, ema_decay=0.9,
)
with open_session(config, model=transformer, model_identity=declaration, rows=rows) as session:
    # The caller may invoke shared lifecycle operations in an authorized run.
    # session.train(); session.checkpoint("training.pt")
    response = session.evaluate(strength=0.5)
    session.export_inference("native.pt", selection="raw")

# Native-only reload needs no rows, teacher cache, game, critic or optimizers.
# The same frozen native base is explicitly staged again by the host.
with staged_model(transformer, model_identity=declaration):
    with open_inference("native.pt", operation="response") as native:
        response = native.response(
            strength=0.5, latent=native_latent, timestep=native_timestep,
            condition=native_neutral_encoding,
        )
```

After updates, raw or EMA exports use the shared v2 native-only artifact.
Training checkpoints keep the exact original configuration and sources;
resume uses `open_session(config, resume=..., model=..., model_identity=...,
rows=...)`. Invalid configuration, source, model and declared geometry are
refused before hook attachment; the shared packet/optimizer validation refuses
incompatible state without mutation. Construction failures and close remove
owned forward attributes and restore borrowed modes/checkpoint settings. Cleanup
preserves the caller's original instance-attribute layout and any later foreign
forward replacement. Failed cleanup leaves the product callback retryable;
retry `session.binding.close()` if cleanup raised (the current shared wrapper
marks the session closed before invoking its callback). The wrapper issue is
shared-core follow-up work, not a local shared API patch.
Open at most one binding on a given staged transformer at a time.

`render` is explicitly unavailable. A future separately reviewed native Sana
pipeline sampling binding is needed to qualify image inference. This preview
must not replace the working UNI entrypoint on the basis of these CPU checks.
Whether a future render path should compose CFG again remains an unqualified
integration question; this preview changes no historical render defaults.

## CPU evidence and release impact

The installed-wheel tests exercise the real shared PG game and actual
HyperGAN-built HNDL modules: matched projection outputs and input/particle VJPs,
matched field-critic input and penalty second derivatives, generator/bank and
critic updates, exact uninterrupted/split resume, raw/EMA exports, strengths
0/0.5/1, native-only reload without training state, source/profile/model/shape
and builder-tamper rejection, frozen-base ownership and rollback. The original
23 Sana UNI tests still pass unchanged.

These are small CPU software checks, not a pretrained/GPU qualification. No
pretrained weights were downloaded, training campaign run, weights published,
model deployed, package released or default dependency changed. A merge adds
one optional source distribution and docs. Merge remains dependent on reviewed
core PR148 landing and a coordinated exact repin/source-lock review. This branch
retains `4abc0eb` until that decision; a later or moving core head is not admitted.
