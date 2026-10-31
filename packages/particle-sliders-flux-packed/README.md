# Flux packed-token prerequisite preview

**Shared training is blocked. This is not a migrated trainer.** Core0.12 at
`7a8a321d04323d49319d0b08e6093a0ddce36b27` (draft PR148) cannot admit native
packed Flux `[1,image_tokens,64]` responses. Its domain gate accepts only 2D
vectors and genuine 4D/5D fields; its response capability descriptor has the
same limitation. No shared training backend is registered by this package.

This isolated opt-in distribution supplies executable prerequisites: actual
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` / HNDL0.10 graphs
for ordinary rank4/alpha1 down/up LoRA and a native Diffusers0.38 packed forward
adapter. It verifies all 49 shared Python/JSON/HNDL files, the builder closure,
the local graph, and six relevant native forward source files before hooking.
It uses a unique `ps_flux_packed_preview` namespace and changes no default
environment, historical file, checkpoint or export identity.

Install in a separate CPU environment, from the repository root:

```sh
python3.12 -m venv .venv-flux
.venv-flux/bin/python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv-flux/bin/python -m pip install './packages/particle-sliders-flux-packed[test]'
.venv-flux/bin/python -m pytest -c packages/particle-sliders-flux-packed/pyproject.toml packages/particle-sliders-flux-packed/tests
```

The exact core VCS source is publicly reachable through a draft; no core or
preview package release accompanies this work. Use installed wheels, not the
repository's root pytest Python path (which selects its historical core).

`FluxLoRA` accepts an already staged frozen float32 CPU `FluxTransformer2DModel`
and builds its graphs **unattached**. Its `borrowed()` context installs removable
hooks and restores all original module modes on success or failure. The model's
original forwards and parameter/gradient flags remain owned by the caller.
Only current native `FluxAttention` linear leaves are selected, with `add_*`
excluded as in the historical selector. Diffusers0.38 renamed the class used by
the old trainer's `Attention` match; the old trainer is preserved byte-for-byte.
The new namespace does not claim historical checkpoint compatibility.

`PackedFluxResponse` consumes precomputed native encodings and token IDs directly.
It preserves the packed image token and feature axes and forwards `[text_tokens,3]`
and `[image_tokens,3]` positional IDs as required by this exact Diffusers API.
Guidance-embedding and non-guidance models are distinguished. It returns raw
packed velocity at `raw_t/1000`; it never invokes a scheduler, pools, unpacks or
relabels axes. Its inherited frozen-positive teacher convention belongs to an
explicit new prerequisite cohort. The historical `prompt_pair.loss` trainer's
**scheduler-step latent** objective is unchanged and is not declared equivalent.

Call `preflight_training((1, tokens, 64))` before loading or touching a native model.
It raises `PackedTrainingUnavailable`, including for a one-token fixture. A
reviewed native packed continuous domain, matching penalty treatment, critic and
response lifecycle capability are needed centrally. Flattening to a small vector
or adding spatial dimensions is not admission. Native shared updates, resume,
EMA, export, critic second derivatives and shared native-only inference are
therefore **unqualified and unexecuted** here. No dummy success path substitutes
for them. The CPU fixtures exercise real randomly initialized Flux API calls;
they do not qualify pretrained weights, native resolution, GPU precision,
checkpointed recomputation, rendering or media quality.
