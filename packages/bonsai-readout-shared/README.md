# Bonsai shared readout profile

This is a separate opt-in distribution. It executes the installed shared
`ConditionalPG094` game, shared routed `HNDLParticleProjection`, shared response
session/inference lifecycle, and an actual HyperGAN-built HNDL vector critic.
It contains no GAN losses, optimizers, penalties or training loop.

**Native Bonsai training remains blocked.** Its real pooled readout is
`[1,5120]`. The pinned core refuses continuous vectors of width >=170 before
fetching readouts, constructing adapters, or contacting a server. A separately
named width16 fixture qualifies software contracts only. There is no truncation,
field reshape, categorical fallback or width override.

The pinned environment is core0.12 commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` (core draft PR148), official PG0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0,
Python >=3.11,<3.15. The entire installed core Python source closure and the
HyperGAN/HNDL clean-build closure are verified. No source overlay is admitted.
The qualified core wheel SHA256 is
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.

Use a fresh environment, from the repository root:

```bash
python3.12 -m venv .venv-bonsai-shared
.venv-bonsai-shared/bin/python -m pip install ./packages/bonsai-readout-shared
```

The project has no dependency on an old core plus an incompatible new-core
extra. Do not install the legacy `hndl` extra (HNDL0.8.2) in this environment.
The wheel installs only the existing Bonsai native backend and the new shared
binding, under their existing `conceptmod.textsliders` import names. It does not
package or import `particle_bridge_gan`, TinyQwen, H3 or other trainers.
The `src` links select those two repository modules without maintaining copies;
the source archive and wheel contain their actual bytes and build independently.

Create a small `rows.json` array of distinct positive/neutral text pairs, then:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

config = SessionConfig("bonsai-readout-fixture", "new-bonsai-software-fixture-v1",
                       rows_file="rows.json", steps=4, batch=2, seed=7,
                       ema_decay=0.9)
with open_response(config) as session:
    session.train(stop_after=2)
    session.checkpoint("training.pt")
with open_response(config, resume="training.pt") as session:
    session.train()
    session.export_inference("head.pt", selection="ema")
# Native-only adapter inference needs neither rows.json nor a game/critic.
with open_inference("head.pt", operation="response") as inference:
    vector = inference.response(text="a plain caption", strength=0.5)
```

For the native admission check, select `bonsai-gguf-shared` with `weights` set
to the pinned `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and `server_url` pointing to an
already owned Prism fork server. The current result is an explicit width5120
refusal. There is no download, server launch or native model execution here.

The architecture remains a torch-side residual on a frozen readout, with
rank8/alpha8, one shared128x4 bank, zero-initialized up matrix and exact strength0
bypass. The HNDL projection is the existing shared router/bridge architecture;
the critic matches shared `ConditionalReadoutCritic(hidden=16)` via
Linear16/SiLU/Linear1 with graph, shapes, plan and verified builder identity in
`extra_repr()`. It is distinct from the historical local critic.

This new profile uses the public `bispf_optimized` recipe and fixed Gaussian
reference from the shared core. It is **not an objective-preserving upgrade** of
the historical Bonsai Rp/annealed-noise/VIC game. Positive-input adaptation and
frozen positive targets remain explicit; the new critic conditions on the frozen
neutral readout. Strengths0/0.5/1 return numerical readouts, not generated tokens
or a modified GGUF model. Old safetensors cannot resume here. Preserve their
original source, dependencies and `train_lora_bonsai_gguf.py` entrypoint.

Required follow-up: centrally reviewed explicit continuous-domain source/profile
admission for width5120, new exact core signature/provenance, and separate real
Prism/PTQ1_0/native-model qualification. A generic modern conditional vector
critic in core can replace this thin graph only under a fresh reviewed source
cohort. None of these follows from CPU fixtures. No GPU, pretrained model,
concept quality, audio, package release, weights publication or deployment is
qualified by this work.

To reproduce the installed-package CPU checks, build the wheel/source archive,
install the wheel with this profile's exact dependencies, and run a copy of
`tests/test_bonsai_shared.py` outside the checkout with `pytest -c /dev/null`.
Set `BONSAI_REFERENCE_ROOT` to the checkout solely for the one matched-weight
comparison against the unmodified historical head. The product import assertion
requires `site-packages`; this prevents pytest's repository `pythonpath` from
silently substituting source for the installed product. The legacy tests run
separately from the repository root using their original entrypoint.
