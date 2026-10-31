# Bonsai shared identity/readout-only preview

This is a separate opt-in distribution. It executes the installed shared
`ConditionalPG094` game, shared routed `HNDLParticleProjection`, shared response
session/inference lifecycle, and an actual HyperGAN-built HNDL vector critic.
It contains no GAN losses, optimizers, penalties or training loop.

**This preserves a positive-input identity/readout experiment, not demonstrated
neutral-to-positive concept learning.** `prepare()` captures the frozen positive
readout as `target`, and `generated()` sends that same `prepared.target` into the
residual head. With a zero up matrix, the output already equals the target and
the paired response error is exactly zero before any update. The neutral readout
conditions the critic; it is not the generator input in this path. Parameter
updates, successful resume, derivative parity and an inference export are
software evidence only, and do not demonstrate learning a concept edit.

The historical Bonsai implementation uses the same positive-input placement.
A separately named neutral-to-target path would change the task and remains
held for the user's decision. This preview changes neither inputs nor game
math to create that path.

**Native Bonsai training remains blocked.** Its real pooled readout is
`[1,5120]`. The pinned core refuses continuous vectors of width >=170 before
fetching readouts, constructing adapters, or contacting a server. A separately
named width16 fixture qualifies software contracts only. There is no truncation,
field reshape, categorical fallback or width override.

The pinned environment is core0.12 commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` (core draft PR148), official PG0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0,
Python >=3.11,<3.15. Before core entrypoint imports, cooperative checks compare
installed source files, executable bytecode caches and import origins. The
shared builder separately checks its on-disk source/clean-build record. These
checks reject ordinary path shadows and stale executable caches; they do not
attest arbitrary previously executed or later monkeypatched in-memory code and
are not a Python security boundary. No source overlay is admitted.
The archived core wheel SHA256 below is an informational reference, not a wheel
admission check; fresh VCS builds can have different ZIP timestamps:
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.

Use a fresh environment, from the repository root:

```bash
python3.12 -m venv .venv-bonsai-shared
.venv-bonsai-shared/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' './packages/bonsai-readout-shared[test]'
```

The project has no dependency on an old core plus an incompatible new-core
extra. Do not install the legacy `hndl` extra (HNDL0.8.2) in this environment.
The wheel owns only the `bonsai_readout_shared` namespace. It does not install a
`conceptmod` package that could hide a sibling preview. The small build backend
materializes the unchanged historical Bonsai loader into `_native.py`, checking
its pinned source hash. The self-contained source archive includes those actual
bytes. No source symlinks are required, including in a `core.symlinks=false`
checkout. Neither the legacy game, TinyQwen nor H3 is packaged.

Create a small `rows.json` array of distinct positive/neutral text pairs, then:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

config = SessionConfig("bonsai-readout-fixture", "bonsai-identity-readout-software-fixture-v1",
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
already owned Prism fork server, plus explicit expected `model_sha256`. The
current result is an explicit width5120 refusal before reading model bytes. The
dormant native identity path checks the full local file content hash, original
filename/size/header and native source identity, and inference compares that
identity too. A future enabled path also needs independent proof that the server
is serving those same bytes; the local file hash does not attest a remote server.
There is no download, server launch or native model execution here.

The identity/readout architecture remains a torch-side residual on a frozen readout, with
rank8/alpha8, one shared128x4 bank, zero-initialized up matrix and exact strength0
bypass. The HNDL projection is the existing shared router/bridge architecture;
the critic matches shared `ConditionalReadoutCritic(hidden=16)` via
Linear16/SiLU/Linear1 with graph, shapes, plan and verified builder identity in
`extra_repr()`. It is distinct from the historical local critic.

This new profile uses the public `bispf_optimized` recipe and fixed Gaussian
reference from the shared core. It is **not an objective-preserving upgrade** of
the historical Bonsai Rp/annealed-noise/VIC game. Positive-input identity/readout
placement and frozen positive targets remain explicit; the new critic conditions
on the frozen neutral readout. Strengths0/0.5/1 return numerical readouts, not generated tokens
or a modified GGUF model. Old safetensors cannot resume here. Preserve their
original source, dependencies and `train_lora_bonsai_gguf.py` entrypoint.
Artifacts from earlier draft sources, including2593452 andcb3d42c, retain their
original factory and class/source identity. These claim corrections do not
silently rebind or relabel saved artifacts.

Required follow-up: centrally reviewed explicit continuous-domain source/profile
admission for width5120, new exact core signature/provenance, and separate real
Prism/PTQ1_0/native-model qualification. A generic modern conditional vector
critic in core can replace this thin graph only under a fresh reviewed source
cohort. None of these follows from CPU fixtures. No GPU, pretrained model,
concept quality, audio, package release, weights publication or deployment is
qualified by this work.

To reproduce the installed-package CPU checks, build the wheel/source archive,
install the wheel with this profile's exact dependencies, and copy
`packages/bonsai-readout-shared/installed_tests` outside the checkout. Run:

```bash
BONSAI_INSTALLED_TESTS=1 BONSAI_REFERENCE_ROOT="$PWD" .venv-bonsai-shared/bin/python -m pytest -q -c /dev/null /tmp/bonsai-installed-tests
```

`BONSAI_INSTALLED_TESTS=1` selects this optional installed CPU fixture lane. It
is outside the default root testpaths and skips when unselected. A missing
`BONSAI_REFERENCE_ROOT` skips only optional historical architecture/packaging
comparisons. The installed product import assertion requires `site-packages`.
Direct factory calls fork/restore the caller's CPU RNG on success and failure.
The original legacy tests run separately from the repository root:

```bash
python -m pytest -q -rs -c /dev/null tests/test_bonsai_gguf_slider.py
```

`qualification/legacy-test-paths.json` lists the exact22 CPU and two opt-in live
node IDs. Public Git/raw blob byte receipts prove source publication, not runtime
builder execution; those are separate evidence lanes.
