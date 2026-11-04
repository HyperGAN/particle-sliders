# One training command, separate native environments

Use the same installed command for every product with a training profile:

```sh
particle-slider train --config path/to/recipe.json
particle-slider train --config path/to/recipe.json --runner modal
```

A raw native recipe finds `training_profile.json` next to itself. An explicit
`--profile /path/to/training_profile.json` overrides that lookup. Products may
also ship a short `./train --config recipe.json` wrapper. The recipe remains
owned and validated by its product. Training does not run inference, LR search,
publication or sharing. Internal teacher forwards and native rollout collection
can be necessary training steps.

The Modal command requires an independently installed private executor and an
explicitly configured remote image profile. The public package contains no Modal
SDK or account configuration. `submitted` means queued work, not completed
training. The private executor decides whether a remote profile is available.

## Preview and validate

```sh
particle-slider plan --config recipe.json
particle-slider train --config recipe.json --check
particle-slider profiles
```

`plan` resolves finite JSON, product profile, inputs, placement and budgets. It
imports no Torch or product model code, allocates no accelerator and does not
create a run. `--check` invokes the selected product's declared validation
operation in its isolated interpreter. Check support is explicit; it must not
load model weights or allocate a GPU. A successful check does not establish
native model quality. Profiles without a check operation can still be planned.

Profiles describe `native`, `fixture`, `source_only`, or `blocked` evidence.
`source_only` is experimental execution, not an approval barrier. The actual
product factory enforces supported operations and prerequisites. `blocked`
profiles refuse before output creation or worker launch, with their concrete
reason. A fixture is never reported as native training.

## Compatible configuration

The historical exact `{"backend": ..., "cohort": ..., "options": ...}`
SessionConfig and its original CLI behavior remain supported. It is not mutated
to contain executor options. Explicit new executor/profile flags select the new
isolated driver. The versioned envelope is optional:

```json
{
  "schema": "particle-slider-train-v1",
  "profile": "training_profile.json",
  "training": {"steps": 128, "neutral": "...", "positive": "..."},
  "execution": {"runner": "local"},
  "output": {"root": "runs/my-concept", "checkpoint_every": 32}
}
```

The `training` object above illustrates a native product recipe. For a session
profile it is the unchanged SessionConfig object. Total steps must be explicit;
`stop-after` is an optional session checkpoint boundary within that fixed budget,
not a new training horizon. A stopped run reports `checkpointed`, not completed.
Defaults create a fresh timestamped directory beside the config. Explicit
`--out` selects an exact fresh directory; nonempty outputs are never overwritten.

```sh
particle-slider train --config recipe.json --out runs/experiment-1
particle-slider train --config recipe.json --resume /path/to/full-checkpoint
particle-slider train --config recipe.json --runtime-python /path/to/product-venv/bin/python
```

`--runtime-python` selects an existing compatible interpreter; it does not install
or repin dependencies. Each product preserves its own core, PG, native library,
model and data identity. The standalone session driver imports that interpreter's
installed `particle_sliders.lifecycle` or `response`; it does not prepend the new
core source. Both current `train(on_step=...)` and earlier `train(stop_after=...)`
APIs use the same native public update implementation. Product wrappers remain
responsible for their native child-process cleanup and GPU leases.

## Product adoption

Place a dependency-free JSON descriptor at the product root:

```json
{
  "schema": "particle-slider-profile-v1",
  "name": "example-native",
  "product": "example-particle-sliders",
  "config_kind": "native",
  "status": "source_only",
  "command": ["{python}", "tools/train_concept.py"],
  "runtime": {"python": "python3"},
  "target_steps_path": "steps",
  "default_checkpoint_every": 32,
  "capabilities": {"resume": true, "periodic_checkpoint": true, "check": true}
}
```

`command` is argv, never shell text. Its working directory is the profile's
directory unless `runtime.cwd` overrides it. Native workers receive `--config`,
`--output`, `--checkpoint-every`, optional `--resume`, and `--check` when requested.
They retain their own trainer and checkpoint format. A completed worker writes
`result.json` with `status`, `completed_steps`, `checkpoint` and `adapter` paths;
both artifacts must exist. It may write `progress.json` with phase/status and
completed steps. Checkpoint directories expose a completion marker
`checkpoints/step-*/checkpoint.json` or `manifest.json` after all state is durable.
Native workers must handle termination and stop/reap their own descendant groups.

For shared lifecycle products replace native `command` with
`"config_kind":"session", "interface":"image", "backend":"example"`.
Use `interface:"response"` for numerical text/audio responses. Do not invent an
image renderer or waveform output for those profiles. Session checkpoint/export
formats and exact source/cohort resume checks remain native to the selected core.

Optional installed discovery uses entrypoint group
`particle_sliders.training_profiles`; entries return a JSON descriptor from a
stdlib-only module. File profiles avoid co-installing conflicting native products.

## Results and inputs

A controller run writes `request.json`, canonical `training.json`, `train.log`,
`progress.json`, `result.json` and the product's `artifacts/` directory. Session
workers checkpoint every requested interval after successful updates, retaining
model, optimizer, EMA, RNG and step state through the native checkpoint API.
Final completion requires the requested update count and readable checkpoint and
adapter paths; deeper format validation remains product-owned. Failures and
cancellation preserve earlier checkpoints and record a terminal state. Linux
workers are bound to the controller lifetime; normal cancellation stops/reaps
only the owned worker group. Native wrappers must also own their descendants.

Declare portable files explicitly with `input_files` entries containing `name`
and `local_path`, referring to them in training as `input://name`. Resolution
records sizes/SHA256; execution verifies them again before mapping to paths.
Local paths, output directories and GPU indices are transport. The semantic hash
covers the recipe, product descriptor and input content identities. Runtime
identity is what the profile explicitly declares beyond placement python/cwd/
device; this hash is not a substitute for product-native source/model checks.
Large model directories are not implicitly uploaded or scanned. Remote profiles
must declare their own model cache and staging policy.

Third-party runners register `particle_sliders.runners` with an object exposing
`submit(request)`, `status(handle)`, `logs(handle)`, `cancel(handle)`, and
`fetch(handle, destination)`. The worker API is
`execute_resolved_request(request, on_checkpoint=None, on_progress=None)` and runs
the frozen request directly without redispatching to its remote executor.
