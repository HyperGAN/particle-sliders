# HyperGAN develop reporting

The optional reporter uses the actual observation implementation from HyperGAN
develop commit `5f368a3770cf5bdae0448995c9f76e29af8049f7`. It verifies the full Python
source digest before importing `ObservationIO`, `Throughput`, the event reader,
and the canonical summary writer. It runs separately with CUDA hidden and never
imports Torch or the model. Training code, weights, checkpoints and native logs
remain unchanged.

Configure the host once with JSON at
`${XDG_CONFIG_HOME:-~/.config}/particle-slider/reporting.json`, or point
`PARTICLE_SLIDER_REPORTING_CONFIG` to a JSON file:

```json
{
  "kind": "hypergan-develop",
  "runtime_python": "/path/to/reporting-environment/bin/python",
  "runtime_path": "/path/to/pinned/parent-of-hypergan-package",
  "runtime_commit": "5f368a3770cf5bdae0448995c9f76e29af8049f7",
  "runtime_sha256": "9c725516409bfbb80f9740c65a435c0b7219be55d0ca88e2c777dc76aea2cb1e"
}
```

The reporting environment needs that exact HyperGAN source and its `foldlitedb`
dependency. It does not need the product's training environment. The shared CLI
checks reporting before launching the model and waits for the reporter's `ready`
receipt. Configuration precedence is explicit `--reporting-config FILE`, the
request envelope, the product profile, then the host setting. `--no-reporting`
disables it. Reporting configuration is separate from training semantics.

For a controller already running with an older frontend, attach directly:

```sh
CUDA_VISIBLE_DEVICES='' /path/to/reporting-environment/bin/python \
  /path/to/particle_sliders/execution/reporting.py \
  --run /path/to/controller-output --follow \
  --runtime-path /path/to/pinned/parent-of-hypergan-package \
  --runtime-commit 5f368a3770cf5bdae0448995c9f76e29af8049f7 \
  --runtime-sha256 9c725516409bfbb80f9740c65a435c0b7219be55d0ca88e2c777dc76aea2cb1e
```

Add `--check` instead of `--follow` to validate the reporter without an existing
run or GPU allocation. The `--run` path must name a shared controller output
containing `request.json` when actually attaching. The reporter backfills
complete native metric rows, preserves incomplete tails, and continues until
the controller writes a terminal `result.json`. Reattaching resumes from the
last recorded source position without repeating training observations.

Reports live under `RUN/reporting/hypergan/`: HyperGAN `events.jsonl`, immutable
metric catalogs, `manifest.json`, `summary.json`, and `reporter-state.json`.
Use HyperGAN's `events` and `read_run_info` APIs against that directory. The
reporter state distinguishes observed metric steps from the highest verified
durable checkpoint. Native checkpoint references appear as external metadata;
the reporter does not copy or reinterpret their tensors. A checkpoint's stored
hash is verified before it advances the durable frontier.

Only supplied finite scalars are published. Missing losses, sample counts and
durations are not invented. Throughput uses HyperGAN's implementation over
recorded native update durations; their sum is labeled measured update time,
not total job wall time. HyperGAN's numerical configuration and training
controller are not constructed. This is an observation-only run: HyperGAN
checkpoint, stop, resume and sampling controls do not operate the native trainer.

CPU tests exercise real pinned HyperGAN event/catalog/summary readers, backfill,
partial tails, reattachment, duplicate prevention, checkpoint integrity and
reporter shutdown. They do not qualify model quality or GPU execution.
