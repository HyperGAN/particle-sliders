# Bonsai1200 admitted startup failure

Admission `bonsai-gan-1200-cpu-20261017-v1` was consumed by the first exact command
at frozen source `1d289731c0d8a59d91c7f5298565c9a0aa6f706b`, worker SHA256
`f775c857c413348ac86478f3392c49e4f4fc570b9a847bc8b8860db559cde8ad`, recipe SHA256
`7854859e3ae028f8e91326846b85eac66819a34d77c7ce516c100b83c2c4be1e`.

Producer exited1 after0.134 seconds in the startup resource check:
`OSError: output filesystem reserve unavailable`. This happened before Torch
import, cached tensor access, model construction, output run-directory creation,
or any optimizer update. **Completed updates:0.** Restore was never attempted.
There was no retry, source change, budget reduction or training extension.

The proposal selected `/tmp/pg09-bonsai-gan-1200-20261017/bispf_optimized`, but its
100 GiB free-space guard checked the `/tmp` filesystem, which has only30.353 GiB
total and11.476 GiB free. The selected destination could never satisfy that
reserve. This proposal error should have been detected during preparation.
The guard correctly stopped the run. `/mnt/nvme/sliders` measured1974.564 GiB free
on a3665.465 GiB filesystem; no space cleanup or unrelated file change was made.

Preserved evidence: `evidence/pg09-text-video/bonsai-1200-startup-failure/` contains
the exact command/admission receipt, stdout/traceback and filesystem measurements.
Original copies remain at `/tmp/pg09-bonsai-gan-1200-20261017/`. There are no
training checkpoints, held-out metrics or PASS/FAIL transfer results from this
attempt because training did not start. The prior four-update transfer failures
remain historical pilot evidence, not a substitute for this1200-update rerun.

The coordinator and independent QA were notified. An initial coordinator
notification failed host lookup; after checking the recovered host and absence
of that message, a subsequent notification succeeded. No uncertain dispatch is
being used as authorization to rerun.

## Concrete correction for review, not applied or executed

1. Bind the disk reserve to the nearest existing parent of the plan's declared
   output path, instead of hardcoded `/tmp`; retain the100 GiB reserve.
2. Change the recipe output and its exact command arguments to a new isolated
   directory under `/mnt/nvme/sliders/qualification/runs/`.
3. Requalify the changed guard using synthetic tests, regenerate worker/source/
   recipe hashes, freeze the correction, and obtain a **new explicit coordinator
   process admission** before either actual producer or restore command.

The proposed guard delta is saved separately as an unapplied patch. The admitted
worker remains byte-identical. CPU, host-memory, runtime, source/domain, data,
objective,1200-update horizon and replay budgets need no relaxation. The rerun
continues to correct neutral-student input as well as replacing the old objective;
its eventual result must not be described as GAN-only causal evidence.

Tiny/LTX live gates and H3 architecture/license/hardware blockers remain unchanged.
No GPU, GGUF/model, inference server, download, installed-package, service,
permission, publication or merge action occurred.
