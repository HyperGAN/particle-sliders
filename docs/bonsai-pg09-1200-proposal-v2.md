# Bonsai1200 NVMe v2 proposal — executed; transfer failed

Historical proposal: root subsequently admitted and completed its exact commands.
All 1200 updates and the separate 20-update replay passed execution checks;
held-out transfer failed. See [bonsai-pg09-1200-result-v2.md](bonsai-pg09-1200-result-v2.md).
This consumed admission does not authorize any repeat or extension.

The failed v1 attempt is preserved in commit
`192696a69dd6e113e701782f7a98425eac4de94f`: zero updates, no Torch or tensor
payload access, no restore/retry. It selected an unsuitable `/tmp` filesystem;
the 100 GiB reserve correctly rejected startup. No gate is lowered in v2.

Root requested this new proposal and exact destination:
`/mnt/nvme/sliders/qualification/runs/bonsai-pg09-gan-1200-20261017-v2/bispf_optimized`.
Recipe and exact argument arrays are in
`evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal-v2.json`.

Worker SHA256: `a83774aeeb3a17c117ccd4febac35b24d62b2914256d66f3fc1c8f214b464ed6`.
Recipe SHA256: `3c49f2c4c8c7f628dd2cf23b56d50022428349cc635a76c0cc9c314181b051aa`.
All 96 source/metadata pins remain; only the worker source pin changed.
The former recipe and receipts are retained.

## Exact correction

The resource guard resolves the actual plan output, including symlinks, and finds
its nearest existing directory. `/proc/self/mountinfo` plus that directory's
`st_dev` must identify mount `/mnt/nvme` on **`/dev/nvme0n1p2`**. Missing or mismatched
device/mount rejects; there is no fallback. Free space is checked on that actual
output filesystem, retaining the 100 GiB floor. The shared startup resource hook
is used by metadata-only preflight, producer and restore before Torch import or
cached tensor access; update/evaluation resource checks continue throughout.

The real resource metadata preflight passed with `torch_imported=false`,
`tensor_payload_loaded=false`, the expected NVMe device/mount and existing parent
`/mnt/nvme/sliders/qualification/runs`. Exact free-space/available-memory/RSS
measurements and all limits are preserved in
`evidence/pg09-text-video/bonsai-1200-v2-resource-preflight.json`.
No output training directory or producer/restore process was created during it.

Eight CPU tests pass on synthetic data only, covering the four-step and reduced
six-step worker paths, checkpoint/evaluation/replay, fixed production horizon,
memory rejection, actual existing-parent/symlink disk selection, retained100 GiB
reserve, and rejection of an incorrect device even with ample free space.

## Unchanged protocol and budget

One fixed public `bispf_optimized` recipe without overrides or sweep, immutable
core `3e94b13` and the single reviewed `alz-continuous-v1`/`continuous` profile.
Exactly 1200 updates (1800s cap), then a separate 1180→1200 replay (180s cap).
One intra/inter-op/BLAS thread, 8 GiB process peak RSS, 8 GiB host reserve
(16 GiB available required at startup), 100 GiB disk reserve. No retries,
extensions, GGUF/model loading, server, GPU, download or package changes.

Preserve the same three training pairs, three held-out pairs, three canaries,
seed 7, strengths 0/.5/1, frozen positive teachers, full neutral 5120+sign/time
conditioning, native rank 8/alpha 8 head and actual 128×4 routed bank. G and D updates,
frozen responses, checkpoint/export and held-out gates are checked as described in
`bonsai-pg09-1200-proposal.md`: checkpoint every 20, evaluations at 0/20/100/400/800/1200,
and exact independent-process final 20 state/loss/vector/export replay.

This rerun also corrects the archived positive-input student bug by using a live
neutral student against the frozen positive teacher. Eventual PASS/FAIL results
must report both changes and cannot establish GAN-only causality. Readout transfer
is reported whether it passes or fails; generation/human quality stays separate.
The four-step pilot is not a substitute for completing 1200 updates.

Actual v2 execution is **pending new exact root admission**. Tiny/LTX live gates,
H3 architecture/license/hardware blockers and sole coordinator process/GPU
ownership remain unchanged. MOSS/YuE2 jobs and all resident services are untouched.
