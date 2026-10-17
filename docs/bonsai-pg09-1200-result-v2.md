# Bonsai corrected 1200-update result: execution/replay PASS, transfer FAIL

Admission `bonsai-gan-1200-nvme-cpu-20261017-v2` completed the exact two sequential
commands at frozen source `1865a07ae9e147f2bb3a1178a0dcf7587036cd0e`.
The producer completed **all 1200 updates**; the separate process restored
checkpoint 1180 and replayed the final 20. Exit codes were 0/0, wall times 22.012s
and 2.552s (budgets 1800s/180s), peak process RSS 0.6192/0.6129 GiB (cap 8 GiB).
There was no retry, source change, sweep or extension.

Worker SHA256: `a83774aeeb3a17c117ccd4febac35b24d62b2914256d66f3fc1c8f214b464ed6`.
Recipe SHA256: `3c49f2c4c8c7f628dd2cf23b56d50022428349cc635a76c0cc9c314181b051aa`.
All 96 source/metadata hashes matched again after execution. Core remained
`3e94b13eb5d6cd1ceee3f018f019a68603f35696`, using only the reviewed isolated ALZ
continuous profile. No installed package was changed.

## Completed protocol and replay evidence

- Fixed public `bispf_optimized`, 1200-step horizon, no recipe overrides; original
  three training pairs, three held-outs, three canaries, seed 7 and strengths 0/.5/1.
- Native 5120 frozen readouts and unchanged rank 8/alpha 8 head/128×4 routed bank.
  Fixed critic conditions retain complete neutral context plus sign/time.
  G,D and the actual bank changed; frozen responses stayed unchanged.
- All 1200 losses were finite and satisfied the shared GAN-only equations.
  Minimum D penalty was 1.31748e-10, positive throughout. No optimized MSE or
  legacy training loop was executed.
- All 60 checkpoints at 20-step intervals were preserved. Original evaluations,
  complete zero/+1 readout vectors and inference exports were saved at
  steps 0/20/100/400/800/1200. Evaluation preserved training state and RNG.
- The separate final 20-update replay matched full parameters/optimizers/controller/
  prior/sampling/RNG state, losses, held-out vectors and final inference-export
  reload exactly. Safe weights-only loading passed; scale 0 remained exact.

## Unchanged held-out transfer gate

The full run's numeric transfer gate **failed at every evaluation stage**.

| Step | Mean strength+1 relative-edit MSE | Successful cases | Canary relative RMS | Positive-input identity MSE | Gate |
|---|---:|---:|---:|---:|---|
|0|1.000000|0/3|0.000000|0.000000|FAIL|
|20|281.768342|0/3|16.459875|39.723968|FAIL|
|100|231.940786|0/3|14.947490|32.743273|FAIL|
|400|2.222796|0/3|1.033094|0.156325|FAIL|
|800|1.452105|0/3|0.883882|0.114471|FAIL|
|1200|1.943358|0/3|1.276340|0.238356|FAIL|

Final garden/sea/street strength+1 relative-edit MSEs were
**2.108203/1.944979/1.776892**; edit cosines were 0.367447/0.187661/0.291124.
All three fail the per-case MSE≤0.9 requirement, so successes are 0/3 (required ≥2).
Mean MSE 1.943358 fails ≤0.9; canary RMS 1.276340 fails ≤0.25; positive-input identity
MSE 0.238356 fails ≤0.01. Exact zero passes. Diagnostic squared errors were never
optimized. These are complete held-out results, not a convergence or quality claim.

The archived failed 1200-update run had mean MSE 1.000205, 0/3 successes,
canary RMS 0.006394 and identity MSE 4.94830e-6; it also failed transfer. This corrected
rerun feeds **neutral** students against frozen positive teachers, while the
archived run fed positive-caption students. The objective/public recipe and input
protocol both changed. The comparison therefore **does not isolate GAN-vs-MSE
causality**. No generation, in-model GGUF adaptation or human-quality qualification
is claimed. Completing the declared 1200-update horizon did not produce useful
transfer on the fixed gate.

## Frozen artifacts and remaining work

All 87 artifacts (67.73 MiB) remain at
`/mnt/nvme/sliders/qualification/runs/bonsai-pg09-gan-1200-20261017-v2/`.
Code evidence under `evidence/pg09-text-video/bonsai-1200-v2-actual/` contains
the admission/command logs, full producer/restore receipts, all per-update JSONL
metrics, six evaluation-metric files, gate summary and SHA256 artifact index.
The native tensor checkpoints/vectors/exports remain on NVMe, indexed by hash.
The original zero-update `/tmp` failure remains untouched and frozen at 192696a.

Only the already existing cached official readouts were loaded: no GGUF/model
weights, inference server, CUDA context, download, package installation, service,
permission, publication or merge action occurred. Output/device and resource
guards stayed in force. MOSS/YuE2 jobs were untouched.

Core owner and QA received the full-horizon failure and replay results. No further
training or diagnostic model process is admitted. Any next Bonsai arm needs a new
concrete reviewed protocol/admission; no fallback or automatic sweep follows.
Tiny and LTX pretrained/runtime/fit/render gates and H3 native AV/license/hardware
blockers remain as recorded in `text-video-pg09-current-status.md`.
