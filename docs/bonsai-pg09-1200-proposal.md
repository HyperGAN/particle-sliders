# Fixed Bonsai corrected1200-update CPU proposal

**Pending coordinator review and new process admission. No long run has started.**
The four-update canary established execution and replay only; its failed held-out
transfer is retained in the prior receipts. This proposal covers the full archived
1200-update comparison horizon, with one fixed public `bispf_optimized` arm and no
sweep, fallback or automatic extension.

Reviewable recipe: `evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal.json`.
The existing `scripts/verify_bonsai_pg09_cached.py` handles both the old four-step
canary and this explicit source-pinned `--plan`. All losses, optimizers, public
prior/penalty, sampling and training state remain in `ConditionalPG09`; the worker
only dispatches shared updates, diagnostics and persistence. No alternate loop
implements optimizer or loss math.

## Original comparison and corrected protocol

Retain the exact three training pairs in `prompts-bonsai-gguf.yaml`, three archived
held-out garden/sea/street pairs, three existing canaries, seed7, and strengths
0/.5/1. Frozen positive teachers and the same caption contexts use the15 official
cached last-token5120 readouts. Cache identity is
`358f97c219524fd96f9d510824f4bc7f874ad4e44927359e0911a3cb95bde5c7`.
The unchanged native Torch head has rank8/alpha8 and its real128x4 routed bank.
The conditional raw-logit critic remains the fixed canary architecture, hidden16;
it sees complete neutral5120 context plus sign+1/time0, never row IDs.

The corrected student takes the neutral caption at+1 against the frozen positive
teacher at0. The archived failed run trained positive-caption students against
positive teachers; that input bug is not repeated. The old paired-error/VIC
trainer,8000-step anneal and separate EMA are replaced by the canonical public
GAN-only recipe and its1200-step schedule. The recipe has no overrides. Gaussian
sigma and response scale are1, batch2, and D/G use core's private balanced streams.
The fully resolved public recipe is embedded in the proposal and checked against
the actual constructed game before updates. Default public configuration fields
unused by this GAN host do not add objectives: G remains Gadv+required public
prior, D remains Dadv+public continuous penalty. Optimized MSE calls are forbidden.

## Budget and commands

One produce process: **1200 updates,1800s wall-time maximum**. A separate restore
process replays checkpoint1180 through1200: **20 updates,180s maximum**. Total
execution budget is1220 shared updates including verification replay. These are
the only two proposed execution commands; exact argument arrays and environment
are embedded in the recipe. No extra canary, sweep, retry or automatic resume.

Both processes hide CUDA, use one Torch intra-op and one inter-op thread, and
set OMP/MKL/OpenBLAS/NumExpr threads to1. Peak process RSS limit is8GiB, with8GiB
host-memory reserve and100GiB free output-filesystem reserve. Startup requires
16GiB available host memory; peak RSS, host reserve, disk reserve and elapsed time
are checked at startup and after each update/evaluation. The external `timeout`
command enforces each wall-time bound. Memory checks occur between shared
operations; no system-wide limit, service or unrelated process is changed.

Output is new `/tmp/pg09-bonsai-gan-1200-20261017/bispf_optimized`.
Existing output refuses overwrite. A failed command stops the sequence; partial
metrics/checkpoints remain for review without a self-retry. This admission request
does not include any GGUF/model load, inference server, GPU, download, package
patch, service stop or permission change.

## Required evidence

- Preserve all1200 per-update metrics, GAN-only equations, actual G/D/bank movement
  and unchanged frozen training responses. Stop on nonfinite state, disconnected
  gradients, source/domain mismatch or resource failure.
- Save a full shared checkpoint every20 updates. Evaluate the original held-outs
  at0/20/100/400/800/1200, with frozen teacher, Off, strengths0/.5/1, canaries and
  positive-input identity diagnostics. Save full zero/+1 vectors and inference
  exports at every evaluation. Evaluation must preserve game state and RNG.
- The independent restore process must match final20 losses, full model/optimizer/
  controller/prior/sampling/RNG state and held-out vectors exactly. Safe weights-only
  checkpoint/export loads and final inference-export vector reload must pass.
- Publish the original numeric transfer gate result whether it passes or fails;
  squared errors remain diagnostic. Completion requires1200 steps, not a pilot.
  The old `exact_ema_replay` comparison is represented by exact canonical live
  checkpoint/export replay because the fixed public recipe has EMA0. Human and
  generation quality remain unqualified for this cached-head-only run.

## Source and software gates

Core is immutable `3e94b13eb5d6cd1ceee3f018f019a68603f35696`, module SHA256
`426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256`.
The single ALZ overlay remains isolated/uninstalled, patch SHA256
`c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571`.
Use only explicit `alz-continuous-v1`/`continuous`; release/auto still rejects wide
vectors. Core's public validator runs before cache payload access.

The proposal pins96 source/metadata files: worker, native head/bridge/backend,
adapter/critic/cache/evaluation/preparation code, prompts, core module/manifest,
all49 reviewed PG sources, all15 metadata/token-cache pairs, model manifest and
archived recipe/final failure metrics. Metadata and tensor sizes were read during
preparation; no real cached tensor payload or large model file was loaded.
Python3.11.15, Torch2.11.0+cu128, NumPy2.4.4, Transformers5.15.0,
Diffusers0.40.0.dev0 and Safetensors0.8.0 are checked before payload access. Core
continues to bind the actual source/profile/domain/runtime during checkpointing.

Six worker CPU tests pass on synthetic caches only: both public presets exercise
the unchanged four-update path and the generalized checkpoint/evaluation/replay
path with a reduced six-update fixture; additional tests reject changed production
budgets and stop on peak RSS before payload access. These do not stand in for the
proposed1200 real-readout run. The production CLI admits only the fixed1200-step
proposal schema and validates its paths, environment, versions and source hashes.

Tiny pretrained execution remains pending separate runtime/model admission.
LTX's independently reviewed full-coordinate mask/hold protocol is retained;
pretrained text-loader/shape/memory/AV render gates and GPU admission remain pending.
H3 remains blocked on native AV protocol, source/asset/license and hardware review.
GPU scheduling remains solely with the coordinator; current MOSS/YuE2 work is not
modified by this CPU proposal.
